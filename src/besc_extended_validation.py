#!/usr/bin/env python3
"""BESC extended reproducibility study (no claims of journal-ready novelty).

Uses same mandatory edge protection for all methods, equal budgets, independent
real Zachary karate-club network as a small real-world structural test,
and directed SBM synthetic replicas. The SNAP experiment runs when the exact
Stanford files are supplied locally. Neither SNAP data nor final claims are
fabricated. Requires numpy, scipy, pandas, networkx, matplotlib.

Several comparators are approximations inspired by the literature, NOT exact
reproductions of Gionis et al., L-Spar, or Local Degree algorithms.
"""
from __future__ import annotations
import argparse
import heapq
import math
import random
import statistics
import time
from collections import Counter,defaultdict
from pathlib import Path
import sys
import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parent))
from besc_prototype import (synthetic,load_snap,jsd,sorted_edges,protected_edges,
                            sparsify_besc,sparsify_baseline,diagnostic,graph_from_edges,edge_budget)


def generate_protection(g, labels, seed=0, randomized=False):
    if not randomized:
        return protected_edges(g,labels)
    rng=random.Random(seed)
    edges=sorted_edges(g);rng.shuffle(edges)
    forest=set();union=nx.utils.UnionFind(g.nodes())
    for u,v in edges:
        if union[u] != union[v]:
            union.union(u,v);forest.add((u,v))
    pairs={};sources={}
    for u,v in edges:
        pairs.setdefault((labels[u],labels[v]),(u,v))
        sources.setdefault(u,(u,v))
    return forest | set(pairs.values()) | set(sources.values())


def comparator(g,labels,rho,kind,prot,seed):
    if kind in ('random','degree'):
        return sparsify_baseline(g,labels,rho,kind,protected=prot,seed=seed)
    edges=sorted_edges(g)
    target=edge_budget(g,rho,prot)
    remainder=sorted(set(edges)-set(prot))
    if kind == 'jaccard_global':
        neighbors={u:set(g.predecessors(u))|set(g.successors(u)) for u in g.nodes()}
        def score(e):
            a,b=e
            un=neighbors[a] | neighbors[b]
            return len(neighbors[a]&neighbors[b])/len(un) if un else 0.
        rng=random.Random(seed)
        rng.shuffle(remainder)  # random tie breaks
        remainder.sort(key=score,reverse=True)
        return graph_from_edges(g,set(prot)|set(remainder[:target-len(prot)]))
    if kind == 'local_degree_global':
        rng=random.Random(seed);rng.shuffle(remainder)
        remainder.sort(key=lambda e: g.degree(e[1]),reverse=True)
        return graph_from_edges(g,set(prot)|set(remainder[:target-len(prot)]))
    if kind != 'community_quota':
        raise ValueError(kind)
    # Greedy additive proportional quota comparator, distinct from JSD objective.
    m=len(edges);original=Counter((u,labels[v]) for u,v in edges)
    retained=Counter((u,labels[v]) for u,v in prot)
    d0=Counter(u for u,v in edges)
    dh=Counter(u for u,v in prot)
    avail=defaultdict(list)
    rng=random.Random(seed);rng.shuffle(remainder)
    for u,v in remainder:avail[(u,labels[v])].append((u,v))
    ver=Counter();queue=[]
    active=defaultdict(list)
    for key in original:active[key[0]].append(key)
    def marginal(key):
        u,c=key
        n=retained[key];old=original[key];d=d0[u];t=rho*old;td=rho*d
        # normalize group and source quota terms so groups with many links
        # do not trivially overpower smaller sources
        cost_group=(2*(n-t)+1)/(d*d)
        cost_source=(2*(dh[u]-td)+1)/(d*d)
        return (d/m)*(cost_group+0.5*cost_source)
    def push(u):
        for key in active[u]:
            if avail[key]:heapq.heappush(queue,(marginal(key),u,key[1],ver[u]))
    for u in sorted(g.nodes()):push(u)
    chosen=set(prot)
    while len(chosen)<target:
        if not queue:raise RuntimeError('No feasible addable edges available')
        _,u,c,v=heapq.heappop(queue)
        key=(u,c)
        if v!=ver[u] or not avail[key]:continue
        edge=avail[key].pop();chosen.add(edge)
        retained[key]+=1;dh[u]+=1;ver[u]+=1
        push(u)
    return graph_from_edges(g,chosen)


def extra_metrics(g,h,labels):
    before={u:sum(labels[v]!=labels[u] for v in g.successors(u)) for u in g.nodes()}
    after={u:sum(labels[v]!=labels[u] for v in h.successors(u)) for u in g.nodes()}
    # independent downstream: top-20% node overlap for cross-group bridging role
    k=max(1,math.ceil(g.number_of_nodes()*.2))
    top_orig=set(sorted(g.nodes(),key=lambda u:(-before[u],u))[:k])
    top_new=set(sorted(g.nodes(),key=lambda u:(-after[u],u))[:k])
    top_overlap=len(top_orig&top_new)/k
    # Distortion in number of cross-community links, after scaling for budget
    rho=h.number_of_edges()/g.number_of_edges()
    orig_cross=sum(before.values());new_cross=sum(after.values())
    cross_error=abs(new_cross-rho*orig_cross)/max(1,rho*orig_cross)
    # Undirected neighborhood clustering and largest strongly connected component
    # are genuinely out-of-objective topological diagnostics.
    original_undirected=g.to_undirected();new_undirected=h.to_undirected()
    cluster_orig=nx.average_clustering(original_undirected)
    cluster_new=nx.average_clustering(new_undirected)
    orig_scc=max(map(len,nx.strongly_connected_components(g)),default=0)
    new_scc=max(map(len,nx.strongly_connected_components(h)),default=0)
    return {'cross_group_fractional_error':cross_error,
            'cross_group_top20_overlap':top_overlap,
            'clustering_absolute_error':abs(cluster_orig-cluster_new),
            'largest_scc_retention':new_scc/max(1,orig_scc)}


def graph_case(kind,seed):
    if kind=='assortative':return synthetic(seed,p_in=.15,p_out=.01)
    if kind=='mixed':return synthetic(seed,p_in=.11,p_out=.035)
    if kind=='karate':
        g0=nx.karate_club_graph();g=nx.DiGraph();g.add_nodes_from(g0.nodes())
        for u,v in g0.edges():g.add_edge(u,v);g.add_edge(v,u)
        return g,{u:g0.nodes[u]['club'] for u in g.nodes()}
    raise ValueError(kind)


def run_case(g,labels,name,seed,rho,protection_seed,alpha=0.75):
    prot=generate_protection(g,labels,seed=protection_seed,randomized=True)
    if math.ceil(rho*g.number_of_edges()) <len(prot):return [],len(prot)
    methods=('BESC','BESC_entropy_only','random','degree','local_degree_global',
             'jaccard_global','community_quota')
    all_rows=[]
    for method in methods:
        t=time.perf_counter()
        if method=='BESC':h=sparsify_besc(g,labels,rho,alpha=alpha,protected=prot)
        elif method=='BESC_entropy_only':h=sparsify_besc(g,labels,rho,alpha=1,protected=prot)
        else:h=comparator(g,labels,rho,method,prot,seed+1110)
        duration=time.perf_counter()-t
        d=diagnostic(g,h,labels,rho)
        ex=extra_metrics(g,h,labels)
        if not d['weak_components_equal'] or not d['outgoing_coverage'] or d['channel_coverage']!=1:
            raise AssertionError((name,seed,method,rho,'Constraint failed'))
        assert h.number_of_edges()==math.ceil(rho*g.number_of_edges())
        assert set(h.edges()).issubset(set(g.edges()))
        all_rows.append(dict(family=name,seed=seed,rho=rho,protection_seed=protection_seed,
                             method=method,vertices=g.number_of_nodes(),
                             initial_edges=g.number_of_edges(),protected=len(prot),
                             output_edges=h.number_of_edges(),runtime_sec=duration,
                             **d,**ex))
    return all_rows,len(prot)


def aggregate(df, out):
    if df.empty:return pd.DataFrame()
    cols=['node_jsd','degree_retention_error','mixing_jsd','mean_seic_drift',
          'cross_group_fractional_error','cross_group_top20_overlap',
          'clustering_absolute_error','largest_scc_retention','runtime_sec']
    agg=df.groupby(['family','rho','method'],sort=True)[cols].agg(['mean','std']).reset_index()
    agg.columns=['_'.join(a).rstrip('_') for a in agg.columns.to_flat_index()]
    agg['n_runs']=df.groupby(['family','rho','method'],sort=True).size().values
    agg.to_csv(out/'extended_summary.csv',index=False,float_format='%.8f')
    return agg


def paired_bootstrap(df,out,replicates=3000):
    results=[]
    rng=np.random.default_rng(20301)
    for (family,rho),block in df.groupby(['family','rho']):
        besc=block[block.method=='BESC'].set_index('seed')
        for method in sorted(set(block.method)-{'BESC'}):
            other=block[block.method==method].set_index('seed')
            idx=besc.index.intersection(other.index)
            if not len(idx):continue
            diffs=(other.loc[idx,'node_jsd']-besc.loc[idx,'node_jsd']).to_numpy(dtype=float)
            draws=rng.choice(diffs,size=(replicates,len(diffs)),replace=True).mean(axis=1)
            results.append(dict(family=family,rho=rho,baseline=method,n_pairs=len(idx),
                                mean_jsd_improvement=float(diffs.mean()),
                                ci95_low=float(np.quantile(draws,.025)),
                                ci95_high=float(np.quantile(draws,.975))))
    result=pd.DataFrame(results)
    result.to_csv(out/'paired_bootstrap.csv',index=False,float_format='%.8f')
    return result


def plots(summary,out):
    for family in sorted(summary.family.unique()):
        sub=summary[summary.family==family]
        if sub.empty:continue
        fig,ax=plt.subplots(figsize=(9,5.3))
        for method,dfm in sub.groupby('method'):
            xx=dfm.sort_values('rho')
            ax.plot(xx.rho,xx.node_jsd_mean,marker='o',label=method)
        ax.set_xlabel('Retained fraction of directed edges')
        ax.set_ylabel('Weighted node-level JSD (bits; lower is better)')
        ax.set_title(f'{family}: communication profile distortion')
        ax.grid(alpha=.25)
        ax.legend(ncol=2,fontsize=8)
        fig.tight_layout();fig.savefig(out/f'{family}_extended_jsd.png',dpi=170);plt.close(fig)


def alpha_sweep(out, n_replicas=8):
    # Probe sensitivity, not post-hoc select parameter based on test data.
    cases=[]
    for fam in ('assortative','mixed'):
        for i in range(n_replicas):
            g,labels=graph_case(fam,2026+i)
            p=generate_protection(g,labels,seed=550+i,randomized=True)
            for alpha in (0.25,.5,.75,1.):
                rho=.5
                h=sparsify_besc(g,labels,rho,alpha,protected=p)
                d=diagnostic(g,h,labels,rho)
                cases.append(dict(family=fam,seed=i,alpha=alpha,**d))
    pd.DataFrame(cases).to_csv(out/'alpha_sensitivity.csv',index=False,float_format='%.8f')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='extended_results')
    p.add_argument('--seeds',type=int,default=30)
    p.add_argument('--snap-dir')
    p.add_argument('--karate-only',action='store_true')
    p.add_argument('--alpha-sweep',action='store_true')
    p.add_argument('--rho',nargs='+',type=float,default=[.7,.5,.3])
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    dataset_names=['karate'] if args.karate_only else (['snap'] if args.snap_dir else ['assortative','mixed','karate'])
    all_rows=[];skips=[]
    for family in dataset_names:
        count=1 if family in ('karate','snap') else args.seeds
        for i in range(count):
            if family=='snap':g,labels=load_snap(args.snap_dir)
            else:g,labels=graph_case(family,2026+i)
            for rho in args.rho:
                rows,prot_count=run_case(g,labels,family,i,rho,protection_seed=550+i)
                all_rows.extend(rows)
                if not rows:skips.append(dict(family=family,seed=i,rho=rho,
                                                required_fraction=prot_count/g.number_of_edges()))
            if (i+1)%10==0:print(f'{family}: {i+1}/{count} completed',flush=True)
        print(f'{family}: completed',flush=True)
    if not all_rows:raise ValueError('No feasible experiments')
    df=pd.DataFrame(all_rows)
    df.to_csv(out/'extended_raw_results.csv',index=False,float_format='%.8f')
    summary=aggregate(df,out)
    paired_bootstrap(df,out)
    pd.DataFrame(skips,columns=['family','seed','rho','required_fraction']).to_csv(out/'infeasible_cases.csv',index=False)
    if not args.karate_only and not args.snap_dir and args.alpha_sweep:alpha_sweep(out)
    plots(summary,out)
    print('Saved',out,'records',len(df),'infeasible',len(skips),flush=True)
    print(summary[['family','rho','method','node_jsd_mean','mixing_jsd_mean','mean_seic_drift_mean','cross_group_top20_overlap_mean']].to_string(index=False))

if __name__=='__main__':main()
