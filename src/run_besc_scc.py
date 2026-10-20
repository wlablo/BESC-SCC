#!/usr/bin/env python3
"""BESC-SCC: SCC-certificate-constrained sparsification, reproducible experiments.

This is a RESEARCH PROTOTYPE, not a verified novel algorithm or a reproduction
of Hamann et al. The SCC certificate is a textbook union of in/out traversals
inside each original strongly connected component; no mathematical novelty is
claimed for that graph-theoretic ingredient.

All methods in the SCC-constrained comparison use the identical protected mask
for each (network, seed); all preserve every original strongly connected
component exactly (the full SCC partition). An unconstrained BESC reference
uses the old weakly-connected protection mask at the same edge budget.

For CollegeMsg, third-party node class labels do NOT exist in SNAP. This script
constructs communities by fixed-seed Louvain from a static undirected projection
and explicitly marks labels as inferred. No CollegeMsg observations claimed
unless the original CollegeMsg.txt.gz is provided separately.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, math, random, time, sys, platform
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr
sys.path.insert(0,str(Path(__file__).resolve().parent))
import besc_prototype as bp
import besc_extended_validation as ev
try:
    from accelerate import jsd_numba
    rng=np.random.default_rng(2042)
    for _ in range(50):
        a=rng.dirichlet(np.ones(5));b=rng.dirichlet(np.ones(5))
        assert abs(jsd_numba(a,b)-bp.jsd(a,b))<1e-12
    bp.jsd=jsd_numba
except ImportError:
    pass


def scc_certificate(g:nx.DiGraph,seed:int):
    """Return a subgraph containing an in- and an out-arborescence per SCC.

    Every original SCC remains strongly connected: for x,y in SCC, x -> root
    follows in-tree, then root -> y follows out-tree. Since only edges are
    deleted, no two distinct original SCCs can be merged. Thus the SCC
    partition of any supergraph containing this certificate is identical to
    the input SCC partition. Complexity O(|V|+|E|) expected using BFS.
    """
    rng=random.Random(seed)
    fixed=set()
    sizes=[]
    comps=sorted(nx.strongly_connected_components(g),key=lambda c:(min(c),len(c)))
    for scc in comps:
        if len(scc)==1:continue
        sizes.append(len(scc))
        sub=g.subgraph(scc)
        roots=sorted(scc)
        root=roots[rng.randrange(len(roots))]
        # Randomize traversal order reproducibly; no orientation is changed.
        order={v:rng.random() for v in scc}
        def sort_neighbors(items):return iter(sorted(items,key=lambda v:(order[v],v)))
        out_edges=list(nx.bfs_edges(sub,root,sort_neighbors=sort_neighbors))
        in_edges=list(nx.bfs_edges(sub.reverse(copy=False),root,sort_neighbors=sort_neighbors))
        if len(out_edges)!=len(scc)-1 or len(in_edges)!=len(scc)-1:
            raise AssertionError('Found a non-traversable SCC')
        fixed.update(out_edges)
        fixed.update((v,u) for u,v in in_edges)
    return fixed


def scc_protection(g,labels,seed):
    """Protect SCC partition, weak partition, source activity, ordered channels."""
    core=scc_certificate(g,seed=100000+seed)
    standard=ev.generate_protection(g,labels,seed=500000+seed,randomized=True)
    result=core|standard
    if len(result)>g.number_of_edges():raise AssertionError('Protected mask invalid')
    h=bp.graph_from_edges(g,result)
    assert same_scc_partition(g,h)
    return result,{'scc_certificate_edges':len(core),'legacy_protected_edges':len(standard),'union_protected_edges':len(result)}


def same_scc_partition(g,h):
    # Exact membership equivalence, not just a largest-SCC size comparison.
    return {frozenset(c) for c in nx.strongly_connected_components(g)} == \
           {frozenset(c) for c in nx.strongly_connected_components(h)}


def load_polblogs(path):
    raw=nx.read_gml(path)
    order=sorted(raw.nodes())
    node_to_id={a:i for i,a in enumerate(order)}
    g=nx.DiGraph()
    g.add_nodes_from(range(len(order)))
    g.add_edges_from((node_to_id[u],node_to_id[v]) for u,v in raw.edges() if u!=v)
    groups={node_to_id[x]:int(raw.nodes[x]['value']) for x in order}
    return g,groups,{'raw_edges':raw.number_of_edges(),'self_loops_removed':nx.number_of_selfloops(raw),'label_origin':'observed blog class metadata','sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def load_college(path,seed=17):
    """Static directed CollegeMsg graph, duplicate temporal arcs collapsed.

    The temporal network contains 59,835 events but 20,296 static arcs.
    Labels created by deterministic Louvain on the undirected static graph.
    """
    path=Path(path)
    o=gzip.open(path,'rt') if path.suffix=='.gz' else path.open('r')
    rows=[]
    with o as f:
        for row in f:
            if row.strip() and not row.startswith('#'):
                src,dst,*_=row.strip().split()
                rows.append((int(src),int(dst)))
    g=nx.DiGraph()
    g.add_edges_from((u,v) for u,v in rows if u!=v)
    # Source code BESC uses 0..n-1 array indexing; normalize arbitrary SNAP IDs.
    g=nx.convert_node_labels_to_integers(g,ordering='sorted')
    if g.number_of_nodes()!=1899: print('WARNING CollegeMsg nodes unexpected:',g.number_of_nodes(),file=sys.stderr)
    groups=nx.community.louvain_communities(g.to_undirected(),seed=seed)
    labels={u:k for k,c in enumerate(groups) for u in c}
    return g,labels,{'temporal_records':len(rows),'static_arcs':g.number_of_edges(),
        'self_loops_in_raw':sum(u==v for u,v in rows),'label_origin':f'Louvain inferred, undirected projected, seed={seed}',
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def load_case(name,inputdir):
    if name=='email-Eu-core':
        g,l=bp.load_snap(inputdir)
        m={'label_origin':'original observed organizational department labels',
          'edge_file_sha256':hashlib.sha256((inputdir/'email-Eu-core.txt.gz').read_bytes()).hexdigest(),
          'label_file_sha256':hashlib.sha256((inputdir/'email-Eu-core-department-labels.txt.gz').read_bytes()).hexdigest()}
    elif name=='Polblogs':g,l,m=load_polblogs(inputdir/'polblogs.gml')
    elif name=='CollegeMsg':g,l,m=load_college(inputdir/'CollegeMsg.txt.gz')
    else:raise ValueError(name)
    return g,l,m


def extra(g,h,labels,orig_scc):
    original_scc_largest=max(map(len,orig_scc))
    new_scc=list(nx.strongly_connected_components(h))
    # Original SCC memberships encoded as frozen sets.
    exact={frozenset(x) for x in orig_scc} == {frozenset(x) for x in new_scc}
    largest=max(map(len,new_scc))
    du0=dict(g.out_degree());du1=dict(h.out_degree())
    valid=[u for u in g.nodes() if du0[u]>0]
    spr=spearmanr([du0[u] for u in valid],[du1[u] for u in valid]).statistic
    # check original SCC connectivity for each group exactly
    return {'scc_partition_preserved':exact,'largest_scc_retention':largest/original_scc_largest,
            'out_degree_spearman':float(spr)}


def run(name,g,labels,mask_seeds=(0,1,2),levels=(.3,.5,.7)):
    records=[];m=g.number_of_edges()
    orig_scc=list(nx.strongly_connected_components(g))
    methods=('BESC_SCC','entropy_only_SCC','random_SCC','community_quota_SCC','degree_SCC','jaccard_global_SCC','local_degree_global_SCC','BESC_legacy')
    masks=[]
    for trial in mask_seeds:
        protect,info=scc_protection(g,labels,trial)
        old=ev.generate_protection(g,labels,seed=500000+trial,randomized=True)
        masks.append(dict(trial=int(trial),**info,protected_fraction=len(protect)/m))
        print(name,'seed',trial,'PROTECT',len(protect),'/',m,'fraction',round(len(protect)/m,4),flush=True)
        for rho in levels:
            target=math.ceil(rho*m)
            if len(protect)>target:
                for method in methods:records.append({'network':name,'trial':trial,'rho':rho,'method':method,
                 'status':'infeasible','protected_edges':len(protect),'target_edges':target})
                continue
            for method in methods:
                t=time.perf_counter()
                if method=='BESC_SCC':h=bp.sparsify_besc(g,labels,rho,alpha=.75,protected=protect)
                elif method=='entropy_only_SCC':h=bp.sparsify_besc(g,labels,rho,alpha=1.,protected=protect)
                elif method=='BESC_legacy':h=bp.sparsify_besc(g,labels,rho,alpha=.75,protected=old)
                elif method=='random_SCC':h=bp.sparsify_baseline(g,labels,rho,'random',protected=protect,seed=trial+117)
                elif method=='degree_SCC':h=bp.sparsify_baseline(g,labels,rho,'degree',protected=protect,seed=trial+117)
                else:
                    short=method.removesuffix('_SCC')
                    h=ev.comparator(g,labels,rho,short,protect,seed=trial+117)
                sec=time.perf_counter()-t
                diag=bp.diagnostic(g,h,labels,rho)
                ex=extra(g,h,labels,orig_scc)
                if method!='BESC_legacy':
                    if not ex['scc_partition_preserved'] or not diag['weak_components_equal'] or diag['channel_coverage']!=1:
                        raise AssertionError((name,method,trial,rho,'SCC preservation broken'))
                if h.number_of_edges()!=target:raise AssertionError('Budget mismatch')
                records.append(dict(network=name,trial=trial,rho=rho,method=method,status='ok',
                  source_nodes=g.number_of_nodes(),source_edges=m,protected_edges=len(protect) if method!='BESC_legacy' else len(old),
                  target_edges=target,runtime_sec=sec,**diag,**ex))
                print(name,'run',trial,rho,method,'jsd',round(diag['node_jsd'],6),'scc',ex['scc_partition_preserved'],'s',round(sec,2),flush=True)
    return pd.DataFrame(records),masks


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',type=Path,default=Path(__file__).resolve().parent.parent/'input')
    p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent.parent/'results')
    p.add_argument('--networks',nargs='+',default=['email-Eu-core','Polblogs'])
    p.add_argument('--trials',type=int,default=3)
    p.add_argument('--rho',nargs='+',type=float,default=[.3,.5,.7])
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    full=[];manifests={}
    for name in args.networks:
        g,labels,meta=load_case(name,args.input_dir)
        comps=sorted(map(len,nx.strongly_connected_components(g)),reverse=True)
        r,masks=run(name,g,labels,range(args.trials),args.rho)
        full.append(r)
        manifests[name]=dict(**meta,nodes=g.number_of_nodes(),edges=g.number_of_edges(),
          groups=len(set(labels.values())),scc_count=len(comps),largest_scc=comps[0],
          protected_masks=masks)
        pd.concat(full,ignore_index=True).to_csv(args.output_dir/'results_incremental.csv',index=False)
    df=pd.concat(full,ignore_index=True)
    df.to_csv(args.output_dir/'all_runs.csv',index=False,float_format='%.9f')
    good=df[df.status=='ok']
    measurements=['node_jsd','degree_retention_error','mixing_jsd','mean_seic_drift',
         'out_degree_spearman','largest_scc_retention','runtime_sec']
    agg=good.groupby(['network','rho','method'])[measurements].agg(['mean','std']).reset_index()
    agg.columns=['_'.join(c).rstrip('_') for c in agg.columns.to_flat_index()]
    agg.to_csv(args.output_dir/'summary.csv',index=False,float_format='%.9f')
    (args.output_dir/'manifest.json').write_text(json.dumps(dict(datasets=manifests,
           methods='Seven SCC-certificate-constrained methods and one BESC legacy reference; three trials each, rho=.3/.5/.7.',
           methodological_warning='All comparators adapted/implemented here are research heuristics; NOT faithful reimplementations of full published algorithms. Strong certificate is classical.',
           software=dict(python=platform.python_version(),networkx=nx.__version__)),indent=2,ensure_ascii=False),encoding='utf-8')
    print('FINAL SUMMARY',flush=True)
    print(agg[['network','rho','method','node_jsd_mean','mean_seic_drift_mean','largest_scc_retention_mean']].to_string(index=False),flush=True)

if __name__=='__main__':main()
