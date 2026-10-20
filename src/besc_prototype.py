#!/usr/bin/env python3
"""BESC prototype: node-level department-profile preserving directed graph sparsification.

RESEARCH PROTOTYPE, not peer-reviewed; reproducible synthetic benchmarks.
Requires: numpy, networkx. No download required.

Inputs: simple directed unweighted graph and exogenous node-group labels.
Protection constraints: same weakly connected components, each originally-active
source has at least one outgoing edge, each observed ordered group pair remains.

All baselines use the same protected edge set and identical edge budgets.
"""
from __future__ import annotations
import argparse
import csv
import heapq
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
import gzip

import networkx as nx
import numpy as np


def jsd(p, q):
    """Jensen--Shannon divergence in bits, with 0 log 0 convention."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    middle = (p + q) / 2.0
    def kl(a, b):
        positive = a > 0
        return float(np.sum(a[positive] * np.log2(a[positive] / b[positive])))
    return 0.5 * kl(p, middle) + 0.5 * kl(q, middle)


def sorted_edges(g):
    return sorted((int(a), int(b)) for a, b in g.edges() if a != b)


def protected_edges(g, group):
    """Common feasibility constraints, using a deterministic choice of representatives."""
    edges = sorted_edges(g)
    # Directed representatives of a spanning forest of the underlying undirected graph.
    uf = nx.utils.UnionFind(g.nodes())
    keep = set()
    for u, v in edges:
        if uf[u] != uf[v]:
            uf.union(u, v)
            keep.add((u, v))
    by_pair = {}
    by_source = {}
    for u, v in edges:
        by_pair.setdefault((group[u], group[v]), (u, v))
        by_source.setdefault(u, (u, v))
    keep.update(by_pair.values())
    keep.update(by_source.values())
    return keep


def edge_budget(g, rho, protected):
    m = g.number_of_edges()
    budget = math.ceil(rho * m)
    if budget < len(protected):
        raise ValueError(f"Infeasible: budget {budget} < {len(protected)} protected edges, "
                         f"minimum retention {len(protected)/m:.3f}")
    return budget


def graph_from_edges(g, edges):
    h = nx.DiGraph()
    h.add_nodes_from(g.nodes())
    h.add_edges_from(edges)
    return h


def sparsify_besc(g, group, rho, alpha=0.75, protected=None):
    """Greedy edge deletion with cached per-source group candidates.

    Minimize weighted JSD(node's original vs retained outgoing department mix)
    + a per-source retention target distortion. Each deletion recomputes only
    scores for the affected source. The method is heuristic (not global optimum).
    """
    if protected is None:
        protected = protected_edges(g, group)
    target = edge_budget(g, rho, protected)
    all_edges = set(sorted_edges(g))
    remaining = set(all_edges)
    group_ids = sorted(set(group.values()))
    gi = {c: i for i, c in enumerate(group_ids)}
    n = g.number_of_nodes()
    m = len(all_edges)
    k = len(group_ids)
    counts = np.zeros((n, k), dtype=int)
    originals = np.zeros((n, k), dtype=float)
    d0 = np.zeros(n, dtype=int)
    for u,v in all_edges:
        counts[u, gi[group[v]]] += 1
        d0[u] += 1
    for u in g.nodes():
        if d0[u]:
            originals[u,:] = counts[u,:]/d0[u]
    removable = defaultdict(lambda: defaultdict(list))
    for u,v in sorted(all_edges, reverse=True):
        if (u,v) not in protected:
            removable[u][gi[group[v]]].append((u,v))

    def source_cost(u, cnt):
        dc = int(cnt.sum())
        if d0[u] == 0 or dc == 0:
            return float('inf')
        w = d0[u]/m
        return w * (alpha * jsd(originals[u,:], cnt/dc) +
                    (1-alpha)*abs(dc/d0[u]-rho))

    version = [0]*n
    heap = []

    def refresh(u):
        c = counts[u, :]
        before = source_cost(u, c)
        for cl, available in removable[u].items():
            if len(available) and c[cl] > 0 and c.sum() > 1:
                alt = c.copy()
                alt[cl] -= 1
                diff = source_cost(u, alt)-before
                heapq.heappush(heap, (round(diff, 14), u, cl, version[u]))

    for u in g.nodes():
        refresh(u)
    removed = 0
    while len(remaining) > target:
        if not heap:
            raise RuntimeError('No feasible edge deletions remain')
        _,u,cl,ver = heapq.heappop(heap)
        if ver != version[u] or not removable[u][cl]:
            continue
        edge = removable[u][cl].pop()
        remaining.remove(edge)
        counts[u,cl] -= 1
        removed += 1
        version[u] += 1
        refresh(u)
    return graph_from_edges(g, remaining)


def sparsify_baseline(g, group, rho, kind, protected=None, seed=0):
    if protected is None:
        protected = protected_edges(g, group)
    budget = edge_budget(g, rho, protected)
    other = sorted(set(sorted_edges(g)) - protected)
    if kind == 'random':
        rng = random.Random(seed)
        rng.shuffle(other)
    elif kind == 'degree':
        # Keep edges joining high-degree endpoints; deterministic tie ordering.
        other.sort(key=lambda e: (-(g.out_degree(e[0])+g.in_degree(e[1])), e))
    else:
        raise ValueError(kind)
    return graph_from_edges(g, set(protected) | set(other[:budget-len(protected)]))


def seic(g, group):
    """Recompute published SEIC as an OUTCOME only, not the BESC objective.

    Communities with zero intra-group edges are returned as NaN (not treated as zero).
    """
    result = {}
    for cl in sorted(set(group.values())):
        nodes = [u for u in g.nodes() if group[u] == cl]
        if len(nodes) <= 1:
            result[cl] = float('nan')
            continue
        vals = np.array([sum(1 for v in g.successors(u) if group[v] == cl)
                         for u in nodes], dtype=float)
        if vals.sum() == 0:
            result[cl] = float('nan')
            continue
        q = vals[vals>0]/vals.sum()
        result[cl] = float(-sum(q * np.log2(q)) / np.log2(len(nodes)))
    return result


def load_snap(folder):
    """Load the original unweighted directed SNAP email-Eu-core data from .gz or .txt files.

    The original SNAP dataset is not bundled with the prototype.
    """
    folder = Path(folder)
    def source(name):
        for suffix in ('.gz',''):
            file=folder/(name+suffix)
            if file.exists():
                return gzip.open(file,'rt') if suffix else file.open('r')
        raise FileNotFoundError(f"Missing SNAP file: {name}(.gz)")
    group = {}
    with source('email-Eu-core-department-labels.txt') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                node,cl=map(int,line.split())
                group[node]=cl
    g=nx.DiGraph()
    g.add_nodes_from(group)
    with source('email-Eu-core.txt') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                a,b=map(int,line.split())
                if a not in group or b not in group:
                    raise ValueError('Node ID not found in department labels')
                if a!=b:g.add_edge(a,b)
    return g,group


def diagnostic(g, h, group, rho):
    edges_original = sorted_edges(g)
    edges_sparse = sorted_edges(h)
    classes = sorted(set(group.values()))
    # Ordered inter/intra-group channels; capture distributions among neighbors.
    original_classes = Counter((group[u], group[v]) for u,v in edges_original)
    sparse_classes = Counter((group[u], group[v]) for u,v in edges_sparse)
    js_nodes = 0.0
    deg_dist = 0.0
    m0 = len(edges_original)
    clindex = {c:i for i,c in enumerate(classes)}
    g_class = defaultdict(lambda: np.zeros(len(classes)))
    h_class = defaultdict(lambda: np.zeros(len(classes)))
    for u,v in edges_original: g_class[u][clindex[group[v]]] += 1
    for u,v in edges_sparse: h_class[u][clindex[group[v]]] += 1
    for u in g.nodes():
        dg = g.out_degree(u)
        dh = h.out_degree(u)
        if dg:
            js_nodes += (dg/m0)*jsd(g_class[u]/dg,h_class[u]/dh)
            deg_dist += (dg/m0)*abs(dh/dg-rho)
    original_components = sorted(sorted(cc) for cc in nx.weakly_connected_components(g))
    sparsified_components = sorted(sorted(cc) for cc in nx.weakly_connected_components(h))
    same_comp = original_components == sparsified_components
    channels = set(original_classes)
    preserved_ch = set(sparse_classes)
    classes_used = sorted(set(original_classes))
    po = [original_classes[c]/len(edges_original) for c in classes_used]
    ps = [sparse_classes[c]/len(edges_sparse) for c in classes_used]
    mixing_jsd = jsd(po,ps)
    original_seic = seic(g,group)
    new_seic = seic(h,group)
    changes = [abs(original_seic[c]-new_seic[c]) for c in original_seic
               if not math.isnan(original_seic[c]) and not math.isnan(new_seic[c])]
    seic_drift = float(np.mean(changes)) if changes else float('nan')
    return {'node_jsd':js_nodes, 'degree_retention_error':deg_dist,
            'mixing_jsd':mixing_jsd,'mean_seic_drift':seic_drift,
            'channel_coverage':len(preserved_ch & channels)/len(channels),
            'weak_components_equal':same_comp,
            'outgoing_coverage':all(h.out_degree(u)>0 for u in g.nodes() if g.out_degree(u)>0),
            'removed_fraction':1-h.number_of_edges()/g.number_of_edges()}


def synthetic(seed, p_in=0.13, p_out=0.012, sizes=(36,36,36,36,36)):
    rng=np.random.default_rng(seed)
    group={}
    start=0
    for k, size in enumerate(sizes):
        for u in range(start,start+size):group[u]=k
        start+=size
    g=nx.DiGraph()
    g.add_nodes_from(range(start))
    # Reproducible independent directed Bernoulli SBM; no loops.
    for u in range(start):
        for v in range(start):
            if u!=v and rng.random() < (p_in if group[u]==group[v] else p_out):
                g.add_edge(u,v)
    return g,group


def test_invariants():
    assert jsd([0.5,0.5],[0.5,0.5]) < 1e-12
    assert abs(jsd([1,0],[0,1])-1)<1e-12
    for s in range(5):
        g,cl=synthetic(s+100, sizes=(18,18,18))
        protect=protected_edges(g,cl)
        for rho in (.5,.7):
            if math.ceil(rho*g.number_of_edges()) < len(protect):
                continue
            h=sparsify_besc(g,cl,rho,protected=protect)
            d=diagnostic(g,h,cl,rho)
            assert d['weak_components_equal'] and d['outgoing_coverage']
            assert abs(d['channel_coverage']-1)<1e-12
            assert h.number_of_edges()==math.ceil(rho*g.number_of_edges())
            assert set(h.edges()).issubset(set(g.edges()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='results')
    p.add_argument('--seeds',type=int,default=5)
    p.add_argument('--alpha',type=float,default=0.75)
    p.add_argument('--snap-dir',default=None, help='Run on downloaded SNAP EU-core files')
    p.add_argument('--karate',action='store_true',help='Run small real Zachary Karate Club sanity check')
    args=p.parse_args()
    test_invariants()
    out=Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records=[]
    configurations=[('assortative',0.15,0.010),('mixed',0.11,0.035)]
    if args.snap_dir:
        configurations=[('SNAP_email_Eu_core',0,0)]
    if args.karate:
        configurations=[('Karate_club',0,0)]
    for family,inside,outside in configurations:
        for seed in range(1 if (args.snap_dir or args.karate) else args.seeds):
            if args.snap_dir:
                g,labels=load_snap(args.snap_dir)
            elif args.karate:
                raw=nx.karate_club_graph()
                g=nx.DiGraph()
                g.add_nodes_from(raw.nodes())
                for u,v in raw.edges():
                    g.add_edge(u,v)
                    g.add_edge(v,u)
                labels={u:raw.nodes[u]['club'] for u in raw.nodes()}
            else:
                g,labels=synthetic(2026+seed,p_in=inside,p_out=outside)
            protect=protected_edges(g,labels)
            for rho in (0.7,0.5,0.3):
                if math.ceil(rho*g.number_of_edges())<len(protect):
                    print('Skipping infeasible',family,seed,rho)
                    continue
                for method in ('BESC','BESC_no_degree','random','degree'):
                    t=time.perf_counter()
                    if method=='BESC':
                        h=sparsify_besc(g,labels,rho,alpha=args.alpha,protected=protect)
                    elif method=='BESC_no_degree':
                        h=sparsify_besc(g,labels,rho,alpha=1.0,protected=protect)
                    else:
                        h=sparsify_baseline(g,labels,rho,method,protected=protect,seed=seed)
                    elapsed=time.perf_counter()-t
                    d=diagnostic(g,h,labels,rho)
                    records.append({'family':family,'seed':seed,'rho':rho,
                                    'method':method,'vertices':g.number_of_nodes(),
                                    'initial_edges':g.number_of_edges(),'protected':len(protect),
                                    'output_edges':h.number_of_edges(),'runtime_sec':round(elapsed,4),**d})
        print('Finished', family,flush=True)
    with (out/'raw_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader();writer.writerows(records)
    summary=[]
    for family,_,_ in configurations:
        for rho in (.7,.5,.3):
            for method in ('BESC','BESC_no_degree','random','degree'):
                block=[r for r in records if (r['family'],r['rho'],r['method'])==(family,rho,method)]
                if not block:continue
                summary.append({'family':family,'rho':rho,'method':method,'n_runs':len(block),
                                'mean_jsd':statistics.mean(r['node_jsd'] for r in block),
                                'sd_jsd':statistics.stdev(r['node_jsd'] for r in block) if len(block)>1 else 0,
                                'mean_degree_error':statistics.mean(r['degree_retention_error'] for r in block),
                                'mean_mixing_jsd':statistics.mean(r['mixing_jsd'] for r in block),
                                'mean_seic_drift':statistics.mean(r['mean_seic_drift'] for r in block),
                                'mean_runtime_sec':statistics.mean(r['runtime_sec'] for r in block),
                                'all_constraints_pass':all(r['weak_components_equal'] and r['outgoing_coverage'] and r['channel_coverage']==1 for r in block)})
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print('\nMEAN RESULTS (synthetic unless --snap-dir or --karate)')
    for r in summary:
        print(f"{r['family']:12s} {r['rho']:.1f} {r['method']:14s} JSD={r['mean_jsd']:.5f} +- {r['sd_jsd']:.5f} "
              f"degerr={r['mean_degree_error']:.4f} seconds={r['mean_runtime_sec']:.3f} valid={r['all_constraints_pass']}")

if __name__=='__main__':
    main()
