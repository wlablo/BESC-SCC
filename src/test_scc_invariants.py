"""Small deterministic tests for SCC certificate correctness."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import networkx as nx
from run_besc_scc import scc_certificate,scc_protection,same_scc_partition
import besc_prototype as bp

for s in range(20):
    for n in (3,8,15,30):
        g=nx.gnp_random_graph(n,0.1+s*0.02,seed=1000+n+s,directed=True)
        g.remove_edges_from(nx.selfloop_edges(g))
        labels={u:u%3 for u in g.nodes()}
        p,m=scc_protection(g,labels,s)
        assert same_scc_partition(g,bp.graph_from_edges(g,p))
        assert p.issubset(set(g.edges()))
        assert len(scc_certificate(g,s))<=2*(n-1)
        rho=max(0.6,len(p)/max(1,g.number_of_edges()))
        if rho<=1 and g.number_of_edges()>0:
            h=bp.sparsify_besc(g,labels,rho,protected=p)
            assert same_scc_partition(g,h)
            assert h.number_of_edges()==__import__('math').ceil(rho*g.number_of_edges())
print('PASS: 80 randomized graph tests, SCC partition preserved under deletion.')
