"""Optional Numba acceleration for Jensen-Shannon divergence (base 2).
Numerically equivalent to reference version; no changes to sparsification objective.
"""
import math
import numpy as np
from numba import njit

@njit(cache=True)
def jsd_numba(p,q):
    result=0.0
    for i in range(len(p)):
        m=(p[i]+q[i])/2.0
        if p[i]>0.0:
            result+=0.5*p[i]*math.log2(p[i]/m)
        if q[i]>0.0:
            result+=0.5*q[i]*math.log2(q[i]/m)
    return result

if __name__=='__main__':
    import sys,time
    sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parent / 'src'))
    import besc_prototype as bp
    x=np.array([.1,.2,.7]); y=np.array([.3,.7,0.0])
    print('reference',bp.jsd(x,y),'numba',jsd_numba(x,y))
    assert abs(bp.jsd(x,y)-jsd_numba(x,y))<1e-13
    bp.jsd=jsd_numba
    from besc_extended_validation import generate_protection
    g,c=bp.load_snap('/mnt/data/besc_stanford/input')
    prot=generate_protection(g,c,seed=50500,randomized=True)
    t=time.perf_counter();h=bp.sparsify_besc(g,c,.5,protected=prot)
    print('elapsed',round(time.perf_counter()-t,2),'edgelist hash',hash(tuple(bp.sorted_edges(h))),'nodes',h.number_of_nodes(),'edges',h.number_of_edges())
    print('metrics',bp.diagnostic(g,h,c,.5))
