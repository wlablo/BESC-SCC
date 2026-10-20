#!/usr/bin/env python3
"""Regenerate BESC-SCC group-profile JSD comparison at 50% edge retention."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,default=root/'figures'/'Figure_1_regenerated.png')
a=p.parse_args()
df=pd.read_csv(root/'results/reference/three_network_summary.csv')
networks=['email-Eu-core','Polblogs','CollegeMsg']
methods=['BESC_SCC','community_quota_SCC','entropy_only_SCC','BESC_legacy']
labels=['BESC-SCC','Group quota + SCC','Entropy-only + SCC','Legacy BESC']
base=df[(df.rho==.5)&(df.method=='random_SCC')].set_index('network')['node_jsd_mean']
fig,ax=plt.subplots(figsize=(10.6,5.1))
width=.20
import numpy as np
for i,(method,label) in enumerate(zip(methods,labels)):
    sub=df[(df.rho==.5)&(df.method==method)].set_index('network')['node_jsd_mean']
    vals=[sub[x]/base[x] for x in networks]
    ax.bar(np.arange(3)+(i-1.5)*width,vals,width,label=label)
ax.axhline(1,linestyle='--',color='black',lw=1,label='Random + SCC = 1.0')
ax.set_xticks(range(3),['Email-Eu-core','Polblogs','CollegeMsg'])
ax.set_ylabel('Profile JSD relative to random + SCC within each network')
ax.set_ylim(0,1.08)
ax.legend(fontsize=8,ncol=3)
fig.tight_layout()
a.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(a.output,dpi=220)
print('Wrote',a.output)
