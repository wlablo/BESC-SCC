#!/usr/bin/env python3
"""Validate included historical experimental tables without original datasets."""
from pathlib import Path
import pandas as pd

root=Path(__file__).resolve().parents[1]
ref=root/'results/reference'
parts=[pd.read_csv(ref/'college_all_runs.csv'),pd.read_csv(ref/'earlier_two_network_all_runs.csv')]
all_runs=pd.concat(parts,ignore_index=True)
assert len(all_runs)==216, f'Expected 216 rows, got {len(all_runs)}'
assert set(all_runs.network)=={'email-Eu-core','Polblogs','CollegeMsg'}
assert set(all_runs.rho)=={0.3,0.5,0.7}
assert all_runs.status.eq('ok').all()
assert len(all_runs.method.unique())==8
assert all_runs.groupby(['network','rho','method']).size().eq(3).all()
assert all_runs[all_runs.method != 'BESC_legacy'].scc_partition_preserved.astype(str).str.lower().eq('true').all()
# Recompute means to ensure summary tables correspond to the raw stored runs.
summary=pd.read_csv(ref/'three_network_summary.csv')
mean=all_runs.groupby(['network','rho','method']).node_jsd.mean().reset_index(name='computed')
joined=mean.merge(summary[['network','rho','method','node_jsd_mean']],on=['network','rho','method'],validate='1:1')
assert len(joined)==72
assert (joined.computed-joined.node_jsd_mean).abs().max() < 1e-7
print('PASS: 216 exploratory run rows, SCC flags and JSD summary consistency')
