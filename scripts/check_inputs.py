#!/usr/bin/env python3
"""Check whether local third-party dataset files match expected graph statistics."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_besc_scc import load_case
cases=['email-Eu-core','Polblogs','CollegeMsg']
expected={'email-Eu-core':(1005,24929,42),'Polblogs':(1490,19014,2),'CollegeMsg':(1899,20296,16)}
errors=0
for name in cases:
    try:
        g,labels,metadata=load_case(name,ROOT/'data')
        got=(g.number_of_nodes(),g.number_of_edges(),len(set(labels.values())))
        msg='PASS' if got==expected[name] else 'WARNING'
        print(f'{msg}: {name}: nodes, arcs, classes={got}; expected={expected[name]}')
        for k,v in metadata.items():
            if 'sha256' in k:print(' ',k,':',v)
        if got!=expected[name]:errors+=1
    except FileNotFoundError as e:
        print(f'MISSING: {name}: {e}');errors+=1
if errors:sys.exit(1)
