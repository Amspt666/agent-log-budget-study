"""Recompute supplementary comparisons from scored rows, without raw logs or APIs.

Usage: python reproduce_scored_evidence.py --evidence .
Requires log_statistics.py alongside this file. Makes no network requests.
"""
import argparse,csv,json
from pathlib import Path
from itertools import combinations
from log_statistics import group_scores,paired,holm
ap=argparse.ArgumentParser();ap.add_argument('--evidence',default='paper/evidence');args=ap.parse_args()
e=Path(args.evidence);rows=list(csv.DictReader((e/'scored_rows.csv').open(encoding='utf-8')))
for r in rows:
    for k in ['correct','abstain','wrong','within1','visible','empty','censored_correct']:r[k]=int(r[k])
assert len(rows)==9450 and len({r['key'] for r in rows})==9450
expected=json.loads((e/'audit.json').read_text())
def equal(x,y):
    if isinstance(x,float): assert abs(x-y)<1e-12,(x,y)
    elif isinstance(x,list):
        assert len(x)==len(y)
        for a,b in zip(x,y):equal(a,b)
    else: assert x==y,(x,y)
for cell in expected['cells']:
    rs=[r for r in rows if r['provider']==cell['provider'] and r['condition']==cell['condition']]
    assert len(rs)==450
    for field,out in [('correct','accuracy'),('abstain','abstain'),('wrong','wrong'),('visible','visible'),('censored_correct','censored_accuracy')]:
        gs=group_scores([dict(r,correct=r[field]) for r in rs]);equal(sum(gs.values())/len(gs),cell[out])
tests=[]
for target in expected['comparisons']:
    full=[r for r in rows if r['provider']==target['provider'] and r['condition']=='full']
    bounded=[r for r in rows if r['provider']==target['provider'] and r['condition']==target['condition']]
    t=paired(full,bounded);tests.append(t)
    for k,v in t.items():equal(v,target[k])
for i in range(0,18,6):
    for p,t in zip(holm([r['sign_flip_p'] for r in tests[i:i+6]]),expected['comparisons'][i:i+6]):equal(p,t['holm6'])
for p,t in zip(holm([r['sign_flip_p'] for r in tests]),expected['comparisons']):equal(p,t['holm18'])
cap=json.loads((e/'configuration_sensitivity.json').read_text());excluded={r['group'] for r in cap['deviations']}
sub=[r for r in rows if r['provider']=='kimi' and r['group'] not in excluded];ts=[]
for target in cap['tests']:
    t=paired([r for r in sub if r['condition']=='full'],[r for r in sub if r['condition']==target['condition']]);ts.append(t)
    for k,v in t.items():equal(v,target[k])
for p,t in zip(holm([r['sign_flip_p'] for r in ts]),cap['tests']):equal(p,t['holm6'])
strategies=json.loads((e/'bounded_comparisons.json').read_text());ts=[]
for target in strategies:
    a=[r for r in rows if r['provider']==target['provider'] and r['condition']==target['left']+str(target['budget'])]
    b=[r for r in rows if r['provider']==target['provider'] and r['condition']==target['right']+str(target['budget'])]
    t=paired(a,b);ts.append(t)
    for k,v in t.items():equal(v,target[k])
for p,t in zip(holm([r['sign_flip_p'] for r in ts]),strategies):equal(p,t['holm18'])
print('Verified 21 cells, 18 full contrasts, 6 cap-exclusion contrasts, 18 strategy contrasts and Holm adjustments.')
