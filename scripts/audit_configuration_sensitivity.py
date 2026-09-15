"""Post hoc matched-group check for historical output-cap deviations; offline only."""
import csv,json
from pathlib import Path
from log_statistics import group_scores,paired,holm
ROOT=Path(__file__).resolve().parents[1];e=ROOT/'paper/evidence';out=ROOT/'outputs/multivendor_formal_v3'
m=json.loads((out/'manifest.json').read_text());bad=[]
for p in sorted((out/'records').glob('*.json')):
    r=json.loads(p.read_text(encoding='utf-8'))
    if r['request']['max_tokens']!=m['output_cfg'][r['provider']]['max_tokens']:
        bad.append({k:r[k] for k in ['key','id','group','provider','policy','budget','repeat','correct']}|{'max_tokens':r['request']['max_tokens']})
assert len(bad)==8 and all(r['provider']=='kimi' and r['policy']=='full' and r['repeat']==0 and r['max_tokens']==4096 for r in bad)
groups={r['group'] for r in bad}
rows=list(csv.DictReader((e/'scored_rows.csv').open(encoding='utf-8')))
for r in rows:r['correct']=int(r['correct'])
rows=[r for r in rows if r['provider']=='kimi' and r['group'] not in groups]
full=[r for r in rows if r['condition']=='full'];tests=[]
for c in ['prefix6000','prefix12000','suffix6000','suffix12000','spread6000','spread12000']:
    tests.append(dict(condition=c,**paired(full,[r for r in rows if r['condition']==c])))
for r,p in zip(tests,holm([r['sign_flip_p'] for r in tests])):r['holm6']=p
v=group_scores(full)
result=dict(deviations=bad,excluded_groups=len(groups),retained_groups=len(v),retained_trajectories=len({r['id'] for r in full}),full_accuracy=sum(v.values())/len(v),tests=tests)
(e/'configuration_sensitivity.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='deviations'},indent=2))
