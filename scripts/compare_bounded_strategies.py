"""Post hoc same-budget strategy comparisons; no new API calls."""
import csv,json
from pathlib import Path
from itertools import combinations
from log_statistics import paired,holm
ROOT=Path(__file__).resolve().parents[1];e=ROOT/'paper/evidence'
rows=list(csv.DictReader((e/'scored_rows.csv').open(encoding='utf-8')))
for r in rows:r['correct']=int(r['correct'])
tests=[]
for provider in ('glm','qwen','kimi'):
    for budget in (6000,12000):
        for left,right in combinations(('prefix','suffix','spread'),2):
            a=[r for r in rows if r['provider']==provider and r['condition']==left+str(budget)]
            b=[r for r in rows if r['provider']==provider and r['condition']==right+str(budget)]
            tests.append(dict(provider=provider,budget=budget,left=left,right=right,**paired(a,b)))
for r,p in zip(tests,holm([r['sign_flip_p'] for r in tests])):r['holm18']=p
(e/'bounded_comparisons.json').write_text(json.dumps(tests,indent=2),encoding='utf-8')
print('significant globally',[(r['provider'],r['budget'],r['left'],r['right'],r['holm18']) for r in tests if r['holm18']<.05])
