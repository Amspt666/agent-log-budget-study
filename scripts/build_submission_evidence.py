"""Read-only audit of final experiment; generate paper tables without API calls."""
import csv, hashlib, json, sys
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from log_framework import choose, serialize
from log_statistics import group_scores, paired, holm

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/multivendor_formal_v3'
DEST=ROOT/'paper/evidence'
DEST.mkdir(parents=True,exist_ok=True)
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def score(rs,field='correct'):
    v=group_scores([dict(r,correct=r[field]) for r in rs]); return sum(v.values())/len(v)
inputs=read(ROOT/'data/processed/log_study_v1/inputs.json')
tasks={r['id']:r for r in inputs if r['split']=='test'}
gold={r['id']:r['step'] for r in read(ROOT/'data/processed/log_study_v1/labels_private.json')}
manifest=read(OUT/'manifest.json'); plan={r['key']:r for r in manifest['jobs']}
archive=list((OUT/'failed_archive').rglob('*.json')); archived={p.stem for p in archive}
rows=[]; problems=Counter(); final_usage=Counter(); archives_usage=Counter(); prices=read(ROOT/'configs/pricing_multivendor.json')
cost=Counter(); archive_status=Counter(); missing_usage=0
def usage(r,acc,kind):
    u=r.get('usage') or {}; prov=r['provider']; pr=prices[prov]
    ti=u.get('prompt_tokens') or 0; to=u.get('completion_tokens') or 0
    tc=min((u.get('prompt_tokens_details') or {}).get('cached_tokens') or 0,ti)
    acc[prov+'_input']+=ti; acc[prov+'_output']+=to; acc[prov+'_cached']+=tc
    cost[kind]+=((ti-tc)*pr['in']+tc*pr['cached']+to*pr['out'])/1e6
for p in sorted((OUT/'records').glob('*.json')):
    r=read(p); t=tasks[r['id']]; pred=r['prediction']['step']
    assert r['key']==p.stem and p.stem in plan
    assert all(r[k]==v for k,v in plan[p.stem].items())
    assert r['status']=='completed' and r['finish_reason']=='stop' and r['answer_source']=='content'
    assert r['model']==r['model_returned']==manifest['models'][r['provider']]
    assert r['correct']==(pred==gold[r['id']])
    request=r['request']; obs=json.loads(request['messages'][1]['content'])
    expected=choose(t['steps'],r['policy'],r['budget'] or 2)
    assert obs==dict(question=t['question'],total_steps=len(t['steps']),observed_steps=expected)
    assert hashlib.sha256(request['messages'][0]['content'].encode()).hexdigest()==manifest['system_prompt_sha256']
    for k,v in manifest['output_cfg'][r['provider']].items():
        if request.get(k)!=v: problems['request_config_'+r['provider']+'_'+k]+=1
    assert r['log_characters']==len(serialize(expected))
    usage(r,final_usage,'final')
    rows.append(dict(key=p.stem,id=r['id'],group=r['group'],provider=r['provider'],
      condition=r['policy']+str(r['budget']) if r['budget'] else 'full',repeat=r['repeat'],
      correct=int(pred==gold[r['id']]),abstain=int(pred is None),
      wrong=int(pred is not None and pred!=gold[r['id']]),
      within1=int(pred is not None and abs(pred-gold[r['id']])<=1),
      visible=int(any(s['step']==gold[r['id']] for s in expected)),
      empty=int(not expected),chars=r['log_characters'],
      censored_correct=int(pred==gold[r['id']] and p.stem not in archived)))
assert len(rows)==len(plan)==9450 and {r['key'] for r in rows}==set(plan)
assert set(archived)<=set(plan)
for p in archive:
    r=read(p); archive_status[r['status']]+=1
    if r.get('usage'): usage(r,archives_usage,'archive')
    else: missing_usage+=1
conditions=['full','prefix6000','prefix12000','suffix6000','suffix12000','spread6000','spread12000']
providers=['glm','qwen','kimi']; cells=[]; comparisons=[]
for prov in providers:
    for c in conditions:
        sub=[r for r in rows if r['provider']==prov and r['condition']==c]
        assert len(sub)==450 and len(group_scores(sub))==112
        assert all(n==3 for n in Counter(r['id'] for r in sub).values())
        cells.append(dict(provider=prov,condition=c,n=len(sub),accuracy=score(sub),
          abstain=score(sub,'abstain'),wrong=score(sub,'wrong'),within1=score(sub,'within1'),
          visible=score(sub,'visible'),empty=score(sub,'empty'),
          censored_accuracy=score(sub,'censored_correct'),recovered=sum(r['key'] in archived for r in sub)))
        assert abs(sum(cells[-1][x] for x in ['accuracy','abstain','wrong'])-1)<1e-10
    full=[r for r in rows if r['provider']==prov and r['condition']=='full']
    for c in conditions[1:]:
        sub=[r for r in rows if r['provider']==prov and r['condition']==c]
        comparisons.append(dict(provider=prov,condition=c,**paired(full,sub)))
    ps=holm([x['sign_flip_p'] for x in comparisons[-6:]])
    for r,p in zip(comparisons[-6:],ps): r['holm6']=p
    print(prov,'audited',flush=True)
for r,p in zip(comparisons,holm([r['sign_flip_p'] for r in comparisons])):r['holm18']=p
result=dict(cells=cells,comparisons=comparisons,request_config_mismatches=dict(problems),
    final_usage=dict(final_usage),archive_usage=dict(archives_usage),cost=dict(cost),
    archive_status=dict(archive_status),archive_records=len(archive),archive_keys=len(archived),
    archive_missing_usage=missing_usage,manifest_sha256=hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest())
(DEST/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
with (DEST/'scored_rows.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (DEST/'cells.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(cells[0]));w.writeheader();w.writerows(cells)
labels={c:c.replace('6000',' 6k').replace('12000',' 12k').capitalize() for c in conditions}
lines=[]
for c in conditions:
    lines.append(labels[c]+' & '+' & '.join(f"{next(x['accuracy'] for x in cells if x['provider']==p and x['condition']==c):.4f}" for p in providers)+r' \\')
(DEST/'accuracy_rows.tex').write_text('\n'.join(lines),encoding='utf-8')
lines=[]
for r in comparisons:
    lo,hi=r['bootstrap95']
    lines.append(f"{r['provider'].upper()} & {labels[r['condition']]} & {r['difference']:.4f} & [{lo:.4f}, {hi:.4f}] & {r['holm6']:.4f} & {r['holm18']:.4f}"+r' \\')
(DEST/'comparison_rows.tex').write_text('\n'.join(lines),encoding='utf-8')
lines=[]
for r in cells:
    lines.append(f"{r['provider'].upper()} & {labels[r['condition']]} & {r['abstain']:.4f} & {r['wrong']:.4f} & {r['visible']:.4f} & {r['censored_accuracy']:.4f}"+r' \\')
(DEST/'diagnostic_rows.tex').write_text('\n'.join(lines),encoding='utf-8')
plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(7,3.5),sharey=True)
for ax,prov in zip(axes,providers):
    sub=[r for r in comparisons if r['provider']==prov]
    for i,r in enumerate(sub):
        lo,hi=r['bootstrap95'];d=r['difference']
        ax.errorbar(d,i,xerr=[[d-lo],[hi-d]],fmt='o' if r['holm6']<.05 else 's',
          color='#245680' if r['holm6']<.05 else '#666666',capsize=3,markersize=4)
    ax.axvline(0,color='black',lw=.7,ls=':');ax.set_title(prov.upper());ax.set_xlim(-.04,.22)
    ax.set_xlabel('Full minus bounded accuracy');ax.grid(axis='x',alpha=.2)
axes[0].set_yticks(range(6),[labels[c] for c in conditions[1:]]);axes[0].invert_yaxis()
fig.tight_layout();fig.savefig(ROOT/'paper/effects.pdf',bbox_inches='tight');plt.close(fig)
print(json.dumps({k:result[k] for k in ['request_config_mismatches','cost','archive_status']}))
print('holm18 significant',sum(r['holm18']<.05 for r in comparisons))
