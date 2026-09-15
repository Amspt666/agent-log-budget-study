"""Group-safe split and deterministic character-budget development inputs."""
import json,hashlib,random,unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SHA='f4d2b6da464a826580e59b3a0eae15ea2d642d7c'
def main():
    base=ROOT/'data/external/who_when'/SHA
    paths=sorted((base/'Who&When').rglob('*.json'));ds=[json.loads(p.read_text(encoding='utf-8')) for p in paths]
    parent=list(range(len(ds)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for i,d in enumerate(ds):
        keys=[('id',d['question_ID']),('question',' '.join(unicodedata.normalize('NFKC',d['question']).casefold().split())),('history',hashlib.sha256(json.dumps(d['history'],sort_keys=True).encode()).hexdigest())]
        for key in keys:
            if key in seen:parent[find(i)]=find(seen[key])
            seen[key]=i
    groups={}
    for i in range(len(ds)):groups.setdefault(find(i),[]).append(i)
    exposed={find(i) for i,p in enumerate(paths) if p.name=='1.json'}
    remaining=sorted(set(groups)-exposed);random.Random(2026091101).shuffle(remaining)
    dev=exposed|set(remaining[:max(0,25-len(exposed))])
    out=ROOT/'data/processed/log_study_v1';out.mkdir(parents=True,exist_ok=True)
    public=[];labels=[];split={}
    for group,indices in groups.items():
        gid=hashlib.sha256('|'.join(str(paths[i].relative_to(base)) for i in indices).encode()).hexdigest()[:16]
        split[gid]='development' if group in dev else 'test'
        for i in indices:
            d=ds[i];ident=str(paths[i].relative_to(base)).replace('\\','/')
            public.append(dict(id=ident,group=gid,split=split[gid],question=d['question'],steps=[dict(step=j,speaker=e.get('name',e.get('role','unknown')),content=e.get('content','')) for j,e in enumerate(d['history'])]))
            labels.append(dict(id=ident,step=int(d['mistake_step']),agent=d['mistake_agent']))
    assert not {r['group'] for r in public if r['split']=='development'} & {r['group'] for r in public if r['split']=='test'}
    for name,obj in [('inputs.json',public),('labels_private.json',labels),('split.json',split)]:
        p=out/name
        if p.exists():assert json.loads(p.read_text())==obj,'Refuse changed split'
        else:p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    summary=dict(groups=len(groups),development_groups=len(dev),test_groups=len(groups)-len(dev),development_trajectories=sum(r['split']=='development' for r in public),test_trajectories=sum(r['split']=='test' for r in public))
    print(json.dumps(summary))
if __name__=='__main__':main()
