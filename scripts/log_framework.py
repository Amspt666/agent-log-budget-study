"""Label-free log selector and strict output validation for framework v1."""
import json

def serialize(steps):
    return json.dumps(steps,ensure_ascii=False,separators=(',',':'))

def choose(steps,policy,budget):
    if policy not in ('full','prefix','suffix','spread'):raise ValueError('policy')
    if budget<2:raise ValueError('budget')
    if policy=='full':return list(steps)
    order=list(range(len(steps)))
    if policy=='suffix':order.reverse()
    if policy=='spread':
        order=[];queue=[(0,len(steps)-1)]
        while queue:
            lo,hi=queue.pop(0)
            if lo>hi:continue
            mid=(lo+hi)//2;order.append(mid);queue.extend([(lo,mid-1),(mid+1,hi)])
    selected=[]
    for i in order:
        proposed=sorted(selected+[i])
        if len(serialize([steps[j] for j in proposed]))>budget:
            if policy!='spread':break
            continue
        selected=proposed
    return [steps[j] for j in selected]

def parse_prediction(text,total_steps):
    value=json.loads(text)
    if not isinstance(value,dict) or set(value)!= {'step','agent'}:raise ValueError('schema')
    step,agent=value['step'],value['agent']
    if step is None:
        if agent is not None:raise ValueError('inconsistent abstention')
    elif type(step)is not int or not 0<=step<total_steps or not isinstance(agent,str):raise ValueError('values')
    return value
