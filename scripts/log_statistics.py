"""Group-weighted paired descriptive analysis with sign-flip tests."""
import random
from collections import defaultdict
def group_scores(rows):
    tasks=defaultdict(list);groups=defaultdict(list)
    for r in rows:tasks[(r['group'],r['id'])].append(float(r['correct']))
    for (g,_),v in tasks.items():groups[g].append(sum(v)/len(v))
    return {g:sum(v)/len(v) for g,v in groups.items()}
def paired(left,right,seed=1107,draws=20000):
    a,b=group_scores(left),group_scores(right)
    if set(a)!=set(b) or not a:raise ValueError('unmatched groups')
    d=[a[g]-b[g] for g in sorted(a)];n=len(d);observed=sum(d)/n;rng=random.Random(seed)
    boot=sorted(sum(rng.choice(d) for _ in d)/n for _ in range(draws))
    extreme=sum(abs(sum(x*rng.choice((-1,1)) for x in d)/n)>=abs(observed)-1e-12 for _ in range(draws))
    return dict(groups=n,difference=observed,bootstrap95=[boot[int(.025*draws)],boot[int(.975*draws)]],sign_flip_p=(extreme+1)/(draws+1))
def holm(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);out=[0.]*len(values);previous=0.
    for rank,i in enumerate(order):
        previous=max(previous,min(1.,values[i]*(len(values)-rank)));out[i]=previous
    return out
