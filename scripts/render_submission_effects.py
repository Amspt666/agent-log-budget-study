"""Vector effect plot from audited estimates; no re-estimation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
a=json.loads((ROOT/'paper/evidence/audit.json').read_text())
plt.rcParams.update({'font.size':8,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(4.7,2.9),sharey=True)
for ax,prov in zip(axes,['glm','qwen','kimi']):
    sub=[r for r in a['comparisons'] if r['provider']==prov]
    for i,r in enumerate(sub):
        lo,hi=r['bootstrap95'];d=r['difference']
        ax.errorbar(d,i,xerr=[[d-lo],[hi-d]],fmt='o' if r['holm6']<.05 else 's',
          color='#245680' if r['holm6']<.05 else '#666666',capsize=2,markersize=3)
    ax.axvline(0,color='black',lw=.7,ls=':');ax.set_title(prov.upper(),fontsize=9)
    ax.set_xlim(-.04,.22);ax.set_xticks([0,.1,.2]);ax.grid(axis='x',alpha=.2)
axes[0].set_yticks(range(6),['Prefix 6k','Prefix 12k','Suffix 6k','Suffix 12k','Spread 6k','Spread 12k']);axes[0].invert_yaxis()
fig.supxlabel('Full minus bounded accuracy',fontsize=8)
fig.tight_layout(pad=.4);fig.savefig(ROOT/'paper/effects.pdf',bbox_inches='tight');plt.close(fig)
