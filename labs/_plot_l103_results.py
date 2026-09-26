"""Per-seed measured results, with separate axes for unlike protocols."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent;out=P/'figures/l103';out.mkdir(parents=True,exist_ok=True)
r=json.loads((P/'_comparison_l103_results.json').read_text())
fig,ax=plt.subplots(figsize=(8,4),facecolor='#faf9f5');ax.set_facecolor('#faf9f5')
for seed in range(3):
 d={x['arm']:100*x['test']['ap'] for x in r['records'] if x['seed']==seed};ax.plot([0,1],[d['TGAT'],d['TGN']],color='#b7c6c8',lw=1)
for i,arm,color in [(0,'TGAT','#007f80'),(1,'TGN','#bb7121')]:
 vals=[100*x['test']['ap'] for x in r['records'] if x['arm']==arm]
 ax.scatter(np.full(3,i),vals,s=55,color=color,zorder=3)
 for seed,y in enumerate(vals):ax.annotate(f'seed {seed}',(i,y),xytext=(10,0),textcoords='offset points',fontsize=9)
 ax.plot([i-.12,i+.12],[np.mean(vals)]*2,color=color,lw=3)
ax.set(xlim=(-.35,1.55),ylim=(75,91),xticks=[0,1],xticklabels=['TGAT · one layer','TGN · one layer'],ylabel='Pooled test AP (%)',title='Matched course slice · three paired seeds · NOT a paper replay')
ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);fig.tight_layout()
fig.savefig(out/'comparison.svg');fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
path=P/'_paper_l103_results.json'
if path.exists():
 r=json.loads(path.read_text());fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True,facecolor='#faf9f5')
 for ax,lane,key,label,color in [(axes[0],'all','test','All test events','#007f80'),(axes[1],'new','new_test','New-node subset','#bb7121')]:
  vals=[100*x[key]['ap'] for x in r['records']];ax.set_facecolor('#faf9f5');ax.scatter(range(10),vals,color=color);ax.axhline(r['summary'][lane]['paper_ap_percent'],ls='--',color='#725389',label='Original paper target')
  ax.axhline(np.mean(vals),color=color,label='Replay mean');ax.set(title=label,xlabel='Independent initialization seed',xticks=range(10));ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8)
 axes[0].set_ylabel('Mean of batch AP (%)');fig.suptitle('Complete released-protocol Wikipedia replay · historical identity INCOMPARABLE',fontsize=11);fig.tight_layout()
 fig.savefig(out/'paper.svg');fig.savefig(out/'paper.png',dpi=170);plt.close(fig)
for name in ['comparison','paper']:
 svg=out/f'{name}.svg'
 if svg.exists():svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
