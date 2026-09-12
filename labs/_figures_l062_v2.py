"""Portable computational diagrams and newly measured split points for L062."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures/l062';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})

def save(fig,name):fig.savefig(OUT/(name+'-v2.png'),dpi=160,bbox_inches='tight');plt.close(fig)

def figures():
 fig,axs=plt.subplots(1,2,figsize=(10,4.2),gridspec_kw={'width_ratios':[1.1,1]})
 mask=np.array([[1,1,0,0]]*4);ax=axs[0];ax.imshow(mask,cmap='Greens',vmin=0,vmax=1)
 ax.set_xticks(range(4),['context 1','context 2','query 1','query 2'],rotation=25);ax.set_yticks(range(4),['context 1','context 2','query 1','query 2']);ax.set_xlabel('Sending keys / values');ax.set_ylabel('Receiving query vectors')
 for i in range(4):
  for j in range(4):ax.text(j,i,'read' if j<2 else 'blocked',ha='center',va='center',fontsize=10,color='white' if j<2 else '#555')
 ax.set_title('v1 removes both query self edges')
 ax=axs[1];ax.axis('off');lines=['One head, width 2','receiver [1, 0]','keys [0, 0], [√2, 0]','scaled scores [0, 1]','weights [.268941, .731059]','values [1, 0], [0, 2]','weighted output [.268941, 1.462117]','then: add receiver residual → norm']
 for i,line in enumerate(lines):ax.text(0,1-i*.125,line,va='top',weight='bold' if i in [0,6] else 'normal',fontsize=11)
 fig.suptitle('Trace the permitted information, then compute its weighted value',y=1.07);fig.tight_layout();save(fig,'attention')
 fig,ax=plt.subplots(figsize=(9.4,4.4));ax.axis('off')
 boxes=[('Context-only moments','context [1,3] → mean 2, sample SD √2\nquery 101 → 70.7106; context remains ±.7071'),('Supported numeric recipe','standardize/clamp → remove constants\noptional context-fitted power → soften outliers'),('Flexible input · separate K=6 fixture','active [1,−2], k=2 → multiply by K/k=3\n[3,−6] → pad → [3,−6,0,0,0,0]')]
 for i,(title,body) in enumerate(boxes):
  y=.88-i*.31;ax.text(.5,y+.035,title,ha='center',va='center',weight='bold',fontsize=13);ax.text(.5,y-.115,body,ha='center',va='center',fontsize=11,bbox=dict(boxstyle='round,pad=.4',fc='#edf6f4',ec='#83b0a6'))
  if i<2:ax.annotate('',xy=(.5,y-.235),xytext=(.5,y-.185),arrowprops=dict(arrowstyle='->',color='#087e83',lw=2))
 ax.text(.5,-.075,'Actual checkpoint K=100. Query moments and sqrt(K/k) would change this predictor.',ha='center',fontsize=10);save(fig,'preprocessing')
 fig,axs=plt.subplots(1,2,figsize=(10,4));ax=axs[0];ax.axis('off')
 text=['Class labels rotate: y′ = (y+s) mod 3','View A: [3, 0, 0], shift 0','View B: [1, 0, 0], shift 1','Undo B answer mapping: [0, 0, 1]','Mean aligned logits: [1.5, 0, .5]','Temperature 1 for this fixture','Actual wrapper temperature: .8']
 for i,line in enumerate(text):ax.text(0,1-i*.145,line,va='top',fontsize=11,weight='bold' if i in [0,4] else 'normal')
 z=np.array([1.5,0,.5]);p=np.exp(z)/np.exp(z).sum();pa=np.exp([3,0,0])/np.exp([3,0,0]).sum();pb=np.exp([0,0,1])/np.exp([0,0,1]).sum()
 ax=axs[1];x=np.arange(3);ax.bar(x-.16,p,.32,label='softmax(mean logits)',color='#087e83');ax.bar(x+.16,(pa+pb)/2,.32,label='mean probabilities',color='#c7833d');ax.set_xticks(x,['class 0','class 1','class 2']);ax.set_ylim(0,.8);ax.set_ylabel('Probability');ax.legend(fontsize=9,loc='upper right');ax.set_title('Different reduction, different predictor')
 for i,v in enumerate(p):ax.text(i-.16,v+.015,f'{v:.4f}',ha='center',fontsize=9)
 fig.tight_layout();save(fig,'ensemble')
 r=json.loads((ROOT/'_verify_l062_v2_results.json').read_text());fig,axs=plt.subplots(1,3,figsize=(10,4),sharey=True)
 for ax,name in zip(axs,['diabetes','blood_transfusion','wdbc']):
  for seed in [0,1,2]:
   rows=sorted([a for a in r['records'] if a['dataset']==name and a['seed']==seed],key=lambda a:a['views'])
   ax.plot([1,4],[a['log_loss'] for a in rows],'-o',label=f'split {seed}',alpha=.8)
  ax.set_xticks([1,4]);ax.set_xlabel('Ensemble views');ax.set_title(name.replace('_',' '));ax.set_ylim(0,.75);ax.grid(axis='y',alpha=.2)
 axs[0].set_ylabel('Test log loss (nats; lower is better)');axs[-1].legend(fontsize=10)
 fig.suptitle('New full-data 50/50 splits · same pretrained weights and paired test rows',fontsize=13);fig.tight_layout();save(fig,'results')
if __name__=='__main__':figures()
