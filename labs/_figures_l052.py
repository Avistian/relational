"""Portable computation diagrams; measured figures are generated from saved evidence."""
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/relational-matplotlib')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;OUT=HERE/'figures/l052';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fbfcfa','axes.facecolor':'#fbfcfa'})
def save(fig,name):fig.savefig(OUT/f'{name}.png',dpi=150,bbox_inches='tight');plt.close(fig)

def architecture():
 fig,ax=plt.subplots(figsize=(9,9));ax.set(xlim=(0,10),ylim=(0,10.7));ax.axis('off')
 ax.text(.2,10.2,'TabR-S · one query, a training memory',fontsize=19,weight='bold')
 stages=[(8.4,'1  Shared input map','Query x [B,p] and training X [N,p]\nTrain-fitted normal quantiles → Linear(p,d)\nh [B,d]; candidate representations [N,d]'),
 (6.8,'2  Learned neighbor search','k = K(h) [B,d]; keys K(H) [N,d]\nExclude matching row IDs → smallest m squared distances\nSelected keys [B,m,d]; selected labels [B,m]'),
 (5.2,'3  Weight × value for each neighbor','sᵢ = −Σⱼ(kⱼ − kᵢⱼ)² → softmax across m\nvᵢ = Wᵧ(yᵢ) + T(k − kᵢ) [B,m,d]\nr = Σᵢ dropout(softmax(s))ᵢ vᵢ [B,d]'),
 (3.6,'4  Add the original query representation','z = h + r [B,d]\nThe residual keeps a retrieval-free route.\nTraining-memory labels enter only through Wᵧ.'),
 (2.0,'5  One residual predictor block','z ← z + Linear(ReLU(Linear(LayerNorm(z))))\nDropout after ReLU; final block dropout = 0 in this lab\nDimensions d → 2d → d'),
 (.4,'6  Prediction head','LayerNorm → ReLU → Linear(d,1)\nClassification: logit → sigmoid; regression: scalar\nTraining and inference both retrieve; no pretraining stage')]
 for y,title,body in stages:
  ax.add_patch(plt.Rectangle((.2,y),9.6,1.45,facecolor='#e4f0eb' if y==5.2 else '#eef0ef',edgecolor='#96b4aa',lw=1))
  ax.text(.4,y+1.2,title,weight='bold',fontsize=12);ax.text(.4,y+.9,body,va='top',fontsize=10.5,linespacing=1.35)
  if y>.4:ax.annotate('',xy=(5,y-.24),xytext=(5,y),arrowprops={'arrowstyle':'->','color':'#256854'})
 save(fig,'architecture')

def mechanisms():
 fig,axs=plt.subplots(1,2,figsize=(10,4.8));fig.suptitle('Self-exclusion changes the information available',fontsize=17)
 for ax,exclude in zip(axs,[False,True]):
  ids=np.array(['A','SELF','B','C']);keys=np.arange(4);scores=-(1-keys)**2
  order=np.argsort(-scores,kind='stable');order=order[ids[order]!='SELF'] if exclude else order;chosen=order[:2]
  w=np.exp(scores[chosen]);w=w/w.sum();heights=np.zeros(4);heights[chosen]=w
  ax.bar(ids,heights,color=['#29806b' if i!='SELF' else '#bb5b4a' for i in ids]);ax.set_ylim(0,1);ax.set_ylabel('Softmax weight');ax.set_title('Legal: exclude own row' if exclude else 'Leaky: allow own row')
  for i in range(4):ax.text(i,heights[i]+.035,f'{heights[i]:.3f}',ha='center')
  ax.text(.02,-.24,'Keys: 0, 1, 2, 3. Query key: 1. m=2.\nLabels: 0, 1, 1, 0. Ties shown in row order.',transform=ax.transAxes,fontsize=10)
 fig.tight_layout(rect=[0,.05,1,.92]);save(fig,'exclusion')
 fig,axs=plt.subplots(1,2,figsize=(10,4.7));fig.suptitle('A value is a label embedding plus a correction',fontsize=16)
 for ax,beta in zip(axs,[0,.5]):
  x=np.arange(2);label=np.array([0.,1.]);correction=beta*np.array([1.,-1.]);value=label+correction
  ax.bar(x-.22,label,.22,label='Label embedding',color='#778f9f');ax.bar(x,correction,.22,label='Correction',color='#bd7955');ax.bar(x+.22,value,.22,label='Corrected value',color='#29806b');ax.set_xticks(x,['A: key=0, y=0','B: key=2, y=1']);ax.set_ylim(-.7,1.3);ax.axhline(0,color='#888',lw=.7);ax.set_title(f'β={beta}: weighted sum = {value.mean():.3f}')
  ax.text(.02,-.22,'q=1; Δkeys=(+1,−1); weights=(½,½).\nScalar T(Δ)=βΔ only for this illustration.',transform=ax.transAxes,fontsize=10)
 axs[0].legend(fontsize=8);fig.tight_layout(rect=[0,.05,1,.92]);save(fig,'values')
 fig,ax=plt.subplots(figsize=(9,3.5));ax.set_title('Past event ≠ available label',fontsize=17)
 for y,(name,event,known) in enumerate([('A',1,3),('B',2,6),('C',5,7)]):
  ax.plot([event,known],[y,y],color='#9faeaa',lw=5);ax.scatter(event,y,color='#517c99',s=65,zorder=3,label='Event occurs' if y==0 else None);ax.scatter(known,y,color='#287d68',marker='s',s=65,zorder=3,label='Label becomes known' if y==0 else None)
 ax.axvline(4,color='#ad5644',ls='--');ax.text(4.12,2.5,'Predict at day 4\nOnly A is eligible',color='#913f31');ax.set(xlim=(0,9),ylim=(-.5,3),yticks=range(3),yticklabels=['A','B','C'],xlabel='Day');ax.legend(loc='lower right',fontsize=9);save(fig,'availability')

def results():
 p=HERE/'_verify_l052_results.json'
 if not p.exists():return
 result=json.loads(p.read_text());fig,axs=plt.subplots(1,3,figsize=(11,4))
 for ax,(name,r) in zip(axs,result['results'].items()):
  for i,(arm,s) in enumerate(r['summary'].items()):
   vals=[x['score'] for x in r['runs'][arm]];ax.scatter([i-.06,i,i+.06],vals,s=22,color='#29806b');ax.errorbar(i,s['mean'],yerr=s['sd'],fmt='_',color='#333',capsize=5)
  ax.set_xticks(range(3),['MLP','TabR-S','XGB']);ax.set_title(name);ax.set_ylabel(r['metric']+(' ↓' if r['metric']=='RMSE' else ' ↑'))
 fig.suptitle('Measured local scores · dots = seeds; bars = sample SD',fontsize=14);fig.tight_layout();save(fig,'scores')
 st=result['stats'];fig,ax=plt.subplots(figsize=(8,3.2));r=np.array(st['mean_ranks']);ax.scatter(r,np.arange(3),s=75,color='#29806b');ax.set(yticks=np.arange(3),yticklabels=st['arms'],xlim=(.8,3.2),xticks=[1,2,3],xlabel='Mean dataset rank (lower is better)');ax.set_title(f"Three datasets; Friedman p={st['friedman_p']:.3f}; CD={st['cd']:.3f}")
 start=1.02;ax.plot([start,start+st['cd']],[2.55,2.55],color='#333');ax.text(start,2.7,'Nemenyi critical difference, α=.05',fontsize=10)
 ax.set_ylim(-.5,3.1);fig.tight_layout();save(fig,'ranks')

if __name__=='__main__':architecture();mechanisms();results()
