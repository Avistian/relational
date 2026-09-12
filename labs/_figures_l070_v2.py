"""L070 computation figures: fixed examples plus actual paired measurements."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).parent;OUT=ROOT/'figures/l070';OUT.mkdir(exist_ok=True,parents=True)
BLUE='#245b86';RED='#a23b35';GREEN='#2b7960';GREY='#606771'
plt.rcParams.update({'font.size':12,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'#fcfbf7'})
def save(fig,name):fig.savefig(OUT/(name+'-v2.png'),dpi=160,bbox_inches='tight');plt.close(fig)
def panel(ax,title):ax.axis('off');ax.set_title(title,loc='left',weight='bold',pad=15)
def card(ax,x,y,w,h,text,color=BLUE):
 ax.add_patch(plt.Rectangle((x,y),w,h,facecolor='white',edgecolor=color,lw=1.6));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=12,color=color,linespacing=1.5)
def arrow(ax,a,b):ax.annotate('',b,a,arrowprops=dict(arrowstyle='->',color=GREY,lw=1.5))
def figures():
 fig,ax=plt.subplots(figsize=(12,6));panel(ax,'Follow row 17: selection happens before its target can be scored');ax.set(xlim=(0,12),ylim=(0,6))
 card(ax,.1,4.5,2.2,1.1,'Raw data + IDs\ntrain / val / test\n60% / 20% / 20%')
 card(ax,3,4.5,2.6,1.1,'Training rows only\nmedian + scaling\nfit candidates 0, 1')
 card(ax,6.3,4.5,2.6,1.1,'Validation X, y\nloss [0.223, 0.308]\nchoose candidate 0')
 card(ax,9.5,4.5,2.3,1.1,'Freeze\nparameters + recipe\nrow + class map')
 card(ax,.1,1.8,2.2,1.1,'Test row 17\nfeatures only\ntarget held aside',GREEN)
 card(ax,3,1.8,2.6,1.1,'Selected predictor\nfitted weights OR\nfrozen θ + context')
 card(ax,6.3,1.8,2.6,1.1,'Probability p₁₇ = .8\npositive class = 1\nrow ID retained',GREEN)
 card(ax,9.5,1.8,2.3,1.1,'Evaluator sees y₁₇=1\n−ln(.8) = .223\nno path to selection',RED)
 for a,b in [((2.3,5.05),(3,5.05)),((5.6,5.05),(6.3,5.05)),((8.9,5.05),(9.5,5.05)),((2.3,2.35),(3,2.35)),((5.6,2.35),(6.3,2.35)),((8.9,2.35),(9.5,2.35)),((10.65,4.5),(4.3,2.9))]:arrow(ax,a,b)
 ax.text(.1,.65,'Every archived probability is scored on the same original row. A raw-label check and an original-model replay answer different questions.',fontsize=11)
 save(fig,'pipeline')
 fig,axes=plt.subplots(3,1,figsize=(12,9));labels=[('TabM-mini-v2: optimize dataset-specific weights','X: B×F\nexpand B×8×F','first adapter R\nshared 3 blocks','8 independent heads\nlogits B×8×2','softmax each head\nmean → B×2'),('Nature v2 / 2.5: feature groups survive repeated attention','context + query\ngrouped feature tokens','feature attention\nwithin every row','row attention\nwithin every group','query target token\nMLP → probabilities'),('TabPFN-3: compress features, then perform row ICL','circular triplets\nN×F×128 + train y','3 induced blocks\n128 summaries / group','3 row blocks → 4 CLS\nN×F×128 → N×512','24 ICL blocks\nretrieval → P(class)')]
 for ax,(title,*texts) in zip(axes,labels):
  panel(ax,title);ax.set(xlim=(0,12),ylim=(0,2.1))
  for i,txt in enumerate(texts):card(ax,.1+i*3,1,2.7,.95,txt);arrow(ax,(2.8+i*3,1.475),(3.1+i*3,1.475)) if i<3 else None
  note='Training: average member losses. Inference: average member probabilities; no context-row attention.' if ax is axes[0] else 'Repeated feature/row attention: scores scale N·G² + G·N·C. N = C + Q; C context rows. Query targets are absent.' if ax is axes[1] else 'TabICLv2 shares the compression route; uses 12 ICL blocks and an MLP/hierarchy head. V3 decoder uses one-hot context labels.'
  ax.text(.1,.25,note,fontsize=11)
 fig.tight_layout(h_pad=2);save(fig,'architectures')
 fig,axes=plt.subplots(1,2,figsize=(11,4.3));a=axes[0];a.bar(['row 1: A','row 2: B','row 3: A'],[.2,.3,.5],color=[BLUE,RED,BLUE]);a.set(ylim=(0,1),ylabel='Attention weight',title='One head distributes one unit of mass')
 for i,v in enumerate([.2,.3,.5]):a.text(i,v+.03,str(v),ha='center')
 a=axes[1];a.bar(['A','B','C absent'],[.7,.3,0],color=[BLUE,RED,GREY]);a.set(ylim=(0,1),ylabel='Pre-log class mass',title='Add the weights of rows with the same label')
 for i,(v,txt) in enumerate(zip([.7,.3,0],['.2 + .5 = .7','.3','0 context votes'])):a.text(i,v+.04,txt,ha='center')
 fig.suptitle('Synthetic retrieval trace · before source-specific floor, log and calibration',weight='bold');fig.tight_layout();save(fig,'decoder')
 fig,axes=plt.subplots(1,2,figsize=(11,4.7));p=np.array([[.2,.8],[.1,.6]]);loss=-np.stack([np.log1p(-p[:,0]),np.log(p[:,1])],1)
 a=axes[0];a.bar(np.arange(2)-.18,loss[:,0],.36,label='y=0: −ln(1−p)');a.bar(np.arange(2)+.18,loss[:,1],.36,label='y=1: −ln(p)');a.set(xticks=[0,1],xticklabels=['candidate A','candidate B'],ylabel='Row loss (nats)',ylim=(0,.62),title='Validation rows [0, 1]');a.legend(fontsize=10)
 a=axes[1];a.bar(['A','B'],loss.mean(1),color=[GREEN,GREY]);a.set(ylabel='Mean validation loss (nats)',ylim=(0,.45),title='Select the first minimum; freeze A')
 for i,v in enumerate(loss.mean(1)):a.text(i,v+.025,f'{v:.4f}',ha='center')
 fig.suptitle('Prediction probabilities differ; the target order and loss rule stay fixed',weight='bold');fig.tight_layout();save(fig,'selection')
 fig,axes=plt.subplots(1,2,figsize=(11,4.7));a=axes[0]
 vals=np.array([[-.12,-.1,-.08],[.0,.02,.04],[-.06,-.04,-.02],[.05,.06,.07],[-.02,0,.02]])
 for i,row in enumerate(vals):a.scatter([i]*3,row,color=BLUE);a.plot([i-.2,i+.2],[row.mean()]*2,color=RED,lw=3)
 a.axhline(0,color=GREY,ls=':');a.set(xticks=range(5),xticklabels=list('ABCDE'),xlabel='Dataset: five blocks',ylabel='Paired loss gap',title='Three downstream seeds stay inside a block')
 a=axes[1];panel(a,'One bootstrap draw: [C, A, C, E, B]');means=vals.mean(1);sample=means[[2,0,2,4,1]]
 for i,(name,v) in enumerate(zip(['C','A','C','E','B'],sample)):card(a,.02,.78-i*.15,.95,.12,f'{name}: seed-mean gap {v:+.2f}')
 a.text(.02,.01,f'Resampled mean = {sample.mean():+.3f}; repeated C is intentional.\nRecompute this for 2,000 dataset draws.',fontsize=11)
 fig.suptitle('Synthetic paired-bootstrap trace · these are not the measured losses',weight='bold');fig.tight_layout();save(fig,'uncertainty')
 fig,ax=plt.subplots(figsize=(10,4.5));batches=np.arange(101);ax.plot(batches,100+batches,label='A: 100 + B',color=BLUE,lw=2);ax.plot(batches,10+4*batches,label='B: 10 + 4B',color=RED,lw=2);ax.scatter([30],[130],color=GREEN,s=70);ax.annotate('30 batches: both 130 s',(30,130),(37,85),arrowprops=dict(arrowstyle='->'));ax.set(xlabel='Identical query batches after one preparation',ylabel='Total lifecycle seconds',title='Synthetic workload: preparation versus repeated prediction');ax.legend();ax.grid(alpha=.15);fig.tight_layout();save(fig,'cost')
 path=ROOT/'_verify_l070_v2_results.json'
 if not path.exists() or json.loads(path.read_text())['status']!='COMPLETE':return
 r=json.loads(path.read_text());s=r['summary'];v=np.array(s['values']);names=[n.split('/')[0] for n in s['datasets']];arms=s['arms'];colors=plt.cm.tab10(np.arange(7))
 fig,axes=plt.subplots(1,2,figsize=(14,6),gridspec_kw={'width_ratios':[1.9,1]})
 for j,arm in enumerate(arms):
  x=np.arange(5)+(j-3)*.105
  axes[0].errorbar(x,v[:,j].mean(1),yerr=v[:,j].std(1,ddof=1),fmt='o',capsize=2,color=colors[j],label=arm,markersize=4)
  for k in range(3):axes[0].scatter(x+(k-1)*.018,v[:,j,k],color=colors[j],s=9,alpha=.5)
 axes[0].set(xticks=range(5),xticklabels=['diabetes','blood','kc1','phoneme','WDBC'],ylabel='Test log loss (nats)',title='Seed means ± sample SD; all seed points retained');axes[0].legend(fontsize=9,ncol=2);axes[0].grid(alpha=.15)
 order=sorted(arms,key=lambda a:s['mean_ranks'][a]);ranks=[s['mean_ranks'][a] for a in order]
 axes[1].scatter(ranks,range(7),color=[colors[arms.index(a)] for a in order]);axes[1].set(yticks=range(7),yticklabels=order,xlim=(.7,7.3),xlabel='Mean dataset rank (lower is better)',title=f"Friedman p = {s['friedman_p']:.3f}")
 axes[1].invert_yaxis();cd=s['nemenyi_cd'];axes[1].plot([1,1+cd],[-.65,-.65],color=RED,lw=3);axes[1].text(1,-.95,f'Nemenyi CD={cd:.2f}',color=RED,fontsize=10);axes[1].grid(alpha=.15)
 fig.suptitle('Corrected seven-arm hybrid panel · six archived arms + fresh TabM-mini-v2',weight='bold');fig.tight_layout();save(fig,'results')
 fig,axes=plt.subplots(1,2,figsize=(12,5))
 for j,arm in enumerate(['XGBoost-fresh-control','TabM-mini-v2']):
  a=axes[j]
  for i,name in enumerate(s['datasets']):
   rows=[z for z in r['records'] if z['arm']==arm and z['dataset']==name];gaps=[z['intervention']['delta_loss'] for z in rows]
   a.scatter([i]*3,gaps,color=colors[j]);a.errorbar(i,np.mean(gaps),yerr=np.std(gaps,ddof=1),fmt='_',markersize=18,capsize=4,color=RED)
  a.axhline(0,color=GREY,ls=':');a.set(xticks=range(5),xticklabels=['diabetes','blood','kc1','phoneme','WDBC'],ylabel='Erased − baseline loss (nats)',title=arm);a.grid(alpha=.15)
 fig.suptitle('Fresh fixed-model intervention · erase the train-most-correlated feature',weight='bold');fig.tight_layout();save(fig,'intervention')
if __name__=='__main__':figures()
