"""Portable computation-revealing L069 figures; generated arithmetic, not decorations."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures/l069';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans','savefig.facecolor':'#fcfbf7'})
BLUE='#245b86';RED='#a23b35';GREEN='#2b7960';GREY='#606771'
def save(fig,name):fig.savefig(OUT/(name+'-v2.png'),dpi=160,bbox_inches='tight');plt.close(fig)
def panel(ax,title):ax.axis('off');ax.set_title(title,loc='left',weight='bold',pad=15)
def card(ax,x,y,w,h,text,color=BLUE):
 ax.add_patch(plt.Rectangle((x,y),w,h,facecolor='white',edgecolor=color,lw=1.6));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=12,color=color,linespacing=1.5)
def arrow(ax,a,b):ax.annotate('',b,a,arrowprops=dict(arrowstyle='->',color=GREY,lw=1.5))
def figures():
 fig,ax=plt.subplots(figsize=(11,5.4));panel(ax,'Trace query row 17: which labels can reach the model?');ax.set_xlim(0,11);ax.set_ylim(0,5)
 card(ax,.1,3.4,2.25,1.1,'Original table\nX: N × F, y: N\nrow IDs retained')
 card(ax,3.0,3.4,3.0,1.1,'Context IDs: [2, 8, 23, …]\nX context + y context\nheld class excluded')
 card(ax,.1,1.4,2.25,1.15,'Query IDs: [17, …]\nX query only\nintervention applied',RED)
 card(ax,3.0,1.4,3.0,1.15,'FULL HISTORICAL v2\n12 blocks · all 81 tensors\nquery targets censored')
 card(ax,6.6,1.4,2.0,1.15,'Probabilities\nrow 17: [.2, .8]\nclass map: [A, B]')
 card(ax,8.95,1.4,1.95,1.15,'Evaluator\ntrue y₁₇ = C\nunsupported',RED)
 arrow(ax,(2.35,3.95),(3,3.95));arrow(ax,(1.22,3.4),(1.22,2.55));arrow(ax,(4.5,3.4),(4.5,2.55));arrow(ax,(2.35,1.97),(3,1.97));arrow(ax,(6,1.97),(6.6,1.97));arrow(ax,(8.6,1.97),(8.95,1.97))
 ax.text(.1,.5,'Every query is retained. Detection score = 1 − max(p).\nNaming loss uses semantic class support; missing C gets zero mass before declared clipping.',fontsize=12)
 save(fig,'pipeline')
 fig,(a,b)=plt.subplots(1,2,figsize=(11,5),gridspec_kw={'width_ratios':[1.35,1]});panel(a,'Same probabilities, different detector')
 pmax=np.array([.5,.34,.85,.55,.8,.7]);novel=np.array([1,1,0,0,0,1]);tab=a.table(cellText=[[i+1,'novel' if y else 'known',f'{m:.2f}',f'{1-m:.2f}',int(.4<=m<=.6)] for i,(m,y) in enumerate(zip(pmax,novel))],colLabels=['Row','Truth','max p','1−max p','Interval'],loc='center',cellLoc='center');tab.auto_set_font_size(False);tab.set_fontsize(12);tab.scale(1,1.8)
 from sklearn.metrics import roc_curve
 for name,score,color in [('continuous',1-pmax,BLUE),('interval',((pmax>=.4)&(pmax<=.6)).astype(int),RED)]:
  x,y,_=roc_curve(novel,score);b.plot(x,y,'o-',label=name,color=color)
 b.plot([0,1],[0,1],':',color=GREY);b.set(xlim=(-.02,1.02),ylim=(-.02,1.05),xlabel='Known false-positive rate',ylabel='Novel true-positive rate',title='AUC: 8/9 → 1/2');b.legend(loc='lower right');fig.text(.08,.03,'Row 2 is most uncertain but outside [.4,.6]. Binary AUC = (TPR + TNR)/2 = (1/3 + 2/3)/2.',fontsize=11)
 fig.tight_layout(rect=(0,.08,1,1));save(fig,'novelty')
 fig,ax=plt.subplots(figsize=(10.5,4.2));panel(ax,'Remove one sensor; retain the query row and other feature');ax.set_xlim(0,10.5);ax.set_ylim(0,4)
 card(ax,.1,1.3,2.7,1.7,'CONTEXT sensor\n[2, 4, 6]\nmean = (2+4+6)/3 = 4')
 card(ax,3.45,1.3,2.5,1.7,'BEFORE query\nrow 17: [1, 10]\nrow 22: [7, 20]')
 card(ax,7.0,1.3,3.2,1.7,'AFTER query\nrow 17: [4, 10]\nrow 22: [4, 20]',RED)
 arrow(ax,(2.8,2.15),(3.45,2.15));arrow(ax,(5.95,2.15),(7,2.15));ax.text(.1,.35,'Training rows and weights remain fixed. Replacement destroys a distinction; it does not delete a required column.\nIf query 7 becomes 9, context mean stays 4. A test-fitted mean would change to 5.',fontsize=11)
 save(fig,'features')
 fig,axes=plt.subplots(1,3,figsize=(11,4.3),sharex=True,sharey=True);rng=np.random.default_rng(69);x=rng.normal(size=(90,2))
 for ax,kind in zip(axes,['IID','Covariate: X₀ + 1.5','Concept: reverse y']):
  q=x.copy()
  if kind.startswith('Covariate'):q[:,0]+=1.5
  y=q[:,0]>0
  if kind.startswith('Concept'):y=~y
  for k,color in [(0,BLUE),(1,RED)]:ax.scatter(q[y==k,0],q[y==k,1],s=22,facecolors='none' if k==0 else color,edgecolors=color,marker='o',label=f'class {k}')
  ax.axvline(0,color=GREY,ls='--');ax.set(title=kind,xlabel='Coordinate 0',xlim=(-3.5,4.8),ylim=(-3.6,3.6))
 axes[0].set_ylabel('Coordinate 1');axes[0].legend(fontsize=10);fig.text(.07,.015,'Illustrative seed 69 coordinates. Concept keeps every point in place and swaps its label; frozen f(X) cannot reveal that swap.',fontsize=10.5);fig.tight_layout(rect=(0,.06,1,1));save(fig,'shift')
 fig,(a,b)=plt.subplots(1,2,figsize=(10.5,4.4));panel(a,'100 rows; every prediction is class 0');tab=a.table(cellText=[[90,0],[10,0]],rowLabels=['True 0','True 1'],colLabels=['Pred 0','Pred 1'],loc='center',cellLoc='center');tab.auto_set_font_size(False);tab.set_fontsize(15);tab.scale(1,2.1);a.text(.05,.12,'TP₁=0, FN₁=10, FP₁=0\nTP₀=90, FN₀=0, FP₀=10',transform=a.transAxes,fontsize=12)
 names=['Accuracy','Balanced accuracy','Macro F1','Class 1 F1'];vals=[.9,.5,(180/190)/2,0];b.barh(names[::-1],vals[::-1],color=[GREY,GREEN,BLUE,RED][::-1]);b.set_xlim(0,1.15);b.set_xlabel('Score (higher is better)')
 for i,v in enumerate(vals[::-1]):b.text(v+.025,i,f'{v:.4f}',va='center')
 fig.tight_layout();save(fig,'objectives')
 result=ROOT/'_verify_l069_v2_results.json'
 if result.exists():
  r=json.loads(result.read_text());summary=r['summary'];fig,axes=plt.subplots(1,3,figsize=(13,4.8));colors={'v2':BLUE,'xgboost':RED}
  for ax,dataset in zip(axes,['cmc','winequality-red','winequality-white']):
   for arm,offset in [('v2',-.12),('xgboost',.12)]:
    row=next(v for v in summary if v['axis']=='novelty' and v['dataset']==dataset and v['arm']==arm and v['condition']=='leave-one-class-out')
    for i,m in enumerate(['auc','interval_auc']):
     values=row[m+'_seed_values'];assert min(values+[row[m+'_mean']-row[m+'_sd']])>=.3 and max(values+[row[m+'_mean']+row[m+'_sd']])<=.9,'Novelty detail axis would clip evidence';ax.scatter(np.array([i+offset]*3)+np.linspace(-.035,.035,3),values,color=colors[arm],s=22,alpha=.8)
     ax.errorbar(i+offset,row[m+'_mean'],yerr=row[m+'_sd'],fmt='_',markersize=15,color=colors[arm],capsize=4,label=arm if i==0 else None)
   ax.axhline(.5,ls=':',color=GREY);ax.set_xticks([0,1],['Continuous','Interval']);ax.set_ylim(.3,.9);ax.set_title(dataset);ax.set_ylabel('Novelty ROC-AUC')
  axes[0].legend();fig.suptitle('Full rows, every held class; dots are split-seed means',weight='bold');fig.tight_layout();save(fig,'results')
  fig,axes=plt.subplots(1,3,figsize=(13,4.8),sharey=True)
  for ax,dataset in zip(axes,['iris','cmc','winequality-red']):
   for arm,color,style in [('v2',BLUE,'-'),('xgboost',RED,'--')]:
    records=[v for v in r['records'] if v['axis']=='features' and v['dataset']==dataset and v['arm']==arm]
    levels=sorted({v['removed_fraction'] for v in records});means=[];sd=[]
    for level in levels:
     values=[v['accuracy'] for v in records if v['removed_fraction']==level]
     # Iris0% and20% name the same actual mask; keep unique seed values.
     values=[next(v['accuracy'] for v in records if v['removed_fraction']==level and v['seed']==seed) for seed in [42,2023,789]]
     means.append(np.mean(values));sd.append(np.std(values,ddof=1));ax.scatter([level]*3,values,s=15,color=color,alpha=.45)
    ax.errorbar(levels,means,yerr=sd,fmt=style+'o',color=color,capsize=3,label=arm)
   ax.set(title=dataset,xlabel='Actual fraction removed',xlim=(-.03,1.03),ylim=(0,1.03));ax.set_xticks([0,.25,.5,.75,1]);ax.grid(alpha=.15)
  axes[0].set_ylabel('Accuracy');axes[0].legend();fig.suptitle('Feature-loss curves: same context and query IDs across levels',weight='bold');fig.tight_layout();save(fig,'feature_results')
 else:
  for name in ['results','feature_results']:
   fig,ax=plt.subplots(figsize=(11,3));panel(ax,'Author panel in progress — no unmeasured result shown');save(fig,name)
if __name__=='__main__':figures()
