"""Computational figures for the complete L065 extraction pipeline."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures/l065';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#faf9f6','axes.facecolor':'#faf9f6'})

def save(fig,name):fig.savefig(OUT/(name+'-v2.png'),dpi=150,bbox_inches='tight');plt.close(fig)
def box(ax,x,y,w,h,text,color='#e7eef5'):
 from matplotlib.patches import FancyBboxPatch
 assert 0<=x<x+w<=1 and 0<=y<y+h<=1
 previous=getattr(ax,'_l065_boxes',[])
 assert all(x+w<=a or a+c<=x or y+h<=b or b+d<=y for a,b,c,d in previous), 'Overlapping pipeline boxes'
 ax._l065_boxes=previous+[(x,y,w,h)]
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.01',facecolor=color,edgecolor='#607080'));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.8,color='#435665'))

def build():
 fig,ax=plt.subplots(figsize=(11,8));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
 box(ax,.08,.88,.84,.09,'Outer rows → stratified 64% train / 16% validation / 20% test')
 box(ax,.04,.71,.43,.12,'TEN training folds\ncontext = other 9; query = held fold\nrecord IDs; never supply own target')
 box(ax,.55,.71,.41,.12,'Evaluation context = all train\nquery = validation OR test\nlabels remain outside encoder')
 arrow(ax,(.28,.88),(.25,.83));arrow(ax,(.73,.88),(.75,.83))
 box(ax,.1,.51,.8,.14,'Context constant filter → paired feature groups + target token\n12 frozen blocks: feature attention → context row attention → MLP\nH after each block: 1 × (C+Q) × (G+1) × 192')
 arrow(ax,(.25,.71),(.3,.65));arrow(ax,(.75,.71),(.7,.65))
 box(ax,.03,.32,.57,.13,'Query target states: H[:, C:, −1, :]\nScatter training folds → Z_train\nDirect query states → Z_valid and Z_test', '#e3f0e8')
 box(ax,.68,.32,.29,.13,'Native target → GELU head\n192 → 768 → 10; active classes\nlogits / 0.9 → softmax')
 arrow(ax,(.35,.51),(.31,.45));arrow(ax,(.82,.51),(.82,.45))
 box(ax,.03,.12,.57,.14,'Select 1–3 layers → width 192r\nZ_train fits scaler/head; Z_valid chooses\nFrozen transforms/head apply to Z_test', '#e3f0e8')
 arrow(ax,(.31,.32),(.31,.26))
 box(ax,.68,.12,.29,.14,'Untouched test rows\ncompare native / head\nsave all probabilities')
 arrow(ax,(.6,.19),(.68,.19));arrow(ax,(.82,.32),(.82,.26))
 ax.text(.5,.035,'No gradients update the encoder. The head learns weights from outer training labels.',ha='center',fontsize=11)
 save(fig,'architecture')
 fig,axes=plt.subplots(1,2,figsize=(11,4.1));values=np.array([[0,0],[0,0],[1,0],[1,-2]])
 axes[0].imshow(values,cmap='coolwarm',vmin=-2,vmax=2,aspect='auto');axes[0].set(xticks=[0,1],xticklabels=['target rank','missing flag'],yticks=range(4),yticklabels=['context y=0','context y=0','context y=1','query y unknown'],title='Target encoder INPUTS (worked example)')
 for i in range(4):
  for j in range(2):axes[0].text(j,i,str(values[i,j]),ha='center',va='center',color='black',fontsize=17)
 result=ROOT/'_check_l065_v2_results.json'
 if result.exists():d=json.loads(result.read_text());axes[1].plot(range(1,13),d['context_own_label_per_layer_delta'],'o-',label='same row in context');axes[1].plot(range(1,13),np.zeros(12),'s--',label='same row excluded from context')
 axes[1].set(xlabel='Complete block number',ylabel='Max |Δ hidden coordinate|',title='Measured: flip only the row’s own label');axes[1].legend(fontsize=10);fig.tight_layout();save(fig,'roles')
 fig,ax=plt.subplots(figsize=(11,4));ax.axis('off');rows=[['fold 0','[0,2,4,5]','[1,3]','[1,3]'],['fold 1','[1,3,4,5]','[0,2]','[0,2]'],['fold 2','[0,1,2,3]','[4,5]','[4,5]']]
 tab=ax.table(cellText=rows,colLabels=['Call','Context positions','Query positions','Scatter destinations'],loc='upper center',cellLoc='center');tab.auto_set_font_size(False);tab.set_fontsize(13);tab.scale(1,2.1)
 ax.text(.5,.36,'Returned order: [1,3] + [0,2] + [4,5] = [1,3,0,2,4,5]',ha='center',fontsize=15,color='#a24435');ax.text(.5,.16,'Correct stored order after scatter: [0,1,2,3,4,5]',ha='center',fontsize=15,color='#23714e');ax.set_title('Synthetic identity trace: same shape, different label alignment',pad=15);save(fig,'scatter')
 fig,ax=plt.subplots(figsize=(11,4));ax.axis('off');rows=[['[6]','0.1','0.80','0.84','selected (simpler tie)'],['[9,12]','1.0','0.80','0.86','not selected'],['[12]','1.0','0.78','0.89','not selected']]
 tab=ax.table(cellText=rows,colLabels=['Layers','C','Validation accuracy','Test accuracy','Decision'],loc='center',cellLoc='center');tab.auto_set_font_size(False);tab.set_fontsize(12);tab.scale(1,2.3)
 ax.text(.5,.04,'Synthetic scores: test changes can never select another head. Highest validation score, then fixed tie rule.',ha='center',fontsize=11);ax.set_title('A test winner is not a legal selection rule',pad=5);save(fig,'selection')
 p=ROOT/'_verify_l065_v2_results.json'
 if p.exists():
  data=json.loads(p.read_text());names=['raw','vanilla','layer_6','layer_9','layer_12','combined','native'];fig,axes=plt.subplots(1,3,figsize=(14,5),sharey=True)
  for ax,dataset in zip(axes,data['config']['datasets']):
   records=[r for r in data['records'] if r['dataset']==dataset]
   for k,r in enumerate(records):ax.plot([r['methods'][n]['accuracy'] for n in names],np.arange(len(names))+(k-1)*.16,'o',label='seed '+str(r['seed']))
   ax.set(title=dataset,xlim=(0,1.01),yticks=range(7),yticklabels=names,xlabel='Test accuracy');ax.grid(axis='x',alpha=.2);ax.legend(fontsize=10);ax.tick_params(labelleft=True)
  axes[0].invert_yaxis();fig.suptitle('Fresh full-data local experiment · individual seeds, no confidence interval');fig.tight_layout();save(fig,'results')
  fig,axes=plt.subplots(1,3,figsize=(13,4.4),sharey=True)
  for ax,dataset in zip(axes,data['config']['datasets']):
   records=[r for r in data['records'] if r['dataset']==dataset]
   for j,name in enumerate(['vanilla','layer_12','raw']):
    for k,r in enumerate(records):
     m=r['methods'][name];ax.plot([m['fitted_train_accuracy'],m['accuracy']],[j+(k-1)*.15]*2,'-',color=['#a34b36','#227653','#4673a8'][j],alpha=.65);ax.plot(m['fitted_train_accuracy'],j+(k-1)*.15,'x',color=['#a34b36','#227653','#4673a8'][j]);ax.plot(m['accuracy'],j+(k-1)*.15,'o',color=['#a34b36','#227653','#4673a8'][j])
   ax.set(title=dataset,yticks=range(3),yticklabels=['context layer 12','query layer 12','raw features'],xlim=(0,1.02),xlabel='Accuracy');ax.grid(axis='x',alpha=.2);ax.tick_params(labelleft=True)
  axes[0].invert_yaxis();fig.suptitle('Fitted training (×) and untouched test (●) · each pair shares an outer split');fig.tight_layout();save(fig,'gaps')
if __name__=='__main__':build()
