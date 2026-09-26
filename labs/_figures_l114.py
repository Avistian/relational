"""Portable computation and measured-result figures; no inferred model explanations."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;D=P/'figures/l114';D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none','svg.hashsalt':'l114','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'#fcfbf8'})
TEAL='#176c68';GOLD='#ae6625';INK='#213743'
def save(fig,name):
 fig.savefig(D/f'{name}.svg',bbox_inches='tight',metadata={'Date':None});fig.savefig(D/f'{name}.png',bbox_inches='tight',dpi=155,metadata={'Software':'L114'});plt.close(fig)
def box(ax,x,y,w,h,title,detail,color=TEAL):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.015',facecolor='#eef5f3',edgecolor=color,linewidth=1.4));ax.text(x+w/2,y+h*.65,title,ha='center',va='center',weight='bold',color=INK);ax.text(x+w/2,y+h*.26,detail,ha='center',va='center',fontsize=10,color=INK)
def arrow(ax,x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':INK,'lw':1.5})
fig,ax=plt.subplots(figsize=(8.4,7.6));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title('Two feature paths; different training populations',loc='left',weight='bold',pad=16)
box(ax,.05,.82,.9,.13,'128 numeric features per paper','MLP training: 90,941 rows  •  evaluation: 169,343 rows')
arrow(ax,.5,.82,.5,.77)
box(ax,.05,.60,.9,.16,'Linear 128 → 256  ·  BN  ·  ReLU  ·  dropout 0.5','Mix coordinates → normalize → clip negatives → random mask')
arrow(ax,.5,.60,.5,.55)
box(ax,.05,.38,.9,.16,'Linear 256 → 256  ·  BN  ·  ReLU  ·  dropout 0.5','Training BN sees train rows only; evaluation uses saved statistics')
arrow(ax,.5,.38,.5,.33)
box(ax,.05,.18,.9,.14,'Linear 256 → 40  ·  log-softmax','Train: mean NLL on 90,941 labels  •  predict: largest class score')
ax.text(.05,.09,'GCN contrast: each linear transform is coupled to normalized',color=TEAL,weight='bold');ax.text(.05,.04,'neighbor aggregation; training BN sees all 169,343 feature rows.',color=TEAL)
save(fig,'architecture')
fig,(ax,b)=plt.subplots(2,1,figsize=(8.4,6.1),gridspec_kw={'height_ratios':[1.5,1]});ax.set(xlim=(-.7,3.7),ylim=(-.65,.8));ax.axis('off');ax.set_title('One graph, three different neighborhood questions',loc='left',weight='bold')
for i in [0,1]:ax.plot([i,i+1],[0,0],color=INK,lw=2,zorder=0)
for i,(name,label,known) in enumerate([('A',0,True),('B',0,False),('C',1,True),('D',1,False)]):
 ax.scatter(i,0,s=1800,color=TEAL if label==0 else GOLD,edgecolor=INK,zorder=1);ax.text(i,0,f'{name}\ny={label}',ha='center',va='center',color='white',weight='bold');ax.text(i,-.48,'train' if known else 'held out',ha='center',fontsize=11)
b.axis('off');table=b.table(cellText=[['A','1','1/1 = 1','0/1 = 0'],['B','2','1/2 = .5','2/2 = 1'],['C','1','0/1 = 0','0/1 = 0'],['D','0','undefined','undefined']],colLabels=['Node','Degree','True-label homophily','Train-neighbor fraction'],cellLoc='center',loc='center',colWidths=[.12,.15,.36,.37]);table.auto_set_font_size(False);table.set_fontsize(11);table.scale(1,1.6)
fig.text(.08,.01,'Diagnostic neighbors exclude self-loops and duplicate records. Synthetic worked example.',fontsize=10)
fig.tight_layout(rect=(0,.05,1,1));save(fig,'neighbors')
fig,(a,b)=plt.subplots(1,2,figsize=(8.4,4.5),gridspec_kw={'width_ratios':[1.2,1]});a.barh(['100 nodes','10 nodes'],[90,10],color=[TEAL,GOLD]);a.set(xlim=(0,100),xlabel='Slice accuracy (%)');a.invert_yaxis();a.text(90,.0,' 90 / 100',va='center',fontsize=10);a.text(10,1,' 1 / 10',va='center',fontsize=10)
b.axis('off');b.text(.05,.8,'Same predictions',weight='bold',fontsize=15);b.text(.05,.62,'Node-weighted:',weight='bold');b.text(.05,.49,'91 / 110 = 82.7%',color=TEAL,fontsize=17);b.text(.05,.28,'Equal-slice mean:',weight='bold');b.text(.05,.15,'(90% + 10%) / 2 = 50%',color=GOLD,fontsize=14)
fig.suptitle('Changing the denominator changes the question',weight='bold');fig.tight_layout();save(fig,'composition')
s=json.loads((P/'evidence/l114/summary.json').read_text());fig,axes=plt.subplots(2,1,figsize=(8.4,7.6))
for ax,pop in zip(axes,['valid','test']):
 rows=[r for r in s['slice_rows'] if r['population']==pop and r['family']=='homophily' and r['n']];xs=np.arange(len(rows))
 for key,shift,color,label in [('gcn',-.14,TEAL,'GCN (L112 weights)'),('mlp',.14,GOLD,'MLP (fresh L114)')]:
  means=[r[key+'_mean_percent'] for r in rows];sd=[r[key+'_sd_pp'] for r in rows];ax.errorbar(xs+shift,means,yerr=sd,fmt='o',capsize=4,color=color,label=label)
 ax.set_xticks(xs,[r['slice']+'\nn='+str(r['n']) for r in rows]);ax.set_ylim(0,100);ax.set_ylabel('Accuracy (%)');ax.set_title(pop+' · complete population',loc='left');ax.grid(axis='y',alpha=.2)
axes[0].legend(fontsize=10,loc='upper left');axes[1].set_xlabel('True-label homophily (retrospective only)')
fig.suptitle('Global advantage can reverse within a slice',weight='bold');fig.text(.06,.01,'Points: ten-run mean. Whiskers: sample seed SD, not a node-sampling confidence interval.',fontsize=10);fig.tight_layout(rect=(0,.03,1,.96));save(fig,'slices')
print('Four portable figures generated')
