"""Measured diagnostics and coherent worked computation in portable figures."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;D=P/'figures/l152';D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l152','axes.spines.top':False,'axes.spines.right':False})
INK='#17324d';TEAL='#007f82';GOLD='#a35919'
def box(a,x,y,w,h,text):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02',facecolor='#eef7f6',edgecolor=TEAL,linewidth=1.5);a.add_patch(p);t=a.text(x+w/2,y+h/2,text,ha='center',va='center',color=INK);a._pairs=getattr(a,'_pairs',[])+[(p,t)]
def arrow(a,x,y,u,v):a.annotate('',(u,v),(x,y),arrowprops={'arrowstyle':'->','color':INK,'lw':1.5})
def save(f,name):
 f.canvas.draw();r=f.canvas.get_renderer()
 for a in f.axes:
  for p,t in getattr(a,'_pairs',[]):
   b=p.get_window_extent(r);q=t.get_window_extent(r);assert b.x0<=q.x0 and b.x1>=q.x1 and b.y0<=q.y0 and b.y1>=q.y1,(name,t.get_text())
 for ext in ['png','svg']:f.savefig(D/f'{name}.{ext}',dpi=160,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else None)
 plt.close(f)
 svg=D/f'{name}.svg';svg.write_text('\n'.join(x.rstrip() for x in svg.read_text().splitlines())+'\n')
f,a=plt.subplots(figsize=(9,9));a.set(xlim=(0,9),ylim=(0,9));a.axis('off')
a.text(.2,8.7,'A DRIVER QUERY BECOMES A SCALAR PREDICTION',weight='bold',fontsize=14,color=INK)
box(a,.3,7.2,8.3,1,'(driverId, cutoff t) → query-owned temporal neighborhoods\nFull nine-table graph: 74,063 rows; fanouts 128 then 64')
arrow(a,4.45,7.2,4.45,6.8)
box(a,.3,5.65,8.3,1.15,'Typed Frame ResNets → row vectors [rows,128]\nAdd relative-time encodings, retaining each owner’s cutoff\nTwo GraphSAGE layers: sum neighbors, then sum relations')
arrow(a,4.45,5.65,4.45,5.25)
box(a,.3,3.75,8.3,1.5,'Inside one relation: illustrative scalar trace\nLegal neighbor coordinates [2,6] → sum 8\nUnit neighbor weight; destination contribution 1 → output 9\nFuture coordinate 20 is excluded; including it gives 29\nActual layers use learned 128-dimensional transformations')
arrow(a,4.45,3.75,4.45,3.3)
box(a,.3,2.35,8.3,.95,'Root vectors [batch,128] → regression MLP → [batch,1]\nNo sigmoid: prediction has finishing-position units')
arrow(a,2.5,2.35,2.5,1.9);arrow(a,6.5,2.35,6.5,1.9)
box(a,.3,.4,3.9,1.5,'TRAINING\nRaw prediction → L1 loss\nAdam .005; ten epochs\nUpdate encoders + GNN + head')
box(a,4.7,.4,3.9,1.5,'EVALUATION\nClip to train percentiles\nValidation selects checkpoint\nTest: MAE + diagnostics');save(f,'architecture')
y=np.array([0,1,1,9]);x=np.linspace(0,10,401);mae=np.abs(x[:,None]-y).mean(1);mse=((x[:,None]-y)**2).mean(1)
f,axes=plt.subplots(1,2,figsize=(10,4.5))
for a,v,label,target,color in [(axes[0],mae,'Mean absolute error',1,TEAL),(axes[1],mse,'Mean squared error',2.75,GOLD)]:
 a.plot(x,v,color=color,lw=2);a.axvline(target,color=INK,ls='--',label=f'Minimum at {target:g}');a.set(xlabel='Constant prediction',ylabel=label,title='Same outcomes: [0,1,1,9]');a.legend();a.grid(alpha=.15)
f.suptitle('A skewed sample separates the median from the mean',color=INK);f.tight_layout();save(f,'loss')
s=json.loads((P/'evidence/l152/summary.json').read_text());rows=s['bins']['0']['test'];f,axes=plt.subplots(2,1,figsize=(9,7.2),gridspec_kw={'height_ratios':[2,1]})
x=np.arange(len(rows));bottom=np.zeros(len(rows))
for key,color in [('below',TEAL),('equal','#94b9b1'),('above',GOLD)]:
 v=np.array([r[key] or 0 for r in rows]);axes[0].bar(x,v,bottom=bottom,label=key,color=color);bottom+=v
axes[0].axhline(.5,color=INK,ls='--',label='Half the outcomes');axes[0].set(ylim=(0,1),ylabel='Fraction of outcomes',title='Measured seed 0 test bins · edges fixed from validation');axes[0].legend(ncol=4,fontsize=9,loc='upper center',bbox_to_anchor=(.5,1.18))
axes[0].set_xticks(x,[f"{i}\nn={r['n']}" for i,r in enumerate(rows)])
axes[1].bar(x,[r['median_violation'] or 0 for r in rows],color=INK);axes[1].set(ylabel='Median violation',xlabel='Prediction bin (low → high)',ylim=(0,.5));axes[1].set_xticks(x)
f.text(.08,.015,'Descriptive finite-sample diagnostic; counts and dependence limit conclusions. No fitted correction.',fontsize=10);f.tight_layout(rect=[0,.035,1,1]);save(f,'calibration')
f,a=plt.subplots(figsize=(9,5));v=s['metrics']['test'];a.scatter(range(5),v['values'],s=65,color=TEAL,label='Fresh primary seeds');a.errorbar(5.3,v['mean'],yerr=v['sample_sd'],fmt='D',color=GOLD,capsize=6,label='Mean ± sample seed SD');a.axhline(4.022,ls='--',color=INK,label='Paper test MAE 4.022');a.axhspan(3.822,4.222,alpha=.08,color=TEAL,label='Descriptive ±.20 band');a.set_xticks([0,1,2,3,4,5.3],['0','1','2','3','4','mean']);a.set(xlabel='Training seed',ylabel='Test MAE · finishing-position units',title='Complete Table 7 selected experiment · lower is better');a.legend(fontsize=10);a.grid(axis='y',alpha=.15);f.tight_layout();save(f,'results')
print('Four figures: architecture, loss target, measured median bins, seed scores')
