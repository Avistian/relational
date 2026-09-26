"""Measured diagnostic curves and computation-specific portable figures."""
import json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;D=P/'figures/l116';D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none','svg.hashsalt':'l116','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'#fcfbf8'})
TEAL='#176c68';GOLD='#a85c21';INK='#233846';RED='#a5383d'
def save(fig,name):
 fig.savefig(D/f'{name}.svg',bbox_inches='tight',metadata={'Date':None});fig.savefig(D/f'{name}.png',bbox_inches='tight',dpi=155,metadata={'Software':'L116'});plt.close(fig)
def box(ax,x,y,w,h,title,detail,color=TEAL):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',facecolor='#edf5f2',edgecolor=color,linewidth=1.4));ax.text(x+w/2,y+h*.73,title,ha='center',va='center',weight='bold',color=INK,fontsize=12);ax.text(x+w/2,y+h*.32,detail,ha='center',va='center',fontsize=10.5,color=INK)
def arrow(ax,x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':INK,'lw':1.5})
fig,ax=plt.subplots(figsize=(8.4,10));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title('Trace one update, then inspect what changed',loc='left',weight='bold',pad=18)
box(ax,.04,.86,.92,.10,'Inputs: X and fixed graph S','169,343 × 128 features; binary graph + one loop + D⁻¹ᐟ² A D⁻¹ᐟ²')
box(ax,.04,.65,.92,.14,'Forward: THREE graph convolutions','S(HWᵀ) + b → BN → ReLU → dropout .5, twice\nThird S(HWᵀ) + b → log-softmax: 169,343 × 40')
arrow(ax,.5,.86,.5,.80)
box(ax,.04,.47,.92,.11,'Supervision: select 90,941 TRAIN rows','L = mean(−log p[training ID, true class])\nHeld-out labels never enter L',GOLD);arrow(ax,.5,.65,.5,.59)
box(ax,.04,.27,.43,.12,'Backward','g = ∂L/∂θ\nMeasure ‖g‖₂, finite values');box(ax,.55,.27,.41,.12,'Optimizer step','θafter = Adam(θbefore, g)\nMeasure ‖θafter − θbefore‖₂',RED)
arrow(ax,.25,.47,.25,.40);arrow(ax,.48,.33,.54,.33)
box(ax,.04,.06,.92,.12,'Evaluate → select → restore → predict','eval mode; first best validation epoch over 500 updates\nSave weights + BN buffers; argmax gives final classes')
arrow(ax,.76,.27,.76,.19);ax.text(.04,.01,'An absent step breaks the red box even when backward succeeds.',fontsize=11,color=RED);save(fig,'trace')
# Full-data interventions: paired seed and source, one changed operation.
fig,axes=plt.subplots(3,1,figsize=(8.4,10),sharex=True)
for arm,color in [('broken',RED),('repaired',TEAL)]:
 r=json.loads((P/f'evidence/l116/{arm}/seed-101/result.json').read_text());h=r['history'];epochs=[v['epoch'] for v in h]
 for ax,key in zip(axes,['loss','gradient_l2','update_l2']):ax.plot(epochs,[v[key] for v in h],label=arm,color=color,lw=2)
for ax,label in zip(axes,['Training NLL','Gradient L2 norm','Parameter-change L2 norm']):ax.set_ylabel(label);ax.grid(alpha=.2)
axes[0].legend();axes[0].set_title('Same GCN and seed; only the optimizer step changes',loc='left',weight='bold');axes[-1].set_xlabel('Epoch · complete ogbn-arxiv · course intervention')
fig.tight_layout();save(fig,'updates')
t=json.loads((P/'_teaching_l116_results.json').read_text())
fig,ax=plt.subplots(figsize=(8.4,5.4))
for gain,color in [(.5,TEAL),(1.,INK),(2.,RED)]:
 rows=[r for r in t['gradient_chain'] if r['gain']==gain];ax.plot([r['depth'] for r in rows],[r['input_gradient'] for r in rows],marker='o',label=f'gain a = {gain:g}',color=color)
ax.set_yscale('log');ax.set_xlabel('Depth k');ax.set_ylabel('Input derivative ∂hₖ/∂x (log scale)');ax.set_title('hₖ = a × hₖ₋₁: backward gain is aᵏ',loc='left',weight='bold');ax.grid(alpha=.2);ax.legend();fig.tight_layout();save(fig,'gradient')
fig,axes=plt.subplots(2,1,figsize=(8.4,8));r=t['smoothing'];k=[v['depth'] for v in r]
for i,name in enumerate(['A: degree 2','B: degree 3','C: degree 2']):axes[0].plot(k,[v['h'][i] for v in r],marker='o',label=name)
axes[0].set_ylabel('Raw node value h');axes[0].legend();axes[0].set_title('Symmetric propagation preserves a degree-shaped limit',loc='left',weight='bold');axes[0].grid(alpha=.2)
axes[1].plot(k,[max(v['degree_corrected_variance'],1e-20) for v in r],marker='o',color=TEAL);axes[1].set_yscale('log');axes[1].set_ylabel('Variance of h / √degree');axes[1].set_xlabel('Repeated propagation depth · no learned weights');axes[1].grid(alpha=.2)
fig.text(.10,.01,'Three-node path + self-loops; start [2,4,8]. Display floor for variance: 10⁻²⁰.',fontsize=10);fig.tight_layout(rect=(0,.04,1,1));save(fig,'smoothing')
fig,ax=plt.subplots(figsize=(8.4,7.3));ax.axis('off');ax.set_title('Local position is not global node identity',loc='left',weight='bold',pad=20)
rows=[['0','4','0','seed → supervised'],['1','1','1','seed → supervised'],['2','5','1','context only'],['3','0','1','context only']]
table=ax.table(cellText=rows,colLabels=['Local row','n_id[row]','Global label','Role'],cellLoc='center',colWidths=[.17,.20,.20,.40],bbox=[.01,.42,.98,.49]);table.auto_set_font_size(False);table.set_fontsize(12)
for (row,col),cell in table.get_celld().items():
 cell.set_edgecolor('#ccd7d4');cell.set_facecolor('#d7eae4' if row in [1,2] else '#faf9f6')
 if row==0:cell.set_text_props(weight='bold');cell.set_facecolor('#e5ebee')
ax.text(.02,.28,'Correct: global_labels[n_id[:2]] = [0, 1]',fontsize=14,color=TEAL,weight='bold');ax.text(.02,.18,'Faulty:   global_labels[:2] = [1, 1]',fontsize=14,color=RED,weight='bold');ax.text(.02,.06,'Use output[:2] for both rows. Context can send messages;\nits label does not become a supervised training target.',fontsize=12);save(fig,'sampling')
s=json.loads((P/'evidence/l116/summary.json').read_text());fig,ax=plt.subplots(figsize=(8.4,5))
for i,(pop,color) in enumerate([('valid',TEAL),('test',GOLD)]):
 r=s['summary'][pop];ax.scatter([i+(j-4.5)*.035 for j in range(10)],[v[pop+'_percent'] for v in s['seeds']],color=color,alpha=.6);ax.errorbar(i+.24,r['mean_percent'],yerr=r['sample_sd_pp'],fmt='D',capsize=5,color=color);ax.hlines(r['target_percent'],i-.28,i+.32,color=color,linestyle='--')
ax.set_xticks([0,1],['Validation','Test']);ax.set_ylabel('Accuracy (%)');ax.set_title('Repaired trainer: ten fresh 500-epoch GCN fits',loc='left',weight='bold');ax.grid(axis='y',alpha=.2)
fig.text(.08,.02,'Dots: seeds. Diamond: mean ± sample seed SD. Dashed: OGB Table 6 mean.',fontsize=10);fig.tight_layout(rect=(0,.07,1,1));save(fig,'results');print('Six figures built')
