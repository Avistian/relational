"""Actual first-batch computation and full selected experiment, with portable figures."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;OUT=P/'figures/l131';OUT.mkdir(parents=True,exist_ok=True)
t=json.loads((P/'evidence/l131/paper/seed-0/stack-trace.json').read_text());s=json.loads((P/'evidence/l131/summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l131'})
INK='#183947';BLUE='#e0edf3';GREEN='#e2f1e6';GOLD='#f7ebcf';RED='#a54134'
def save(f,name):
 f.tight_layout();f.savefig(OUT/(name+'.svg'),metadata={'Date':None});f.savefig(OUT/(name+'.png'),dpi=150,metadata={'Software':'L131'});plt.close(f)
def canvas(title,h=7):
 f,a=plt.subplots(figsize=(11,h));f.patch.set_facecolor('#faf9f6');a.set(xlim=(0,11),ylim=(0,h));a.axis('off');a.set_title(title,loc='left',weight='bold',fontsize=17,pad=18);return f,a
def box(a,x,y,w,h,text,c=BLUE):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',fc=c,ec=INK));a.text(x+w/2,y+h/2,text,ha='center',va='center',color=INK,fontsize=11)
def arrow(a,x,y,xx,yy):a.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':INK,'lw':2})
f,a=canvas('The measured stack: a root loss reaches neighboring row encoders',7.5)
box(a,.2,5.9,3,1,'512 driver queries\nSeparate cutoff + future label',GOLD)
box(a,3.9,5.9,3,1,f"Disjoint temporal contexts\n{sum(x['rows'] for x in t['inputs'].values()):,} row occurrences")
box(a,7.6,5.9,3,1,'Table-specific TensorFrames\nKeys removed from features');arrow(a,3.3,6.4,3.8,6.4);arrow(a,7,6.4,7.5,6.4)
box(a,6.3,3.9,4.3,1.1,f"Per-table four-block ResNets\nresults: {t['inputs']['results']['rows']:,} × 128\ndrivers: {t['inputs']['drivers']['rows']:,} × 128")
box(a,.2,3.9,4.6,1.1,'Age → positional encoding → linear\nH + time vector, both width 128',GREEN)
arrow(a,9,5.8,9,5.1);arrow(a,6.2,4.45,4.9,4.45)
box(a,.2,1.9,4.6,1.1,'Two typed GraphSAGE layers\nΣ neighbors → Σ relations\nLayerNorm → ReLU');arrow(a,2.5,3.8,2.5,3.1)
box(a,6.3,1.9,4.3,1.1,'First 512 driver vectors → linear head\n512 × 128 → 512 × 1\nMean L1 over 512 targets',GOLD);arrow(a,4.9,2.45,6.2,2.45)
a.annotate('Backward: loss → head → GNN → row encoders',xy=(2.4,1.8),xytext=(7,0.75),ha='center',color=RED,arrowprops={'arrowstyle':'->','color':RED,'connectionstyle':'arc3,rad=-.3','lw':2})
a.text(.2,.15,'Actual seed-0 first minibatch; raw prediction path. Evaluation clipping is a separate operation.',fontsize=10,color=INK);save(f,'architecture')
f,a=canvas('Age is indexed by query ownership, not database identity',5.2)
rows=[]
for kind in ['results','races']:
 v=t['time_inputs'][kind]
 for i in range(min(2,len(v['owner']))):
  q=v['owner'][i];cut=v['cutoff_seconds'][q];stamp=v['node_seconds'][i]
  rows.append([kind,str(i),str(q),str(stamp),str(cut),f'{(cut-stamp)/86400:.1f}'])
tab=a.table(cellText=rows,colLabels=['Table','Occurrence','Owner q','Node seconds','Cutoff[q]','Age days'],loc='center',cellLoc='center',bbox=[0,.28,1,.5]);tab.auto_set_font_size(False);tab.set_fontsize(10)
a.text(.1,4.5,'age[i] = (seed_time[batch_index[i]] − node_time[i]) / 86,400',fontsize=14,color=INK)
a.text(.1,.55,'Recorded integer timestamps; displayed ages rounded to one decimal.\nThe loader enforces eligibility; the time embedding represents the age.',fontsize=11,color=INK);save(f,'time')
f,axes=plt.subplots(4,1,figsize=(10,7));f.patch.set_facecolor('#faf9f6')
kind='results';stages=['row_encoder','time_encoder','time_added','layer_1_relu'];labels=['Row encoder H','Time vector T','H + T','After relation sum, norm and ReLU']
values=[np.array(t['stages'][stage][kind]['first_rows']) for stage in stages]
limit=max(np.abs(x).max() for x in values[:3])
for ax,v,label in zip(axes,values,labels):
 ax.imshow(v,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto');ax.set_yticks(range(3),['row 0','row 1','row 2']);ax.set_xticks(range(6),[str(x) for x in range(6)]);ax.set_title(label,loc='left',fontsize=12)
 for i in range(3):
  for j in range(6):ax.text(j,i,f'{v[i,j]:.2f}',ha='center',va='center',fontsize=10,color='white' if abs(v[i,j])>limit*.6 else INK)
axes[-1].set_xlabel('First six of 128 channels; the same first three results occurrences')
f.suptitle('Measured activations: check addition before interpreting the GNN',fontsize=16,weight='bold');save(f,'activations')
f,ax=plt.subplots(figsize=(10,5));f.patch.set_facecolor('#faf9f6');names=['encoder','time','gnn','head'];x=np.arange(4)
ax.bar(x-.18,[t['gradients'][n]['finite_l2'] for n in names],.36,label='Connected',color='#386c83')
ax.bar(x+.18,[t['detached_gradients'][n]['finite_l2'] for n in names],.36,label='Row vectors detached',color='#cba25d')
ax.set_xticks(x,['Row encoders','Time encoders','GNN','Head']);ax.set_ylabel('L2 norm over FINITE gradient entries');ax.legend();ax.set_title('Same forward output; a different backward path',loc='left',weight='bold',pad=18)
ax.text(.01,-.22,f"Original and traced encoder gradients contain {t['gradients']['encoder']['nonfinite']} matched nonfinite entries.\nExcluded from these norms and counted explicitly; parity is not gradient health.",transform=ax.transAxes,fontsize=10,color=RED);f.subplots_adjust(bottom=.25);save(f,'gradients')
f,axes=plt.subplots(1,2,figsize=(10,4.7));f.patch.set_facecolor('#faf9f6')
for ax,split in zip(axes,['val','test']):
 m=s['metrics'][split];ax.scatter(range(5),[r[split] for r in s['seeds']],color='#386c83',label='Fresh seed');ax.errorbar([5.2],[m['mean']],yerr=[m['sample_sd']],fmt='D',capsize=5,color='#97662d',label='Mean ± seed SD');ax.axhline(m['target'],ls='--',color='#64785d',label='Paper mean');ax.set_xticks([0,1,2,3,4,5.2],['0','1','2','3','4','mean']);ax.set_title(split+' MAE · detail scale');ax.set_ylabel('Finishing-position units');ax.legend(fontsize=8)
f.suptitle('Full selected experiment · five fresh ten-epoch fits',weight='bold');save(f,'scores')
print('Five measured figures built')
