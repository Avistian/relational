"""Portable mechanism figures; numerical result figure reads only collected evidence."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;O=P/'figures/l110';O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','svg.hashsalt':'l110','axes.spines.top':False,'axes.spines.right':False})
INK='#233b45';TEAL='#087e82';RUST='#b65b2c';GRAY='#647579'
def canvas(title,subtitle,h):
 f,ax=plt.subplots(figsize=(10,h));f.patch.set_facecolor('#fbfcf9');ax.set(xlim=(0,10),ylim=(0,h));ax.axis('off');ax.text(.2,h-.35,title,color=INK,fontsize=17,weight='bold');ax.text(.2,h-.72,subtitle,color=GRAY,fontsize=10.5);return f,ax
def box(ax,x,y,w,h,s,col=TEAL):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',facecolor='white',edgecolor=col,lw=1.5));ax.text(x+.13,y+h-.13,s,va='top',color=INK,fontsize=10.5,linespacing=1.55)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':2,'color':TEAL})
def save(f,n):
 f.savefig(O/(n+'.svg'),bbox_inches='tight',metadata={'Date':None});f.savefig(O/(n+'.png'),dpi=150,bbox_inches='tight',metadata={'Software':'L110'});plt.close(f)
f,ax=canvas('TGN-attn: the current event enters only after scoring','Wikipedia · one layer · two attention heads · dimensions per event unless stated',8.5)
box(ax,.3,5.65,4.35,1.65,'OLD queued message (688)\n[own 172 | other 172 | edge 172 | time 172]\nGRU(message, memory 172) → state 172\nPrior positive interactions only')
box(ax,5.25,5.65,4.35,1.65,'QUESTION: user u, page v, time t\nAlso sample one negative page v−\nGather ten historical edges with time < t\nCurrent event features are not read here')
box(ax,.3,3.05,9.3,1.8,'HISTORY ATTENTION for u, v and v−\nQuery = [own state 172 | time(0) 172] → 344\nKey/value = [neighbor state 172 | past edge 172 | time(t−τ) 172] → 516\nTwo-head attention + merge with own state → embedding 172')
arrow(ax,(2.5,5.5),(2.5,4.98));arrow(ax,(7.5,5.5),(7.5,4.98))
box(ax,.3,.6,4.35,1.65,'SCORE FIRST\nDecoder(zᵤ, zᵥ) and decoder(zᵤ, zᵥ−)\nSigmoid → p+, p−\nTraining: BCE+ + BCE− → gradients')
box(ax,5.25,.6,4.35,1.65,'QUEUE AFTERWARD\nNow read current edge features (172)\nBuild messages for real endpoints only\nDetach between training batches',RUST)
arrow(ax,(2.5,2.92),(2.5,2.4));arrow(ax,(4.72,1.3),(5.13,1.3));ax.text(.35,.12,'Inference: fixed parameters, evolving temporal state. Negative candidates never create messages.',color=GRAY,fontsize=10)
save(f,'architecture')
f,ax=canvas('Which predictor did validation select?','Illustrative epochs 12 and 17 · state consistency is separate from test-label leakage',5.8)
box(ax,.35,2.7,4.2,1.85,'Selected epoch 12\nweights θ₁₂ + memory M₁₂\nclocks T₁₂ + queued messages Q₁₂\nTogether: one reproducible predictor')
box(ax,5.3,2.7,4.2,1.85,'Stopping epoch 17\nweights θ₁₇ + memory M₁₇\nclocks T₁₇ + queued messages Q₁₇\nValidation has stopped improving',RUST)
box(ax,.35,.5,4.2,1.3,'Clean restore: θ₁₂, M₁₂, T₁₂, Q₁₂\nClone all state; restore each branch\nExpected next prediction is recovered')
box(ax,5.3,.5,4.2,1.3,'Release restore: θ₁₂, M₁₂, T₁₂, Q₁₇\nstate_dict omits the Python queue\nA mixed state survives early stopping',RUST)
arrow(ax,(2.4,2.55),(2.4,1.95));arrow(ax,(7.4,2.55),(7.4,1.95));save(f,'checkpoint')
f,ax=canvas('A strict sampler cannot protect a leaking memory path','Worked stream [1, 2, 2, 2, 3, 4] · nominal batch size two · timestamps in arbitrary units',5.4)
box(ax,.4,2.7,9.0,1.35,'Fixed slicing: [1, 2]  |  [2, 2]  |  [3, 4]\nFirst group queues time 2 → second group scores time 2\nHistory lookup excludes time 2; memory may still consume it.',RUST)
box(ax,.4,.65,9.0,1.35,'Keep ties together: [1, 2, 2, 2]  |  [3, 4]\nPrevious maximum 2 < next minimum 3\nAll time-2 questions use the same pre-group temporal state.')
ax.text(.5,.12,'Soft batch size: preserve order and every event; extend through the last timestamp group.',color=GRAY,fontsize=10.5);save(f,'ties')
report=P/'evidence/l110/summary.json'
if report.exists():
 r=json.loads(report.read_text());f,axs=plt.subplots(1,2,figsize=(10,4.8));f.patch.set_facecolor('#fbfcf9')
 for ax,lane in zip(axs,['all','new']):
  for i,arm in enumerate(['release','clean']):
   vals=[x[arm][lane]['batch_ap_percent'] for x in r['paired']];ax.scatter([i+(j-4.5)*.025 for j in range(len(vals))],vals,color=TEAL if i==0 else RUST,label=arm)
   if vals:ax.errorbar(i,sum(vals)/len(vals),yerr=r['summary'][arm][lane]['sample_sd_pp'],fmt='s',color=INK,capsize=5)
  ax.axhline({'all':98.46,'new':97.81}[lane],ls='--',color=GRAY,label='Paper target (release only)');ax.set_xticks([0,1],['Release','Clean']);ax.set_ylabel('Batch-mean AP (%)');ax.set_title('All-event' if lane=='all' else 'New-node');ax.grid(axis='y',alpha=.15)
 handles,labels=axs[0].get_legend_handles_labels();f.legend(handles,labels,fontsize=9,loc='lower center',ncol=3,bbox_to_anchor=(.5,0),frameon=False);f.suptitle('Fresh paired runs: individual seeds and mean ± seed SD',fontsize=15,color=INK);f.tight_layout(rect=(0,.09,1,.93));save(f,'results')
else:
 f,ax=canvas('Fresh experiment evidence is still running','This placeholder carries no measured score.',3);ax.text(.5,1.3,'Ten seeds × two protocols · full Wikipedia populations',color=INK);save(f,'results')
print('Built four L110 SVG/PNG pairs')
