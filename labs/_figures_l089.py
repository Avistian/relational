"""Portable computation diagrams; synthetic traces separate from fresh PPI results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures/l089';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#faf8f3','axes.facecolor':'#faf8f3','text.color':'#203347','axes.labelcolor':'#203347'})
colors=['#176b83','#a65a24','#65539b']
def save(fig,name):fig.savefig(OUT/f'{name}.png',dpi=160,bbox_inches='tight',facecolor=fig.get_facecolor());plt.close(fig)
def path(ax,selected,title):
 ax.set_xlim(-.5,5.5);ax.set_ylim(-.6,.7);ax.axis('off');ax.set_title(title,loc='left',weight='bold',fontsize=13)
 for i in range(5):
  kept=i in selected and i+1 in selected
  ax.plot([i,i+1],[0,0],color='#203347' if kept else '#c7c4bf',lw=4 if kept else 2,linestyle='-' if kept else '--',zorder=1)
  if i==1 and kept:ax.text(1.5,.32,'restored',ha='center',fontsize=10,color='#a65a24')
 for i in range(6):
  ax.scatter(i,0,s=650,c=colors[i//2] if i in selected else '#e2dfd8',edgecolor='white',lw=2,zorder=2)
  ax.text(i,0,str(i),color='white' if i in selected else '#444',va='center',ha='center',weight='bold')
 for j in range(3):ax.text(2*j+.5,-.4,f'C{j}',ha='center',color=colors[j])
f,axes=plt.subplots(2,1,figsize=(9,4.8));path(axes[0],{0,1},'One cluster: 2 nodes · 1 of 5 original edges');path(axes[1],{0,1,2,3},'Union of two clusters: 4 nodes · 3 of 5 original edges');f.suptitle('Slice the original graph on the UNION',weight='bold',fontsize=18);f.tight_layout();save(f,'partition')
f,axes=plt.subplots(1,2,figsize=(10,4.4));rows=[[.5,1,0],[1/3,2/3,1/3]]
for ax,row,title in zip(axes,rows,['C0 only: node 2 is absent','C0 + C1: edge 1–2 restored']):
 ax.axis('off');ax.set_title(title,fontsize=14,weight='bold',pad=20)
 table=ax.table(cellText=[['h₀ = 1','h₁ = 2','h₂ = 4'],[f'{x:.3f}' for x in row]],rowLabels=['State','S row'],loc='center',cellLoc='center',bbox=[.12,.42,.86,.35])
 table.auto_set_font_size(False);table.set_fontsize(13)
 for (r,c),cell in table.get_celld().items():cell.set_edgecolor('#d1cabc');cell.set_facecolor('#e9f1ef' if r==1 else '#fffdf8')
 total=np.dot(row,[1,2,4]);ax.text(.5,.2,f'Sh = {total:g}',transform=ax.transAxes,ha='center',fontsize=24,color=colors[0]);ax.text(.5,.05,'then concatenate [Sh, h₁]',transform=ax.transAxes,ha='center',fontsize=12)
f.suptitle('A cut changes the messages AND their weights',fontsize=18,weight='bold');f.tight_layout(rect=[0,0,1,.9]);save(f,'support')
f,ax=plt.subplots(figsize=(11,8));ax.set_xlim(0,11);ax.set_ylim(0,8);ax.axis('off')
def box(x,y,w,h,title,body,color='#e6efed'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.10',facecolor=color,edgecolor='#acb9b7',lw=1))
 ax.text(x+.18,y+h-.35,title,fontsize=12,weight='bold',va='center');ax.text(x+.18,y+h-.68,body,fontsize=10.5,va='top',linespacing=1.30)
def arrow(x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':'#203347','lw':2})
ax.text(0,7.75,'PPI release: cached input → cluster-restricted hidden messages',fontsize=18,weight='bold')
box(.1,5.8,4.6,1.35,'INPUT · 44,906 training proteins','50 features → train-only standardization\nRaw training adjacency Aₜ · 121 labels')
box(6,5.8,4.6,1.35,'METIS · training graph only','50 disjoint parts · one part per update\nLater layers use its enhanced support S', '#f3eadc')
arrow(4.8,6.5,5.9,6.5)
box(.1,3.65,4.6,1.45,'CACHE ONCE · before partitioning','[Aₜ X, X] : Nₜ × 100\nNeighbor sum  |  self features\nSelect this batch’s b rows')
arrow(2.4,5.8,2.4,5.2)
box(6,3.65,4.6,1.45,'LAYER 1 · trainable weights','100 → 2048\nDropout .2 → W₀ → LayerNorm → ReLU\nNo further adjacency in this first stage')
arrow(4.8,4.35,5.9,4.35);arrow(8.3,5.8,8.3,5.2);ax.text(8.5,5.4,'batch IDs',fontsize=10)
box(.1,1.45,4.6,1.55,'LAYERS 2–4 · cluster support S','S H : b × 2048  |  H : b × 2048\nConcatenate → b × 4096 → Dropout → W\n→ LayerNorm → ReLU : b × 2048')
arrow(8.3,3.65,8.3,3.3);arrow(8.3,3.3,2.4,3.3);arrow(2.4,3.3,2.4,3.0)
box(6,1.45,4.6,1.55,'LAYER 5 · logits and training loss','[S H, H] → Dropout → W₄ → b × 121\nNo LayerNorm or ReLU\nBinary cross-entropy → Adam update','#f3eadc')
arrow(4.8,2.2,5.9,2.2)
ax.text(.15,.8,'INFERENCE  ·  Same learned weights, dropout off; cache from full graph.',fontsize=12,weight='bold')
ax.text(.15,.35,'Validation: 2 partitions.  Test: 1 full-graph partition → logits > 0 → pooled micro-F1.',fontsize=12)
save(f,'architecture')
r=json.loads((ROOT/'_teaching_l089_results.json').read_text())['rows'];f,axes=plt.subplots(1,2,figsize=(10,4.8));labels=['Full','Random','Cluster q1','Cluster q5']
for i,row in enumerate(r):
 vals=np.array(row['test_f1'])*100;axes[0].scatter(np.full(3,i)+np.array([-.09,0,.09]),vals,s=45,c=colors[0]);axes[0].errorbar(i,vals.mean(),yerr=vals.std(ddof=1),fmt='_',capsize=5,color='#203347')
axes[0].set(xticks=range(4),xticklabels=labels,ylabel='Test micro-F1 (%)',ylim=(0,100));axes[0].set_title('Three seeds · mean ± sample SD',fontsize=13);axes[0].tick_params(axis='x',rotation=18)
axes[1].bar(range(4),[x['hidden_state_proxy_bytes']/2**20 for x in r],color=['#bcc3c3',colors[1],colors[0],colors[2]])
axes[1].set(xticks=range(4),xticklabels=labels,ylabel='Largest-batch hidden-state proxy (MiB)');axes[1].set_title('Array estimate, NOT peak RAM',fontsize=13);axes[1].tick_params(axis='x',rotation=18)
f.suptitle('Controlled teaching run · 10 passes · width 32 · 2 layers',fontsize=16,weight='bold');f.tight_layout();save(f,'results')
