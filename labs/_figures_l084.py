"""Portable diagrams expose the GAT graph mask, head axes and training boundary."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
LAB=Path(__file__).resolve().parent;OUT=LAB/'figures/l084';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'figure.facecolor':'#f8fafc','axes.facecolor':'#f8fafc','text.color':'#13243a','axes.spines.top':False,'axes.spines.right':False})
def box(ax,x,y,w,h,text,color='#dcecf5',size=12):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.015',facecolor=color,edgecolor='#9bafc3'))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)
def arrow(ax,a,b,color='#526780'):
 ax.annotate('',b,a,arrowprops={'arrowstyle':'->','color':color,'lw':2})
def save(fig,name):
 fig.savefig(OUT/f'{name}.png',dpi=150,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(12,5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.text(0,1.04,'ONE RECEIVER · NEIGHBOR SOFTMAX',fontsize=19,weight='bold')
colors=['#e2e8f0','#cceae4','#ffe3ad'];vals=[0,1,2];weights=np.exp(vals)/np.exp(vals).sum()
for i,(v,a,c) in enumerate(zip(vals,weights,colors)):
 x=.02+i*.33
 box(ax,x,.70,.28,.17,f'Sender {i}'+(' = self' if i==0 else '')+f'\nz = {v}',c)
 box(ax,x,.39,.28,.15,f'e = {v}     α = {a:.3f}',c);arrow(ax,(x+.14,.7),(x+.14,.55))
 ax.text(x+.14,.29,f'α × z = {a*v:.3f}',ha='center')
 arrow(ax,(x+.14,.25),(.49,.15))
box(ax,.3,.0,.38,.14,'Weighted sum = 1.575','#d8d8ef')
ax.text(.02,.60,'q = 0, k = z; LeakyReLU leaves these nonnegative scores unchanged',fontsize=11)
ax.text(.72,.07,'Uniform mean = 1.000\nDropout off; zero biases',fontsize=11)
save(fig,'trace')
fig,ax=plt.subplots(figsize=(14,9));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.text(0,1.035,'GAT ON CORA · COMPLETE COMPUTATION',fontsize=21,weight='bold')
box(ax,.01,.82,.29,.15,'All node features\n2,708 × 1,433\nRow-normalized bag of words')
box(ax,.36,.82,.29,.15,'Citation graph + self-loops\n13,264 directed messages\n(receiver, sender)', '#e2e8f0')
box(ax,.73,.82,.25,.15,'Training access\nAll features / graph\nOnly 140 training labels','#ffe3ad')
for i,(x,c) in enumerate([(.03,'#cceae4'),(.36,'#d8d8ef'),(.69,'#ffe3ad')]):
 title=['HEAD 1','HEAD 2','… HEAD 8'][i]
 box(ax,x,.40,.27,.32,title+'\nIndependent input dropout\nz = X W   [2,708 × 8]\neᵢⱼ = LeakyReLU(qᵢ + kⱼ)\nsoftmax over j ∈ N(i)\nDrop α and z independently\nΣⱼ αᵢⱼ zⱼ + bias → ELU',c,11)
 arrow(ax,(.15,.82),(x+.135,.73));arrow(ax,(.50,.82),(x+.22,.73),'#aa7650')
 arrow(ax,(x+.135,.40),(.48,.33))
box(ax,.24,.22,.49,.11,'CONCAT hidden heads → 2,708 × 64','#dcecf5')
box(ax,.24,.04,.49,.11,'OUTPUT HEAD · same attention computation\n64 → 7 logits/node; no ELU','#cceae4');arrow(ax,(.48,.22),(.48,.15))
box(ax,.77,.03,.21,.17,'Train: CE on 140 labels\n+ L2, Adam update\n500 labels select\n1,000 labels test', '#ffe3ad',10);arrow(ax,(.73,.095),(.77,.095))
ax.text(.01,.025,'Evaluation: dropout off\nClass softmax → prediction\nOutput heads: mean, not concat',fontsize=10)
ax.text(.01,.995,'Distinct parameters across heads; shared parameters across nodes and edges within a head.',fontsize=11)
save(fig,'architecture')
p=LAB/'_attention_l084.json'
if p.exists():
 r=json.loads(p.read_text());fig,ax=plt.subplots(figsize=(11,4.5));ax.bar([str(s) for s in r['senders']],r['coefficients'],color=['#c77732' if s==0 else '#327e8b' for s in r['senders']]);ax.axhline(1/len(r['senders']),ls='--',color='#64748b',label='Uniform weights');ax.set(title='TRAINED ROUTING · receiver 0 / hidden head 1 / seed 0',xlabel='Sender node ID (orange = self)',ylabel='Evaluation attention coefficient');ax.legend();fig.tight_layout();save(fig,'attention')
p=LAB/'_paper_l084_results.json'
if p.exists():
 r=json.loads(p.read_text());fig,ax=plt.subplots(figsize=(11,4));a=np.array([v['test_accuracy'] for v in r['runs']])*100;ax.scatter(range(len(a)),a,color='#327e8b',s=20);ax.axhline(83,color='#c77732',label='Paper mean: 83.0%');ax.axhline(a.mean(),ls='--',color='#13243a',label=f'Port mean: {a.mean():.3f}%');ax.set(xlabel='Declared local seed',ylabel='Test accuracy (%)',title='ONE FIXED CORA SPLIT · seed variation, not dataset uncertainty');ax.legend();fig.tight_layout();save(fig,'results')
print(OUT)
