"""Editable, model-specific architecture and numeric traces with portable PNG exports."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Circle
P=Path(__file__).resolve().parent;OUT=P/'figures/l092';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none'})
ink='#193248';teal='#087f8c';rust='#a34430';gold='#bc881b'
def base(title,subtitle):
 f,ax=plt.subplots(figsize=(13,7));f.patch.set_facecolor('#fbfcfa');ax.set_xlim(0,13);ax.set_ylim(0,7);ax.axis('off')
 ax.text(.2,6.65,title,fontsize=21,weight='bold',color=ink);ax.text(.2,6.2,subtitle,fontsize=11,color='#526776');return f,ax
def box(a,x,y,w,h,title,body,c=teal):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08,rounding_size=.12',fc='white',ec=c,lw=1.6))
 a.text(x+.12,y+h-.3,title,fontsize=12,weight='bold',color=c);a.text(x+.12,y+h-.65,body,fontsize=11,va='top',color=ink,linespacing=1.6)
def arrow(a,x,y,u,v,c=ink):a.annotate('',xy=(u,v),xytext=(x,y),arrowprops={'arrowstyle':'->','lw':1.8,'color':c})
def save(f,name):
 f.savefig(OUT/f'{name}.png',dpi=150,bbox_inches='tight');f.savefig(OUT/f'{name}.svg',bbox_inches='tight');plt.close(f)
f,a=base('HAN · choose neighbors, then combine meanings','ACM release port · every path keeps its own attention heads until the semantic fusion')
box(a,.25,2.5,2.5,2.1,'PAPER INPUT','X: 3,025 × 1,870\nRaw keyword features\nTwo endpoint graphs')
box(a,3.4,4.15,3,1.55,'PAP · same author','29,281 edges with self\n8 heads × 8 coordinates',teal)
box(a,3.4,1.95,3,1.55,'PSP · same subject','2,210,761 edges with self\n8 heads × 8 coordinates',rust)
arrow(a,2.85,4,3.3,4.7,teal);arrow(a,2.85,2.85,3.3,2.7,rust)
box(a,7.05,2.85,2.6,2.3,'SEMANTIC MIX','Z: N × 2 × 64\nScorer: 64→128→1\nβ: N × 2 in release\nSum paths → N × 64',gold)
arrow(a,6.5,4.8,6.95,4.35,teal);arrow(a,6.5,2.7,6.95,3.4,rust)
box(a,10.25,4.15,2.35,1.55,'CLASSIFIER','64 → 3 logits\nCE + L2; 600 labels')
box(a,10.25,1.85,2.35,1.65,'KNN PROBE','2,125 frozen vectors\nk=5; four fractions\n10 splits per fraction',rust)
arrow(a,9.75,4.3,10.15,4.8);arrow(a,9.75,3.25,10.15,2.7,rust)
a.text(.3,.95,'Inside each head: project → score allowed pairs → receiver softmax → weighted sum → ELU',color=ink,fontsize=13)
a.text(.3,.5,'Validation selects weights; test labels do not train HAN. KNN then fits its own labeled probe subset.',color=ink,fontsize=11)
a.text(.3,.12,'Paper-global alternative: mean node scores before path softmax → one shared β [2].',color=gold,fontsize=12)
save(f,'architecture')
f,a=base('PAP · two walks can still mean one neighbor','Worked construction · endpoint adjacency retains reachability, not the number of shared authors')
positions={'p0':(1,4),'a0':(4,5),'a1':(4,3),'p1':(7,4),'p2':(7,1.7)}
for first,last in [('p0','a0'),('p0','a1'),('a0','p1'),('a1','p1'),('a1','p2')]:
 x,y=positions[first];u,v=positions[last];a.plot([x,u],[y,v],color=teal,lw=2,alpha=.7)
for name,(x,y) in positions.items():
 a.add_patch(Circle((x,y),.38,fc='#e1f2f1' if name[0]=='p' else '#f5e9c8',ec=teal,lw=2));a.text(x,y,name,ha='center',va='center',weight='bold',color=ink)
box(a,8.5,2.1,4,3.45,'COLLAPSE ENDPOINTS','p0 → a0 → p1\np0 → a1 → p1\n\nWalk count: 2\nBinary endpoint edge: 1\nSender p1 contributes once')
a.text(.3,.75,'Self is reachable: p0 → a0 → p0. All three papers share a1 in the matrix example.',color=ink,fontsize=12)
a.text(.3,.3,'Remove p2–a1: its off-diagonal neighbors disappear. Add another shared author: no new endpoint.',color=ink,fontsize=11)
save(f,'paths')
f,a=base('Semantic attention · the order of operations changes the model','Fixed node/path scores · teal = PAP · rust = PSP · this is arithmetic, not fitted ACM evidence')
box(a,.3,3.8,3.3,1.8,'SCORES [node, path]','node 0: [2, 0]\nnode 1: [0, 0]')
box(a,4.25,3.8,3.5,1.8,'PAPER: MEAN FIRST','mean scores: [1, 0]\nshared β: [0.731, 0.269]',gold)
box(a,8.4,3.8,4.1,1.8,'RELEASE: SOFTMAX FIRST','node 0 β: [0.881, 0.119]\nnode 1 β: [0.500, 0.500]')

for y,label,b in [(2.7,'Paper: either node',.731059),(1.8,'Release: node 0',.880797),(0.9,'Release: node 1',.5)]:
 a.text(.4,y+.18,label,color=ink);a.barh(y+.2,b*7,left=4,height=.4,color=teal);a.barh(y+.2,(1-b)*7,left=4+b*7,height=.4,color=rust);a.text(11.2,y+.13,f'{b:.3f}',color=teal,weight='bold')
a.text(.4,.25,'Mean released PAP weight = 0.690 ≠ paper-global PAP weight = 0.731.',color=ink,fontsize=13,weight='bold')
save(f,'semantic')
f,axs=plt.subplots(1,2,figsize=(12,5),sharey=True)
for ax,weights,title in zip(axs,[[.25,.5,.25],[1/7,2/7,4/7]],['Baseline scores: [0, ln 2, 0]','Change last score: [0, ln 2, ln 4]']):
 ax.bar(['self p0','sender p1','sender p2'],weights,color=[teal,teal,rust]);ax.set_ylim(0,.8);ax.set_title(title);ax.set_ylabel('Receiver-row attention weight')
 for j,(v,h) in enumerate(zip(weights,[1,3,5])):ax.text(j,v+.035,f'{v:.3f} × {h}',ha='center',color=ink)
 ax.text(.5,.92,f'Weighted message = {sum(v*h for v,h in zip(weights,[1,3,5])):.3f}',transform=ax.transAxes,ha='center',weight='bold',color=ink)
f.suptitle('Node attention · fixed values [1, 3, 5], one changed score',weight='bold');f.tight_layout(rect=(0,0,1,.93));save(f,'neighbors')
result=P/'_paper_l092_results.json'
if result.exists():
 d=json.loads(result.read_text());f,axs=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
 for ax,key,target in zip(axs,['macro_f1','micro_f1'],['paper_macro','paper_micro']):
  for i,row in enumerate(d['table3_acm']):
   vals=[r[key]*100 for run in d['runs'] for r in run['knn'] if r['fraction']==row['fraction']]
   ax.scatter(i+np.linspace(-.11,.11,len(vals)),vals,s=25,color=teal,alpha=.7)
   ax.scatter(i,row[target]*100,marker='D',s=60,color=rust)
  ax.set_xticks(range(4),['20%','40%','60%','80%']);ax.set_xlabel('KNN training fraction');ax.set_title(key.replace('_',' ').title());ax.grid(axis='y',alpha=.2)
 axs[0].set_ylabel('F1 (%)');f.suptitle('ACM · KNN repetitions from the full released training schedule',weight='bold');f.text(.12,.015,'Teal: measured probe splits. Rust: paper target. Protocol verdict: INCOMPARABLE.',fontsize=11)
 f.tight_layout(rect=(0,.05,1,.93));save(f,'results')
