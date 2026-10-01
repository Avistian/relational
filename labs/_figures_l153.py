"""Portable computation diagrams and measured reproduction boundary."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;D=P/'figures/l153';D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l153','axes.spines.top':False,'axes.spines.right':False})
INK='#17324d';TEAL='#007f82';GOLD='#a35919'
def box(a,x,y,w,h,text,color=TEAL):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02',facecolor='#eef7f6',edgecolor=color,linewidth=1.5);a.add_patch(p);t=a.text(x+w/2,y+h/2,text,ha='center',va='center',color=INK);a._pairs=getattr(a,'_pairs',[])+[(p,t)]
def arrow(a,x,y,u,v):a.annotate('',(u,v),(x,y),arrowprops={'arrowstyle':'->','color':INK,'lw':1.5})
def save(f,name):
 f.canvas.draw();renderer=f.canvas.get_renderer()
 for ax in f.axes:
  for p,t in getattr(ax,'_pairs',[]):
   b=p.get_window_extent(renderer);q=t.get_window_extent(renderer);assert b.x0<=q.x0 and b.x1>=q.x1 and b.y0<=q.y0 and b.y1>=q.y1,(name,t.get_text())
 for ext in ['svg','png']:f.savefig(D/f'{name}.{ext}',dpi=150,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else None)
 plt.close(f)
f,a=plt.subplots(figsize=(11,9.5));a.set(xlim=(0,11),ylim=(0,10));a.axis('off');a.text(.1,9.7,'FROM A FACILITY QUERY TO A RANKED SPONSOR LIST',weight='bold',fontsize=15,color=INK)
box(a,.2,8.55,10.4,.85,'Query (facility, cutoff t) · full relational graph · sponsor catalog\nThree root groups: sites, positive sponsors, sampled sponsors')
arrow(a,5.4,8.55,5.4,8.2)
box(a,.2,6.75,10.4,1.4,'ONE SHARED ENCODER, APPLIED TO THREE TEMPORAL NEIGHBORHOODS\nTyped column encoders → per-table ResNets → [nodes,128]\n+ relative-time features + learned sponsor ID embeddings\nTwo sum-GraphSAGE layers, fanouts 128/64 → root head [B,128]')
for x in [1.9,5.4,8.9]:arrow(a,x,6.75,x,6.35)
box(a,.2,5.5,3.25,.8,'SITE vectors U\n[B,128]');box(a,3.8,5.5,3.25,.8,'POSITIVE vectors V+\n[B,128]');box(a,7.4,5.5,3.2,.8,'SAMPLED vectors V−\n[B,128]')
arrow(a,3.3,5.5,4,5.05);arrow(a,5.4,5.5,5.4,5.05);arrow(a,8.9,5.5,7,5.05)
box(a,.6,3.55,9.6,1.45,'TRAIN: positive[i] = dot(U[i], V+[i]) → [B,1]\nnegative[i,j] = dot(U[i], V−[j]) → [B,B]\nmean softplus(negative − positive) → scalar loss → shared updates\nExample: positive 2, negative 0 → loss 0.127; reversed → 2.127')
box(a,.6,1.25,9.6,1.65,'EVALUATE: all 53,241 sponsors → vectors [53,241,128]\nSite blocks × sponsor vectors → full-catalog scores → top 10\nValidation MAP selects latest tied maximum; test does not select\nNo local ContextGNN branch and no probability interpretation',GOLD)
a.text(.6,.55,'Solid arrows: training computation. Evaluation reuses the selected shared weights.',color=INK,fontsize=10);save(f,'architecture')
f,a=plt.subplots(figsize=(10,5.3));a.set(xlim=(0,10),ylim=(0,5));a.axis('off');a.text(.2,4.7,'SHARED NEGATIVES: COLLISION ≠ FUTURE INFORMATION',weight='bold',fontsize=14,color=INK)
a.text(4.1,3.9,'Sampled A',ha='center',weight='bold');a.text(7.6,3.9,'Sampled B',ha='center',weight='bold')
for y,label in [(2.9,'Query 0 · positives {A}'),(1.8,'Query 1 · positives {B,C}')]:a.text(.2,y+.35,label,fontsize=10)
box(a,3.0,2.9,2.2,.7,'Positive collision',GOLD);box(a,6.5,2.9,2.2,.7,'Unlabeled pair');box(a,3.0,1.8,2.2,.7,'Unlabeled pair');box(a,6.5,1.8,2.2,.7,'Positive collision',GOLD)
a.text(.2,.9,'Both cutoffs 10. If observed availability(B)=11, both B comparisons are future.',fontsize=11,color=INK)
a.text(.2,.35,'Unknown availability is not zero violations. Source sampling keeps collisions.',fontsize=11,color=INK);save(f,'negatives')
f,a=plt.subplots(figsize=(10,5.3));a.set(xlim=(0,10),ylim=(0,5));a.axis('off');a.text(.2,4.7,'AP@3: EVERY RELEVANT HIT CONTRIBUTES ITS PREFIX PRECISION',weight='bold',fontsize=13,color=INK)
for x,rank,label,term in [(.3,1,'A · relevant','1/1'),(3.5,2,'B · irrelevant','0'),(6.7,3,'C · relevant','2/3')]:box(a,x,2.8,2.8,1.25,f'Rank {rank}\n{label}\nNumerator term {term}')
a.text(.3,1.9,'Truth {A,C} → denominator min(k=3, positives=2)=2',fontsize=13,color=INK)
a.text(.3,1.15,'AP=(1+0+2/3)/2=5/6 · Hit=1 · Recall=2/2=1',fontsize=14,color=TEAL)
a.text(.3,.45,'AP@2=.50. Remove B and AP@2=1.00 — the candidate problem changed.',fontsize=11,color=GOLD);save(f,'ranking')
s=json.loads((P/'evidence/l153/summary.json').read_text());decision=json.loads((P/'evidence/l153/cost-decision.json').read_text())
f,a=plt.subplots(figsize=(10,5.3));a.axis('off');a.set(xlim=(0,10),ylim=(0,5));a.text(.2,4.7,'WHAT THE MEASURED EVIDENCE CAN SUPPORT',weight='bold',fontsize=14,color=INK)
if s['full_selected_reproduction']=='COMPLETE':
 vals=[r['scores']['test']*100 for r in s['records']];a.text(.3,3.7,'Five completed test MAP@10 scores: '+', '.join(f'{v:.2f}%' for v in vals),color=TEAL)
 a.text(.3,2.8,f"Mean {s['portfolio']['test']['mean']*100:.2f}% · {s['portfolio']['paper_comparison']}",color=TEAL,fontsize=15)
else:
 box(a,.3,3.15,9.2,1,'FULL FIVE-SEED REPRODUCTION: INCOMPLETE\nTest MAP comparison: NOT_RUN',GOLD)
 a.text(.4,2.6,f"Pilot: 32 training batches; full { s['pilot']['counts']['val']:,} validation queries",color=INK,fontsize=12)
 a.text(.4,1.95,f"Projected five-run compute: ${decision['projected_five_run_compute_usd']:.2f} before overhead",color=INK,fontsize=12)
 a.text(.4,1.3,'Budget gate: '+decision['decision']+' · approved total cap $10',color=GOLD,fontsize=13)
a.text(.4,.55,'Mechanism, label and metric checks do not substitute for final training evidence.',color=INK,fontsize=11);save(f,'evidence')
