"""Source-grounded schematic plus the measured comparison-policy contrast."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle,FancyArrowPatch
P=Path(__file__).resolve().parent;F=P/'figures/l191';F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.hashsalt':'l191'})
green='#247766';blue='#305d85';ink='#203d34';gold='#bc8039';bg='#f5f8f6'
fig,ax=plt.subplots(figsize=(7,10));fig.patch.set_facecolor(bg);ax.set_facecolor(bg);ax.set_xlim(0,10);ax.set_ylim(0,15);ax.axis('off')
def text(x,y,s,size=10,**kw):ax.text(x,y,s,fontsize=size,color=ink,va='center',**kw)
def box(x,y,w,h,label,color=green):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.1',facecolor='white',edgecolor=color,linewidth=1.4));text(x+w/2,y+h/2,label,ha='center')
def arrow(a,b,color=green):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color=color,linewidth=1.5))
text(.2,14.65,'KumoRFM-2 · follow the query',17,weight='bold')
text(.2,14.12,'Paper-described forward path · symbolic shapes',10)
# Context/query cells depict structure, not invented learned attention weights.
text(.2,13.5,'1   Time-valid rows + known context targets',12,weight='bold')
for r in range(3):
 for c in range(4):
  x=.3+c*.8;y=11.7+r*.45
  ax.add_patch(Rectangle((x,y),.7,.36,facecolor=('#dceae5' if r!=0 else '#f2dbb9'),edgecolor='white'))
  if c==3:text(x+.35,y+.18,'?' if r==0 else str((r+1)%2),9,ha='center')
text(.3,11.33,'3 illustrated rows × cells; query in amber',9)
box(4.5,11.85,2,1,'Child rows\n(time ≤ cutoff)');arrow((3.65,12.5),(4.4,12.5),blue)
text(7.1,12.35,'Known y enters\nearly; query y\nstays hidden.',10)
arrow((5,11.15),(5,10.65))
text(.2,10.2,'2   Task-conditioned table attention',12,weight='bold')
# crossed directions, with real schematic dimensions
for r in range(3):
 for c in range(3):ax.add_patch(Rectangle((.5+c*.55,8.65+r*.4),.44,.3,facecolor='#c6ded5',edgecolor='white'))
arrow((.48,8.35),(2,8.35));arrow((2.45,8.65),(2.45,9.7),blue)
text(3,9.45,'Mix columns within rows',10);text(3,8.92,'Mix rows within columns',10)
text(3,8.35,'Cell states [R, C, d] → row states [R, d]',10)
arrow((5,7.9),(5,7.45))
text(.2,7.05,'3   Relate rows, then context examples',12,weight='bold')
box(.45,5.7,1.65,.65,'query',gold);box(3.5,6,1.9,.65,'child A');box(3.5,4.98,1.9,.65,'child B')
arrow((3.38,6.32),(2.2,6.05),blue);arrow((3.38,5.3),(2.2,5.92),blue)
text(6.2,6.12,'Foreign-key attention',10);text(6.2,5.6,'related rows → query',9)
box(.45,3.75,1.65,.65,'query',gold);box(3.5,3.75,1.9,.65,'context + y');arrow((3.38,4.08),(2.2,4.08))
text(6.2,4.23,'Cross-sample attention',10);text(6.2,3.72,'[K + 1, d] → query [d]',9)
arrow((1.25,5.58),(1.25,4.53),gold)
arrow((5,3.2),(5,2.75))
box(1.5,1.85,7, .65,'4   Prediction head → query score',gold)
text(.2,1.05,'Pretraining: learn weights on synthetic + real tasks.',10)
text(.2,.55,'Selected tables: frozen weights; context uses train + validation.',9)
fig.subplots_adjust(left=.02,right=.98,top=.99,bottom=.01)
for ext in ['svg','png']:fig.savefig(F/('architecture.'+ext),dpi=180,metadata={'Date':None} if ext=='svg' else {})
plt.close(fig)
r=json.loads((P/'evidence/l191/report.json').read_text());t=r['tables'][0];fig,ax=plt.subplots(figsize=(7,3.7),layout='constrained');fig.patch.set_facecolor(bg);ax.set_facecolor(bg)
labels=['Open foundation pool','Open supervised pool','Combined open pool']
for i,pool in enumerate(['foundation','supervised','all']):
 row=t['pools'][pool]
 ax.plot([row['oracle_gap'],row['single_gap']],[i,i],color='#92b4a7',linewidth=4)
 ax.scatter(row['single_gap'],i,s=95,color=green,label='Single method' if i==0 else None,zorder=3)
 ax.scatter(row['oracle_gap'],i,s=70,color=gold,marker='D',label='Taskwise oracle' if i==0 else None,zorder=3)
 ax.text(row['single_gap']+.12,i+.03,f"{row['single_gap']:.3f}",color=ink,fontsize=10)
ax.set_yticks(range(3),labels);ax.invert_yaxis();ax.set_xlim(0,4.7);ax.axvline(0,color=ink);ax.set_xlabel('Kumo minus comparator · AUROC percentage points');ax.set_title('Same Table 3 cells, different comparison question',loc='left',weight='bold',pad=14);ax.spines[['right','top','left']].set_visible(False);ax.grid(axis='x',alpha=.2);ax.legend(frameon=False,loc='lower right');ax.set_ylim(2.7,-.5)
for ext in ['svg','png']:fig.savefig(F/('gaps.'+ext),dpi=180,metadata={'Date':None} if ext=='svg' else {})
plt.close(fig)
# Narrow-layout equivalents retain readable text instead of shrinking desktop labels.
fig,ax=plt.subplots(figsize=(3.8,10));fig.patch.set_facecolor(bg);ax.set_xlim(0,10);ax.set_ylim(0,23);ax.axis('off')
text(.3,22.35,'KumoRFM-2',16,weight='bold');text(.3,21.6,'Follow the hidden query target',10)
text(.3,20.55,'1 · Context + query',12,weight='bold')
for r in range(3):
 for c in range(4):
  x=.5+c*1.25;y=18+r*.52;ax.add_patch(Rectangle((x,y),1.1,.42,facecolor='#f2dbb9' if r==0 else '#c6ded5',edgecolor='white'))
  if c==3:text(x+.55,y+.21,'?' if r==0 else str((r+1)%2),9,ha='center')
text(.5,17.3,'Known context labels enter early.\nAmber query label stays hidden.',10)
arrow((5,16.4),(5,15.8))
text(.3,15.2,'2 · Table attention',12,weight='bold')
text(.5,14.25,'Cells [R, C, d]',11)
box(.6,12.4,8.6,1.2,'Across columns ↔ across rows')
text(.5,11.55,'Task-conditioned rows [R, d]',10)
arrow((5,10.95),(5,10.35))
text(.3,9.8,'3 · Relational attention',12,weight='bold')
box(.6,8.2,3,.9,'query',gold);box(6.2,8.2,3,.9,'child rows');arrow((6,8.65),(3.8,8.65),blue)
text(4.3,7.55,'Foreign-key\nattention',9)
box(.6,5.85,3,.9,'query',gold);box(6.2,5.85,3,.9,'context + y');arrow((6,6.3),(3.8,6.3));arrow((2.1,8.05),(2.1,6.93),gold)
text(.5,5.15,'Mix across context examples.\n[K + 1, d] → query state [d]',10)
arrow((5,4.4),(5,3.85))
box(.6,2.8,8.6,.75,'4 · Head → predicted target',gold)
text(.5,1.9,'Pretraining learns weights.',10);text(.5,1.1,'Selected inference freezes weights.\nEach subgraph respects its cutoff.',9)
fig.subplots_adjust(left=.02,right=.98,bottom=.01,top=.99)
fig.savefig(F/'architecture-mobile.svg',metadata={'Date':None});fig.savefig(F/'architecture-mobile.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(3.8,4.5),layout='constrained');fig.patch.set_facecolor(bg);ax.set_facecolor(bg)
for i,pool in enumerate(['foundation','supervised','all']):
 row=t['pools'][pool];ax.plot([row['oracle_gap'],row['single_gap']],[i,i],color='#92b4a7',linewidth=3);ax.scatter(row['single_gap'],i,s=50,color=green,label='Single' if i==0 else None);ax.scatter(row['oracle_gap'],i,s=35,color=gold,marker='D',label='Oracle' if i==0 else None);ax.text(row['single_gap']+.12,i,f"{row['single_gap']:.3f}",va='center',fontsize=9)
ax.set_yticks(range(3),['Foundation','Supervised','Combined'],fontsize=9);ax.set_ylim(2.8,-.8);ax.set_xlim(0,5.1);ax.set_title('Same cells, different pool',fontsize=12,loc='left',weight='bold');ax.set_xlabel('Kumo advantage\nAUROC percentage points',fontsize=10);ax.spines[['right','top','left']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.legend(frameon=False,loc='lower right',fontsize=9)
fig.savefig(F/'gaps-mobile.svg',metadata={'Date':None});fig.savefig(F/'gaps-mobile.png',dpi=180);plt.close(fig)
