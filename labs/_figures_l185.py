"""Portable mechanism and evidence figures; original synthetic SCM."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;D=P/'figures/l185';D.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fbfaf7','axes.facecolor':'#fbfaf7','savefig.facecolor':'#fbfaf7','svg.hashsalt':'l185-fixed'})
ink='#17394a';teal='#187f79';orange='#b64b2f'

def save(fig,name):
 fig.savefig(D/(name+'.png'),dpi=155,bbox_inches='tight');fig.savefig(D/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});plt.close(fig)

fig,ax=plt.subplots(figsize=(9,6));ax.set(xlim=(-.3,10),ylim=(-.35,7));ax.axis('off')
def box(x,y,w,h,title,body,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',facecolor='white',edgecolor=color,linewidth=1.8))
 ax.text(x+.16,y+h-.25,title,ha='left',va='top',color=color,weight='bold',fontsize=13)
 ax.text(x+.16,y+h-.72,body,ha='left',va='top',fontsize=11,color=ink)
def arrow(start,end,color=teal):ax.annotate('',end,start,arrowprops={'arrowstyle':'->','color':color,'lw':2})
ax.text(0,6.8,'One company, twenty customers',fontsize=19,weight='bold',color=ink)
box(3.2,4.45,3.2,1.6,'Company demand U','U = 0 or 1, equally likely\nMeasured at time 0',ink)
box(.05,1.8,2.95,1.65,'Company badge B','Copies U 95% of the time\nNo causal arrow to Y',orange)
box(6.6,1.8,3,1.65,'Customer action A','P(A=1 | U=0) = 10%\nP(A=1 | U=1) = 90%',teal)
arrow((3.5,4.45),(1.6,3.55),orange);arrow((6.0,4.45),(8.1,3.55))
box(3.25,.05,3.3,1.6,'Customer purchase Y','p = .05 + .85U + .05A\nY = 1 if fixed noise E < p',ink)
arrow((4.9,4.4),(4.9,1.8));arrow((7.5,1.8),(6.7,.95))
ax.text(.2,.55,'Set B: cut U → B.\nNo path continues to Y.',color=orange,fontsize=11)
ax.text(7.1,.5,'Set A: cut U → A.\nA → Y remains.',color=teal,fontsize=11)
save(fig,'mechanism')

fig,axs=plt.subplots(1,2,figsize=(9,4.8),gridspec_kw={'wspace':.45})
for ax in axs:ax.set_ylim(0,100);ax.set_ylabel('Purchase probability (%)')
axs[0].bar(['A=0\nobserved','A=1\nobserved'],[13.5,86.5],color=[ink,orange],width=.55)
axs[0].set_title('Different demand mixes',loc='left',weight='bold',fontsize=13)
for i,v in enumerate([13.5,86.5]):axs[0].text(i,v+2,f'{v:g}%',ha='center')
axs[0].text(0,49,'73-point\ngap',ha='center',color=orange,weight='bold')
axs[1].bar(['A=0\nstandardized','A=1\nstandardized'],[47.5,52.5],color=[ink,teal],width=.55)
axs[1].set_title('Same 50/50 demand mix',loc='left',weight='bold',fontsize=13)
for i,v in enumerate([47.5,52.5]):axs[1].text(i,v+2,f'{v:g}%',ha='center')
axs[1].text(.5,78,'5-point effect',ha='center',color=teal,weight='bold')
fig.text(.08,.01,'Exact population arithmetic: low demand 5% → 10%; high demand 90% → 95%.',fontsize=11,color=ink)
fig.subplots_adjust(bottom=.23,top=.9);save(fig,'adjustment')

r=json.loads((P/'evidence/l185/report.json').read_text());fig,axs=plt.subplots(1,2,figsize=(9,4.6))
for i,(key,label) in enumerate([('test_badge_auroc','Badge'),('test_action_auroc','Action'),('test_demand_action_auroc','Demand + action')]):
 vals=[x[key] for x in r['per_seed']];axs[0].scatter(vals,[i+(j-2)*.07 for j in range(5)],s=40,color=[orange,ink,teal][i])
axs[0].set_yticks(range(3),['Badge','Action','Demand + action']);axs[0].set_xlim(.5,1);axs[0].set_xlabel('Held-out AUROC\n(higher = better ranking)');axs[0].set_title('Prediction',loc='left',weight='bold');axs[0].grid(axis='x',alpha=.2)
for i,(key,label) in enumerate([('naive_action_difference','Observed action gap'),('adjusted_action_effect','Adjusted action'),('paired_action_effect','Paired action'),('paired_badge_effect','Paired badge')]):
 vals=[100*x[key] for x in r['per_seed']];axs[1].scatter(vals,[i+(j-2)*.07 for j in range(5)],s=40,color=orange if i==0 else teal)
axs[1].set_yticks(range(4),['Observed gap','Adjusted action','Paired action','Paired badge']);axs[1].set_xlim(-5,80);axs[1].axvline(5,color=ink,ls=':',label='True action effect: 5 points');axs[1].set_xlabel('Purchase-rate difference\n(percentage points)');axs[1].set_title('Action evidence',loc='left',weight='bold');axs[1].grid(axis='x',alpha=.2)
axs[1].legend(loc='upper center',bbox_to_anchor=(.5,-.38),fontsize=9,frameon=False)
fig.subplots_adjust(left=.18,right=.98,wspace=.85,bottom=.25,top=.88);save(fig,'results')
