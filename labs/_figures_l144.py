"""Portable mechanism diagrams. Common values shared with interactive ranking."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent;D=P/'figures/l144';D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.hashsalt':'l144'})
INK='#17324d';TEAL='#007f82';GOLD='#ad682b'
def save(f,name):
 f.tight_layout()
 for ext in ['png','svg']:f.savefig(D/f'{name}.{ext}',dpi=150,metadata={'Date':None} if ext=='svg' else None)
 plt.close(f)
f,axes=plt.subplots(3,1,figsize=(9,7.6),gridspec_kw={'height_ratios':[.7,1.5,1]})
for a in axes:a.axis('off')
a=axes[0];a.text(0,1,'ONE QUERY → TWO SCORE PATHS → ONE RANKING',color=INK,weight='bold',fontsize=16)
a.text(.02,.6,'Facility at cutoff t\nRoot identity + row features + age',bbox=dict(boxstyle='round,pad=.6',fc='#eef7f6',ec=TEAL),va='center')
a.annotate('4-hop temporal subgraph\n128 → 64 → 32 → 16 neighbors',xy=(.36,.6),xytext=(.53,.6),arrowprops=dict(arrowstyle='<-',color=INK),va='center')
a=axes[1];a.text(.02,.96,'Shared ResNet row encoders → typed sum-GraphSAGE → H [sampled nodes × h]',color=INK,fontsize=12)
a.text(.02,.5,'LOCAL: item in this query’s graph\n\nContextual item hᵤᵢ ∈ ℝʰ\nhead(hᵤᵢ) + hᵤ · hᵤᵢ\n+ user-specific local offset',va='center',bbox=dict(boxstyle='round,pad=.7',fc='#eef7f6',ec=TEAL))
a.text(.57,.5,'DISTANT: item outside graph\n\nProject root zᵤ ∈ ℝᵈ\nShallow item eᵢ ∈ ℝᵈ\nzᵤ · eᵢ + tower offset',va='center',bbox=dict(boxstyle='round,pad=.7',fc='#fff5e8',ec=GOLD))
a.annotate('',xy=(.24,.78),xytext=(.43,.9),arrowprops=dict(arrowstyle='->',color=TEAL,lw=1.7));a.annotate('',xy=(.76,.78),xytext=(.53,.9),arrowprops=dict(arrowstyle='->',color=GOLD,lw=1.7))
a.text(.02,.03,'One item can be local for one facility and distant for another.',color=INK)
a=axes[2];a.text(.02,.82,'Initialize B × N tower scores; replace the (owner, item) local entries.',weight='bold',color=INK)
a.text(.02,.47,'TRAIN: sparse multi-positive cross-entropy → all model parameters\nEVALUATE: sigmoid → top 10 sponsor IDs → keyed MAP@10',linespacing=1.7,color=INK)
a.text(.02,.02,'ShallowItem baseline retains the tower matrix. No local overwrite or learned offsets.\nThis release scores the full sponsor vocabulary; sampled softmax is a separate option.',fontsize=11,color=INK)
save(f,'architecture')
f,axes=plt.subplots(1,3,figsize=(10,3.7));matrices=[np.array([[1,2,3,4],[4,3,2,1]]),np.array([[0,1,0,0],[0,0,1,0]]),np.array([[1,10.5,3,4],[4,3,19.5,1]])]
for a,m,title in zip(axes,matrices,['Tower scores','Local ownership mask','After replacement']):
 a.imshow(m,cmap='GnBu',vmin=0,vmax=20 if title!='Local ownership mask' else 1);a.set_title(title,color=INK);a.set_xticks(range(4),list('ABCD'));a.set_yticks([0,1],['Facility 0','Facility 1'])
 for (i,j),v in np.ndenumerate(m):a.text(j,i,f'{v:g}',ha='center',va='center',color='white' if v>10 else INK)
f.suptitle('Owner-specific indexing: local values 10/20, offsets +0.5/−0.5',color=INK);save(f,'ownership')
f,a=plt.subplots(figsize=(9,4));x=np.arange(4);a.bar(x-.18,[1,2,3,4],.36,color='#c5d3df',label='Tower only');a.bar(x+.18,[3,2,1,4],.36,color=[TEAL,GOLD,TEAL,GOLD],label='ContextGNN illustration');a.set_xticks(x,['A · local','B · distant','C · local','D · distant']);a.set_ylabel('Score (logit)');a.set_ylim(0,5);a.set_title('Fusion can lower a local item’s score: C moves from 3 to 1');a.legend(frameon=False);save(f,'ranking')
f,a=plt.subplots(figsize=(9,3.7));a.axis('off');a.text(.02,.91,'MAP@3: reward correct items early',weight='bold',fontsize=17,color=INK)
a.text(.02,.65,'Relevant sponsors: {A, C}       Recommended: [A, B, C]',fontsize=14,color=INK)
a.text(.02,.4,'Rank                         1                 2                 3\nRelevant?                  yes               no                yes\nPrecision at hit         1 / 1              —                2 / 3',linespacing=1.6,color=INK)
a.text(.02,.02,'AP = (1 + 2/3) / min(2, 3) = 5/6. MAP averages AP across query rows.',color=TEAL,weight='bold');save(f,'metric')
