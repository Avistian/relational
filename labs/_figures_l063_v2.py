"""Portable, computation-exposing figures; all measured values read from fresh evidence."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures/l063';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'#fbfaf7'})
BLUE='#176b8b';ORANGE='#b9582c';GREEN='#397c50';DARK='#243545'
def canvas(title,sub,height=6.4):
 f,a=plt.subplots(figsize=(11,height));f.patch.set_facecolor('#fbfaf7');a.set_facecolor('#fbfaf7');a.set(xlim=(0,11),ylim=(0,height));a.axis('off');f.suptitle(title,fontsize=18,fontweight='bold',x=.07,ha='left',y=.97);f.text(.07,.89,sub,fontsize=11,color=DARK);return f,a

def box(a,x,y,w,h,text,color=BLUE,size=12):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor='white',edgecolor=color,lw=1.7));a.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color=DARK)
def arrow(a,x,y,xx,yy,label=None):
 a.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops=dict(arrowstyle='->',color=DARK,lw=1.6))
 if label:a.text((x+xx)/2,(y+yy)/2+.12,label,ha='center',fontsize=10)
def save(f,name):f.savefig(OUT/(name+'-v2.png'),dpi=150,bbox_inches='tight');plt.close(f)

def build():
 f,a=canvas('One sampled world → many rows → a prediction task','Released numerical path. W[child,parent]; the same W, b and activation apply to every row.',7.2)
 box(a,.2,5.4,10.4,.6,'WORLD φ: family + dimensions + fixed edge masks/weights/biases + noise scales + activation',size=12)
 box(a,.2,3.9,2.4,.9,'N root draws\nC : N × d₀')
 box(a,3.1,3.9,3.1,.9,'First affine output H₀\nC W₀ᵀ + b₀')
 box(a,6.8,3.7,3.8,1.3,'Repeat: a(H) Wᵀ + b + ε\nN × d_in → N × d_out\nactivate → affine → add noise',size=11)
 arrow(a,2.6,4.35,3.05,4.35);arrow(a,6.2,4.35,6.75,4.35);arrow(a,8.7,5.4,8.7,5.02)
 box(a,.2,1.9,5.1,1.1,'SCM: concatenate H₁,H₂,…\nchoose feature nodes X and target node z\nC and H₀ excluded from selectable pool',size=11)
 box(a,5.8,1.9,4.8,1.1,'BNN: X = root causes C\nz = final scalar network output\nno arbitrary intermediate observation',size=11)
 arrow(a,8.7,3.7,8.2,3.03);arrow(a,7.2,3.7,3.4,3.03)
 box(a,.2,.2,10.4,.9,'z → K−1 sampled bounds → strict comparisons → shared class permutation\nLab probe: context (X,y) | query X → logistic probabilities; y_query used only to score',size=11)
 arrow(a,2.8,1.9,4.0,1.12);arrow(a,8.2,1.9,7.0,1.12)
 a.set_ylim(-1.1,7.2);f.set_size_inches(11,8.1)
 box(a,.2,-1.0,10.4,.8,'Paper offline fitting (NOT_RUN): context + query X → qθ; query y → cross-entropy → update θ\nDeployment: real context + query X → frozen qθ → probabilities; sampled graph φ stays hidden',size=10.5)
 save(f,'generator')
 f,axes=plt.subplots(1,3,figsize=(11,5));f.suptitle('A fixed edge mask changes every row of this world',fontweight='bold',fontsize=17,y=.98)
 mats=[np.array([[1,2],[3,4]]),np.array([[1,0],[0,1]]),np.array([[2,0],[0,8]])]
 for ax,m,title in zip(axes,mats,['Raw weights R','Keep mask M','R × M / (1 − √.25)']):
  ax.imshow(m,cmap='Blues',vmin=0,vmax=8);ax.set_title(title,fontsize=12);ax.set_xticks([0,1],['parent0','parent1']);ax.set_yticks([0,1],['child0','child1'])
  for i in range(2):
   for j in range(2):ax.text(j,i,str(m[i,j]),ha='center',va='center',fontsize=21,color='white' if m[i,j]>4 else DARK)
 f.text(.06,.08,'Independent-edge branch: p=.25 ⇒ divisor=.5. First affine layer bypasses this rule.\nBlock branch differs: keep diagonal blocks and divide by √keep_fraction, including layer0.',fontsize=12);f.subplots_adjust(bottom=.26,wspace=.6,top=.80);save(f,'sparsity')
 f,a=canvas('Observation selects the information, not the underlying graph','Known synthetic graph; this is an illustrative SCM, not a claim of a recovered real-world graph.')
 for x,y,t in [(4.4,4.8,'U\nhidden cause'),(1.0,2.9,'X₁\nproxy'),(4.4,2.9,'Y\ntarget'),(8,2.9,'X₂\neffect')]:box(a,x,y,1.8,.8,t,GREEN if t.startswith('Y') else BLUE)
 arrow(a,4.4,5,2.8,3.7);arrow(a,5.3,4.8,5.3,3.75);arrow(a,6.2,3.3,7.95,3.3)
 box(a,.5,.5,4.5,1.2,'Table A observes (X₁,X₂), predicts Y\nX₁ carries information about U;\nX₂ carries information about Y.',size=11)
 box(a,5.6,.5,4.6,1.2,'Table B observes only X₁, predicts Y\nSame world; less observed information.\nPrediction ≠ do(Y=c) inference.',size=11)
 save(f,'roles')
 f,ax=plt.subplots(figsize=(11,5.3));z=np.array([-1.,-.1,.2,.5,1.]);r=(z[:,None]>np.array([-.1,.5])).sum(1);perm=np.array([2,0,1]);ax.scatter(z,np.ones(5),s=160,c=[BLUE,BLUE,ORANGE,ORANGE,GREEN],zorder=3)
 for b in [-.1,.5]:ax.vlines(b,1.08,1.43,color=DARK,ls='--');ax.text(b,1.48,f'bound {b}',ha='center')
 for i,v in enumerate(z):ax.text(v,.72,f'z={v:g}\nrank {r[i]}\nlabel {perm[r[i]]}',ha='center',fontsize=12)
 ax.set(xlim=(-1.25,1.25),ylim=(.2,1.7));ax.set_yticks([]);ax.set_xticks([]);ax.spines[['left','bottom']].set_visible(False);ax.set_title('Strict bounds first; arbitrary class names second',fontsize=18,pad=18,fontweight='bold');f.text(.07,.05,'Bounds are sampled from episode targets. Equality remains in the lower interval.\nRepeated bounds can create empty intervals; a class permutation never changes which rows share a class.',fontsize=12);f.subplots_adjust(bottom=.25);save(f,'classes')
 f,axes=plt.subplots(1,2,figsize=(11,5.6));w=np.array([1/7,6/7]);axes[0].bar(np.arange(2)-.17,[.4,.6],width=.34,label='context only',color=BLUE);axes[0].bar(np.arange(2)+.17,w,width=.34,label='+ query features',color=ORANGE);axes[0].set_xticks([0,1],['world0','world1']);axes[0].set_ylim(0,1);axes[0].set_ylabel('Posterior world weight');axes[0].legend(fontsize=10)
 axes[1].bar([0,1],[.58,11/14],color=[BLUE,ORANGE]);axes[1].set_xticks([0,1],['context only','+ query features']);axes[1].set_ylim(0,1);axes[1].set_ylabel('P(query class = 1)')
 for i,v in enumerate([.58,11/14]):axes[1].text(i,v+.03,f'{v:.4f}',ha='center')
 f.suptitle('A query feature can change the posterior over worlds',fontweight='bold',fontsize=17);f.text(.07,.035,'Prior (.5,.5) × context likelihood (.4,.6) × query likelihood (.2,.8) = (.04,.24).\nNormalize to (1/7,6/7); average class probabilities (.1,.9) to obtain 11/14.',fontsize=12);f.subplots_adjust(bottom=.24,wspace=.4,top=.85);save(f,'posterior')
 result=json.loads((ROOT/'_verify_l063_v2_results.json').read_text());f,axes=plt.subplots(1,2,figsize=(11,5.6),sharex=True)
 for ax,family in zip(axes,['SCM','BNN']):
  rows=[r for r in result['records'] if r['family']==family]
  for i,r in enumerate(rows):ax.plot([r['nll'],r['shuffled_nll']],[i,i],color='#aaa',lw=1);ax.scatter(r['nll'],i,c=BLUE,s=35);ax.scatter(r['shuffled_nll'],i,c=ORANGE,s=35)
  ax.set_title(f'{family} generated tasks');ax.set_xlabel('Query log loss (nats; lower is better)');ax.set_yticks(range(12),[str(r['seed']) for r in rows]);ax.set_ylabel('World seed');ax.grid(axis='x',alpha=.15)
 axes[0].scatter([],[],c=BLUE,label='ordinary context');axes[0].scatter([],[],c=ORANGE,label='shuffled labels');axes[0].legend(fontsize=10)
 f.suptitle('Paired logistic probes: shuffling often hurts, but not always',fontsize=17,fontweight='bold');f.text(.06,.025,'24 fresh worlds, 256 rows each, 128 context rows. Same query targets and fixed logistic recipe within each pair.\nThese are synthetic-world diagnostics, not PFNs trained on different priors and not Table4 reproduction.',fontsize=11);f.subplots_adjust(bottom=.23,wspace=.32,top=.85);save(f,'results')
if __name__=='__main__':build()
