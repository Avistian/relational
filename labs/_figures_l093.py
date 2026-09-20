"""HGT architecture, attention, time and sampling: editable SVG + portable PNG."""
from pathlib import Path
import json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;OUT=P/'figures/l093';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none'})
ink='#20394b';teal='#007f86';rust='#b54c32';gold='#a87319'
def base(title,subtitle,h=7):
 f,a=plt.subplots(figsize=(13,h));f.patch.set_facecolor('#fcfcf8');a.set_xlim(0,13);a.set_ylim(0,h);a.axis('off');a.text(.15,h-.4,title,fontsize=22,weight='bold',color=ink);a.text(.15,h-.8,subtitle,fontsize=11,color='#566775');return f,a
def box(a,x,y,w,h,title,body,color=teal):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06,rounding_size=.12',ec=color,fc='white',lw=1.6));a.text(x+.12,y+h-.28,title,weight='bold',fontsize=12,color=color);a.text(x+.12,y+h-.57,body,va='top',fontsize=11,color=ink,linespacing=1.5)
def arrow(a,x,y,u,v,color=ink):a.annotate('',xy=(u,v),xytext=(x,y),arrowprops={'arrowstyle':'->','lw':1.7,'color':color})
def save(f,name):
 f.savefig(OUT/f'{name}.svg',bbox_inches='tight');f.savefig(OUT/f'{name}.png',dpi=140,bbox_inches='tight');plt.close(f)
f,a=base('HGT · schema-conditioned messages, end to end','Pictured: paper-sized modern reconstruction using release operators; d = 256, H = 8, D = 32',9)
box(a,.25,5.9,2.65,1.9,'OAG + target papers','5 node types; typed edges\nFeatures X: N × F\nEvent or inherited time')
box(a,3.4,5.9,3,1.9,'SAMPLE + MASK','Budget separately by type\nReconstruct induced edges\nRemove target L2 links ↔',rust)
box(a,7,5.9,2.3,1.9,'ADAPT BY TYPE','F → 256; tanh\nDropout 0.2\nZ: n × 256')
box(a,9.9,5.9,2.6,1.9,'REPEAT 3 LAYERS','One-hop edges each time\nComposes implicit routes\nNew weights per layer')
for x,u in [(2.95,3.3),(6.5,6.9),(9.4,9.8)]:arrow(a,x,6.85,u,6.85)
box(a,.25,2.55,3.0,2.45,'ONE EDGE: s → t','zₛ + RTE(Tₜ − Tₛ + 120)\nSource type: Kₛ, Vₛ\nTarget type: Qₜ\nEach: E × 8 × 32')
box(a,3.85,3.9,3.65,1.15,'ATTENTION BRANCH','q · (k Wᴬᵀᵀᵣ) × μᵣ / √32',gold)
box(a,3.85,2.25,3.65,1.15,'MESSAGE BRANCH','v Wᴹˢᴳᵣ : E × 8 × 32',rust)
arrow(a,3.35,4.35,3.75,4.5,gold);arrow(a,3.35,3.1,3.75,2.8,rust)
box(a,8.05,2.25,4.45,2.8,'MEET AT THE RECEIVER','Softmax across ALL incoming relations\nΣₛ αₛₜ messageₛₜ → n × 256\nGELU → target-type A projection\nDropout → gated residual → LayerNorm')
arrow(a,7.6,4.5,7.95,4.5,gold);arrow(a,7.6,2.8,7.95,2.8,rust)
arrow(a,11.1,5.8,11.1,5.15)
a.text(.25,1.5,'Prediction handoff: gather seed-paper vectors → Linear 256 → number of L2 fields → log-softmax',fontsize=13,color=ink,weight='bold')
a.text(.25,.95,'Training: normalized multi-label targets → KL divergence. Validation alone chooses the checkpoint.',color=ink)
a.text(.25,.48,'Inference: selected weights, dropout off → rank all candidate fields → NDCG + MRR on later papers.',color=ink)
save(f,'architecture')
f,a=base('One target · both relation types compete in one softmax','Synthetic one-head trace after type projections; D = 2; relation matrices initially identity')
box(a,.25,3.7,3.65,1.95,'AUTHOR → PAPER','q = [1, 0], k = [0, 1]\nv = [2, 0], prior μ = 1\nscore = 0',teal)
box(a,.25,1.25,3.65,1.95,'PAPER → PAPER','q = [1, 0], k = [√2 ln 2, 0]\nv = [6, 0], prior μ = 1\nscore = ln 2',rust)
box(a,4.65,2.3,3.05,2.25,'SHARED NORMALIZER','exp(scores) = [1, 2]\nα = [1/3, 2/3]\nSum α = 1',gold)
arrow(a,4,4.6,4.55,3.9);arrow(a,4,2.2,4.55,2.9)
box(a,8.35,2.3,4.1,2.25,'WEIGHT THE MESSAGES','(1/3)[2,0] + (2/3)[6,0]\n= [14/3, 0] ≈ [4.667, 0]\nBefore GELU / residual')
arrow(a,7.8,3.4,8.25,3.4)
a.text(.25,.65,'Change only the citation prior to 2: exp(scores) = [1,4], α = [.2,.8], message = [5.2,0].',color=ink)
a.text(.25,.2,'A separate softmax per relation would give [1,1] here: total weight 2. That is a different computation.',color=rust)
save(f,'attention')
f,a=base('Relative time · one source, two receiving contexts','Release initialization, d = 4; paired sin/cos frequencies and 1/√d scale; learned projection follows')
box(a,.3,3.85,3.35,1.7,'2012 → 2015','ΔT = 3 years\nTable index = 3 + 120 = 123')
box(a,.3,1.65,3.35,1.7,'2012 → 2018','ΔT = 6 years\nTable index = 6 + 120 = 126',rust)
for i,(idx,col) in enumerate([(123,teal),(126,rust)]):
 vec=np.empty(4);vec[::2]=np.sin(idx*np.array([1,.01]))/2;vec[1::2]=np.cos(idx*np.array([1,.01]))/2
 box(a,4.3,3.85-i*2.2,4,1.7,f'BASIS AT INDEX {idx}', '['+', '.join(f'{v:.3f}' for v in vec)+']\nT-Linear: 4 → 4',col)
 arrow(a,3.75,4.7-i*2.2,4.2,4.7-i*2.2,col)
box(a,9,2.3,3.35,2.55,'ADD TO SOURCE','Same source vector\nDifferent time vector\nDifferent key AND value\nQuery stays target-based',gold)
arrow(a,8.4,4.7,8.9,4.2);arrow(a,8.4,2.5,8.9,3)
a.text(.3,.85,'Zero gap still means index120, not a zero vector. Removing RTE means bypassing the addition.',color=ink)
a.text(.3,.35,'Time features do not enforce a cutoff: edge eligibility, feature availability and label masking need audits.',color=rust)
save(f,'time')
f,a=base('HGSampling · normalize locally, budget by type','Illustrative budget update with no pre-sampling cap: two selected papers share author b')
box(a,.3,3.2,3.4,2.3,'EXPAND TWO PAPERS','p₀ has authors {a, b}: +1/2\np₁ has authors {b, c}: +1/2\nBoth contributions reach b')
box(a,4.35,3.2,3.6,2.3,'AUTHOR BUDGET','B(a,b,c) = [.5, 1, .5]\nSquare = [.25, 1, .25]\nP(a,b,c) = [1/6, 2/3, 1/6]',gold)
box(a,8.6,3.2,3.8,2.3,'SAMPLE + RECONSTRUCT','Draw within each node type\nUpdate budgets; repeat depth\nRetain induced typed edges',rust)
arrow(a,3.8,4.35,4.25,4.35);arrow(a,8.05,4.35,8.5,4.35)
a.text(.3,2.25,'Uniform author sampling gives [1/3,1/3,1/3]. HGSampling favors the shared, densely connected author.',color=ink)
a.text(.3,1.65,'Types have separate quotas. A million paper candidates do not consume the author quota.',color=ink)
a.text(.3,1.05,'Release detail: cap each expanded relation neighborhood before adding its normalized budget.',color=ink)
a.text(.3,.45,'Timestamp detail: untimed neighbors inherit context; induced-edge reconstruction needs a separate audit.',color=rust)
save(f,'sampling')
p=P/'_teaching_l093_results.json'
if p.exists() and json.loads(p.read_text())['status']=='COMPLETE':
 r=json.loads(p.read_text());f,axs=plt.subplots(1,2,figsize=(12,4.6));f.patch.set_facecolor('#fcfcf8')
 labels=['HGT','R-GCN','HGT\n(no RTE)','Frequency'];arms=['hgt','rgcn','hgt_no_rte'];baseline=json.loads((P/'_baseline_l093_results.json').read_text())
 for ax,metric in zip(axs,['ndcg','mrr']):
  for seed in [0,1,2]:
   values=[next(v['test_'+metric] for v in r['runs'] if v['arm']==arm and v['seed']==seed) for arm in arms]
   values.append(next(v[metric] for v in baseline['runs'] if v['seed']==seed))
   ax.plot(range(4),values,'o-',alpha=.65,label=f'seed {seed}')
  ax.set_xticks(range(4),labels);ax.set_ylabel('NN test '+metric.upper()+' (detail scale)');ax.set_ylim((.35,.55) if metric=='ndcg' else (.985,1.002));ax.grid(axis='y',alpha=.25);ax.spines[['top','right']].set_visible(False)
 axs[0].legend();f.suptitle('Local NN teaching comparison · matched sample streams · 3 model seeds',weight='bold');f.tight_layout();save(f,'results')
