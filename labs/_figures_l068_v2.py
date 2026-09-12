"""Portable numerical figures generated from the actual source/evidence operators."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/l068-matplotlib')
from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from relkit import driftpfn_l068_v2 as c
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures/l068';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#faf9f5','axes.facecolor':'#faf9f5','savefig.facecolor':'#faf9f5'})
COLORS=['#537c9f','#243b5a','#007c78','#ae5d38','#8c69a4'];ARMS=['base_no_time','base_time','drift','drift_zero','noT2V'];LABELS=['Base, no time','Base + time','Drift','Drift, zero time','NoT2V checkpoint']
def save(fig,name):fig.savefig(OUT/(name+'-v2.png'),dpi=160,bbox_inches='tight');plt.close(fig)
def box(ax,x,y,w,h,text,color='#e8ecee'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.015',facecolor=color,edgecolor='#657787'));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
def architecture():
 fig,ax=plt.subplots(figsize=(10,10));ax.set(xlim=(0,10),ylim=(0,11));ax.axis('off')
 ax.set_title('Complete released Drift-Resilient TabPFN\nWhere does a future timestamp enter the prediction?',fontsize=16,pad=20)
 box(ax,.2,9.4,4.4,1,'X: B × (C+Q) × F\nPairs → pack / source impute / z-score')
 box(ax,5.2,9.4,4.5,1,'c: B × (C+Q)\nSource min–max → 100 time coordinates')
 box(ax,.2,7.8,6.7,1,'Repeat 100 time coordinates into EACH numeric pair\n[2 numeric ; 100 time] → Linear(102,192) + group ID','#dceee8')
 box(ax,7.3,7.8,2.4,1,'Context y / query\nquery [mean-rank, −2]\nLinear(2,192)')
 for x,y,xx,yy in [(2.4,9.4,2.4,8.8),(7.4,9.4,5.4,8.8),(3.6,7.8,4.6,7.2),(8.5,7.8,6.1,7.2)]:ax.annotate('',(xx,yy),(x,y),arrowprops=dict(arrowstyle='->',lw=2))
 ax.text(5,7,'State B × (C+Q) × (G+1) × 192',ha='center',weight='bold')
 box(ax,.6,2,8.7,4.5,'',color='#f0ece1');ax.text(.9,6.2,'REPEAT 12 TIMES · all checkpoint weights',weight='bold')
 box(ax,1,5.05,7.9,.75,'Feature attention: groups + target within each row')
 box(ax,1,3.95,7.9,.75,'Row attention: context K/V only\nContext: 6 K/V heads · Query: first K/V head shared')
 box(ax,1,2.85,7.9,.75,'Token MLP: 192 → GELU(768) → 192')
 ax.text(5,2.3,'Each sublayer: residual addition → non-affine LayerNorm',ha='center')
 for y in [5.05,3.95]:ax.annotate('',(5,y-.35),(5,y),arrowprops=dict(arrowstyle='->'))
 box(ax,1,.45,8,1,'Take QUERY TARGET token → 192 → GELU(768) → 10\nKeep K observed classes → softmax → B × Q × K','#dceee8')
 ax.annotate('',(5,1.45),(5,2),arrowprops=dict(arrowstyle='->',lw=2))
 save(fig,'architecture')
def timefigure():
 fig,axs=plt.subplots(1,2,figsize=(11,4.2));t=np.linspace(0,8,200);u=t/4
 axs[0].plot(t,u,label='Source fit: (c−0)/(4−0)',lw=2);axs[0].plot(t,np.clip(u,0,1),'--',label='Incorrect [0,1] clipping')
 axs[0].scatter([0,2,4,6],[0,.5,1,1.5],s=55);axs[0].annotate('query 6 → 1.5',(6,1.5),xytext=(4,1.85),arrowprops=dict(arrowstyle='->'));axs[0].set(xlabel='Raw domain time c',ylabel='Normalized time τ',title='Only source endpoints define the clock');axs[0].legend(fontsize=9,loc='lower right')
 values=[1.5,np.sin(1.5*np.pi/2),np.sin(1.5*np.pi),np.sin(1.5*2*np.pi)];axs[1].bar(range(4),values,color=['#243b5a','#007c78','#007c78','#007c78']);axs[1].set(xticks=range(4),xticklabels=['linear','sin(πτ/2)','sin(πτ)','sin(2πτ)'],ylim=(-1.3,1.9),ylabel='Encoded coordinate',title='Illustrative weights, τ = 1.5')
 for i,v in enumerate(values):axs[1].text(i,v+(.08 if v>=0 else -.17),f'{v:.4f}',ha='center')
 fig.tight_layout();save(fig,'time')
def scm():
 d=c.sample_scm(2)['trace'];fig,axs=plt.subplots(2,1,figsize=(11,8),gridspec_kw={'height_ratios':[1,1.2]});a=axs[0];a.axis('off');a.set(xlim=(0,12),ylim=(0,3))
 box(a,.2,1,2,1,'Time c\nH: shared causes');box(a,3,1,2.8,1,'12 correlated outputs\nshared r(c) → s(c) → ω(c)');box(a,6.6,1,2.5,1,'Mask relations {0,3}\nactive edges {0,1,8,9}');box(a,9.6,1,2.1,1,'G_c fixed per domain\nfresh row noise → X,y')
 for x,xx in [(2.2,3),(5.8,6.6),(9.1,9.6)]:a.annotate('',(xx,1.5),(x,1.5),arrowprops=dict(arrowstyle='->',lw=2))
 a.set_title('Sparse causal relationships, correlated functional-edge shifts',fontsize=15)
 ax=axs[1];w=np.array(d['weights']);tt=d['domains']
 for e,label in [(0,'U→X hidden0 (shifted)'),(1,'U→X hidden1 (shifted)'),(8,'X→Y hidden0 (shifted)'),(2,'V→X hidden0 (fixed)')]:ax.plot(tt,w[:,e],marker='o',label=label)
 ax.set(xlabel='Domain time c; same H for every point',ylabel='Functional edge weight',title='Seed 2 reconstruction: different outputs share hidden temporal causes');ax.legend(ncol=2,fontsize=10);fig.tight_layout();save(fig,'scm')
def split():
 fig,ax=plt.subplots(figsize=(11,4.8));d=c.load_dataset(ROOT,'electricity');u=np.unique(d['c']);counts=[]
 for j,frac in enumerate([.4,.55,.7]):
  n=int(len(u)*frac);s=c.temporal_split(d['c'],n,j,None)
  for k,color in [('train','#007c78'),('id','#dfb257'),('ood','#b16a54')]:
   vals=np.array([sum(d['c'][s[k]]==v) for v in u]);bottom=np.array([sum(d['c'][s['train']]==v) for v in u]) if k=='id' else 0
   ax.bar(np.arange(len(u))+j*17,vals,bottom=bottom,color=color,label=k if j==0 else None)
  ax.axvline(j*17+n-.5,color='#243b5a',ls='--');ax.text(j*17+7,95,f'{n} source / {len(u)-n} future domains',ha='center')
 ax.set(xticks=[7,24,41],xticklabels=['Repetition 0 · 40%','Repetition 1 · 55%','Repetition 2 · 70%'],ylabel='Rows per original weekly domain',ylim=(0,110),title='Electricity: different cutoffs, frozen context, all rows retained');ax.legend(ncol=3,loc='lower right');fig.tight_layout();save(fig,'split')
def results():
 s=json.loads((ROOT/'_verify_l068_v2_results.json').read_text());fig,axs=plt.subplots(1,4,figsize=(15,5),sharey=True)
 for ax,d in zip(axs,['electricity','parking','chess','blobs']):
  for i,a in enumerate(ARMS):
   vals=[r['log_loss'] for r in s['records'] if (r['dataset'],r['arm'],r['split'])==(d,a,'ood')]
   ax.scatter(i+np.array([-.12,0,.12]),vals,color=COLORS[i],s=26);ax.errorbar(i,np.mean(vals),yerr=np.std(vals,ddof=1),fmt='_',color=COLORS[i],capsize=4,markersize=18)
  ax.set(title=d.capitalize()+(' (synthetic)' if d=='blobs' else ''),xticks=range(5),xticklabels=LABELS);ax.tick_params(axis='x',labelrotation=65)
 axs[0].set_ylabel('OOD log loss ↓ (nats)');fig.suptitle('Full pretrained checkpoints · each dot is a declared cutoff/checkpoint repetition\nBar = sample SD; three real datasets, not twelve independent dataset replicates',fontsize=14);fig.tight_layout();save(fig,'results')
def boundary():
 r=json.loads((ROOT/'_boundary_l068_v2_results.json').read_text());data=c.load_dataset(ROOT,'blobs');grid=np.array(r['grid']);fig,axs=plt.subplots(2,3,figsize=(12,8));from matplotlib.colors import ListedColormap
 cmap=ListedColormap(['#b9d4e6','#c6dfb6','#eecac0'])
 for i,arm in enumerate(['base_time','drift']):
  for j,d in enumerate([4,8,13]):
   row=next(v for v in r['records'] if v['arm']==arm and v['domain']==d);p=np.array(row['probabilities']);ax=axs[i,j]
   ax.pcolormesh(grid[:,0].reshape(r['shape']),grid[:,1].reshape(r['shape']),p.argmax(1).reshape(r['shape']),cmap=cmap,vmin=0,vmax=2,shading='nearest',rasterized=True)
   take=data['c']==d;ax.scatter(data['x'][take,0],data['x'][take,1],c=data['y'][take],cmap=ListedColormap(['#325c89','#44753a','#944630']),vmin=0,vmax=2,s=8)
   ax.set(title=f'{arm} · domain {d}\nGrid confidence {row["mean_confidence"]:.3f}',xlabel='Feature 1',ylabel='Feature 2',xlim=(-20,20),ylim=(-26,17))
 fig.suptitle('Same source domains 0–3 and same weights: only requested future time changes',fontsize=14);fig.tight_layout();save(fig,'boundary')
def ranks():
 s=json.loads((ROOT/'_analysis_l068_v2_results.json').read_text());fig,axs=plt.subplots(1,2,figsize=(11,5))
 for ax,m in zip(axs,['auc','log_loss']):
  r=s['stats'][m];ax.scatter(r['mean_ranks'],range(5),s=70,c=COLORS);ax.set(yticks=range(5),yticklabels=LABELS,xlim=(.5,5.5),xlabel='Mean rank across 3 real datasets (lower better)',title=f'{m} · Friedman p={r["friedman_p"]:.3f}')
  ax.hlines(-.7,1,1+r['nemenyi_cd'],color='#243b5a',lw=3);ax.text(1,-.9,f'Nemenyi CD = {r["nemenyi_cd"]:.3f}',fontsize=10);ax.set_ylim(4.7,-1.3);ax.grid(axis='x',alpha=.2)
 fig.suptitle('Average repetitions first, rank within datasets, then weight datasets equally\nAll pairwise mean-rank gaps are below CD; nonsignificance is not equivalence',fontsize=13);fig.tight_layout();save(fig,'ranks')
if __name__=='__main__':
 for fn in [architecture,timefigure,scm,split,results,boundary,ranks]:fn()
 print('Seven portable figures generated')

def horizon():
 s=json.loads((ROOT/'_verify_l068_v2_results.json').read_text());fig,axs=plt.subplots(2,2,figsize=(11,7))
 for ax,d in zip(axs.ravel(),['electricity','parking','chess','blobs']):
  for arm,color in [('base_time',COLORS[1]),('drift',COLORS[2]),('noT2V',COLORS[4])]:
   row=next(v for v in s['records'] if (v['dataset'],v['arm'],v['seed'],v['split'])==(d,arm,0,'ood'));v=row['domain_metrics'];ax.plot([z['domain'] for z in v],[z['log_loss'] for z in v],marker='o',ms=3,color=color,label=arm)
  ax.set(title=d.capitalize(),xlabel='Actual future domain index; fixed context',ylabel='Domain log loss ↓');ax.legend(fontsize=9)
 fig.suptitle('Repetition 0 only: how does error evolve after its fixed cutoff?\nNeighboring domains share a context and are not independent datasets.',fontsize=13);fig.tight_layout();save(fig,'horizon')
if __name__=='__main__':horizon()
