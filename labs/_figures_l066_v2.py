"""Computational, portable L066 figures; synthetic fixtures and measured data distinguished."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures/l066';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#faf9f6','axes.facecolor':'#faf9f6'})
def save(fig,name):fig.savefig(OUT/(name+'-v2.png'),dpi=150,bbox_inches='tight');plt.close(fig)
def box(ax,x,y,w,h,text,color='#e4edf5'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008',facecolor=color,edgecolor='#607080'));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=12)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=2,color='#435665'))
def build():
 fig,ax=plt.subplots(figsize=(11,10));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
 box(ax,.04,.88,.68,.09,'Context X: C × F   |   Query X: Q × F\nContext-only filter → scale → soften outliers')
 box(ax,.79,.88,.17,.09,'Context labels\ny: C', '#f7e5c2')
 box(ax,.04,.67,.68,.15,'COLUMN: shared scalar projection → width 128\n3 × [I reads context → M; all cells read M]\n128 inducing vectors; 4 heads; pre-LN + FFN\nW=LN(Linear(V)); B=LN(Linear(V)); E=xW+B')
 arrow(ax,(.38,.88),(.38,.82));ax.text(.77,.735,'E: N × F × 128',fontsize=11)
 box(ax,.04,.45,.68,.15,'ROW: prepend 4 CLS tokens to each row\n3 blocks; 8 heads; Q/K adjacent-pair RoPE\nbase 100,000; head width 16; positions include CLS\nLN on each final CLS → concatenate 4 × 128')
 arrow(ax,(.38,.67),(.38,.60));ax.text(.77,.515,'H: N × 512\nNo labels yet',fontsize=11)
 box(ax,.04,.22,.68,.16,'ICL: context H + Linear(one-hot(y)); query H\n12 blocks; width 512; 4 heads; pre-LN + FFN\nAll N queries read only C context keys/values\nFinal LN → 512 → GELU(1024) → 10 logits')
 arrow(ax,(.38,.45),(.38,.38));arrow(ax,(.96,.88),(.96,.30));arrow(ax,(.96,.30),(.73,.30))
 box(ax,.14,.06,.48,.09,'Keep K class logits for Q query rows\nlogits / 0.9 → softmax → Q × K probabilities','#e4f0e4');arrow(ax,(.38,.22),(.38,.15))
 ax.set_title('Original February TabICL checkpoint · complete numeric single-view inference',pad=16)
 ax.text(.79,.08,'N = C + Q\n2 ≤ K ≤ 10\nWeights frozen',fontsize=11)
 save(fig,'architecture')
 u=np.array([[-1.,0.],[0.,1.],[2.,1.],[1.,-1.]]);i=np.eye(2)
 def soft(x):v=np.exp(x-x.max(-1,keepdims=True));return v/v.sum(-1,keepdims=True)
 a=soft(i@u[:3].T/np.sqrt(2));m=a@u[:3];b=soft(u@m.T/np.sqrt(2));out=b@m
 fig,axes=plt.subplots(1,3,figsize=(13,4.4))
 for ax,z,title,xlabs,ylabs in zip(axes,[a,m,b],['1. Inducing readers → context','2. M = weights × context values','3. Every cell → two memories'],[['U₀','U₁','U₂'],['coordinate 0','coordinate 1'],['M₀','M₁']],[['I₀ = (1,0)','I₁ = (0,1)'],['M₀','M₁'],['context U₀','context U₁','context U₂','query (1,−1)']]):
  ax.imshow(z,cmap='Blues',vmin=0,vmax=max(1,z.max()),aspect='auto');ax.set(xticks=range(z.shape[1]),xticklabels=xlabs,yticks=range(z.shape[0]),yticklabels=ylabs,title=title)
  for row in range(z.shape[0]):
   for col in range(z.shape[1]):ax.text(col,row,f'{z[row,col]:.4f}',ha='center',va='center',color='white' if z[row,col]>.7 else 'black',fontsize=13)
 fig.suptitle('Synthetic attention-only trace · query output = (1.0814, 0.8698)');fig.tight_layout();save(fig,'inducing')
 fig,axes=plt.subplots(1,2,figsize=(11,4.2));theta=np.linspace(0,2*np.pi,300);axes[0].plot(np.cos(theta),np.sin(theta),color='#999999',lw=1)
 for p,color in [(0,'#3a719c'),(1,'#2b825a'),(2,'#b7573b')]:
  axes[0].arrow(0,0,np.cos(p),np.sin(p),head_width=.07,length_includes_head=True,color=color);axes[0].text(np.cos(p)*1.15,np.sin(p)*1.15,f'p={p}',ha='center',color=color)
 axes[0].set(xlim=(-1.35,1.35),ylim=(-1.35,1.35),aspect='equal',title='First pair: (1,0) rotates by p radians',xlabel='coordinate 0',ylabel='coordinate 1')
 positions=np.arange(9);axes[1].plot(positions,np.cos(positions),'o-');axes[1].axhline(0,color='#999',lw=1);axes[1].set(title='Dot product with the p=0 vector',xlabel='Relative position',ylabel='cos(relative position)',xticks=positions)
 fig.tight_layout();save(fig,'rope')
 fig,axes=plt.subplots(1,2,figsize=(11,4.3));mask=np.zeros((5,5));mask[:,:3]=1
 axes[0].imshow(mask,cmap='Blues',vmin=0,vmax=1);axes[0].set(xticks=range(5),xticklabels=['C₀','C₁','C₂','Q₀','Q₁'],yticks=range(5),yticklabels=['C₀','C₁','C₂','Q₀','Q₁'],xlabel='Key/value row',ylabel='Reader row',title='Dataset attention: 1 allowed, 0 blocked')
 for a in range(5):
  for b0 in range(5):axes[0].text(b0,a,int(mask[a,b0]),ha='center',va='center',color='white' if b0<3 else 'black')
 axes[1].axis('off');axes[1].text(.02,.95,'Known labels: [0, 1, 0]\n\nContext state = H + Linear(one-hot(y))\nQuery state = H, without a target\n\nQ₀ reads C₀, C₁, C₂; never Q₀ or Q₁.\nIts own H survives through residuals.\n\n12 blocks repeat the same boundary.\nFinal head: LN → 512 → 1024 → 10\nKeep K logits; temperature 0.9.',va='top',fontsize=13);fig.tight_layout();save(fig,'roles')
 c=np.arange(40,1001,20);q=100;f=8;n=c+q
 curves=[12*f*128*(c+n),24*n*(f+4)**2,48*n*c]
 fig,axes=plt.subplots(1,2,figsize=(12,4.5))
 for y,label in zip(curves,['Column: 3 × 4 × F × 128(C+N)','Row: 3 × 8 × N(F+4)²','ICL: 12 × 4 × NC']):axes[0].plot(c,y/1e6,label=label)
 axes[0].set(xlabel='Context rows C (Q=100; F=8 fixed)',ylabel='Million attention-score elements',title='Arithmetic count, including blocks and heads');axes[0].legend(fontsize=9)
 vals=[11059200,1728000,9600000];axes[1].barh(['column','row','ICL'],np.array(vals)/1e6,color=['#3a719c','#2b825a','#b7573b']);axes[1].set(xlabel='Million score elements',title='Worked state: C=400, Q=100, F=8',xlim=(0,14))
 for j,v in enumerate(vals):axes[1].text(v/1e6+.15,j,f'{v/1e6:.4f}',va='center')
 fig.tight_layout();save(fig,'cost')
 data=json.loads((ROOT/'_verify_l066_v2_results.json').read_text());fig,axes=plt.subplots(2,3,figsize=(13,7.6))
 for j,dataset in enumerate(data['config']['datasets']):
  for seed in data['config']['seeds']:
   rows=[r for r in data['records'] if r['dataset']==dataset and r['seed']==seed];x=[len(r['context_ids']) for r in rows]
   for ax,metric in [(axes[0,j],'log_loss'),(axes[1,j],'accuracy')]:ax.plot(x,[r[metric] for r in rows],'o-',label=f'seed {seed}');ax.set(xlabel='Context rows',ylabel=metric.replace('_',' '));ax.grid(alpha=.2)
  axes[0,j].set_title(dataset.replace('_',' '));axes[0,j].legend(fontsize=10);axes[1,j].set_ylim(.65,1.01)
 fig.suptitle('Fresh original-checkpoint measurements · paired contexts; fixed queries within each seed');fig.tight_layout();save(fig,'results')
 summary=json.loads((ROOT/'_analysis_l066_v2_results.json').read_text());fig,axes=plt.subplots(1,2,figsize=(12,4.3));ranks=summary['mean_ranks'];axes[0].set(xlim=(.8,3.2),ylim=(-.4,2.8),yticks=[0,1,2],yticklabels=['full context','37.5% context','12.5% context'],xticks=[1,2,3],xlabel='Mean dataset rank (lower loss is better)',title='Three datasets; seeds averaged before ranking')
 axes[0].plot(ranks[::-1],[0,1,2],'o');cd=summary['nemenyi_cd'];axes[0].plot([1,1+cd],[2.5,2.5],lw=3);axes[0].text(1.95,2.55,f'Nemenyi CD={cd:.3f}',ha='center',fontsize=11)
 for j,dataset in enumerate(data['config']['datasets']):
  rows=[r for r in summary['summary'] if r['dataset']==dataset];bottom=np.zeros(3)
  for key,label,color in [('column_seconds','column','#3a719c'),('row_seconds','row','#2b825a'),('icl_seconds','ICL','#b7573b')]:
   values=[r['stage_seconds'][key] for r in rows];axes[1].bar(np.arange(3)+j*4,values,bottom=bottom,color=color,label=label if j==0 else None);bottom+=values
 axes[1].set(xticks=[1,5,9],xticklabels=['diabetes','blood','WDBC'],ylabel='Seconds (mean over three seeds)',title='Each group: small → middle → full context');axes[1].legend(fontsize=10)
 fig.suptitle(f"Exploratory Friedman p={summary['friedman_p']:.4f} · CPU timings are this run, not paper speedups");fig.tight_layout();save(fig,'ranks')
 symmetry=json.loads((ROOT/'_symmetry_l066_results.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4.2));ctx=np.array(symmetry['context_x']);axes[0].axis('off');table=axes[0].table(cellText=[[int(a),int(b),int(y)] for (a,b),y in zip(ctx,symmetry['context_y'])],colLabels=['Feature A','Feature B','Context y'],loc='center',cellLoc='center');table.auto_set_font_size(False);table.set_fontsize(14);table.scale(1,2.2);axes[0].set_title('Identical context column distributions')
 for j,r in enumerate(symmetry['records']):axes[1].bar(np.arange(2)+(j-.5)*.3,np.array(r['query_probabilities'])[:,1],width=.3,label='RoPE on' if r['rope'] else 'RoPE off')
 axes[1].set(xticks=[0,1],xticklabels=['query [0,1]','query [1,0]'],ylabel='Probability of class 1',ylim=(0,1.06),title='Original trained weights; only RoPE toggled');axes[1].legend();fig.tight_layout();save(fig,'symmetry')
if __name__=='__main__':build()
