"""Portable figures: computed mechanisms, complete routing, measured uncertainty."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/relational-matplotlib")
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE=Path(__file__).parent
DEST=HERE/'figures/l049';DEST.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
COLORS=['#245e7c','#388462','#c17438']

def save(fig,name):
    fig.savefig(DEST/(name+'.png'),dpi=160,bbox_inches='tight',pad_inches=.15)
    plt.close(fig)

# Routing diagrams keep side inputs separate from the prediction path.
def routed_architecture(name, title, nodes, edges, footer, height):
    fig,ax=plt.subplots(figsize=(5,height));ax.set_xlim(-2,102);ax.set_ylim(5,100);ax.axis('off')
    for a,b,label in edges:
        x,y,w,h,_,_=nodes[a];xx,yy,ww,hh,_,_=nodes[b]
        start=(x+w/2,y);end=(xx+ww/2,yy+hh)
        if abs(y-yy)<2: start=(x+w,y+h/2);end=(xx,yy+hh/2)
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#62757e','lw':1.5})
        if label:ax.text((start[0]+end[0])/2,(start[1]+end[1])/2,label,fontsize=8,ha='center',va='center',bbox={'facecolor':'white','edgecolor':'none','pad':1})
    if name=='excel-architecture':
        ax.plot([94,97,97,94],[56,56,26,26],color='#388462',lw=1.5)
        ax.text(98,41,'×2',ha='left',va='center',fontsize=10,color='#256c50')
    labels=[]
    for key,(x,y,w,h,head,body) in nodes.items():
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.6',facecolor='#edf5ee' if key in ['mask','weight','aggregate'] else '#edf3f7',edgecolor='#a7b9bf'))
        a=ax.text(x+w/2,y+h*.73,head,fontsize=10,weight='bold',ha='center',va='center')
        b=ax.text(x+w/2,y+h*.35,body,fontsize=9,ha='center',va='center',linespacing=1.4)
        labels.append((key,(x,y,w,h),a,b))
    fig.canvas.draw()
    for key,(x,y,w,h),a,b in labels:
        lo=ax.transData.transform((x,y));hi=ax.transData.transform((x+w,y+h))
        for text in (a,b):
            box=text.get_window_extent(fig.canvas.get_renderer())
            assert box.x0>=lo[0]-1 and box.x1<=hi[0]+1 and box.y0>=lo[1]-1 and box.y1<=hi[1]+1,(name,key,text.get_text())
    ax.set_title(title,loc='left',fontsize=13,weight='bold',pad=15)
    fig.text(.5,.04,footer.replace(r'\n','\n'),ha='center',fontsize=9)
    save(fig,name)

routed_architecture('excel-architecture','ExcelFormer · numeric lab forward pass',{
 'input':(12,88,76,10,'Ordered row [B,C]','Training MI orders columns, strongest first'),
 'token':(12,74,76,10,'Gated tokens [B,C,32]','zᵢ = (xᵢwᵢ+bᵢ) ⊙ tanh(xᵢuᵢ+cᵢ)'),
 'mix':(12,60,76,10,'Optional Feat-Mix · training only','Swap tokens with donor; mix target by MI'),
 'mask':(12,43,76,13,'Masked attention residual','a = softmax(QKᵀ/√dₕ + mask)V\nz ← z + output_projection(a)'),
 'gate':(12,26,76,13,'Gated residual','a,g = split(linear(LayerNorm(z)))\nz ← z + a ⊙ tanh(g)'),
 'head':(12,9,76,13,'Column pool → prediction','[B,C,32] → [B,32] → LayerNorm\nPReLU → linear logit → sigmoid')},
 [('input','token',''),('token','mix',''),('mix','mask',''),('mask','gate',''),('gate','head','after 2 blocks')],
 'At inference Feat-Mix is the identity. No cross-row attention.\\nFirst attention omits pre-norm; later attention uses it.',8)

routed_architecture('trompt-architecture','Trompt · one cell and its prediction',{
 'row':(0,87,44,10,'Row values','Numeric / categorical\nInput [B,C]'),
 'prev':(55,87,44,10,'Prompt + state','Eₚ [P,d]\nOprev [B,P,d]'),
 'encode':(0,69,44,12,'Encode and expand','E: [B,C,d]\nÊ: [B,P,C,d]'),
 'fuse':(55,69,44,12,'Fuse · Eq.2','H = dense([Eₚ; O])\n+ Eₚ + O; O = Oprev'),
 'weight':(55,50,44,12,'Column weights','M = softmax(H Ecolᵀ)\n[B,P,C]; normalize C'),
 'aggregate':(12,31,76,13,'Weighted column sum · Eq.5','O[b,p,k] = Σc M[b,p,c] Ê[b,p,c,k]\n[B,P,d] also feeds the next cell'),
 'head':(12,12,76,13,'Shared downstream head','Softmax over prompts P → weighted sum\n[B,d] → dense → ReLU → class scores')},
 [('row','encode',''),('prev','fuse',''),('fuse','weight','column vectors enter here'),('encode','aggregate','Ê'),('weight','aggregate','M'),('aggregate','head','')],
 'Repeat cells (paper L=6). Train: sum cell losses.\\nInfer: average predictions. Lab codes Eq.4/5 only.',8)

fig,axes=plt.subplots(2,2,figsize=(7,6),gridspec_kw={'height_ratios':[2,1]})
v=np.array([2.,4.,9.]);m=np.tril(np.ones((3,3)));m=m/m.sum(1,keepdims=True)
for ax,masked,title in zip(axes[0],[True,False],['Mask on · strongest first','Mask off · same zero logits']):
 a=m if masked else np.ones((3,3))/3
 ax.imshow(a,vmin=0,vmax=1,cmap='BuGn');ax.set_xticks(range(3),['Strong','Middle','Weak']);ax.set_yticks(range(3),['Strong','Middle','Weak']);ax.set_xlabel('Sender');ax.set_ylabel('Receiver');ax.set_title(title,fontsize=11)
 for i in range(3):
  for j in range(3):ax.text(j,i,f'{a[i,j]:.3f}',ha='center',va='center',color='white' if a[i,j]>.65 else '#122e39')
for ax,a,title in zip(axes[1],[m,np.ones((3,3))/3],['Weak value 9 → 18, mask on','Weak value 9 → 18, mask off']):
 a0=a@v;a1=a@np.array([2,4,18]);ax.plot(range(3),a0,'o-',label='Baseline 9',color=COLORS[0]);ax.plot(range(3),a1,'s--',label='Changed 18',color=COLORS[2]);ax.set_xticks(range(3),['Strong','Middle','Weak']);ax.set_ylim(0,9);ax.set_title(title,fontsize=10);ax.set_ylabel('Output value');ax.grid(axis='y',alpha=.2)
axes[1,0].legend(fontsize=8);fig.suptitle('A blocked route cannot carry the intervention',fontsize=14);fig.tight_layout();save(fig,'spa')

fig,axes=plt.subplots(2,1,figsize=(6,5.5));values=np.array([2,4,9])
for ax,t in zip(axes,[0,2]):
 logits=np.array([t,0,0]);w=np.exp(logits-logits.max());w/=w.sum();contrib=w*values
 ax.barh(range(3),contrib,color=COLORS);ax.set_yticks(range(3),['Feature 1','Feature 2','Feature 3']);ax.invert_yaxis();ax.set_xlim(0,4.4)
 for i in range(3):ax.text(contrib[i]+.07,i,f'{w[i]:.3f} × {values[i]} = {contrib[i]:.3f}',va='center',fontsize=10)
 ax.set_title(f'Query [{t}, 0] → weights softmax([{t}, 0, 0])\nWeighted output = {contrib.sum():.3f}',loc='left',fontsize=11)
 ax.set_xlabel('Contribution to one output coordinate');ax.grid(axis='x',alpha=.15)
fig.suptitle('One fused prompt, fixed columns and feature values',fontsize=13);fig.tight_layout();save(fig,'prompt')

fig,ax=plt.subplots(figsize=(6.3,2.8));ax.barh([0,1],[.6,1/3],color=[COLORS[1],COLORS[2]]);ax.set_yticks([0,1],['Importance share\n6 / (6 + 3 + 1)','Feature count\n1 / 3']);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel('Mixed target, with donor labels 1 and 0');ax.set_title('Keep only the strongest of three features',loc='left')
ax.text(.62,0,'0.600',va='center');ax.text(.35,1,'0.333',va='center');fig.tight_layout();save(fig,'mix')

path=HERE/'_verify_l049_results.json'
if path.exists():
 r=json.loads(path.read_text())
 for key,name,title in [('summary','scores','Author run · released paper splits'),('transfer_summary','transfer','Separate MovieLens task · same 12,000 sampled events')]:
  per=r[key]['per_dataset'];fig,axes=plt.subplots(len(per),1,figsize=(7,2.0*len(per)),squeeze=False)
  for ax,(ds,models) in zip(axes[:,0],per.items()):
   for i,(model,d) in enumerate(models.items()):
    ax.errorbar(d['mean'],i,xerr=d['sample_std'],fmt='o',capsize=4,color=COLORS[i]);ax.scatter(d['scores'],np.full(3,i)+np.linspace(-.12,.12,3),s=18,color=COLORS[i],alpha=.7)
   ax.set_yticks(range(3),['Excel · no DA','Excel · Feat-Mix','CatBoost']);ax.invert_yaxis();ax.set_xlim(*({'pima':(.79,.85),'breast':(.990,1.002),'banknote':(.995,1.002),'random':(.65,.70),'temporal':(.65,.70)}[ds]));ax.set_title(ds+' · detail view',loc='left',fontsize=11);ax.grid(axis='x',alpha=.2)
  axes[-1,0].set_xlabel('Test AUROC · points = seeds; bars = ± sample SD')
  fig.suptitle(title+'\n'+('Task-specific axis ranges · compare models within each panel' if name=='scores' else 'Both panels share the same zoomed AUROC axis'),fontsize=12);fig.tight_layout();save(fig,name)
