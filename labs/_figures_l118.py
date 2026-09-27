"""Portable computation figures, generated from one worked example."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import numpy as np
P=Path(__file__).resolve().parent/'figures/l118';P.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l118'})
ink='#203448';blue='#216b91';green='#147d64';muted='#e8edf0';orange='#a55d16'
def save(fig,name):
 fig.savefig(P/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});fig.savefig(P/(name+'.png'),dpi=160,bbox_inches='tight',metadata={'Software':'L118'});plt.close(fig)
def box(ax,x,y,w,h,text,color=muted,size=11):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.015',facecolor=color,edgecolor=ink,lw=.8));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color=ink)
def arrow(ax,a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color=blue,lw=1.3))
# Each panel repeats the same graph; edge direction is the foreign-key direction.
positions={0:(.42,.48),1:(.14,.48),2:(.71,.48),3:(.71,.2),4:(.42,.2),5:(.14,.78)}
labels=['Customer A','Order A','Country','Customer B','Order B','Line A']
edges=[(1,0),(0,2),(3,2),(4,3),(5,1)]
fig,axes=plt.subplots(3,1,figsize=(8,9))
for ax,selected,title in zip(axes,[{0},{0,1,5},{0,1,2,5}],['1  Start from the target','2  Follow incoming references to closure','3  Follow outgoing references to closure']):
 ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title(title,loc='left',color=ink,fontweight='bold')
 for u,v in edges:
  a=np.array(positions[u]);b=np.array(positions[v]);delta=b-a
  if abs(delta[0])>abs(delta[1]):offset=np.array([np.sign(delta[0])*.121,0])
  else:offset=np.array([0,np.sign(delta[1])*.081])
  arrow(ax,a+offset,b-offset)
 for i,(x,y) in positions.items():box(ax,x-.105,y-.065,.21,.13,labels[i], '#c7e9df' if i in selected else '#f2f4f5',10)
 ax.text(.02,.02,'Selected: '+', '.join(labels[i] for i in sorted(selected)),fontsize=10,color=green)
fig.text(.12,.005,'After Country is found, do not restart the incoming pass: Customer B stays excluded.',fontsize=10,color=ink)
save(fig,'extraction')
fig,ax=plt.subplots(figsize=(8,10));ax.set(xlim=(0,10),ylim=(0,13));ax.axis('off')
box(ax,.2,11.45,9.6,1.2,'One applicant graph · N rows across seven table types\nReleased Home Credit query: undirected paths of length ≤ 2',size=12)
arrow(ax,(5,11.4),(5,10.9))
box(ax,.2,9.55,4.5,1.3,'Categorical column\nindex → embedding ≤ 32\nmissing / unknown → index 0')
box(ax,5.1,9.55,4.7,1.3,'Numeric column\n(value − median) / IQR\n+ missing flag; clip to [−5, 5]')
arrow(ax,(2.45,9.5),(4,8.9));arrow(ax,(7.45,9.5),(6,8.9))
box(ax,1,7.8,8,1.05,'Separate MLP for each table: d → 4d → 256\nSELU activations; hidden and embedding dropout 0.5')
arrow(ax,(5,7.75),(5,7.15))
box(ax,.4,5.55,9.2,1.55,'N × 256 node matrix H\nAdd reverse edges and self loops before normalizing\nH′ = SELU(D⁻½ A D⁻½ H W + b), then dropout\nOne shared GCN layer; relation IDs are not used by W', '#d9edf7',12)
arrow(ax,(3,5.5),(2.5,4.8));arrow(ax,(7,5.5),(7.5,4.8))
box(ax,.3,3.55,4.4,1.2,'Gate branch\n256 → 256 → 256 → 1\nsoftmax over this graph’s nodes')
box(ax,5.3,3.55,4.4,1.2,'Value branch\n256 → 256 → 256\nSELU after each linear map')
arrow(ax,(2.5,3.5),(4,2.95));arrow(ax,(7.5,3.5),(6,2.95))
box(ax,1,1.85,8,1.05,'Weighted sum: r = Σᵥ αᵥ value(H′ᵥ) ∈ ℝ²⁵⁶\nAll selected nodes contribute; not only the applicant', '#c7e9df',12)
arrow(ax,(5,1.8),(5,1.2))
box(ax,1,.15,8,1,'Linear 256 → 2 logits → class probabilities\nTrain: cross entropy + AdamW; select by validation AUROC')
save(fig,'architecture')
fig,axes=plt.subplots(1,2,figsize=(9,4));a=np.array([[1,1,1],[1,1,0],[1,0,1.]])
d=a.sum(1);norm=a/np.sqrt(d[:,None]*d[None,:]);x=np.array([1,2,4]);out=norm@x
axes[0].imshow(norm,cmap='Blues',vmin=0,vmax=1)
for i in range(3):
 for j in range(3):axes[0].text(j,i,f'{norm[i,j]:.3f}',ha='center',va='center')
axes[0].set_xticks(range(3),['root','leaf 1','leaf 2']);axes[0].set_yticks(range(3),['root','leaf 1','leaf 2']);axes[0].set_title('Normalized incoming weights')
axes[1].axis('off');axes[1].text(.02,.9,'One coordinate, W = 1, b = 0',fontweight='bold');axes[1].text(.02,.7,'Input: [1, 2, 4]\nDegrees including self: [3, 2, 2]',linespacing=1.8)
axes[1].text(.02,.38,f'Root = 1/3 + 2/√6 + 4/√6\n         = {out[0]:.3f}\n\nAll outputs: {np.round(out,3).tolist()}',linespacing=1.5)
fig.tight_layout();save(fig,'normalization')
print('Built extraction, architecture and numeric message trace')
