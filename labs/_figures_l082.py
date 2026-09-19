"""Computation-specific GCN figures and seed-zero hidden-state diagnostic."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch
from relkit import gcn_l082 as gcn
LAB=Path(__file__).resolve().parent;OUT=LAB/'figures/l082'
BLUE='#245b78';INK='#173449';TEAL='#167d86';ORANGE='#ae542c'

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=160,bbox_inches='tight',facecolor='#faf9f5');plt.close(fig)

def box(ax,x,y,w,h,text,color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',facecolor=color,edgecolor='none'))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',color='white',fontsize=11,linespacing=1.6)

def arrow(ax,x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':TEAL,'lw':2})

def build():
    OUT.mkdir(parents=True,exist_ok=True);plt.rcParams.update({'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK})
    fig,ax=plt.subplots(figsize=(12,6));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.text(.02,.95,'GCN / one graph, two learned channel maps',fontsize=20,weight='bold')
    box(ax,.02,.66,.27,.18,'Cora: 2,708 papers\nX: 2,708 × 1,433\nrow-normalized features')
    box(ax,.02,.36,.27,.18,'Undirected binary A\nT = A + I; d = T1\nS = diag(d⁻½) T diag(d⁻½)',TEAL)
    box(ax,.37,.66,.25,.18,'Sparse input dropout .5\nX W₀ → S(X W₀)\nReLU: 2,708 × 16')
    box(ax,.70,.66,.27,.18,'Hidden dropout .5\nH W₁ → S(H W₁)\nlogits: 2,708 × 7')
    arrow(ax,.29,.75,.37,.75);arrow(ax,.62,.75,.70,.75)
    arrow(ax,.29,.46,.49,.66);arrow(ax,.29,.46,.83,.66)
    box(ax,.37,.32,.25,.22,'140 train labels\nmean CE + ½λ‖W₀‖²\nAdam updates W₀, W₁',ORANGE)
    box(ax,.70,.32,.27,.22,'500 validation labels\nprevious-ten mean stop\nretain LAST weights',TEAL)
    arrow(ax,.83,.66,.83,.54);arrow(ax,.75,.66,.55,.54)
    box(ax,.37,.04,.60,.16,'Evaluation: dropout OFF → same S and weights → argmax\n1,000 test labels used only for final accuracy',INK)
    arrow(ax,.83,.32,.83,.20)
    ax.text(.02,.27,'W₀: 1,433 × 16\nW₁: 16 × 7\nno bias · no graph pooling\nλ = 0.0005',fontsize=11,linespacing=1.7,va='top')
    save(fig,'architecture')
    fig,axs=plt.subplots(1,2,figsize=(11,4.3));a=axs[0];a.axis('off');a.set(xlim=(-.5,3.5),ylim=(-1,1.4))
    a.plot([0,1,2],[0,0,0],color=TEAL,lw=3)
    for i,(name,value,degree) in enumerate(zip('ABCD',[2,4,8,10],[2,3,2,1])):
        a.scatter(i,0,s=1000,color=BLUE if i!=1 else ORANGE,zorder=3);a.text(i,0,name,color='white',ha='center',va='center',fontsize=15)
        a.text(i,.55,f'x={value}',ha='center');a.text(i,-.55,f'd̃={degree}',ha='center');a.text(i,-.85,'+ self-loop',ha='center',fontsize=9)
    a.set_title('Add loops first; D remains a node',loc='left',weight='bold')
    a=axs[1];parts=[2/np.sqrt(6),4/3,8/np.sqrt(6)];a.bar(['A → B','B → B','C → B'],parts,color=[BLUE,ORANGE,TEAL]);a.set_ylabel('Contribution to B (W=1)');a.set_title('Sum = 5.415816; ordinary mean = 4.666667',fontsize=11)
    for i,v in enumerate(parts):a.text(i,v+.06,f'{v:.6f}',ha='center')
    a.set_ylim(0,4);fig.tight_layout();save(fig,'trace')
    r=json.loads((LAB/'_paper_l082_results.json').read_text());scores=np.array([x['test_accuracy']*100 for x in r['runs']])
    fig,ax=plt.subplots(figsize=(10,3.8));ax.scatter(np.arange(len(scores)),scores,s=15,color=TEAL);ax.axhline(81.5,color=ORANGE,label='Paper target 81.5%');ax.axhline(scores.mean(),color=BLUE,ls='--',label=f'Port mean {scores.mean():.3f}%');ax.set(xlabel='Initialization seed (one fixed split)',ylabel='Test accuracy (%)',title='Complete Cora experiment • no selected seeds');ax.legend();save(fig,'results')
    torch.set_num_threads(1);data=gcn.load_cora(LAB);instances=[];original=gcn.GCN
    class CapturedGCN(original):
        def __init__(self,*args):super().__init__(*args);instances.append(self)
    try:
        gcn.GCN=CapturedGCN;result=gcn.train_cora(data,0)
    finally:gcn.GCN=original
    model=instances[0];x,s,y,tr,va,te=data
    with torch.no_grad():h=torch.relu(gcn.propagate(s,x,model.w0)).numpy()
    centered=h-h.mean(0);u,sv,vt=np.linalg.svd(centered,full_matrices=False);z=centered@vt[:2].T
    fig,ax=plt.subplots(figsize=(9,5));points=ax.scatter(z[:,0],z[:,1],c=y.numpy(),cmap='tab10',s=7,alpha=.7);ax.set(xlabel='Hidden-state PC1',ylabel='Hidden-state PC2',title='All Cora nodes • seed 0 • colors shown after training');fig.colorbar(points,ax=ax,label='Class index');save(fig,'embedding')
    np.savez_compressed(OUT/'hidden-seed0.npz',hidden=h,pca=z,labels=y.numpy())
    (OUT/'embedding-evidence.json').write_text(json.dumps({'seed':0,'epochs':result['epochs'],'test_accuracy':result['test_accuracy'],'projection':'PCA via centered SVD; all nodes; visualization only','pca_variance_fraction':(sv[:2]**2/(sv**2).sum()).tolist()},indent=2)+'\n')
if __name__=='__main__':build()
