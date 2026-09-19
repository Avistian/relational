"""Portable computed diagrams, generated from the lesson's declared values."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).parent;OUT=ROOT/'figures/l071';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':12,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
BLUE='#175778';ORANGE='#a65719';GREEN='#246d52'
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=150,bbox_inches='tight',facecolor='white');plt.close(fig)

def figures():
    fig,ax=plt.subplots(figsize=(7,4));ax.axis('off')
    x=np.array([.2,.8,.8]);donor=np.array([.9,.8,.1]);m=np.array([1,1,0]);xt=x*(1-m)+donor*m
    rows=[x,donor,m,xt,(xt!=x).astype(int)]
    table=ax.table(cellText=[[f'{v:g}' for v in r] for r in rows],rowLabels=['Original x','Donor values','Selected mask','Corrupted row','Changed target'],colLabels=['Column 1','Column 2','Column 3'],loc='center',cellLoc='center',colWidths=[.19]*3)
    table.auto_set_font_size(False);table.set_fontsize(13);table.scale(1,1.9)
    for key,cell in table.get_celld().items():
        if key[0] in (4,5):cell.set_facecolor('#e7f2f5')
    ax.set_title('Column 2 is selected but unchanged',pad=18,loc='left',weight='bold')
    ax.text(0,-.06,'The encoder receives the corrupted row. Targets: original x and changed mask.',transform=ax.transAxes,fontsize=10)
    save(fig,'corruption')
    fig,ax=plt.subplots(figsize=(7,8));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def box(x,y,w,h,s,color=BLUE):
        ax.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=11,color=color,bbox=dict(boxstyle='round,pad=.65',facecolor='#f4f8fa',edgecolor=color))
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=BLUE,lw=1.7))
    box(.1,.87,.8,.07,'Numeric input X [B,d] scaled to [0,1]\nLocal digits: d=64; public MNIST: d=784')
    box(.1,.71,.8,.07,'Columnwise corruption → X̃ [B,d]\nTargets: changed mask and original X')
    arrow((.5,.85),(.5,.80))
    box(.1,.55,.8,.07,'Shared encoder: ReLU(X̃W + b)\nW [d,d] → Z [B,d]')
    arrow((.5,.69),(.5,.64))
    box(.0,.36,.45,.10,'Mask logits [B,d]\nBCE with changed mask')
    box(.55,.36,.45,.10,'Sigmoid values [B,d]\nMSE with all of X')
    arrow((.4,.54),(.23,.48));arrow((.6,.54),(.77,.48))
    ax.text(.5,.30,'Pretraining: update encoder + both heads',ha='center',fontsize=11,color=ORANGE)
    ax.axhline(.25,color='#bbbbbb',lw=1)
    box(.08,.1,.84,.08,'Downstream: clean X → saved encoder → predictor\nd → 100 ReLU → 100 ReLU → C logits → softmax',GREEN)
    ax.text(.5,.035,'Frozen encoder in release; fine-tuning is a labeled extension.',ha='center',fontsize=10)
    ax.set_title('Two pretext heads; one reusable encoder',loc='left',weight='bold')
    save(fig,'architecture')
    fig,axs=plt.subplots(2,1,figsize=(7,5.4),layout='constrained')
    q=np.array([.7,.2,.1]);t=np.array([1,0,0]);terms=-(t*np.log(q)+(1-t)*np.log(1-q))
    squared=(np.array([.3,.7,.6])-x)**2
    for ax,vals,title,color in [(axs[0],terms,'Mask BCE: target [1,0,0]; prediction [.7,.2,.1]',BLUE),(axs[1],squared,'Reconstruction: original [.2,.8,.8]; prediction [.3,.7,.6]',GREEN)]:
        ax.bar(['Column 1','Column 2','Column 3'],vals,color=color)
        ax.set_title(title,fontsize=11,loc='left');ax.set_ylim(0,max(vals)*1.4);ax.set_ylabel('Loss contribution')
        for i,v in enumerate(vals):ax.text(i,v+max(vals)*.05,f'{v:.4f}',ha='center')
    fig.suptitle(f'BCE mean {terms.mean():.4f} + 2 × MSE mean {squared.mean():.4f} = {terms.mean()+2*squared.mean():.4f}',fontsize=12)
    save(fig,'loss')
    fig,ax=plt.subplots(figsize=(7,4.6));ax.axis('off')
    lines=[('1,797 digit rows','Split before any fitting'),('450 test rows','Only final predictions and scores'),('809 unlabeled rows','Pretraining + optional consistency; no task labels'),('538 label-reserve rows','Nested budgets: 50 → 150 → 500'),('40/10 · 120/30 · 400/100','Training / validation labels; both count')]
    for i,(a,b) in enumerate(lines):
        y=.9-i*.19;ax.text(.02,y,a,weight='bold',fontsize=12);ax.text(.02,y-.06,b,fontsize=11,color=BLUE)
    ax.set_title('Local information contract per seed',loc='left',pad=15,weight='bold');save(fig,'split')
    result_path=ROOT/'_verify_l071_results.json'
    if result_path.exists():
        result=json.loads(result_path.read_text());fig,axs=plt.subplots(2,1,figsize=(7,8),layout='constrained')
        arms=['scratch','vime_finetune','vime_frozen','recon_finetune','vime_semi']
        colors=['#333333',BLUE,ORANGE,GREEN,'#7b4b94'];xs=np.array([50,150,500])
        for arm,col in zip(arms,colors):
            rows=[next(r for r in result['summary'] if r['arm']==arm and r['budget']==int(b)) for b in xs]
            axs[0].errorbar(xs,[100*r['accuracy'] for r in rows],yerr=[100*r['sd'] for r in rows],marker='o',capsize=3,label=arm,color=col)
            for seed in result['config']['seeds']:
                rr=[next(r for r in result['records'] if r['arm']==arm and r['budget']==int(b) and r['seed']==seed) for b in xs]
                axs[0].scatter(xs,[100*r['accuracy'] for r in rr],s=15,alpha=.45,color=col)
        axs[0].set(xscale='log',xticks=xs,xticklabels=xs,xlabel='Total labeled rows (including validation)',ylabel='Test accuracy (%)',ylim=(0,100))
        axs[0].legend(fontsize=9,ncol=2);axs[0].set_title('Measured digits results: dots = seeds; bars = sample SD',loc='left',fontsize=11)
        for seed in result['config']['seeds']:
            gaps=[]
            for b in xs:
                lookup={r['arm']:r['accuracy'] for r in result['records'] if r['seed']==seed and r['budget']==b}
                gaps.append(100*(lookup['vime_finetune']-lookup['scratch']))
            axs[1].plot(xs,gaps,marker='o',label=f'Seed {seed}')
        axs[1].axhline(0,color='black',lw=1)
        axs[1].set(xscale='log',xticks=xs,xticklabels=xs,xlabel='Total labeled rows',ylabel='Paired gain (percentage points)')
        axs[1].legend(fontsize=10);axs[1].set_title('Fine-tuned VIME minus scratch on the same test rows',loc='left',fontsize=11)
        for ax in axs: ax.minorticks_off()
        save(fig,'results')
if __name__=='__main__':figures()
