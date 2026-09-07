"""Portable computation figures and measured evidence; all values generated."""
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/relational-matplotlib')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from relkit.realmlp import coslog4,smooth_clip
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures/l053';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fffdf8','axes.facecolor':'#fffdf8','savefig.facecolor':'#fffdf8'})
BLUE='#225b78';ORANGE='#b65c29';GREEN='#377355'
def save(fig,name):fig.savefig(OUT/f'{name}.png',dpi=160,bbox_inches='tight');plt.close(fig)

def figures():
    from matplotlib.patches import FancyBboxPatch
    fig,ax=plt.subplots(figsize=(8,9));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def box(x,y,w,h,color='#e7eff0'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008',facecolor=color,edgecolor='#b8c7cd'))
    def txt(x,y,t,size=11,color='#222',weight='normal',ha='left'):
        ax.text(x,y,t,fontsize=size,color=color,weight=weight,ha=ha,va='center')
    def arrow(x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops=dict(arrowstyle='->',color=BLUE,lw=1.6))
    txt(.02,.975,'Numeric RealMLP-TD-S',18,weight='bold')
    txt(.02,.94,'Trace one row; B rows share the same fitted operations.',10)
    box(.02,.815,.96,.095)
    txt(.04,.89,'INPUT + TRAIN-FITTED PREPROCESSING  [B, p]',11,weight='bold')
    txt(.04,.85,'x = 4, median = 0, IQR = 2  →  z = 2  →  u = 1.664',11,BLUE)
    arrow(.5,.808,.5,.78)
    box(.02,.652,.96,.12)
    txt(.04,.745,'LEARNED SCALES: h₀ = u ⊙ s  [B, p]',11,weight='bold')
    txt(.04,.705,'Illustrative vector: u = [1, 2, −1, 0], s = [2, 1, 1, 1]',10)
    txt(.04,.674,'h₀ = [2, 2, −1, 0]    ·    s starts at 1 in the actual model',11,BLUE)
    arrow(.5,.645,.5,.612)
    txt(.04,.594,'THREE HIDDEN STAGES',11,weight='bold')
    for i,shape in enumerate(['[B, p] → [B, d]','[B, d] → [B, d]','[B, d] → [B, d]']):
        x=.03+i*.33;box(x,.493,.28,.075)
        txt(x+.14,.548,'NTP → activation',10,ha='center')
        txt(x+.14,.516,shape,9,ha='center')
        if i<2:arrow(x+.288,.53,x+.324,.53)
    txt(.04,.464,'SELU for classification; Mish for regression',10)
    box(.02,.274,.96,.162,'#f3ece0')
    txt(.04,.413,'INSIDE ONE NTP OUTPUT COORDINATE',11,weight='bold')
    headers=['hᵢ','2','2','−1','0'];weights=['Wᵢⱼ','1','−1','2','1'];products=['hᵢWᵢⱼ','2','−2','−2','0']
    for y,row in zip([.377,.348,.319],[headers,weights,products]):
        for x,text in zip([.05,.34,.50,.66,.82],row):txt(x,y,text,11,BLUE if row is products else '#222')
    txt(.04,.245,'Sum = −2   →   divide by √4: −1   →   add bⱼ = 0.5: −0.5',11,BLUE)
    arrow(.5,.225,.5,.197)
    box(.02,.083,.96,.105)
    txt(.04,.163,'ZERO-INITIALIZED NTP HEAD',11,weight='bold')
    txt(.04,.129,'[B, d] → [B, 2] logits → class probabilities',10)
    txt(.04,.099,'             or [B, 1] target → unscale to original units',10)
    txt(.02,.048,'Published d = 256; learning d = 64. No dropout in TD-S.',10)
    txt(.02,.017,'Supervised training only; inference uses the row, no memory.',10)
    fig.tight_layout();save(fig,'architecture')

    fig,ax=plt.subplots(figsize=(8,4.3));z=np.linspace(-12,12,500)
    ax.plot(z,z,color='#999',ls=':',label='No clipping');ax.plot(z,np.clip(z,-3,3),color=ORANGE,ls='--',label='Hard clip');ax.plot(z,smooth_clip(z),color=BLUE,lw=2.5,label='Smooth clip')
    v=float(smooth_clip(2.));ax.scatter([2],[v],color=BLUE,zorder=5);ax.annotate(f'z = 2 → {v:.3f}',(2,v),xytext=(4,0),arrowprops={'arrowstyle':'->'})
    ax.set(xlim=(-12,12),ylim=(-4,4),xlabel='Robust-scaled coordinate z',ylabel='Network input u',title='Extreme values compress; their order remains')
    ax.legend(ncol=3,fontsize=9,loc='lower right');fig.tight_layout();save(fig,'preprocessing')

    fig,(ax,bx)=plt.subplots(2,1,figsize=(8,6),gridspec_kw={'height_ratios':[1.6,1]})
    ts=np.linspace(0,1,700);ys=np.array([coslog4(float(t)) for t in ts])
    ax.plot(ts,ys,color=BLUE,lw=2);valleys=[(2**j-1)/15 for j in range(5)]
    ax.scatter(valleys,np.zeros(5),color=ORANGE);ax.set(xlabel='Fraction of optimizer steps completed',ylabel='Schedule multiplier',title='Four cycles, with increasingly long intervals')
    for j,t in enumerate(valleys):ax.annotate(f'{t:.3f}',(t,0),xytext=(0,10 if j!=1 else 30),textcoords='offset points',ha='center',fontsize=9)
    t=.25;lr=.04*coslog4(t);names=['Feature scale ×6','Weight ×1','Bias ×0.1'];rates=[6*lr,lr,.1*lr]
    bx.barh(names,rates,color=[GREEN,BLUE,ORANGE]);bx.invert_yaxis()
    for i,v in enumerate(rates):bx.text(v+.002,i,f'{v:.5f}',va='center')
    bx.set(xlim=(0,.15),xlabel='Actual learning rate at t = 0.25 (classification base = 0.04)')
    fig.tight_layout();save(fig,'schedule')

    fig,ax=plt.subplots(figsize=(8,4));ax.axis('off')
    rows=[['META-TRAIN DATASETS','Train rows fit weights','Validation rows choose recipe','Freeze the recipe across datasets'],
          ['NEW META-TEST DATASET','Train rows fit new weights','Validation rows choose epoch','Test rows score the fixed procedure']]
    table=ax.table(cellText=rows,colLabels=['Dataset level','Fit','Select','Report'],loc='center',cellLoc='left',colWidths=[.26,.23,.24,.27])
    table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,3.4)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#d4d1c8')
        if r==0:cell.set_facecolor('#e7eff0')
        text=cell.get_text().get_text()
        if r>0:
            import textwrap
            cell.get_text().set_text('\n'.join(textwrap.wrap(text,21)))
    ax.set_title('A new row split cannot make a reused dataset meta-test',pad=20,weight='bold')
    fig.tight_layout();save(fig,'boundaries')

    result=json.loads((ROOT/'_verify_l053_results.json').read_text());arms=result['ranks']['arms']
    fig,axes=plt.subplots(1,3,figsize=(11,4.6))
    for ax,(name,data) in zip(axes,result['results'].items()):
        for i,(arm,color) in enumerate(zip(arms,[BLUE,ORANGE,GREEN])):
            stat=data['summary'][arm];v=[r['error'] for r in data['runs'][arm]]
            ci=stat['ci95'];ax.errorbar(i,stat['mean'],yerr=[[stat['mean']-ci[0]],[ci[1]-stat['mean']]],fmt='o',color=color,capsize=4)
            ax.scatter(i+np.array([-.09,0,.09]),v,color=color,s=18,alpha=.7)
        ax.set_title(name);ax.set_xticks(range(3),['TD-S\nlocal','XGB\nfixed','XGB\ntuned']);ax.set_ylabel(data['metric']+' ↓');ax.grid(axis='y',alpha=.15)
    fig.suptitle('Measured local errors · dots = seeds; bars = conditional 95% t intervals',fontsize=13)
    fig.tight_layout();save(fig,'results')

    fig,ax=plt.subplots(figsize=(8,3.8));ranks=result['ranks'];cd=ranks['nemenyi_cd']
    for i,a in enumerate(arms):
        r=ranks['means'][a];ax.scatter(r,i,color=[BLUE,ORANGE,GREEN][i]);ax.text(r+.035,i,f'{a}: {r:.3f}',va='center')
    ax.plot([1,1+cd],[3,3],color='#333');ax.text(1+cd/2,3.16,f'CD = {cd:.3f}',ha='center')
    ax.set(xlim=(.9,3.9),ylim=(-.5,3.65),xlabel='Mean rank across 3 datasets (lower is better)',yticks=[],title=f"Friedman p = {ranks['friedman_p']:.5f} · exploratory, low power")
    ax.set_xticks([1,2,3]);fig.tight_layout();save(fig,'ranks')

    closer=ROOT/'_paper_repro_l053_closer_summary.json'
    if closer.exists():
        data=json.loads(closer.read_text());s=data['summary'];fig,ax=plt.subplots(figsize=(7,3.5))
        ax.scatter([0,1,2],[r['error'] for r in data['runs']],color=BLUE,s=55)
        ax.axhline(s['mean'],color=ORANGE,label=f"mean {s['mean']:.4f}");ax.set(xticks=[0,1,2],xlabel='Model seed',ylabel='California RMSE ↓',title='Larger TD-S run · 6000 rows, width 256, 256 epochs')
        ax.legend();fig.tight_layout();save(fig,'closer')

if __name__=='__main__':figures()
