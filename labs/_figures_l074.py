"""Portable computation figures; arithmetic and scores generated from their inputs."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,Circle
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures/l074'
TEAL='#087e83';GOLD='#b77522';INK='#1d3442'

def canvas(title,w=13,h=6):
    fig,ax=plt.subplots(figsize=(w,h));fig.patch.set_facecolor('#fafbf8');ax.set_facecolor('#fafbf8')
    ax.set(xlim=(0,13),ylim=(0,h));ax.axis('off');ax.set_title(title,loc='left',fontsize=19,color=INK,pad=16)
    return fig,ax

def text(ax,x,y,s,size=13,**kw):return ax.text(x,y,s,fontsize=size,color=INK,va='center',**kw)
def arrow(ax,a,b,color=TEAL):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,lw=2,color=color))
def save(fig,name):fig.tight_layout();fig.savefig(OUT/(name+'.png'),dpi=155,bbox_inches='tight');plt.close(fig)

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    fig,ax=canvas('One row → two observed leaves → a contextual center',h=6)
    text(ax,.2,5.4,'Row: colour = red     volume = 2     region = missing',15)
    for x,y,label in [(2,3.6,'red\n[2, −1]'),(2,1.3,'2 × volume\n[6, 2]'),(6,2.45,'row center\n[10, 0]')]:
        ax.add_patch(Circle((x,y),.72,facecolor='#d7ece7',edgecolor=TEAL,lw=2));text(ax,x,y,label,12,ha='center')
    arrow(ax,(2.7,3.5),(5.25,2.8));text(ax,3.6,4,'colour [1, 2]',12)
    arrow(ax,(2.7,1.45),(5.25,2.1));text(ax,3.6,.8,'volume [3, 1]',12)
    text(ax,7.3,3.9,'Elementwise products',15)
    text(ax,7.3,3.25,'[2, −1] ⊙ [1, 2] = [2, −2]')
    text(ax,7.3,2.6,'[6, 2] ⊙ [3, 1] = [18, 2]')
    text(ax,7.3,1.85,'Mean = ([2, −2] + [18, 2]) / 2')
    text(ax,7.3,1.15,'Released center = [10, 0]',15)
    text(ax,.2,.2,'Illustrative d=2. Missing region has no leaf. Paper plain leaf mean: [4, 0.5].',12)
    save(fig,'graph')
    fig,ax=canvas('CARTE: what is learned, what is transferred, what predicts',h=8)
    ax.axhspan(4.7,7.8,color='#edf2e2');ax.axhspan(.4,4.45,color='#e8f3f1')
    text(ax,.2,7.5,'BACKGROUND PRETRAINING — paper overview',15)
    # A tiny neighborhood plus its truncated view instead of a generic box chain.
    for offset in [0,2.4]:
        center=(1+offset,6.05)
        for dx,dy in [(-.6,.65),(.65,.6),(.5,-.6)][:3 if offset==0 else 2]:
            arrow(ax,(center[0]+dx,center[1]+dy),center);ax.plot(center[0]+dx,center[1]+dy,'o',color=GOLD,ms=8)
        ax.plot(*center,'o',color=TEAL,ms=14)
    text(ax,.2,5,'YAGO neighborhood     edge-deleted view',11)
    arrow(ax,(4.1,6.1),(5,6.1));text(ax,5.2,6.3,'Shared 300-wide encoder\n12 attention layers / 12 heads',13)
    arrow(ax,(8.5,6.1),(9.25,6.1));text(ax,9.4,6.2,'Projection +\ncontrastive objective',13)
    arrow(ax,(6.7,5.3),(6.7,4.3));text(ax,7,4.65,'selected checkpoint weights',11)
    text(ax,.2,4.1,'TARGET ROW — exact one-readout lab variant',15)
    text(ax,.25,3.2,'Text value → FastText(value)\nNumber → scalar × FastText(column)\nEdge → FastText(column)',12)
    arrow(ax,(3.7,3.1),(4.4,3.1))
    text(ax,4.5,3.3,'Initial node / edge maps\nlinear → GELU → LayerNorm\nwidth 300',12)
    arrow(ax,(7.7,3.1),(8.35,3.1));text(ax,8.5,3.35,'12 heads × 25 coordinates\nz = sender ⊙ edge\nq·k / √25 → neighbor softmax',12)
    arrow(ax,(10,2.55),(10,1.95));text(ax,8.5,1.6,'weighted values → concat(300)\nLayerNorm → FFN → LayerNorm',12)
    arrow(ax,(8.2,1.6),(7.3,1.6));text(ax,4.6,1.6,'select center: row vector (300)\nfrozen probe OR trainable encoder',12)
    arrow(ax,(4.3,1.6),(3.6,1.6));text(ax,.3,1.6,'ridge OR linear head\n→ scalar transformed price',12)
    text(ax,.2,.15,'Paper pretraining is not rerun. Local model omits ordinary blocks, uses zero dropout and strictly loads its selected weights.',10)
    save(fig,'architecture')
    fig,ax=canvas('One attention head: normalize within each receiver',h=4.8)
    w=np.exp(np.array([1,0])/np.sqrt(2));w/=w.sum();o=w*np.array([2,4])
    rows=[['Sender','key','value','scaled logit','weight','contribution'],['A','[1, 0]','[2, 0]','0.7071',f'{w[0]:.4f}',f'[{o[0]:.4f}, 0]'],['B','[0, 1]','[0, 4]','0',f'{w[1]:.4f}',f'[0, {o[1]:.4f}]']]
    table=ax.table(cellText=rows,cellLoc='center',bbox=[0,.32,1,.48]);table.auto_set_font_size(False);table.set_fontsize(13)
    for (r,c),cell in table.get_celld().items():cell.set_facecolor('#d7ece7' if r==0 else '#fafbf8');cell.set_edgecolor('#b8cfcd')
    text(ax,.1,4.4,'Receiver query = [1, 0]; head width = 2; divide dot products by √2.',14)
    text(ax,.2,.9,f'Output = [{o[0]:.4f}, {o[1]:.4f}]     Weights sum to {w.sum():.1f}.',16)
    text(ax,.2,.2,'A different receiver with one neighbor gives that neighbor weight 1. Never normalize across unrelated rows.',11)
    save(fig,'attention')
    r=json.loads((ROOT/'_verify_l074_results.json').read_text());fig,axes=plt.subplots(1,3,figsize=(14,5.5),sharey=False)
    labels=['Text center','Random probe','Pretrain probe','Scratch','Pretrain tune','CatBoost'];order=['fasttext_ridge','random_probe','pretrained_probe','scratch','pretrained_ft','catboost']
    for ax,d in zip(axes,r['config']['datasets']):
        for i,arm in enumerate(order):
            vals=[z['r2'] for z in r['records'] if z['dataset']==d and z['arm']==arm]
            ax.scatter(np.arange(len(vals))*.08+i-.08,vals,color=TEAL,s=23)
            ax.errorbar(i,np.mean(vals),yerr=np.std(vals,ddof=1),fmt='_',color=INK,capsize=4,ms=17)
        if d=='wine_pl':
            ax.set_yscale('symlog',linthresh=.5)
            ax.set_yticks([-40,-10,-1,0,.5]);ax.set_yticklabels(['−40','−10','−1','0','0.5'])
        ax.set_title(d.replace('wine_','').replace('_',' ')+(' (symmetric-log y)' if d=='wine_pl' else ''));ax.set_xticks(range(6),labels,rotation=50,ha='right');ax.axhline(0,color='#aaa',lw=.7);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Held-out R² (higher is better)');fig.suptitle('Author measurements: dots = paired split/model repetitions; bars = mean ± SD',fontsize=14)
    save(fig,'results')
    fig,axes=plt.subplots(1,2,figsize=(13,5.2),gridspec_kw={'width_ratios':[1,1.4]})
    for i,d in enumerate(r['config']['datasets']):
        values={a:{z['seed']:z['r2'] for z in r['records'] if z['dataset']==d and z['arm']==a} for a in ['pretrained_ft','scratch']}
        gains=np.array([values['pretrained_ft'][seed]-values['scratch'][seed] for seed in r['config']['seeds']])
        half=4.3026527299*gains.std(ddof=1)/np.sqrt(len(gains))
        axes[0].scatter(gains,[i]*len(gains),s=24,color=TEAL)
        axes[0].errorbar(gains.mean(),i,xerr=half,fmt='|',color=INK,capsize=5)
    axes[0].set_yticks(range(3),['Poland','Wine.com','Vivino']);axes[0].axvline(0,color='gray',lw=1)
    axes[0].set_xlabel('Pretrained FT − scratch R²');axes[0].set_title('Paired gains + descriptive t95 intervals')
    ordered=sorted(r['ranks']['mean'].items(),key=lambda kv:kv[1]);cd=r['ranks']['nemenyi_cd']
    for i,(name,rank) in enumerate(ordered):
        axes[1].plot([1,rank],[i,i],color='#b8cfcd');axes[1].plot(rank,i,'o',color=TEAL)
        axes[1].text(rank+.08,i,name,va='center',fontsize=10)
    axes[1].plot([1,1+cd],[6,6],color=GOLD,lw=3);axes[1].text(1,6.45,f'Nemenyi CD = {cd:.2f} (6 methods; 3 datasets)',fontsize=11)
    axes[1].set(xlim=(.8,max(7.5,1+cd+.2)),ylim=(-.5,7),xlabel='Mean dataset rank (lower is better)',title='Exploratory critical-difference view')
    axes[1].set_yticks([]);axes[1].grid(axis='x',alpha=.2)
    fig.suptitle('Uncertainty is conditional on three related wine tables and three repetitions',fontsize=13)
    save(fig,'audit')
if __name__=='__main__':build()
