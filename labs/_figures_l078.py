"""Portable computation snapshots and architecture; generated from the same arithmetic."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;DEST=ROOT/'figures/l078'
def build():
    DEST.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,4));ax.axis('off')
    rows=[['A','2','B: 4','4','3'],['B','4','A: 2; C: 8','5','4.5'],['C','8','B: 4','4','6'],['D','10','none','0 (declared)','5']]
    table=ax.table(cellText=rows,colLabels=['Node','Old state','Incoming old values','Mean','½ self + ½ mean'],loc='center',cellLoc='center');table.auto_set_font_size(False);table.set_fontsize(12);table.scale(1,2)
    ax.set_title('One synchronous round · A—B—C, isolated D\nEvery message reads the old snapshot',pad=20)
    fig.tight_layout();fig.savefig(DEST/'trace.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));terms=np.array([2/np.sqrt(6),4/3,8/np.sqrt(6)])
    ax.bar(['A → B\n2 / √6','B → B\n4 / 3','C → B\n8 / √6'],terms,color=['#267d76','#aa7733','#267d76'])
    for i,t in enumerate(terms):ax.text(i,t+.05,f'{t:.6f}',ha='center')
    ax.set_ylim(0,4);ax.set_ylabel('Contribution to B');ax.set_title(f'GCN: sum = {terms.sum():.6f}; ordinary mean = {14/3:.6f}\nAugmented degrees [2,3,2,1]; scalar W=1')
    fig.tight_layout();fig.savefig(DEST/'normalization.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6));ax.axis('off')
    stages=[('INPUT · X [2708,1433]','Row-normalize word features; construct S from graph + self-loops'),('LAYER 1 · H [2708,16]','Dropout(X) → × W0 [1433,16] → S × result → ReLU'),('LAYER 2 · logits [2708,7]','Dropout(H) → × W1 [16,7] → S × result'),('PREDICTION / TRAINING','Softmax per node / cross-entropy on 140 training nodes + L2(W0)')]
    for i,(title,detail) in enumerate(stages):
        y=.9-i*.23
        ax.text(.5,y,title+'\n'+detail,ha='center',va='center',bbox=dict(boxstyle='round,pad=.7',fc='#edf6f5',ec='#277c76'),fontsize=12)
        if i<3:ax.annotate('',xy=(.5,y-.14),xytext=(.5,y-.07),arrowprops={'arrowstyle':'->','color':'#277c76'})
    ax.text(.5,.015,'Cora two-layer GCN · shared S at both layers · no biases · dropout only during training',ha='center',fontsize=11)
    fig.tight_layout();fig.savefig(DEST/'architecture.png',dpi=150);plt.close(fig)
    p=ROOT/'_paper_l078_results.json'
    if p.exists():
        r=json.loads(p.read_text());a=np.array([v['test_accuracy'] for v in r['runs']])*100
        fig,ax=plt.subplots(figsize=(9,4));ax.scatter(range(len(a)),a,s=18,label='Executed port: each seed');ax.axhline(81.5,color='#aa5533',ls='--',label='Paper Table 2: 81.5%');ax.axhline(a.mean(),color='#277c76',label=f'Port mean {a.mean():.3f}%');ax.set(xlabel='Initialization seed (same split)',ylabel='Test accuracy (%)',ylim=(75,86));ax.legend(fontsize=10);fig.tight_layout();fig.savefig(DEST/'results.png',dpi=150);plt.close(fig)
if __name__=='__main__':build()
