"""Portable computation trace and per-seed measured result panels."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent

def build():
    out=ROOT/'figures/l080';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,5));ax.axis('off')
    rows=[('TRAIN · 128 rows','Fit medians and scales\nFit gradient models / build ICL context','#15616d'),
          ('VALIDATION · 48 rows','A: .31     B: .33\nSelect A; freeze recipe and epoch','#8a5a00'),
          ('TEST · 48 rows','A: .40     B: .25 (illustrative oracle)\nReport .40; do not switch to B','#9b2226')]
    for i,(title,body,color) in enumerate(rows):
        x=.02+i*.335
        ax.text(x,.84,title,color=color,weight='bold',transform=ax.transAxes)
        ax.text(x,.70,body,va='top',fontsize=10,transform=ax.transAxes,bbox=dict(boxstyle='round,pad=.6',facecolor='#f5f5f0',edgecolor=color))
    ax.annotate('',xy=(.64,.44),xytext=(.04,.44),xycoords='axes fraction',arrowprops={'arrowstyle':'->','lw':2,'color':'#15616d'})
    ax.text(.03,.34,'Frozen train-fitted transform applies to validation and test.',transform=ax.transAxes)
    ax.text(.03,.20,'Within each panel: identical rows and features for all four arms.\nAcross regimes: different populations; no causal drift identification.',transform=ax.transAxes)
    ax.text(.03,.04,'Illustrative losses; row counts match the declared local exam.',fontsize=9,color='#555',transform=ax.transAxes)
    fig.tight_layout();fig.savefig(out/'protocol.png',dpi=155);plt.close(fig)
    r=json.loads((ROOT/'_verify_l080_results.json').read_text());fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
    for ax,(name,regime) in zip(axes.flat,[(d,s) for d in r['tasks'] for s in ['random','temporal']]):
        for i,arm in enumerate(r['arms']):
            y=[v['error'] for v in r['records'] if (v['dataset'],v['regime'],v['arm'])==(name,regime,arm)]
            ax.scatter(np.arange(len(y))*.07+i-.07,y,s=26,color=['#15616d','#bb3e03','#6c3f85','#005f73'][i])
            ax.plot([i-.17,i+.17],[np.mean(y)]*2,color='black',lw=2)
        ax.set_title(name+' · '+regime);ax.set_ylabel('Test log loss (nats; lower is better)');ax.set_xticks(range(4),['XGB','FT-T','TabM','PFN v2']);ax.tick_params(labelbottom=True);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fresh local comparison · points = downstream seeds; bars = means',fontsize=14)
    fig.text(.5,.01,'128 / 48 / 48 rows; numeric + binary inputs; two candidates. No published benchmark reproduction.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.04,1,.96]);fig.savefig(out/'results.png',dpi=150);plt.close(fig)
if __name__=='__main__':build()
