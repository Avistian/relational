"""Portable computation snapshots from exact audited results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
LAB=Path(__file__).resolve().parent

def build():
    out=LAB/'figures/l079';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(9,6));ax.axis('off')
    steps=[('1  INFORMATION','At arrival: transaction + past merchant history','Future labels excluded'),('2  EVALUATION','Earlier → later periods; honor label delay','Unseen merchants reported'),('3  FEASIBILITY','Measure full-path p95 ≤ 5 ms with margin','Context + preprocessing included'),('4  SELECTION','Best validation loss among feasible candidates','Freeze before test')]
    for i,(title,detail,boundary) in enumerate(steps):
        y=.9-i*.235
        ax.text(.02,y,title,color='#1a5276',weight='bold',transform=ax.transAxes)
        ax.text(.02,y-.065,detail,transform=ax.transAxes)
        ax.text(.02,y-.12,boundary,color='#666',fontsize=10,transform=ax.transAxes)
        if i<3:ax.annotate('',xy=(.85,y-.18),xytext=(.85,y-.04),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#1a5276','lw':2})
    fig.tight_layout();fig.savefig(out/'decision.png',dpi=170);plt.close(fig)
    r=json.loads((LAB/'_verify_l079_results.json').read_text());arms=list(r['matched']['random']['mean_ranks']);y=np.arange(len(arms))
    fig,ax=plt.subplots(figsize=(9,5))
    for key,offset,color,label in [('random',-.1,'#1a5276','Matched random · 3 datasets'),('temporal',.1,'#a04b26','Temporal · same 3 names')]:
        values=[r['matched'][key]['mean_ranks'][a] for a in arms]
        ax.scatter(values,y+offset,label=label,color=color,s=75)
        for x,yy in zip(values,y+offset):ax.text(x+.09,yy,f'{x:.2f}',va='center',fontsize=10,color=color)
    ax.set_yticks(y,arms);ax.invert_yaxis();ax.set_xlim(.7,5.3);ax.set_xticks(range(1,6));ax.set_xlabel('Mean within-dataset rank · lower is better');ax.legend(loc='lower right',fontsize=10)
    ax.set_title('Population matched; rows and time periods still differ',fontsize=13);fig.tight_layout();fig.savefig(out/'ranks.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5));names=['Trees','TabM','ICL'];lat=[2,8,35]
    ax.barh(names,lat,color=['#1a5276','#b9c3cc','#b9c3cc']);ax.invert_yaxis();ax.axvline(5,color='#a04b26',ls='--',label='Budget = 5 ms')
    for i,(x,loss) in enumerate(zip(lat,[.32,.29,.27])):ax.text(x+.6,i,f'{x} ms; loss {loss:.2f}',va='center',fontsize=11)
    ax.set_xlim(0,49);ax.set_xlabel('p95 latency (ms) · synthetic fixture');ax.legend();ax.set_title('Only Trees is eligible at 5 ms; lower loss alone is insufficient',fontsize=12)
    fig.tight_layout();fig.savefig(out/'budget.png',dpi=170);plt.close(fig)

if __name__=='__main__':build()
