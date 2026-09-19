import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
COLORS={'raw':'#475569','random':'#a16207','scarf_frozen':'#9333ea','scratch':'#2563eb','scarf_ft':'#047857','tree':'#be123c'}
def build():
    out=ROOT/'figures/l073';out.mkdir(parents=True,exist_ok=True)
    r=json.loads((ROOT/'_verify_l073_results.json').read_text())
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fffdf8','axes.facecolor':'#fffdf8'})
    def save(fig,name):fig.savefig(out/f'{name}.png',dpi=155,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4));ax.axis('off')
    lines=[('Training features · 700 rows','fit scaler + donor pool + SCARF pretraining'),('Training labels · 70 rows','fit head; update encoder only in fine-tuning'),('Validation labels · 100 rows','select C for probes; no gradient updates'),('Test · 200 unseen rows','transform → predict → score after choices freeze')]
    for i,(a,b) in enumerate(lines):
        y=3-i;ax.text(.02,y,a,weight='bold',va='center');ax.annotate(b,xy=(.43,y),xytext=(.51,y),va='center',arrowprops={'arrowstyle':'<-','color':'#047857'})
    ax.set(xlim=(0,1.2),ylim=(-.6,3.6));ax.set_title('Access ledger · illustrative 1,000-row experiment',loc='left',weight='bold',pad=20)
    save(fig,'boundary')
    fig,ax=plt.subplots(figsize=(10,3.8));order=[8,2,7,9,5,4,1,0,3,6]
    for row,n in enumerate([2,5,10]):
        for j,v in enumerate(order):
            ax.add_patch(plt.Rectangle((j,row),.9,.75,color='#047857' if j<n else '#e2e8f0'))
            ax.text(j+.45,row+.37,str(v),ha='center',va='center',color='white' if j<n else '#334155')
        ax.text(10.1,row+.37,f'{n}/10 labels',va='center')
    ax.set(xlim=(-.1,12),ylim=(-.3,3.2),yticks=[],xticks=[]);ax.invert_yaxis();ax.set_title('One label-blind row order → nested prefixes\nNumbers are row IDs; green cells have available labels.',loc='left',pad=12)
    save(fig,'nesting')
    fractions=r['config']['fractions'];datasets=r['config']['datasets']
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),sharey=True)
    for ax,d in zip(axes,datasets):
        for arm,color in COLORS.items():
            ss=[next(s for s in r['summary'] if s['dataset']==d and s['fraction']==f and s['arm']==arm) for f in fractions]
            means=100*np.array([s['mean'] for s in ss]);sd=100*np.array([s['sd'] for s in ss])
            ax.plot(np.array(fractions)*100,means,'o-',color=color,label=arm,ms=4)
            ax.fill_between(np.array(fractions)*100,means-sd,means+sd,color=color,alpha=.07)
        ax.set(title=d,xlabel='Training rows labeled (%)',ylim=(40,102))
    axes[0].set_ylabel('Test accuracy (%)');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=6,fontsize=10);fig.suptitle('Measured accuracy · means and ±1 sample SD over three repetitions');fig.tight_layout(rect=(0,.1,1,.94));save(fig,'curves')
    fig,axes=plt.subplots(1,3,figsize=(13,4.6),sharey=True)
    for ax,d in zip(axes,datasets):
        for offset,(t,c,color) in enumerate([('scarf_ft','scratch','#047857'),('scarf_frozen','random','#9333ea')]):
            ss=[next(s for s in r['contrasts'] if s['dataset']==d and s['fraction']==f and s['treatment']==t and s['control']==c) for f in fractions]
            xx=np.array(fractions)*100+offset*1.5;means=100*np.array([s['mean'] for s in ss]);bounds=100*np.array([s['interval'] for s in ss])
            ax.errorbar(xx,means,yerr=np.maximum(0,np.array([means-bounds[:,0],bounds[:,1]-means])),fmt='o-',color=color,capsize=3,label=t+' − '+c)
            for x,s in zip(xx,ss):ax.scatter(x+np.array([-1,0,1]),100*np.array(s['gains']),s=10,color=color,alpha=.6)
        ax.axhline(0,color='#334155',lw=1);ax.set(title=d,xlabel='Training rows labeled (%)')
    axes[0].set_ylabel('Paired accuracy gain (percentage points)');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2);fig.suptitle('Points: paired seeds · bars: descriptive t95 intervals, not simultaneous bands');fig.tight_layout(rect=(0,.1,1,.94));save(fig,'gains')
    fig,ax=plt.subplots(figsize=(10,4.4));z=r['rank_audits'][2]
    for i,(a,v) in enumerate(zip(z['arms'],z['mean_ranks'])):ax.scatter(v,i,color=COLORS[a],s=60);ax.text(v+.06,i,a,va='center')
    ax.plot([1,1+z['cd']],[6,6],color='#a16207',lw=3);ax.text(1,6.2,f"Nemenyi CD = {z['cd']:.2f}",va='bottom');ax.set(xlim=(.8,7.5),ylim=(-.6,7),yticks=[],xlabel='Mean within-dataset rank (lower is better)');ax.set_title(f"40% budget · three datasets, seeds averaged first\nFriedman p = {z['friedman_p']:.3f}; exploratory, not a broad ranking",loc='left');save(fig,'ranks')
if __name__=='__main__':build()
