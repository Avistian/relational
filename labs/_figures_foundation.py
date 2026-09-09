"""Generate numerical teaching figures from operators and measured result artifacts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from _foundation_config import ARCH,TITLES
from relkit.foundation_core import context_mask,attention,posterior_predictive
import torch

ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.facecolor':'#fffdf8','axes.facecolor':'#fffdf8','savefig.facecolor':'#fffdf8'})
COLORS=['#24677b','#b75032','#677d35','#775991','#aa792b']


def save(fig,n,name):
    directory=ROOT/f'figures/l{n:03}';directory.mkdir(parents=True,exist_ok=True)
    fig.savefig(directory/f'{name}.png',dpi=150,bbox_inches='tight');plt.close(fig)


def architecture(n):
    blocks=ARCH[n];fig,ax=plt.subplots(figsize=(8.2,8));ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,10)
    for i,(name,shape,note) in enumerate(blocks):
        y=9.6-i*1.92
        ax.add_patch(FancyBboxPatch((.2,y-1.4),9.6,1.35,boxstyle='round,pad=.12',facecolor='#edf3f3',edgecolor='#acc1c5'))
        ax.text(.45,y-.22,name,fontsize=11,fontweight='bold',color=COLORS[0],va='top')
        ax.text(.45,y-.64,shape,fontsize=12,va='top')
        ax.text(.45,y-1.03,note,fontsize=10,va='top',color='#555')
        if i<len(blocks)-1:ax.annotate('',xy=(5,y-1.85),xytext=(5,y-1.55),arrowprops=dict(arrowstyle='->',color=COLORS[1],lw=2))
    fig.suptitle(f'L{n:03} · End-to-end architecture and information boundary',fontsize=14,y=1.01)
    save(fig,n,'architecture')


def mechanism(n):
    fig,ax=plt.subplots(figsize=(8.3,4.6));ax.set_title('Synthetic worked example · trace the operation',loc='left',pad=20)
    if n in [62,64,65]:
        mask=context_mask(3,2).numpy();ax.imshow(mask,cmap=matplotlib.colors.ListedColormap(['#eee5dc','#4a8290']),vmin=0,vmax=1)
        ax.set_xticks(range(5),['context 0','context 1','context 2','query 0','query 1'],rotation=25,ha='right')
        ax.set_yticks(range(5),['context 0','context 1','context 2','query 0','query 1']);ax.set_xlabel('Key/value source');ax.set_ylabel('Reading row')
        for i in range(5):
            for j in range(5):ax.text(j,i,'READ' if mask[i,j] else 'BLOCK',ha='center',va='center',fontsize=9,color='white' if mask[i,j] else '#655')
    elif n==61:
        s=np.arange(5);p=(1+s)/6;ax.plot(s,s/4,'o--',color=COLORS[1],label='Empirical rate s/4');ax.plot(s,p,'o-',color=COLORS[0],label='Beta(1,1) posterior (s+1)/6')
        ax.set(xlabel='Successes in four observed labels',ylabel='Next-label probability',ylim=(-.04,1.04),xticks=s);ax.legend()
        ax.annotate('s=3: posterior 4/6 = .667',xy=(3,4/6),xytext=(.1,.85),arrowprops=dict(arrowstyle='->'))
    elif n==63:
        ax.axis('off');cells=[['node','noise','parent calculation','output'],['0','1','root: no parents','1'],['1','.2','2 × 1 + .2','2.2'],['2','−.1','−1 × 2.2 − .1','−2.3']]
        table=ax.table(cellText=cells,loc='center',cellLoc='center',colWidths=[.12,.15,.46,.2]);table.auto_set_font_size(False);table.set_fontsize(13);table.scale(1,2.5)
        ax.text(.02,.08,'Linear fixture. The nonlinear lab replaces the parent sum by tanh(sum).',transform=ax.transAxes,fontsize=11)
    elif n==66:
        ax.axis('off');q=torch.tensor([[0.,1.],[1.,0.]]);k=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
        w=(q@k.T/np.sqrt(2)).softmax(-1).numpy();memory=attention(q,k,k).numpy()
        lines=['3 context tokens U; 2 inducing vectors I','I = [[0,1], [1,0]]; U = [[1,0], [0,1], [1,1]]',
          'Stage 1 weights = softmax(I Uᵀ / sqrt(2))',np.array2string(w,precision=3),'Memory M = weights × U',np.array2string(memory,precision=3),
          'Stage 2: each cell reads only M; unrelated queries cannot change M.']
        ax.text(.03,.95,'\n\n'.join(lines),transform=ax.transAxes,va='top',fontsize=12,fontfamily='DejaVu Sans Mono')
    elif n==67:
        x=np.array([0.,1.,2.,4.]);distance=(x-1)**2;ax.bar(range(4),distance,color=[COLORS[0],'#ccc',COLORS[0],COLORS[1]])
        ax.set(xticks=range(4),xticklabels=['row 0\nx=0','row 1\nx=1','row 2\nx=2','row 3\nx=4'],ylabel='Squared distance to query x=1')
        ax.text(1,1,'excluded\nby identity',ha='center');ax.text(.02,.92,'Eligible k=2 → row IDs [0,2] (stable tie)',transform=ax.transAxes)
    elif n==68:
        t=np.arange(5);w=1.5*np.cos(t*.65);ax.plot(t,w,'o-',color=COLORS[1]);ax.axhline(0,color='#777',lw=1)
        ax.set(xlabel='Temporal domain',ylabel='Changed edge weight',xticks=t);ax.text(.03,.05,'Same exogenous noise; only the edge trajectory changes.',transform=ax.transAxes)
    elif n==69:
        probs=np.array([.8,.8,1e-12]);loss=-np.log(probs);ax.bar(['Known A','Known B','Unsupported C'],loss,color=[COLORS[0],COLORS[0],COLORS[1]])
        ax.set_ylabel('Per-row negative log probability (nats)')
        for i,v in enumerate(loss):ax.text(i,v+.3,f'{v:.3f}',ha='center')
        ax.text(.02,.91,'Unsupported class: epsilon = 1e−12\nDo not drop the row.',transform=ax.transAxes)
    elif n==59:
        values=[.33,.26,.31];ax.bar(['candidate A','candidate B','candidate C'],values,color=[COLORS[0],COLORS[1],COLORS[0]])
        ax.axhline(.3,color='#555',linestyle='--',label='Equal true error .30');ax.set_ylim(.2,.36);ax.set_ylabel('Error (zoomed scale)');ax.legend(loc='upper right')
        ax.text(1,.265,'selected',ha='center',color=COLORS[1])
    else:
        data=np.array([[1,2,3],[3,1,2]]);ax.imshow(data,cmap='YlGnBu',vmin=1,vmax=3)
        ax.set_xticks(range(3),['A','B','C']);ax.set_yticks(range(2),['dataset one','dataset two']);ax.set_xlabel('Mean ranks: A=2, B=1.5, C=2.5')
        for i in range(2):
            for j in range(3):ax.text(j,i,str(data[i,j]),ha='center',va='center',fontsize=22,color='white' if data[i,j]==3 else '#111')
        ax.set_title('Rank within datasets, then give each dataset one vote',loc='left')
    fig.tight_layout();save(fig,n,'mechanism')


def score_points(n,records,value='error'):
    datasets=list(dict.fromkeys(r['dataset'] for r in records));arms=list(dict.fromkeys(r['arm'] for r in records))
    fig,axes=plt.subplots(len(datasets),1,figsize=(9,2.15*len(datasets)+.6),squeeze=False)
    for ax,name in zip(axes[:,0],datasets):
        for j,arm in enumerate(arms):
            vals=[r[value] for r in records if r['dataset']==name and r['arm']==arm]
            if not vals:continue
            mean=np.mean(vals);sd=np.std(vals,ddof=1) if len(vals)>1 else 0
            ax.errorbar(j,mean,yerr=sd,fmt='o',color=COLORS[j%5],capsize=5)
            ax.scatter(np.arange(len(vals))*.07+j-.07,vals,color=COLORS[j%5],s=22,alpha=.7)
        labels=[a.replace('TabPFN-2.5-synthetic','PFN-2.5\nsynthetic').replace('TabPFN-','PFN-').replace('TabICL-','ICL-') for a in arms]
        ax.set_title(name,loc='left',fontsize=12);ax.set_xticks(range(len(arms)),labels,fontsize=9);ax.set_ylabel('Test loss');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Measured seed points and sample SD · fixed rows · separate task scales',fontsize=12)
    fig.tight_layout();save(fig,n,'results')


def comparison(n):
    """Paired raw gaps preserve units; rank uncertainty uses dataset blocks."""
    from scipy.stats import t
    r=json.loads((ROOT/f'_verify_l{n:03}_results.json').read_text())
    arms=sorted({v['arm'] for v in r['records']} - {'XGBoost'})
    datasets=list(dict.fromkeys(v['dataset'] for v in r['records']))
    fig,axes=plt.subplots(len(datasets),1,figsize=(9,2*len(datasets)),squeeze=False)
    for ax,name in zip(axes[:,0],datasets):
        base={v['seed']:v['error'] for v in r['records'] if v['dataset']==name and v['arm']=='XGBoost'}
        for j,arm in enumerate(arms):
            vals=[v['error']-base[v['seed']] for v in r['records'] if v['dataset']==name and v['arm']==arm]
            half=t.ppf(.975,len(vals)-1)*np.std(vals,ddof=1)/np.sqrt(len(vals))
            ax.errorbar(j,np.mean(vals),yerr=half,fmt='o',capsize=4,color=COLORS[j%5])
            ax.scatter(np.arange(len(vals))*.06+j-.06,vals,s=18,color=COLORS[j%5])
        ax.axhline(0,color='#777',lw=1)
        ax.set_xticks(range(len(arms)),[a.replace('TabPFN-2.5-synthetic','PFN-2.5\nsynthetic').replace('TabPFN-','PFN-') for a in arms],fontsize=9)
        ax.set_title(name,loc='left',fontsize=11);ax.set_ylabel('Loss gap vs XGB')
    fig.suptitle('Paired seed gaps and conditional 95% t intervals\nBelow zero favors the named model · units are task-specific',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.975));save(fig,n,'comparison')
    fig,axes=plt.subplots(len(r['summary']),1,figsize=(9,4*len(r['summary'])),squeeze=False)
    for ax,(regime,s) in zip(axes[:,0],r['summary'].items()):
        pairs=sorted(s['mean_ranks'].items(),key=lambda v:v[1]);k=len(pairs)
        ax.scatter([v for _,v in pairs],range(k),s=65,color=COLORS[0])
        ax.set_yticks(range(k),[a for a,_ in pairs]);ax.invert_yaxis();ax.set_xlim(.5,max(k+.5,1+s['nemenyi_cd']+.5))
        ax.plot([1,1+s['nemenyi_cd']],[-.8,-.8],color=COLORS[1],lw=3)
        ax.text(1,-1.05,f"Nemenyi CD = {s['nemenyi_cd']:.2f}; {s['datasets']} dataset blocks",fontsize=10)
        ax.set_ylim(k-.5,-1.5);ax.set_xlabel('Mean rank; a gap smaller than CD is not significant by this test')
        ax.set_title(regime+' · exploratory complete-panel rank comparison',loc='left');ax.grid(axis='x',alpha=.2)
    fig.tight_layout();save(fig,n,'ranks')


def results(n):
    p=ROOT/f'_verify_l{n:03}_results.json'
    if not p.exists():return False
    r=json.loads(p.read_text())
    if n in [62,64,65,67,70]:score_points(n,r['records']);return True
    fig,ax=plt.subplots(figsize=(9,5))
    if n==58:
        arms=r['arms'];a=[r['mean_ranks'][k] for k in arms];b=[r['xgb_favored_45_ranks'][k] for k in arms];x=np.arange(len(arms))
        ax.plot(x,a,'o-',label=f'All {r["datasets"]} released tasks',color=COLORS[0]);ax.plot(x,b,'s--',label='45 outcome-selected tasks',color=COLORS[1]);ax.set_xticks(x,arms,rotation=15);ax.set_ylabel('Mean rank in six-model pool (lower better)');ax.legend()
    elif n==59:
        rows=r['summary'];x=[a['candidates'] for a in rows]
        ax.plot(x,[a['validation_error'] for a in rows],'o-',label='Selected validation error',color=COLORS[1]);ax.plot(x,[a['test_error'] for a in rows],'o-',label='Independent test error',color=COLORS[0]);ax.set_xscale('log',base=4);ax.set_xticks(x,[str(v) for v in x]);ax.set(xlabel='Number of noise candidates searched',ylabel='Mean classification error across 200 trials');ax.legend()
    elif n==60:
        plt.close(fig);fig,axes=plt.subplots(1,2,figsize=(10,4.7))
        for ax,(regime,s) in zip(axes,r['summary'].items()):
            names=list(s['mean_ranks']);vals=list(s['mean_ranks'].values());ax.barh(names,vals,color=COLORS[:len(names)]);ax.invert_yaxis();ax.set_title(f'{regime}: {s["datasets"]} task blocks');ax.set_xlabel('Mean rank (lower better)');ax.set_xlim(0,5.3)
            for i,v in enumerate(vals):ax.text(v+.04,i,f'{v:.2f}',va='center',fontsize=10)
    elif n==61:
        plt.close(fig);fig,axes=plt.subplots(1,3,figsize=(11,3.8),sharey=True)
        for ax,length in zip(axes,[4,32,64]):
            values=np.array([[z['prediction'] for z in run['grid'] if z['n']==length] for run in r['runs']]);s=np.arange(length+1)
            ax.plot(s/length,(s+1)/(length+2),'--',color=COLORS[1],label='Exact posterior');ax.plot(s/length,values.mean(0),color=COLORS[0],label='Trained mean');ax.fill_between(s/length,values.min(0),values.max(0),color=COLORS[0],alpha=.2);ax.set_title(f'n={length}'+(' · unseen length' if length==64 else ''));ax.set_xlabel('Observed success fraction');ax.set_ylim(-.03,1.03)
        axes[0].set_ylabel('Next-success probability');axes[0].legend(fontsize=9)
    elif n==63:
        a=np.array(r['base']);b=np.array(r['intervention']);ax.scatter(a[:,1],a[:,2],s=12,alpha=.5,label='Original edge −1.5',color=COLORS[0]);ax.scatter(b[:,1],b[:,2],s=12,alpha=.5,label='Edge removed; same noise',color=COLORS[1]);ax.set(xlabel='Unchanged parent node 1',ylabel='Child node 2');ax.legend()
    elif n==66:
        plt.close(fig);fig,axes=plt.subplots(1,2,figsize=(10,4));rows=r['records']
        for seed in [0,1,2]:
            v=[z for z in rows if z['seed']==seed]
            for ax,key in zip(axes,['log_loss','predict_seconds']):ax.plot([z['context'] for z in v],[z[key] for z in v],'o-',label=f'seed {seed}',alpha=.8);ax.set_xlabel('Labeled context rows')
        axes[0].set_ylabel('Test log loss (60 fixed queries)');axes[1].set_ylabel('Prediction seconds (CPU)');axes[0].legend()
    elif n==68:
        rows=r.get('pfn_ablation',{}).get('records',[])
        for j,arm in enumerate(['stationary-prior','changing-edge-prior']):
            for i,condition in enumerate(['stationary','drifting']):
                values=[v['log_loss'] for v in rows if v['arm']==arm and v['condition']==condition]
                if values:ax.errorbar(i+(j-.5)*.2,np.mean(values),yerr=np.std(values,ddof=1),fmt='o',capsize=5,color=COLORS[j],label=arm if i==0 else None);ax.scatter(np.arange(len(values))*.025+i+(j-.5)*.2,values,color=COLORS[j],s=18)
        ax.set_xticks([0,1],['stationary evaluation','drifting evaluation']);ax.set_ylabel('Query-label log loss');ax.legend();ax.set_title('Reduced PFNs: matched compute, different pretraining priors')
    elif n==69:
        names=list(r['datasets']);arms=['XGBoost','TabPFN-v2'];x=np.arange(len(names))
        for j,(arm,condition) in enumerate((a,c) for a in arms for c in ['missing-column','scaled-column']):
            gaps=[]
            for name in names:
                values=[]
                for seed in [0,1,2]:
                    v={z['condition']:z['log_loss'] for z in r['records'] if z['dataset']==name and z['arm']==arm and z['seed']==seed};values.append(v[condition]-v['clean'])
                gaps.append(np.mean(values))
            ax.plot(x,gaps,'o-',label=arm+' / '+condition,color=COLORS[j])
        ax.axhline(0,color='#777',lw=1);ax.set_xticks(x,names);ax.set_ylabel('Mean test log-loss change from clean');ax.legend(fontsize=9)
    ax.set_title(ax.get_title() or f'L{n:03} · Author-measured evidence; see protocol and uncertainty limits',fontsize=12)
    fig.tight_layout();save(fig,n,'results');return True


def build():
    for n in range(58,71):
        mechanism(n)
        if n in ARCH:architecture(n)
        if not results(n):print(f'L{n:03}: evidence not ready')
        if n in [60,70]:comparison(n)

if __name__=='__main__':build()
