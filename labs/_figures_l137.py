"""Mechanistic static companions and measured plots, generated from evidence."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent;D=P/'figures/l137';D.mkdir(parents=True,exist_ok=True)
r=json.loads((P/'evidence/l137/errors.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.hashsalt':'l137','axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    fig.savefig(D/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});fig.savefig(D/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
    p=D/(name+'.svg');p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')
def table(ax,rows,headers):
    ax.axis('off');t=ax.table(cellText=rows,colLabels=headers,loc='center',cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(12);t.scale(1,2.2)
    for (i,j),c in t.get_celld().items():c.set_edgecolor('#c5d3d9');c.set_facecolor('#e3f2ee' if i==0 else 'white')
f,ax=plt.subplots(figsize=(7.7,3.4));ax.set_title('Same query → two absolute losses → signed difference',loc='left',weight='bold');table(ax,[['A at t1','5','8 → 3','6 → 1','+2'],['B at t1','10','8 → 2','14 → 4','−2']],['Query','Target','GNN → loss','FE → loss','GNN − FE']);ax.text(.5,.02,'Mean gap = (+2 − 2) / 2 = 0. A tie in the mean hides different failures.',ha='center',transform=ax.transAxes,fontsize=11);save(f,'paired')
f,ax=plt.subplots(figsize=(7.7,3.4));ax.set_ylim(-.7,1.3);ax.set_xlim(2008,2013);ax.axvspan(2008,2010,color='#e3f2ee');ax.axvline(2010,color='#b85d20',ls='--');ax.scatter([2008.5,2009.5],[.35,.35],s=80,color='#007f82');ax.scatter([2011,2012],[.35,.35],marker='x',s=80,color='#b85d20');ax.scatter([2012],[.85],marker='v',s=90,color='#17324d');ax.text(2010,.98,'Snapshot ends\n2010-01-01',ha='center');ax.text(2012,1.02,'Later query',ha='center');ax.annotate('',xy=(2012,-.05),xytext=(2009.5,-.05),arrowprops={'arrowstyle':'<->'});ax.text(2010.75,-.33,'Recorded recency grows; new races stay hidden',ha='center',fontsize=11);ax.set_yticks([]);ax.set_xlabel('Illustrative event year');ax.set_title('A frozen history is not evidence of inactivity',loc='left',weight='bold');save(f,'snapshot')
f,axes=plt.subplots(2,1,figsize=(7.7,5.6));f.subplots_adjust(hspace=.45);table(axes[0],[['A','0, 0','0','2'],['B','6','6','1']],['Driver','Paired losses','Loss sum','Query count']);axes[0].set_title('Carry all queries when sampling a driver',loc='left',weight='bold');table(axes[1],[['A + A','0 / 4','0'],['A + B','6 / 3','2'],['B + B','12 / 2','6']],['Bootstrap draw','Pooled loss / rows','Mean']);axes[1].set_title('Whole-driver draws retain query weighting',loc='left',weight='bold');save(f,'clusters')
f,axes=plt.subplots(1,2,figsize=(8,4),sharey=True)
for ax,split in zip(axes,['val','test']):
    rows=r['splits'][split]['seeds']
    for row in rows:ax.plot([0,1],[row['gnn'],row['fe']],color='#adb8bd',lw=1)
    ax.scatter(np.zeros(5),[x['gnn'] for x in rows],color='#007f82',label='GNN');ax.scatter(np.ones(5),[x['fe'] for x in rows],color='#b85d20',label='FE');ax.set_xticks([0,1],['Basic GNN','SQL + LightGBM']);ax.set_xlim(-.4,1.4);ax.set_title(split.upper());ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('MAE (position units; lower is better)');f.suptitle('Five fresh runs per pipeline · lines pair run labels only',weight='bold',fontsize=13);f.tight_layout();save(f,'seeds')
f,axes=plt.subplots(1,2,figsize=(9.5,5.2),sharey=True);names=list(r['splits']['val']['slices'])
for ax,split in zip(axes,['val','test']):
    for i,name in enumerate(names):
        row=r['splits'][split]['slices'][name];ci=row['interval']
        if ci:ax.errorbar(row['mean'],i,xerr=[[row['mean']-ci['low']],[ci['high']-row['mean']]],fmt='o',color='#007f82' if name=='high_history' else '#647781',capsize=3)
        else:ax.text(.03,i,'UNSUPPORTED: '+str(row['rows'])+' rows',va='center',fontsize=10)
    ax.axvline(0,color='#b85d20',ls='--');ax.set_title(split.upper());ax.set_xlabel('GNN − FE absolute loss');ax.grid(axis='x',alpha=.15)
axes[0].set_yticks(range(len(names)),[n.replace('_',' ') for n in names]);axes[0].invert_yaxis();f.suptitle('Paired means · descriptive 95% driver-cluster intervals',weight='bold',fontsize=13);f.tight_layout();save(f,'slices')
