"""Portable query, metric, coverage, information and evidence diagrams."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;D=P/'figures/l136';D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.hashsalt':'l136','axes.spines.top':False,'axes.spines.right':False})
navy='#17324d';teal='#007f82';amber='#b85d20'
def save(fig,name):
 fig.savefig(D/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});fig.savefig(D/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)
 path=D/(name+'.svg');path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
def table(ax,rows,headers,widths=None):
 ax.axis('off');t=ax.table(cellText=rows,colLabels=headers,cellLoc='center',loc='center',colWidths=widths);t.auto_set_font_size(False);t.set_fontsize(12);t.scale(1,2.2)
 for (r,c),cell in t.get_celld().items():cell.set_edgecolor('#c5d3d9');cell.set_facecolor('#e4f3ee' if r==0 else 'white')
 return t
f,axes=plt.subplots(2,1,figsize=(7,6));f.subplots_adjust(hspace=.55)
axes[0].set_title('1 · Submitted rows can arrive in any order',loc='left',weight='bold');table(axes[0],[['(9, 100)','30'],['(4, 100)','10'],['(4, 200)','20']],['(entity, cutoff)','Prediction'])
axes[1].set_title('2 · Join on both keys → official query order',loc='left',weight='bold');table(axes[1],[['(4, 100)','10'],['(4, 200)','20'],['(9, 100)','30']],['Official query','Aligned prediction'])
axes[1].text(.5,-.18,'Driver 4 appears twice. Entity alone loses the question.',ha='center',transform=axes[1].transAxes,color=navy);save(f,'alignment')
f,axes=plt.subplots(2,1,figsize=(7,5.5));f.subplots_adjust(hspace=.55)
axes[0].set_title('Fixed predictions → fixed absolute errors',loc='left',weight='bold');table(axes[0],[['0','1','1'],['2','4','2'],['4','4','0']],['Target','Prediction','Absolute error'])
ax=axes[1];ax.barh(['Scale = 2 (baseline)','Scale = 4 (changed)'],[.5,.25],color=[teal,amber]);ax.invert_yaxis();ax.set(xlim=(0,.65),xlabel='NMAE = fixed MAE 1 / scale')
for i,v in enumerate([.5,.25]):ax.text(v+.02,i,f'{v:.2f}',va='center')
ax.text(0,-.65,'Synthetic illustration: changing the yardstick changes the score.\nReproduction must retain the official denominator.',transform=ax.transAxes,color=navy,fontsize=11);save(f,'normalization')
f,ax=plt.subplots(figsize=(7,4.8));table(ax,[['Complete','0.10','0.50','0.30','YES'],['Omit B','0.10','missing','0.10*','NO']],['File set','Task A','Task B','Mean','Valid?'],[.22,.15,.18,.23,.22]);ax.set_title('The smaller partial average is not a board result',loc='left',weight='bold');ax.text(.5,.02,'*Diagnostic partial mean only.\nReal regression board: exactly nine equally weighted tasks.',ha='center',transform=ax.transAxes,color=navy);save(f,'coverage')
s=json.loads((P/'evidence/l136/leaderboard.json').read_text());f,ax=plt.subplots(figsize=(7,4.4));ids=sorted(s['entries'],key=lambda i:s['entries'][i]['mean']);vals=[s['entries'][i]['mean'] for i in ids]
ax.barh([s['entries'][i]['name'] for i in ids],vals,color=[teal,'#70a7a8',navy]);ax.invert_yaxis();ax.set(xlim=(0,.39),xlabel='Mean of nine task NMAEs · lower is better',title='Published archives independently rescored')
for i,v in enumerate(vals):ax.text(v+.008,i,f'{v:.6f}',va='center')
ax.text(0,-.3,'Same evaluator and task coverage. Training budgets and temporal\ninputs are not established equal. No invented seed error bars.',transform=ax.transAxes,color=navy,fontsize=11);save(f,'leaderboard')
f,ax=plt.subplots(figsize=(7,4.8));ax.set(xlim=(8,15),ylim=(-.7,1.7),yticks=[0,1],yticklabels=['Rolling to query','Frozen at boundary'],xticks=range(8,16),xlabel='Illustrative event time',title='The same query can receive different histories')
ax.hlines([0,1],[8,8],[14,10],color=[teal,navy],lw=8);ax.axvline(10,color=navy,ls='--');ax.axvline(14,color=amber,ls='--');ax.scatter([12,12],[0,1],s=120,color=[teal,'#bbb'],zorder=4)
ax.text(10.05,1.4,'Freeze = 10',fontsize=11);ax.text(12,1.13,'Event 12: hidden',ha='center',fontsize=11);ax.text(12,.15,'Event 12: visible',ha='center',fontsize=11);ax.text(14.05,.8,'Query\n= 14',fontsize=11,color=amber)
ax.text(0,-.35,'Hold query and target fixed; vary eligible history.\nThis illustration does not certify any submitted training pipeline.',transform=ax.transAxes,color=navy,fontsize=11);save(f,'regimes')
r=json.loads((P/'evidence/l136/training.json').read_text());f,axes=plt.subplots(2,1,figsize=(7,6.5));f.subplots_adjust(hspace=.65)
for ax,split,label in zip(axes,['val','test'],['Validation','Test']):
 m=r['metrics'][split];ax.scatter(range(5),m['values'],color=teal,s=70,zorder=3);ax.axhline(m['paper_target'],color=amber,ls='--',label=f"Paper mean {m['paper_target']:.3f}");ax.axhline(m['mean'],color=navy,label=f"Fresh mean {m['mean']:.4f}")
 ax.set(xticks=range(5),xlabel='Fresh training seed',ylabel='Raw MAE',title=f"{label}: {m['mean']:.4f} ± {m['sample_sd']:.4f} sample SD")
 ax.legend(loc='best',fontsize=10)
axes[1].text(0,-.42,'Historical selected experiment; different evidence from archive replay.\nSeed spread conditions on one task and split.',transform=axes[1].transAxes,color=navy,fontsize=11);save(f,'training')
print('Built six portable figures')
