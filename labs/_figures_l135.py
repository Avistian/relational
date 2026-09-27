"""Readable portable experiment diagrams and measured seed plots."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).parent;D=P/'figures/l135';D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.hashsalt':'l135','axes.spines.top':False,'axes.spines.right':False})
navy='#17324d';teal='#007f82';amber='#b85d20'
def save(fig,name):
 fig.savefig(D/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});fig.savefig(D/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)
f,ax=plt.subplots(figsize=(7,4.5));ax.axis('off');ax.set_title('Six configurations · two search seeds each',loc='left',weight='bold',pad=18)
t=ax.table(cellText=[['0.001','2 × 10 epochs','2 × 10 epochs'],['0.005','2 × 10 epochs','DEFAULT · 2 × 10'],['0.010','2 × 10 epochs','2 × 10 epochs']],colLabels=['Learning rate','Fanouts [32,16]','Fanouts [128,64]'],cellLoc='center',loc='center',colWidths=[.22,.38,.4]);t.auto_set_font_size(False);t.set_fontsize(12);t.scale(1,2.6)
for (r,c),cell in t.get_celld().items():
 cell.set_edgecolor('#d0dbdf');cell.set_facecolor('#edf5f6' if r==0 else 'white')
 if (r,c)==(2,2):cell.set_facecolor('#e4f3ee');cell.get_text().set_color(teal)
ax.text(.5,.03,'Fixed: full data · width 128 · two layers · batch 512\nChanged: optimization step size and sampled context',ha='center',va='bottom',transform=ax.transAxes,color=navy);save(f,'space')
f,axes=plt.subplots(2,1,figsize=(7,6),gridspec_kw={'height_ratios':[1,1.25]});f.subplots_adjust(hspace=.7)
ax=axes[0];ax.plot([1,2,3],[4,2,3],'o-',color=teal,lw=2);ax.scatter([2],[2],s=140,facecolor='none',edgecolor=amber,lw=2);ax.set(xticks=[1,2,3],xlabel='Epoch (illustration)',ylabel='Validation MAE',ylim=(1.5,4.5));ax.set_title('1 · Select a checkpoint inside each fit',loc='left',weight='bold');ax.annotate('Keep epoch 2',xy=(2,2),xytext=(2.15,3.4),arrowprops=dict(arrowstyle='->',color=amber),color=amber)
ax=axes[1];ax.axis('off');ax.set_title('2 · Select a configuration across seeds',loc='left',weight='bold');t=ax.table(cellText=[['A','2.0','4.0','3.0'],['B','3.0','2.0','2.5 ← winner']],colLabels=['Config','Seed 1','Seed 2','Mean'],cellLoc='center',loc='center');t.auto_set_font_size(False);t.set_fontsize(12);t.scale(1,2.2)
for (r,c),cell in t.get_celld().items():cell.set_edgecolor('#d0dbdf');cell.set_facecolor('#e4f3ee' if r==2 else 'white')
ax.text(.5,-.16,'Best single run ≠ best mean. No test metric enters either choice.',ha='center',transform=ax.transAxes,color=navy,fontsize=11);save(f,'selection')
f,ax=plt.subplots(figsize=(7,4.2));values=[900*.00022572,12*900*.00022572,10*900*.00022572,3];names=['Pilot (1 worker)','Search (12 workers)','Final (10 workers)','Overhead reserve'];colors=[navy,teal,'#62a8a1',amber]
ax.barh(names,values,color=colors);ax.invert_yaxis();ax.set(xlim=(0,3.7),xlabel='US dollars · maximum reserved cost',title='Budget the decision before observing its score')
for i,x in enumerate(values):ax.text(x+.05,i,f'${x:.3f}',va='center')
ax.text(.0,-.31,f'Total reserved plan: USD {sum(values):.6f} / 10\nUnused headroom funds checks or explicitly recorded retries.',transform=ax.transAxes,color=navy,fontsize=11);save(f,'budget')
s=json.loads((P/'evidence/l135/summary.json').read_text());frozen=json.loads((P/'evidence/l135/frozen.json').read_text())
f,ax=plt.subplots(figsize=(7,5));rank=frozen['ranking']
for i,row in enumerate(rank):
 vals=[r['selection_mae'] for r in frozen['records'] if r['config']==row['config']];ax.plot(vals,[i,i],color='#9aaab5');ax.scatter(vals,[i,i],marker='o',color=teal,s=50);ax.scatter([row['mean_validation_mae']],[i],marker='D',color=amber,s=55)
ax.set(yticks=range(6),yticklabels=[r['config'] for r in rank],xlabel='Selected validation MAE · lower is better',title='Search result: both seeds count');ax.invert_yaxis();ax.text(0,-.28,'Circles = seeds 100, 101 · diamond = arithmetic mean\nAll six configurations completed ten epochs per seed.',transform=ax.transAxes,fontsize=11,color=navy);save(f,'search')
f,axes=plt.subplots(2,1,figsize=(7,7.5));f.subplots_adjust(hspace=.68)
a=s['metrics'][s['default']]['test']['values'];b=s['metrics'][s['winner']]['test']['values'];ax=axes[0]
for seed,(x,y) in enumerate(zip(a,b)):ax.plot([0,1],[x,y],'o-',alpha=.75,label=f'Seed {seed}')
ax.set(xticks=[0,1],xticklabels=['Default · lr 0.005','Frozen winner · lr 0.01'],ylabel='Test MAE',title='Fresh final fits: match the same seed');ax.legend(ncol=3,fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.16))
ax=axes[1];pair=s['paired_test'];ax.axvline(0,color=navy,lw=1);ax.scatter(pair['differences'],range(5),color=teal,s=55);ax.set(yticks=range(5),yticklabels=[f'Seed {i}' for i in range(5)],xlabel='Tuned − default test MAE',title='Left of zero favors the tuned setting');ax.invert_yaxis();ax.text(.5,-.46,f"Mean {pair['mean_difference']:+.4f} · conditional 95% t interval\n[{pair['conditional_t95'][0]:+.4f}, {pair['conditional_t95'][1]:+.4f}] (fit seeds only)",ha='center',transform=ax.transAxes,color=navy,fontsize=11);save(f,'paired')
print('Built five portable figures')
