"""Actual computation traces for manual FE; deterministic portable figures."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;O=P/'figures/l129';O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'l129','font.size':12})
INK='#19384a';BLUE='#dfeaf2';GOLD='#f5e9c7';GREEN='#dceee4';RED='#f4ded7'
def setup(title,h):
 f,a=plt.subplots(figsize=(10,h));f.patch.set_facecolor('#faf9f6');a.axis('off');a.set(xlim=(0,10),ylim=(0,h));a.set_title(title,loc='left',fontsize=17,weight='bold',color=INK,pad=17);return f,a
def box(a,x,y,w,h,text,c=BLUE):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04',facecolor=c,edgecolor=INK));a.text(x+w/2,y+h/2,text,ha='center',va='center',color=INK,fontsize=12)
def arrow(a,x,y,xx,yy):a.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','lw':2,'color':INK})
def save(f,name):
 f.tight_layout();f.savefig(O/(name+'.svg'),metadata={'Date':None});f.savefig(O/(name+'.png'),dpi=150,metadata={'Software':'L129'});plt.close(f)
f,a=setup('From a relational question to one flat model input',8)
box(a,.2,6.3,9.5,1,'Query = (driver, cutoff)\nFuture 60-day mean position is the target; keep it outside features',GOLD)
box(a,.2,4.35,4.25,1.2,'Relational history\nresults · standings · constructors\n+ schedule assumption');box(a,5.15,4.35,4.55,1.2,'Explicit SQL choices\nASOF standing · race-ID slots\nratios · missing history',GREEN);arrow(a,4.55,4.95,5.05,4.95);arrow(a,5,6.25,5,5.8)
box(a,.2,2.3,9.5,1.15,'50 engineered fields + numeric driverId\nTrain-fitted mappings → 14 categorical + 37 numerical columns');arrow(a,5,4.25,5,3.55)
box(a,.2,.3,4.25,1.1,'Train + validation\n10 trials → select → refit',GOLD);box(a,5.15,.3,4.55,1.1,'Test once per selected model\nAlign by both key fields → MAE',GREEN);arrow(a,5,2.2,2.3,1.5);arrow(a,4.55,.85,5.05,.85);save(f,'workflow')
s=json.loads((P/'evidence/l129/sql-audit.json').read_text())['example'];v=s['features']
f,a=setup(f"Real query: driver {s['entity']} at {s['cutoff'][:10]}",7.5)
box(a,.15,5.65,4.45,1.1,'Latest driver standing before t\n'+f"Position {v['driver_position']:g}; points {v['driver_points']:g}");box(a,5.25,5.65,4.45,1.1,'Same race → constructor\n'+f"Position {v['constructor_position']:g}; points {v['constructor_points']:g}");arrow(a,4.7,6.2,5.15,6.2)
box(a,.15,3.7,9.55,1.1,f"Derived comparison: points ratio = {v['driver_points']:g} / {v['constructor_points']:g} = {v['points_ratio']:.2f}\nPosition difference = {v['driver_position']:g} − {v['constructor_position']:g} = {v['position_diff']:g}",GREEN);arrow(a,5,5.55,5,4.9)
for i in [1,2,3]:
 x=.15+(i-1)*3.25
 def fmt(x):return 'missing' if x is None else f'{x:g}'
 box(a,x,1.65,3.05,1.2,f"Race slot {i}\nposition {fmt(v[f'past_{i}_driver_position'])}\ngrid {fmt(v[f'past_{i}_driver_grid'])}",GOLD)
a.text(.15,.55,'Slots use race-ID arithmetic and a two-calendar-month window.\nMissing participation stays missing; these are not the last three personal starts.',fontsize=12,color=INK);save(f,'features')
t=json.loads((P/'evidence/l129/prediction-trace.json').read_text())
n=len(t['first_tree_path']);height=4.2+1.2*n
f,a=setup(f"Follow the first tree, then add all {t['tree_count']} contributions",height)
for i,node in enumerate(t['first_tree_path']):
 y=height-1.6-i*1.2;value='missing' if node['value'] is None else f"{node['value']:.3g}";thr=float(node['threshold'])
 box(a,.4,y,8.8,.82,f"{node['feature']} = {value}   ≤   {thr:.3g} ?     → {node['branch']}",BLUE)
 if i<n-1:arrow(a,4.8,y-.04,4.8,y-.3)
box(a,.4,1.75,8.8,.85,f"Tree 1 leaf contribution = {t['first_tree_leaf']:.6f}",GOLD)
box(a,.4,.2,8.8,1.1,f"Trees 1–10: {t['first_ten_sum']:.6f}   +   trees 11–{t['tree_count']}: {t['remaining_sum']:.6f}\nPrediction = {t['prediction']:.6f}; future target = {t['target']:.1f}",GREEN);arrow(a,4.8,3.4,4.8,2.7);arrow(a,4.8,1.7,4.8,1.4);save(f,'trees')
sumry=json.loads((P/'evidence/l129/summary.json').read_text());f,axes=plt.subplots(1,2,figsize=(10,4.6));f.patch.set_facecolor('#faf9f6')
for ax,split in zip(axes,['val','test']):
 ax.scatter([0]*5,[r[split] for r in sumry['seeds']],color='#ad6045',s=45,label='Five FE searches')
 m=sumry['metrics'][split];c=sumry['comparison'][split]
 ax.errorbar(.18,m['mean'],yerr=m['sample_sd'],fmt='D',color='#ad6045',capsize=5)
 ax.errorbar(1,c['rdl_mean'],yerr=c['rdl_sample_sd'],fmt='D',color='#39846b',capsize=5,label='L127 mean ± seed SD')
 ax.set(xticks=[0,1],xticklabels=['Manual FE','L127 RDL'],xlim=(-.4,1.4),ylabel='MAE · finishing-position units',title=split+' · detail scale');ax.grid(axis='y',alpha=.2)
axes[0].legend(fontsize=9);f.suptitle('Descriptive comparison: matched query keys, different preprocessing / information assumptions',fontsize=11);save(f,'scores')
f,a=setup('Three clocks answer different questions',6.5)
box(a,.15,4.65,9.55,1.1,'ORIGINAL STUDY: active human work\nExplore + invent + implement + prepare; reusable infrastructure excluded',GOLD)
box(a,.15,2.8,9.55,1.1,'AUTHOR REPLAY: machine execution\nStarts with released SQL; measures computation, not feature discovery',BLUE)
box(a,.15,.95,9.55,1.1,'YOUR LOG: new learner observation\nRecord active minutes, assistance, failed ideas and reused setup',GREEN)
a.text(.15,.25,'Do not substitute one clock for another. A blank measurement is not zero.',fontsize=12,color=INK);save(f,'effort')
