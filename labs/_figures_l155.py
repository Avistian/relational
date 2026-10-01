"""Model-specific computation comparison and separate synthetic effort accounting."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent;F=P/'figures/l155';F.mkdir(parents=True,exist_ok=True);r=json.loads((P/'evidence/l155/report.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l155'})
def save(fig,name):
 fig.savefig(F/(name+'.png'),dpi=160,facecolor=fig.get_facecolor());fig.savefig(F/(name+'.svg'),metadata={'Date':None},facecolor=fig.get_facecolor());plt.close(fig)
 path=F/(name+'.svg');path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
# Trace one real validation query through both saved models.
import cairosvg
trace=json.loads((P/'evidence/l155/fe/prediction-trace.json').read_text())
g=np.load(P/'evidence/l155/paper/seed-0/predictions.npz')
time=int(np.datetime64(trace['cutoff'],'ns').astype(np.int64))
idx=np.flatnonzero((g['val_entity']==trace['entity'])&(g['val_time']==time))[0]
svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="940" height="700" viewBox="0 0 940 700" role="img" aria-label="One actual validation query through released manual features and basic RDL">
<rect width="940" height="700" rx="14" fill="#f3f6f3"/>
<g font-family="sans-serif" fill="#173f37">
<text x="30" y="38" font-size="14" letter-spacing="2">ONE QUERY · TWO RELEASED PIPELINES</text>
<text x="30" y="75" font-size="24">Compare the prediction. Keep the information contract.</text>
<rect x="30" y="98" width="880" height="62" rx="8" fill="#173f37"/>
<text x="50" y="124" fill="white" font-size="17">Driver {trace['entity']} · cutoff {trace['cutoff'][:10]} · validation query · primary seed 0</text>
<text x="50" y="148" fill="white" font-size="15">Same target: average next-60-day position = {trace['target']:.3f}</text>
<text x="30" y="196" font-size="20" fill="#176a5c">Manual features → LightGBM</text>
<text x="505" y="196" font-size="20" fill="#a5632c">Relational graph → basic GNN</text>
<g fill="white"><rect x="30" y="215" width="405" height="100" rx="8" stroke="#176a5c"/><rect x="505" y="215" width="405" height="100" rx="8" stroke="#a5632c"/>
<rect x="30" y="352" width="405" height="120" rx="8" stroke="#176a5c"/><rect x="505" y="352" width="405" height="120" rx="8" stroke="#a5632c"/>
<rect x="30" y="510" width="405" height="105" rx="8" stroke="#176a5c"/><rect x="505" y="510" width="405" height="105" rx="8" stroke="#a5632c"/></g>
<g font-size="16"><text x="48" y="242">SQL: history + standings + race slots</text><text x="48" y="271">50 fields + driver ID → 51 columns</text><text x="48" y="300">Train-fitted category / numeric mapping</text>
<text x="523" y="242">Rows + foreign keys + relative times</text><text x="523" y="271">Typed encoders → 128 values / row</text><text x="523" y="300">Owner cutoff follows every sampled hop</text>
<text x="48" y="380">x → leaf₁(x) + leaf₂(x) + …</text><text x="48" y="408">{trace['tree_count']} selected trees; additive prediction</text><text x="48" y="437">First 10 sum {trace['first_ten_sum']:.3f} + rest {trace['remaining_sum']:.3f}</text><text x="48" y="460">10 trials / search · minimum validation MAE</text>
<text x="523" y="380">Two sum-GraphSAGE layers</text><text x="523" y="408">128 / 64 fanouts → root vector [128]</text><text x="523" y="437">Scalar head → training-quantile clipping</text><text x="523" y="460">10 epochs / fit · best validation checkpoint</text>
<text x="48" y="541">Saved FE prediction</text><text x="48" y="581" font-size="30">{trace['prediction']:.3f}</text>
<text x="523" y="541">Saved RDL prediction</text><text x="523" y="581" font-size="30">{float(g['val_pred'][idx]):.3f}</text></g>
<g font-size="28"><text x="220" y="341">↓</text><text x="690" y="341">↓</text><text x="220" y="499">↓</text><text x="690" y="499">↓</text></g>
<text x="30" y="654" font-size="16">One worked query is not the aggregate result. Score all 499 validation / 760 test queries.</text>
<text x="30" y="682" font-size="14">Actual primary predictions, rounded. Pipeline differences remain; human effort is a separate measurement.</text>
</g></svg>'''
(F/'pipelines.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(F/'pipelines.png'))
fig,axes=plt.subplots(1,2,figsize=(10,4.1),layout='constrained');fig.patch.set_facecolor('#f3f6f3')
for ax,split,title in zip(axes,['val','test'],['Validation','Test']):
 ax.set_facecolor('#f3f6f3');ax.spines[['top','right']].set_visible(False)
 for i,(arm,label,color) in enumerate([('fe_mae','Manual FE','#176a5c'),('rdl_mae','Basic RDL','#a5632c')]):
  vals=[x[arm] for x in r['rows'] if x['split']==split];m=r['metrics'][split][arm]
  ax.scatter(i+np.linspace(-.17,.17,5),vals,color=color,zorder=3)
  ax.errorbar(i+.24,m['mean'],yerr=m['sample_sd'],fmt='D',capsize=5,color=color,label='Mean ± seed SD' if i==0 else None)
 ax.set(xticks=[0,1],xticklabels=['Manual FE','Basic RDL'],ylabel='MAE · finishing positions ↓',title=title,xlim=(-.5,1.6));ax.grid(axis='y',alpha=.15)
 axes[0].legend(fontsize=9)
fig.suptitle('Fresh full-data fits · 5 runs per arm · separate split scales',fontsize=15);save(fig,'scores')
fig,ax=plt.subplots(figsize=(9,5.2),layout='constrained');fig.patch.set_facecolor('#f3f6f3');ax.set_facecolor('#f3f6f3')
labels=['FE · marginal','RDL · marginal','RDL · first task','RDL · 10 tasks'];active=[2,.5,.5,.5];shared=[0,0,1.5,.15];y=np.arange(4)
ax.barh(y,active,color='#176a5c',label='Task-specific active human work');ax.barh(y,shared,left=active,color='#b87835',label='Shared setup allocated per task')
for i,(a,s) in enumerate(zip(active,shared)):ax.text(a+s+.06,i,f'{a+s:.2f} h',va='center')
ax.set(yticks=y,yticklabels=labels,xlim=(0,2.7),xlabel='Human hours per task');ax.invert_yaxis();ax.spines[['top','right']].set_visible(False);ax.legend(loc='lower right',fontsize=9)
ax.set_title('SYNTHETIC: marginal ratio 4× → first-task ratio 1×',loc='left',fontsize=15,pad=20)
save(fig,'clocks')
print('Three teaching figures generated')
