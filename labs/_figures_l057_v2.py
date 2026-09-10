"""Generate portable computation traces plus plots from measured artifacts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from relkit.cross_ensemble import binary_loss,greedy_select
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures/l057';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fffefb','axes.facecolor':'#fffefb'})
C=['#24728b','#b06027','#6b5da7','#65727b','#4f865f','#bf4e56']
def save(fig,name):fig.savefig(OUT/(name+'.png'),dpi=170,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(8,7));ax.axis('off');ax.set(xlim=(0,10),ylim=(0,10))
def box(x,y,w,h,s,color='#edf3f5'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.1',facecolor=color,edgecolor='#667580'));ax.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=11)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.6,color='#52636e'))
box(.3,8.3,4.1,1.1,'Development rows [N,d]\nSplit into K=3 OOF folds')
box(5.5,8.3,4.1,1.1,'Outer test features [N_test,d]\nLabels withheld until scoring')
box(.3,6.3,4.1,1.1,'For each fold f: FIT other folds\nPreprocess/context only FIT rows')
box(5.5,6.3,4.1,1.1,'PREDICT f and test features\nNo early stopping on f')
arrow((2.3,8.3),(2.3,7.4));arrow((4.4,6.8),(5.5,6.8));arrow((7.5,8.3),(7.5,7.4))
box(.3,4.15,9.3,1.35,'XGB trees     |     TabM-mini, k8 members     |     TabICL v1.1 context\nEach branch returns P(y=1), not labels or logits\nTabM: [B,8,2] logits → softmax → member mean → [B]')
arrow((7.5,6.3),(7.5,5.5))
box(.3,1.95,4.1,1.2,'Scatter held-out rows → Z [N,3]\nEvery row once per family\nZ + y_dev → greedy weights w [3]')
box(5.5,1.95,4.1,1.2,'Average test probabilities over K\nZ_test [N_test,3]\nApply frozen w → p_test [N_test]')
arrow((2.3,4.15),(2.3,3.15));arrow((7.5,4.15),(7.5,3.15));arrow((4.4,2.3),(5.5,2.3))
ax.text(5,.65,'Only now: test labels + p_test → log loss\nTabICL pretraining is external; no pretrained weights are learned here.',ha='center',va='center',fontsize=11)
ax.set_title('Trace one held-out row through the complete stack',fontsize=15,pad=14);save(fig,'architecture')
y=np.array([0,0,1,1]);a=np.array([.1,.8,.9,.2]);b=np.array([.8,.1,.2,.9]);p=(a+b)/2
fig,axes=plt.subplots(2,1,figsize=(7,6),gridspec_kw={'height_ratios':[1,1.4]});axes[0].axis('off')
rows=[[str(y[i]),f'{a[i]:.2f}',f'{b[i]:.2f}',f'{p[i]:.2f}'] for i in range(4)]
tab=axes[0].table(cellText=rows,colLabels=['y','A','B','½A + ½B'],loc='center',cellLoc='center');tab.scale(1,1.7);tab.auto_set_font_size(False);tab.set_fontsize(12)
idx=np.arange(4)
for j,(v,n) in enumerate([(a,'A alone'),(b,'B alone'),(p,'Mixture')]):
 losses=-(y*np.log(v)+(1-y)*np.log1p(-v));axes[1].bar(idx+(j-1)*.24,losses,.24,label=n,color=C[j])
axes[1].set(xticks=idx,xticklabels=['row 1','row 2','row 3','row 4'],ylabel='Per-row log loss (nats)',ylim=(0,2));axes[1].legend(ncol=3,loc='upper center');fig.suptitle('Synthetic: equal-quality models, different mistakes',fontsize=14);fig.tight_layout();save(fig,'diversity')
_,tr=greedy_select(np.column_stack([a,b]),y,3)
fig,ax=plt.subplots(figsize=(7,4));ax.axis('off');total=np.zeros(4);rows=[]
for r in tr:
 total+=[a,b][r['chosen']];v=total/r['step'];rows.append([r['step'],f"{r['losses'][0]:.4f}",f"{r['losses'][1]:.4f}",'AB'[r['chosen']],', '.join(f'{p:.2f}' for p in v)])
tb=ax.table(cellText=rows,colLabels=['Step','Try A: loss','Try B: loss','Choose','Mixture probabilities'],loc='center',cellLoc='center',colWidths=[.09,.16,.16,.12,.47]);tb.auto_set_font_size(False);tb.set_fontsize(10);tb.scale(1,2.4)
ax.set_title('Synthetic greedy trace: try each new average',fontsize=14);ax.text(.5,.08,'Best prefix: step 2 → counts [1,1] → weights [.5,.5]\nStep 3 can be worse; keep the best prefix, not the last step.',ha='center',transform=ax.transAxes);save(fig,'selection')
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
fig,ax=plt.subplots(figsize=(8,7));ax.axis('off');ax.set(xlim=(0,10),ylim=(0,10))
items=[(8.1,'OTHER DATASETS A + B','Learn ordered portfolio: configurations 1…200','Do not use evaluation dataset C to choose this order.'),(5.5,'DATASET C: DEVELOPMENT ROWS','Fit candidate library; construct aligned OOF matrix Z','Each row excluded from the base fit predicting it.'),(2.9,'DATASET C: OOF LABELS','40 selection steps → best prefix → frozen weights w','These labels TRAIN the combiner; its OOF loss is training loss.'),(.3,'DATASET C: OUTER TEST ROWS','Fold-average family probabilities → Z_test @ w → loss','Only score after choices are frozen. Reused test = exploratory.')]
for y,head,op,note in items:
 ax.add_patch(FancyBboxPatch((.2,y),9.6,1.6,boxstyle='round,pad=.12',facecolor='#eef4f7',edgecolor='#667580'))
 ax.text(5,y+1.24,head,ha='center',fontsize=12,fontweight='bold');ax.text(5,y+.77,op,ha='center',fontsize=11);ax.text(5,y+.28,note,ha='center',fontsize=9)
 if y>.3:ax.annotate('',xy=(5,y-.85),xytext=(5,y-.17),arrowprops=dict(arrowstyle='->',lw=2))
fig.tight_layout();fig.savefig(OUT/'portfolio.png',dpi=170,bbox_inches='tight')

path=ROOT/'_verify_l057_v2_results.json'
if path.exists():
 r=json.loads(path.read_text());s=r['summary'];names=list(s['table']);arms=s['arms']
 fig,axes=plt.subplots(1,len(names),figsize=(11,4),sharey=False)
 for ax,name in zip(axes,names):
  for j,arm in enumerate(arms):
   vals=[row['errors'][arm] for row in r['rows'] if row['dataset']==name]
   ax.scatter(np.full(len(vals),j)+np.linspace(-.08,.08,len(vals)),vals,color=C[j],s=22)
   v=s['table'][name][arm];ax.errorbar(j,v['mean'],yerr=v['sd'],fmt='_',color=C[j],capsize=3,markersize=12)
  ax.set_title(name.replace('_',' '));ax.set_xticks(range(len(arms)),arms,rotation=60,ha='right');ax.set_ylabel('Test log loss ↓');ax.grid(axis='y',alpha=.15)
 fig.suptitle('Measured: three model seeds; bars = sample SD (fixed rows/folds)',fontsize=13);fig.tight_layout();save(fig,'results_v2')
 fig,axes=plt.subplots(2,1,figsize=(8,6),gridspec_kw={'height_ratios':[1,1.2]});ax=axes[0]
 for i,n in enumerate(names):
  v=s['table'][n]['paired_gap'];lo,hi=v['ci95'];ax.errorbar(v['mean'],i,xerr=np.array([[v['mean']-lo],[hi-v['mean']]]),fmt='o',color=C[0],capsize=4)
 ax.axvline(0,color='#333',ls='--');ax.set(yticks=range(len(names)),yticklabels=names,xlabel='Stack − OOF-selected single: test log-loss gap\nPaired seed 95% t intervals; negative favors stack')
 ax=axes[1];rank=s['mean_ranks']
 ax.scatter(list(rank.values()),range(len(rank)),color=C);ax.set(yticks=range(len(rank)),yticklabels=list(rank),xlim=(.8,len(rank)+.2),xlabel='Mean rank of dataset mean errors (lower is better)')
 cd=s['nemenyi_cd'];ax.plot([1,1+cd],[-1,-1],color='#333',lw=3);ax.text(1,-1.45,f'Nemenyi CD={cd:.3f}; Friedman p={s["friedman_p"]:.3f}',fontsize=10);ax.set_ylim(len(rank)-.5,-2)
 fig.tight_layout();save(fig,'uncertainty_v2')
 html=ROOT.parent/'lessons/0057-cross-family-ensembling.html';text=html.read_text()
 table='<table><thead><tr><th>Dataset</th>'+''.join('<th>'+a+'</th>' for a in arms)+'</tr></thead><tbody>'
 for n in names:table+='<tr><td>'+n+'</td>'+''.join(f'<td>{s["table"][n][a]["mean"]:.4f} ± {s["table"][n][a]["sd"]:.4f}</td>' for a in arms)+'</tr>'
 table+='</tbody></table>'
 block='<details id="corrected-results-reveal"><summary>Reveal corrected v2 three-family results</summary><p>Corrected hybrid library: 27 new TabM fold fits; unchanged archived XGB/TabICL probabilities from 54 earlier fit-context constructions. Reused outer test; exploratory. Loss means ± sample SD; all family and combination scores use identical outer-test rows. '+f'Correction elapsed {r["seconds"]:.1f} CPU wall seconds (archive training excluded). INCOMPARABLE to the paper.</p><p>Scroll horizontally to inspect all columns and plots; each chart also opens at full size.</p><div class="ensemble-scroll" role="region" aria-label="Corrected model results" tabindex="0">'+table+'</div><figure tabindex="0"><a href="../labs/figures/l057/results_v2.png"><img class="ensemble-figure" src="../labs/figures/l057/results_v2.png" alt="Measured per-seed test log loss for each family and combiner"></a><figcaption>Points show every seed; bars are sample SD, not confidence intervals. Independent y-axis zooms expose within-dataset differences.</figcaption></figure><figure tabindex="0"><a href="../labs/figures/l057/uncertainty_v2.png"><img class="ensemble-figure" src="../labs/figures/l057/uncertainty_v2.png" alt="Paired test loss gaps with conditional seed intervals and mean ranks with critical difference"></a><figcaption>Dataset means are ranked after seed averaging. Three datasets provide little power.</figcaption></figure></details>'
 import re
 text=re.sub(r'<!-- L057-MEASURED -->.*?<!-- /L057-MEASURED -->','<!-- L057-MEASURED -->',text,flags=re.S)
 text=text.replace('<!-- L057-MEASURED -->','<!-- L057-MEASURED -->'+block+'<!-- /L057-MEASURED -->');html.write_text(text)
print('Figures generated; measured figures require main results.')
