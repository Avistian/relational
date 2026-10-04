"""Build the B19a lesson, reference and portable source-visible notebooks."""
import ast,base64,gzip,hashlib,json,re
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;R=P.parent;S='b19a-predictive-distributions';F=P/'figures/b19a'
r=json.loads((P/'evidence/b19a/diagnostic.json').read_text());paper=json.loads((P/'evidence/b19a/reproduction.json').read_text())
(R/'assets/b19a-evidence.js').write_text('/* Complete finite course diagnostic, not model results. */\nwindow.B19A_EVIDENCE='+json.dumps(r,separators=(',',':'))+';\n')
plt.rcParams.update({'font.size':12,'figure.facecolor':'#fafcf9','axes.facecolor':'#fafcf9','axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(3,1,figsize=(7,8),sharex=True)
for ax,(name,f) in zip(axes,r['forecasts'].items()):
 v=f['values'];p=f['probabilities'];cum=[0]
 for mass in p:cum.append(cum[-1]+mass)
 ax.step([-5,*v,5],[*cum,1],where='post',color='#20684f',lw=2.5,label='Forecast CDF');ax.step([-5,0,5],[0,1,1],where='post',color='#a55b1d',ls='--',lw=2,label='Observed CDF at y=0');ax.set_ylim(-.05,1.1);ax.set_yticks([0,.5,1]);ax.set_ylabel('Probability');ax.set_title(name.capitalize()+' · mean 0',loc='left',weight='bold');ax.grid(alpha=.15)
axes[0].legend(fontsize=10,loc='upper left');axes[-1].set_xlabel('Outcome z · target units');fig.tight_layout();fig.savefig(F/'cdf.png',dpi=145);plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(7,6.2))
names=[x['forecast'].capitalize() for x in r['summary']];colors=['#5781a2','#20684f','#b27a36']
axes[0].bar(names,[x['crps'] for x in r['summary']],color=colors);axes[0].set_ylabel('Expected CRPS ↓');axes[0].set_title('Same RMSE √2; different distribution quality',loc='left',fontsize=13,weight='bold')
for i,x in enumerate(r['summary']):axes[0].text(i,x['crps']+.05,f"{x['crps']:.2f}",ha='center')
axes[0].set_ylim(0,2.5)
axes[1].scatter([x['width'] for x in r['summary']],[x['covered']*100 for x in r['summary']],s=100,c=colors)
for x in r['summary']:axes[1].annotate(x['forecast'].capitalize(),(x['width'],x['covered']*100),xytext=(8,0),textcoords='offset points',fontsize=11)
axes[1].set_xlim(0,11);axes[1].set_ylim(0,115);axes[1].set_xlabel('Mean central 50% interval width · target units');axes[1].set_ylabel('Truth-weighted coverage %');axes[1].axhline(50,color='#888',ls='--');axes[1].set_title('Coverage alone rewards uninformative width',loc='left',fontsize=13);fig.tight_layout();fig.savefig(F/'scores.png',dpi=145);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,7));ax.axis('off')
steps=[('1  Legal features + frozen support','TFM → bin masses / quantiles\nQuantile tree → predicted quantiles'),('2  Frozen distribution interface','Native output → CDF F(z) → mean and Q(q)\nDeclare units, interpolation and tail treatment'),('3  Separate calibration partition','Fixed predictor + calibration labels → interval adjustment\nFinal labels cannot choose this adjustment'),('4  Untouched evaluation partition','F and y → CRPS; mean and y → squared error\nInterval and y → score, coverage and width'),('5  Paired evidence','Same row IDs → fold means → dataset effects / ranks\nMissing values retain their identity')]
for i,(title,desc) in enumerate(steps):
 y=.93-i*.2;ax.text(.04,y,title+'\n'+desc,transform=ax.transAxes,fontsize=11,va='top',bbox=dict(boxstyle='round,pad=.7',facecolor='#edf5f0' if i!=2 else '#fff1dc',edgecolor='#94b3a4'))
 if i<4:ax.text(.5,y-.14,'↓',transform=ax.transAxes,ha='center',fontsize=20,color='#20684f')
fig.tight_layout();fig.savefig(F/'pipeline.png',dpi=145,bbox_inches='tight');plt.close(fig)
def figure(name,caption,portable=False):
 src='data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/b19a/'+name+'.png'
 return f'<figure class="evidence-figure" tabindex="0" role="region" aria-label="{caption}"><img src="{src}" alt="{caption}" style="min-width:560px;max-width:100%;height:auto"><figcaption>{caption} On narrow screens, scroll horizontally.</figcaption></figure>'
pipe='''<section class="evidence-flow" aria-label="Distribution evaluation architecture"><div class="evidence-step"><strong>1 · Legal features and frozen support</strong>TFM: bin masses / quantiles. Quantile tree: quantile predictions. Preserve both model identities.</div><div class="evidence-arrow" aria-hidden="true">↓</div><div class="evidence-step"><strong>2 · Common distribution interface</strong>Native outputs → CDF F(z) → mean and Q(q). Freeze units, interpolation, tails and any quantile repair.</div><div class="evidence-arrow" aria-hidden="true">↓</div><div class="evidence-step boundary"><strong>3 · Separate calibration partition</strong>Frozen predictor + calibration labels → interval adjustment. Final labels have no arrow into this step.</div><div class="evidence-arrow" aria-hidden="true">↓</div><div class="evidence-step"><strong>4 · Untouched evaluation</strong>F,y → CRPS; mean,y → squared error; interval,y → interval score / coverage / width. Pair row keys, average folds inside datasets, then compare datasets.</div></section>'''
widget='''<section class="evidence-board distribution-board" data-distribution-board><h3>Predict the score, then change the forecast</h3><div class="evidence-controls"><label>Forecast <select name="forecast"><option value="narrow">Narrow</option><option value="calibrated">Calibrated</option><option value="wide">Wide</option></select></label><label>Observed outcome <select name="outcome"><option value="-2">−2</option><option value="0" selected>0</option><option value="2">2</option></select></label><button type="button">Reset</button></div><div class="distribution-chart" tabindex="0" role="region" aria-label="Scrollable forecast CDF"><svg viewBox="0 0 620 255" role="img" aria-label="Forecast and observed CDF"></svg></div><p>Green solid: forecast CDF. Orange dashed: observed CDF. CRPS integrates the <strong>squared</strong> vertical gap; shaded area alone is not CRPS.</p><output aria-live="polite">Baseline narrow forecast at y=0: CRPS 0.5, interval score 2. The full static table below remains available.</output><noscript><p>Enable JavaScript to vary the forecast; the worked trace and figures contain the complete baseline.</p></noscript></section>'''
cal='''<section class="evidence-board" data-calibration-board><h3>Change final labels; keep calibration fixed</h3><div class="evidence-controls"><label>Final evaluation outcomes <select><option value="0">Original 0, 7, 8, 20</option><option value="100">Add 100 to every outcome</option></select></label><button type="button">Reset</button></div><output aria-live="polite">Fixed calibration residuals 0,…,8 give radius 7. Original final coverage is 50%; shifted coverage is 0%. The fitted radius stays 7.</output><noscript><p>The adjacent worked example remains readable without JavaScript.</p></noscript></section>'''
table='| Forecast | RMSE ↓ | Expected CRPS ↓ | Expected interval score ↓ | Coverage | Width ↓ |\n|---|---:|---:|---:|---:|---:|\n'
for x in r['summary']:table+=f"| {x['forecast'].capitalize()} | {x['rmse']:.4f} | {x['crps']:.2f} | {x['interval_score']:.2f} | {100*x['covered']:.0f}% | {x['width']:.0f} |\n"
paper_table='| Target | Models × datasets | Published fields matched | Leading mean rank |\n|---|---:|---:|---|\n'
for m,c in paper['comparisons'].items():
 winner=min(c['rows'],key=lambda x:x['actual']['meanrank']);paper_table+=f"| {m.upper()} | 38 × {c['datasets']} | 266 / 266 | {winner['model']}: {winner['actual']['meanrank']:.6f} |\n"
body=(R/'lessons/content'/f'{S}.md').read_text()
def fill(text,portable=False):
 for k,v in {'PIPELINE':figure('pipeline','Distribution-to-score computation and its calibration boundary.',True) if portable else pipe,'DISTRIBUTION_WIDGET':'**Predict first:** calculate the narrow forecast at y=0, then vary the notebook inputs.' if portable else widget,'CALIBRATION_WIDGET':'**Intervention:** change only final labels and verify that the calibrated radius stays fixed.' if portable else cal,'RESULTS':table,'PAPER_TABLE':paper_table,'CDF_FIG':figure('cdf','Three forecast CDFs share mean zero; the observed CDF jumps at y=0.',portable),'SCORE_FIG':figure('scores','Exact finite expectation: CRPS distinguishes equal means; coverage alone rewards width.',portable)}.items():text=text.replace('{{'+k+'}}',v)
 return text

def document(title,body,interactive=False):
 html=render(body).replace('<table>','<div class="evidence-table" tabindex="0" role="region" aria-label="Scrollable comparison table"><table>').replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','b19a-evidence','distribution-scores'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{s}.css">' for s in ['lesson','benchmark-evidence','distribution-scores'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../notebooks.html">Notebook gallery</a></nav>'+html+'</article>'+''.join(f'<script src="../assets/{s}.js"></script>' for s in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document('B19a · Predictive distributions',fill(body),True))
reference='''# B19a · Predictive distribution reference

[Lesson](../lessons/b19a-predictive-distributions.html) · [Notebook](../labs/b19a-predictive-distributions.ipynb) · [Metric contract](../labs/b19a-metric-contract.md)

| Quantity | Evaluates | Boundary |
|---|---|---|
| RMSE | Error of a mean prediction; target units | Blind to uncertainty when means agree |
| CRPS | Full CDF; E abs(X−y) − ½ E abs(X−X′) | Independent X,X′; lower is better; target units |
| Interval score | Two interval quantiles; width + miss penalty | Fixed alpha and endpoint convention |
| Coverage | Fraction inside interval | Can improve by becoming uninformatively wide |
| Sharpness | Concentration; interval width here | Must be interpreted alongside calibration |
| Split calibration | Adjustment fitted on held-out residuals | Keep final labels and adaptive selection separate |

**Hand trace:** alpha=.2, [−1,1], y=2 → width2 +10×miss1 =12. Narrow ±1 forecast at y=0 → CRPS1−½×1=.5. A point forecast has CRPS=absolute error.

**Generalized inverse:** Q(q)=inf{z:F(z)≥q}. For masses(.25,.5,.25) at(−2,0,2), Q(.25)=−2 and Q(.75)=0. Closed 50% interval covers 75% of this discrete truth. Interval propriety elicits quantiles, not the full distribution.

**Calibration:** k=ceil((n+1)(1−alpha)), radius=kth absolute residual; if k>n use infinity. Nine residuals 0…8, alpha=.2 → radius7. Exchangeability and a fixed predictor support a marginal guarantee; this is not conditional coverage or arbitrary-shift robustness.

'''+table+'''

**Comparison contract:** paired row identities; frozen train/validation/calibration/test partitions; versioned TFM and probabilistic-tree outputs; equal selection budgets; primary score chosen before final labels; quantile interpolation/tails; target units and inverse transforms; dataset-level paired effects; explicit missing scores and cost cutoff.

'''+paper_table+'''

Evidence: complete selected published-table reconstruction from 18,480 released scores, not raw-prediction rescoring or fresh training. Full environment/checkpoint identity and whole paper remain unestablished. [Full evidence contract](../labs/b19a-reproduction.md).

Primary sources: [ScoringBench](https://arxiv.org/html/2603.29928v3), [Distributional regression](https://arxiv.org/html/2603.08206v1), [Conformal introduction](https://arxiv.org/abs/2107.07511). Ask the teacher to review your metric contract; revisit in 1/7/30 days.
'''
(R/'reference'/f'{S}.html').write_text(document('B19a · Scoring reference',reference))
source=(P/'relkit/scoring_b19a.py').read_text();parts={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
tests=(P/'_test_b19a.py').read_text().split('class ScoringTests',1)[1].split("if __name__",1)[0];tests='class ScoringTests'+tests
repro=(P/'_reproduce_b19a.py').read_text();replay=next(ast.get_source_segment(repro,n) for n in ast.parse(repro).body if isinstance(n,ast.FunctionDef) and n.name=='replay')
blob=gzip.compress((P/'evidence/b19a/released-scores.json').read_bytes(),mtime=0);digest=hashlib.sha256(blob).hexdigest()
recap=fill(body.split('## Lab ·')[0],True);recap=re.sub(r'<div id="b19a-[^"]+"></div>','',recap);recap=re.sub(r'\]\(\.\./([^)]*)\)',r'](https://avistian.github.io/relational/\1)',recap);recap=recap.replace('](b19-benchmark-evidence.html)','](https://avistian.github.io/relational/lessons/b19-benchmark-evidence.html)')
goals={'crps':'Validate finite support and nonnegative unit mass; use the two expectation terms. Do not sample or call a hidden scoring library.','interval_score':'Validate ordered endpoints and 0<alpha<1; implement width plus both outside-distance penalties.','calibrate':'Accept unique calibration IDs only. Compute the finite-sample residual order statistic. Reject test records; handle k>n as infinity.'}
for solution in [False,True]:
 cells=[]
 def md(s):cells.append(nb.v4.new_markdown_cell(s))
 def py(s):cells.append(nb.v4.new_code_cell(s))
 md(recap);md('## Run contract\n\nSelf-contained Python standard library; no downloads, installs, credentials or repository imports. Embedded figures are author reference measurements. Code below performs a fresh finite diagnostic and replays saved author scores. It never trains a benchmark model. Live Colab frontend NOT_CHECKED. Optional lesson hyperlinks become live only after publication; core execution is offline.')
 py('# @colab-bootstrap\nimport math,itertools,json,unittest,base64,gzip,hashlib\nfrom pathlib import Path')
 md('## PROVIDED · hand-computable tests\n\nRun a CHECK after each implementation. The full experiment calls all three learner functions.');py(tests)
 for name,test in [('crps','test_crps'),('interval_score','test_interval'),('calibrate','test_calibration')]:
  md('## '+('SOLUTION' if solution else 'TODO')+' · '+name+'\n\n'+goals[name]);py(parts[name] if solution else parts[name].split('\n')[0]+f'\n    raise NotImplementedError("Implement {name}")');md('### CHECK');py(f"ScoringTests('{test}').debug()\nprint('{name}: passed')")
 md('## PROVIDED · quantiles and complete finite experiment\n\nRead the data flow: quantiles come from the left inverse CDF; calibration receives no final test labels. All 9 forecast/outcome pairs, 15 probability triples and 18 affine interventions are retained.');py(parts['quantile']);py(parts['run_experiment'])
 py("report=run_experiment()\nPath('b19a-diagnostic.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\\n')\nfor row in report['summary']: print(row)\nassert len(report['propriety_grid'])==15\nassert all(x['crps_scaled']==x['expected'] and x['interval_scaled']==x['interval_expected'] for x in report['unit_interventions'])\nassert report['calibration']['radius_unchanged']\nprint('Calibration:',report['calibration'])")
 md('## Replay · complete saved fold scores\n\n18,480 rows from 38 historical model tables. Packet hash establishes embedded-byte identity; repository verification independently compares every row with authenticated original Parquet bytes. Missing values are None. Portable replay recomputes all mean ranks and medians, not the autorank confidence/effect columns. No raw predictions or fresh model fits are available here.')
 py('packet=base64.b64decode('+repr(base64.b64encode(blob).decode())+')\nassert hashlib.sha256(packet).hexdigest()=='+repr(digest)+'\nrecords=json.loads(gzip.decompress(packet))\nassert len(records)==18480')
 py(replay)
 py("ranking=replay(records)\nPath('b19a-replay.json').write_text(json.dumps(ranking,indent=2)+'\\n')\nfor metric,data in ranking.items():\n    assert len(data['models'])==38 and len(data['datasets'])==97\n    print(metric, data['rows'][:2], 'Dropped:',data['dropped_datasets'])")
 md('## Paired dataset effects · inspect before ranking\n\nCompare released TabICLv2 and quantile CatBoost on the same 97 datasets. Negative CRPS difference favors TabICLv2. Units differ across datasets: retain each raw effect rather than treating their mean as an application cost. These are saved results; this comparison is not the new contract you will preregister.')
 py("from collections import defaultdict\npaired=defaultdict(dict)\nfor dataset,model,fold,c,r2,l in records:\n    if dataset in ranking['crps']['datasets'] and model in ['tabiclv2','catboost_quantile']:\n        paired[dataset].setdefault(model,[]).append(c)\neffects=[{'dataset':d,'tabiclv2_minus_catboost_crps':sum(v['tabiclv2'])/5-sum(v['catboost_quantile'])/5} for d,v in sorted(paired.items())]\nassert len(effects)==97\nPath('b19a-paired-effects.json').write_text(json.dumps(effects,indent=2)+'\\n')\nfor row in effects: print(row)")
 md('## EXIT · freeze a new metric-and-calibration contract\n\nSpecify a paired version-pinned TFM and probabilistic/quantile tree, accessible inputs, untouched evaluation IDs, separate tuning/calibration data, equal tuning budgets, primary score, RMSE/CRPS/interval diagnostics, quantile extraction/tails, units, dataset effects, missing-run policy and total cost cap. State exchangeability assumptions and what would falsify your thesis. Explain why all 798 published table fields matching does not establish fresh training. Ask the teacher for review; revisit after 1/7/30 days. Learner PENDING_WRITTEN_DEFENSE.')
 n=nb.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python3',language='python'),language_info=dict(name='python',version='3.12')));nb.write(n,P/'solutions'/f'{S}.ipynb' if solution else P/f'{S}.ipynb')
print('Built lesson, reference, two notebooks and three portable figures')
