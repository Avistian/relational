"""Build B19b with portable figures, visible implementation and complete saved-score replay."""
import ast,base64,gzip,hashlib,json,re
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,Rectangle
import numpy as np
from relkit.forecast_b19b import source_periods
P=Path(__file__).resolve().parent;R=P.parent;S='b19b-forecasting-contracts';F=P/'figures/b19b';E=P/'evidence/b19b'
r=json.loads((E/'diagnostic.json').read_text());paper=json.loads((E/'reproduction.json').read_text())
plt.rcParams.update({'font.size':11,'figure.facecolor':'#fbfcfa','axes.facecolor':'#fbfcfa','axes.spines.top':False,'axes.spines.right':False})
# End-to-end model and actual feature/target information boundary.
fig=plt.figure(figsize=(8,10));gs=fig.add_gridspec(3,1,height_ratios=[2,1.7,2.3]);ax=fig.add_subplot(gs[0]);ax.axis('off');ax.set_title('TabPFN-TS · one series, all future rows together',loc='left',weight='bold',fontsize=15)
columns=['time','sin phase','cos phase','legal z','target'];rows=[['0','0','1','known','y₀'],['1','0.5','0.87','known','y₁'],['…','…','…','…','…'],['C','known','known','as-of','HIDDEN'],['C+1','known','known','as-of','HIDDEN']]
table=ax.table(cellText=rows,colLabels=columns,loc='center',cellLoc='center',bbox=[0,.12,1,.72]);table.auto_set_font_size(False);table.set_fontsize(11)
for (i,j),cell in table.get_celld().items():
 cell.set_edgecolor('#b4c5bd');cell.set_facecolor('#e3efe8' if i<4 else '#fff0dc')
 if i>=4 and j==4:cell.set_text_props(color='#a34c21',weight='bold')
ax.text(0,-.02,'C context rows + H future rows · illustrative period 12\nReal recipe: index + calendar + history-fitted periods + legal covariates',transform=ax.transAxes,fontsize=11)
ax=fig.add_subplot(gs[1]);ax.axis('off');ax.set_title('Feature fitting stops at the origin',loc='left',weight='bold')
for x,title,detail,color in [(0,'Observed history','y[0:C] → detrend → window\nFFT peaks → periods p₁,…,pₖ','#e3efe8'),(.53,'Frozen future mapping','t → [sin(2πt/p), cos(2πt/p)]\nC × D support; H × D queries','#fff0dc')]:
 ax.add_patch(Rectangle((x,.28),.46,.53,facecolor=color,edgecolor='#7e9d8e'));ax.text(x+.025,.69,title,weight='bold',fontsize=12);ax.text(x+.025,.48,detail,fontsize=10,va='center')
ax.annotate('',xy=(.53,.53),xytext=(.46,.53),arrowprops=dict(arrowstyle='->',color='#306f59',lw=2));ax.text(.02,.06,'No arrow from future targets to fitted periods or support labels.',color='#a34c21',fontsize=11)
ax=fig.add_subplot(gs[2]);ax.axis('off');ax.set_title('Frozen TabPFN-v2 regressor · schematic internal path',loc='left',weight='bold')
steps=[('Encode feature groups + observed target slots','(C+H) × (G+1) × d; query target values masked'),('Repeated: feature attention → row attention → FFN','Within-row mixing → support-to-query transfer → refinement'),('Query target tokens → target-bin probability head','H distributions → median and quantiles 0.1,…,0.9')]
for i,(title,detail) in enumerate(steps):
 y=.81-i*.31;ax.add_patch(Rectangle((.01,y-.14),.98,.23,facecolor='#e8f0f5',edgecolor='#7b9daf'));ax.text(.035,y+.005,title,fontsize=11,weight='bold');ax.text(.035,y-.085,detail,fontsize=10)
 if i<2:ax.annotate('',xy=(.5,y-.21),xytext=(.5,y-.14),arrowprops=dict(arrowstyle='->',lw=2,color='#306f59'))
fig.subplots_adjust(hspace=.38,top=.95,bottom=.04,left=.08,right=.96);fig.savefig(F/'architecture.png',dpi=145);plt.close(fig)
# History-only fitted seasonal coordinate trace.
fig,axes=plt.subplots(3,1,figsize=(8,7.8));t=np.arange(96);h=np.sin(2*np.pi*t/12);periods=source_periods(h);detrended=h-np.polyval(np.polyfit(t,h,1),t);windowed=detrended*np.hanning(96)
axes[0].plot(t,h,label='Observed history',color='#306f59');axes[0].plot(t,windowed,label='Detrend × Hann',color='#ae722d',alpha=.8);axes[0].legend(fontsize=10,ncol=2);axes[0].set_ylabel('Signal');axes[0].set_title('Fit on 96 observations; no future targets enter',loc='left',weight='bold')
freq=np.fft.rfftfreq(192);mag=np.abs(np.fft.rfft(np.pad(windowed,(0,96))));mag[0]=0
axes[1].plot(freq,mag,color='#306f59');axes[1].axvline(1/12,color='#ae722d',ls='--');axes[1].set_xlim(0,.25);axes[1].set_xlabel('Cycles per step (1/12 → period 12)');axes[1].set_ylabel('Magnitude');axes[1].set_title('Extract periods; preserve smaller source-selected peaks',loc='left',fontsize=12)
all_t=np.arange(108);axes[2].plot(all_t,np.sin(2*np.pi*all_t/12),label='sin',color='#306f59');axes[2].plot(all_t,np.cos(2*np.pi*all_t/12),label='cos',color='#4b79a1');axes[2].axvspan(95.5,107.5,color='#f5dcb8',alpha=.65);axes[2].set_xlim(78,108);axes[2].set_ylabel('Feature value');axes[2].set_xlabel('Time index · shaded = future coordinates, not future labels');axes[2].legend(fontsize=10,ncol=2);fig.tight_layout(h_pad=2);fig.savefig(F/'seasonality.png',dpi=145);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,3.8))
for i,o in enumerate([96,120,144]):
 ax.barh(i,o,left=0,height=.45,color='#719e88',label='Available history' if i==0 else None);ax.barh(i,12,left=o,height=.45,color='#e8b677',label='Hidden horizon' if i==0 else None);ax.text(o+6,i,str(o)+'–'+str(o+11),ha='center',va='center',fontsize=9);ax.text(o-2,i,str(o-1),ha='right',va='center',fontsize=9,color='white')
ax.set_yticks(range(3),['Origin96','Origin120','Origin144']);ax.invert_yaxis();ax.set_xlim(0,160);ax.set_xlabel('Time index · final observed index printed in green');ax.set_title('Forecast first, reveal targets later',loc='left',weight='bold',pad=48);ax.legend(loc='lower left',bbox_to_anchor=(0,1.03),ncol=2,fontsize=10);fig.tight_layout();fig.savefig(F/'origins.png',dpi=145);plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(8,6.5));labels=['Seasonal naive','Legal regression','Oracle weather\n(unavailable)'];colors=['#6d879a','#306f59','#ae722d']
for ax,metric in zip(axes,['mase','wql']):
 for i,row in enumerate(r['summary']):
  ax.scatter([i-.1,i,i+.1],[x[metric] for x in row['by_seed']],s=65,c=colors[i]);ax.plot([i-.19,i+.19],[row[metric]]*2,color=colors[i],lw=3)
 ax.set_xticks(range(3),labels);ax.set_ylabel(metric.upper()+' ↓');ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
axes[0].set_title('Course experiment · dots = seeds; line = mean',loc='left',weight='bold');fig.tight_layout(h_pad=2);fig.savefig(F/'scores.png',dpi=145);plt.close(fig)
def figure(name,caption,portable=False):
 src='data:image/png;base64,'+base64.b64encode((F/('architecture-refined.png' if name=='architecture' else name+'.png')).read_bytes()).decode() if portable else '../labs/figures/b19b/'+name+'.png'
 return f'<figure class="evidence-figure forecast-figure" style="max-width:100%;overflow-x:auto" tabindex="0" role="region" aria-label="{caption}"><img src="{src}" alt="{caption}" style="display:block;width:100%;min-width:680px;max-width:100%;height:auto"><figcaption>{caption} On narrow screens, scroll horizontally.</figcaption></figure>'
widget='''<section class="evidence-board" data-forecast-availability><h3>Which cells exist for this decision?</h3><p>Predict the legal inputs before changing the issue time or weather arrival.</p><div class="evidence-controls"><label>Issue time <select name="issue"><option>10</option><option>11</option><option>12</option></select></label><label>Weather input available at <select name="weather"><option value="12">12 · observed later</option><option value="8">8 · hypothetical early input</option></select></label><button type="button">Reset</button></div><div class="forecast-cells"></div><output aria-live="polite">At issue time10: calendar and promotion available; observed weather and delayed measurement unavailable.</output><noscript><p>At issue11 the delayed measurement becomes available; at12 observed weather also arrives. The target remains hidden in this feature-access example.</p></noscript></section>'''
results='| Arm | Mean MASE ↓ | Mean WQL ↓ | WQL by seed0 /1 /2 |\n|---|---:|---:|---|\n'
for x in r['summary']:results+=f"| {x['arm']} | {x['mase']:.4f} | {x['wql']:.4f} | "+' / '.join(f"{a['wql']:.4f}" for a in x['by_seed'])+' |\n'
pt='| Model | WQL rank: released / paper | Relative WQL: released / paper | Relative MASE: released / paper |\n|---|---:|---:|---:|\n'
for x in paper['rows']:pt+='| '+x['model']+' | '+' | '.join(f"{x[m]:.3f} / {x['published'][m]:.3f}" for m in ['mean_wql_rank','relative_wql','relative_mase'])+' |\n'
body=(R/'lessons/content'/f'{S}.md').read_text()
def fill(s,portable=False):
 items={'AVAILABILITY_WIDGET':'**Availability trace:** issue10 permits calendar and promotion only. At11 the delayed measurement arrives; at12 observed weather arrives. Future targets stay hidden.' if portable else widget,'ARCHITECTURE':figure('architecture','From historical observations to future feature rows and predictive distributions.',portable),'SEASONAL_FIG':figure('seasonality','Historical signal → spectral periods → future coordinates.',portable),'ORIGINS_FIG':figure('origins','Three expanding histories and complete twelve-step horizons.',portable),'RESULTS_FIG':figure('scores','All three seeds retained; oracle weather is unavailable at forecast issue time.',portable),'RESULTS':results,'PAPER_TABLE':pt}
 for k,v in items.items():s=s.replace('{{'+k+'}}',v)
 return s
def document(title,text,interactive=False):
 html=render(text).replace('<table>','<div class="evidence-table" tabindex="0" role="region" aria-label="Scrollable comparison table"><table>').replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','forecast-availability'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','benchmark-evidence','forecast-availability'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../notebooks.html">Notebook gallery</a></nav>'+html+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document('B19b · Forecasting contracts',fill(body),True))
reference='''# B19b · Forecast-time reference

[Lesson](../lessons/b19b-forecasting-contracts.html) · [Notebook](../labs/b19b-forecasting-contracts.ipynb) · [Task contract](../labs/b19b-task-contract.md)

| Contract | Rule | Failure case |
|---|---|---|
| Input availability | available_at ≤ issue_time | Future observed weather; late-arriving history |
| Origin o, horizon H | context[0:o]; query[o:o+H] | Random row split; partial final horizon |
| Fitted features | Estimate periods/scales on available history | FFT or normalization fitted on final labels |
| Future covariates | Known schedule or archived forecast vintage | Current revised record substituted for as-of state |
| MASE | Mean absolute median error / training seasonal difference | Scale0; claiming MASE<1 proves test-baseline win |
| WQL | Mean over q of2×pinball sum / sum abs(target) | Zero denominator; treating quantile grid as full CRPS |
| Uncertainty | Preserve series/dataset dependence | Treating97 related tasks as97 independent datasets |
| Evidence | Released scores ≠ raw predictions ≠ fresh inference | Choosing source files because they match |

**Worked trace:** history[1,3,2,4],s=1 → scale5/3. Actual[2,4],median[1,5] → MASE0.6 and median-only WQL1/3.

**Architecture:** C history rows with labels + H future feature rows with labels hidden → fixed TabPFNv2 grouped-feature/row attention → H target-bin distributions. D=28+legal covariates for the full wrapper. Source-visible seasonal fitting uses history only. No fresh TabPFN inference was run here.

**Falsification:** change all unavailable future values; legal features and forecasts must remain fixed. A pass checks that intervention, not every possible leak.

'''+results+'''

**Reproduction:** complete13×97 released score matrix reconstructed. Only2/39 printed figure fields match, both baseline identities. Figure4.1 historical identity and95% intervals remain INCOMPLETE; fresh model evaluation and pretraining NOT_RUN. [Full evidence contract](../labs/b19b-reproduction.md).

**Research exit:** include forecasting in L201 only with a decision/horizon, historical availability, paired legal baselines, frozen metric/aggregation and cost contract. Otherwise write a justified exclusion. Ask the teacher for review; revisit after1/7/30days.

Primary reading: [TabPFN-TS v4](https://arxiv.org/html/2501.02945v4).
'''
(R/'reference'/f'{S}.html').write_text(document('B19b · Forecast-time reference',reference))
source=(P/'relkit/forecast_b19b.py').read_text();parts={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
tests=(P/'_test_b19b.py').read_text().split('class ForecastTests',1)[1].split("if __name__",1)[0];tests='class ForecastTests'+tests
repro=(P/'_reproduce_b19b.py').read_text();replay=next(ast.get_source_segment(repro,n) for n in ast.parse(repro).body if isinstance(n,ast.FunctionDef) and n.name=='replay')
records=(E/'released-scores.json').read_bytes();blob=gzip.compress(records,mtime=0)
packet={str(p.relative_to(P/'sources/b19b')):base64.b64encode(p.read_bytes()).decode() for p in (P/'sources/b19b').rglob('*') if p.is_file()};sourceblob=gzip.compress(json.dumps(packet).encode(),mtime=0)
recap=fill(body.split('## 7 ·')[0],True);recap=re.sub(r'<div id="b19b-[^"]+"></div>','',recap);recap=re.sub(r'\]\(\.\./([^)]*)\)',r'](https://avistian.github.io/relational/\1)',recap);recap=re.sub(r'\]\((b\d[^)]*\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',recap)
for solution in [False,True]:
 cells=[]
 def md(s):cells.append(nb.v4.new_markdown_cell(s))
 def py(s):cells.append(nb.v4.new_code_cell(s))
 md(recap);md('## Run contract\n\nPython3 with NumPy and SciPy; no repository imports, downloads, credentials or model calls. All figures and score/source bytes are embedded. Install numpy/scipy in your own environment if absent; Colab usually provides them. Live Colab frontend NOT_CHECKED. PROVIDED scaffolding is complete; implement three TODO functions, run their CHECKs, then produce the EXIT artifact. Historical source files are evidence; they are not imported by core execution.')
 py('# @colab-bootstrap\nimport math,random,json,unittest,base64,gzip,hashlib\nfrom pathlib import Path\nimport numpy as np\nimport scipy\nprint("NumPy",np.__version__,"SciPy",scipy.__version__)')
 py(tests)
 for name,test,goal in [('available_features','test_availability','Select unique cells available at the issue time. Reject ambiguous identities and missing availability metadata.'),('rolling_origins','test_origins','Construct all complete chronological query horizons. Reject invalid lengths and a partial first horizon.'),('forecast_scores','test_scores','Compute the historical seasonal denominator and asymmetric pinball losses. Validate shape, finite values and ordered quantiles; fail on zero denominators.')]:
  md('## '+('SOLUTION' if solution else 'TODO')+' · '+name+'\n\n'+goal+' The full experiment calls this function; its result is not bypassed by a saved answer.')
  py(parts[name] if solution else parts[name].split('\n')[0]+f'\n    raise NotImplementedError("Implement {name}")');md('### CHECK · hand trace and invalid inputs');py(f"ForecastTests('{test}').debug()\nprint('{name}: passed')")
 md('## PROVIDED · visible paper-wrapper mechanism\n\nThese two functions mirror the archived default spectral extraction and build10seasonal columns. This is source code parity for selected cases, not pretrained model inference. Constant histories can produce numerical residual peaks. Inspect intermediate periods before trusting their meaning.')
 py(parts['source_periods']+'\n\n'+parts['source_seasonal_features'])
 py("history=np.sin(2*np.pi*np.arange(96)/12)\ntrain_features,query_features,periods=source_seasonal_features(history,12)\nassert train_features.shape==(96,10) and query_features.shape==(12,10)\nassert periods[0]==12\nprint('Learned source periods:',periods)")
 md('## PROVIDED · fixed course models and complete experiment\n\nRead the four-column legal model and five-column oracle. Future observed weather appears only in the declared illegal arm. In-sample residual quantiles have no coverage guarantee. No choice is made from final outcomes.')
 for name in ['temporal_features','fit_predict','run_experiment']:py(parts[name])
 py("report=run_experiment()\nassert len(report['predictions'])==324 and len(report['scores'])==27\nassert all(x['unavailable_feature_change_invariant'] and x['legal_predictions_invariant'] for x in report['interventions'])\nPath('b19b-diagnostic.json').write_text(json.dumps(report,indent=2)+'\\n')\nfor row in report['summary']:print(row)")
 md('## Saved-score replay · all13 models and97 tasks\n\nThis packet contains1,261original released task scores. It contains no raw per-series predictions. We recompute all39 aggregate values; the original figure remains source-gated, and no confidence interval method is invented. Hash checks bind the embedded bytes.')
 py('packed=base64.b64decode('+repr(base64.b64encode(blob).decode())+')\nassert hashlib.sha256(packed).hexdigest()=='+repr(hashlib.sha256(blob).hexdigest())+'\nrecords=json.loads(gzip.decompress(packed))')
 py(replay);py("reconstruction=replay(records)\nPath('b19b-replay.json').write_text(json.dumps(reconstruction,indent=2)+'\\n')\nprint('Tasks',reconstruction['tasks'],'models',reconstruction['models'])\nfor row in reconstruction['rows']:print(row)")
 md('## Source archive and full evaluation operator\n\nExtract the authenticated primary source packet for inspection. Core execution does not import its model dependencies. The optional operator writes the full55-configuration/97-task plan. Actual execution requires a separately prepared environment, local data/checkpoint and a new compute decision; it remains OFF. This path is a candidate-release evaluator, not an authenticated historical reproducer.')
 py('archive=base64.b64decode('+repr(base64.b64encode(sourceblob).decode())+')\nassert hashlib.sha256(archive).hexdigest()=='+repr(hashlib.sha256(sourceblob).hexdigest())+'\nfor name,value in json.loads(gzip.decompress(archive)).items():\n    relative=Path(name)\n    assert not relative.is_absolute() and ".." not in relative.parts\n    target=Path("sources/b19b")/relative\n    target.parent.mkdir(parents=True,exist_ok=True)\n    target.write_bytes(base64.b64decode(value))\nmanifest=json.loads(Path("sources/b19b/manifest.json").read_text())\nfor entry in manifest["files"]:\n    assert hashlib.sha256((Path("sources/b19b")/entry["file"]).read_bytes()).hexdigest()==entry["sha256"]\nprint("Authenticated source files",len(manifest["files"]))')
 py('operator='+repr((P/'_paper_b19b.py').read_text())+'\nPath("_paper_b19b.py").write_text(operator)\nimport subprocess,sys\nplan=json.loads(subprocess.check_output([sys.executable,"_paper_b19b.py","--plan"],text=True))\nassert plan["tasks"]==97\nRUN_FRESH_MODEL_EVALUATION=False\nif RUN_FRESH_MODEL_EVALUATION:\n    raise RuntimeError("Read b19b-reproduction.md and configure the explicit local operator; no model calls authorized by this notebook run.")\nprint(plan["status"],plan["tasks"],"tasks; no fresh inference")')
 md('## EXIT · defend a forecasting task\n\nWrite your issue timestamp, horizon, entity/time key, every feature’s availability, rolling validation/final windows, baseline, primary metric, zero policy and total compute cap. Include or exclude forecasting from L201 with a reason. Explain why oracle scores are not deployment evidence and why complete source replay did not reproduce Figure4.1. Ask the teacher for review; learner status remains PENDING_WRITTEN_DEFENSE.')
 n=nb.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python3',language='python'),language_info=dict(name='python',version='3.12')));nb.write(n,P/'solutions'/f'{S}.ipynb' if solution else P/f'{S}.ipynb')
print('Built B19b HTML/reference,4figures and portable notebooks')
