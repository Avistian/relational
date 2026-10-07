"""Build a reproducible regression lesson and self-contained live notebooks."""
import ast,base64,hashlib,json,re,zlib
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l152';S='0152-regression-portfolio';TITLE='Regression portfolio: accurate predictions, honest calibration'
def defs(path):
 s=path.read_text();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
functions=defs(P/'relkit/regression_l152.py');checks=defs(P/'_check_l152.py');summary=json.loads((E/'summary.json').read_text())
results='| Seed | Selected epoch | Selection MAE | Final val MAE | Test MAE | Test RMSE | Test bias |\n|---|---:|---:|---:|---:|---:|---:|\n'
for r in summary['records']:results+=f"| {r['seed']} | {r['selected_epoch']} | {r['selection_mae']:.6f} | {r['val_mae']:.6f} | {r['test_mae']:.6f} | {r['test_rmse']:.6f} | {r['test_bias']:+.6f} |\n"
m=summary['metrics'];results+=f"\n**Five fresh fits:** validation MAE **{m['val']['mean']:.6f} ± {m['val']['sample_sd']:.6f}**; test MAE **{m['test']['mean']:.6f} ± {m['test']['sample_sd']:.6f}**; test RMSE **{m['rmse']['mean']:.6f} ± {m['rmse']['sample_sd']:.6f}**. Mean ± sample seed SD, finishing-position units. Test mean: **{m['paper_score']}** under the frozen ±0.20 descriptive rule. These are new runs, not earlier-lesson scores.\n"
labels=json.loads((E/'label-audit.json').read_text());budget=json.loads((P/'_budget_l152.json').read_text());ceiling=sum(r['upper_usd'] for r in budget['reservations'])+budget['overhead_reserve_usd']
audit=f"All **{labels['independently_rebuilt_labels']:,} labels** were independently reconstructed from raw result rows. All **{summary['predictions']:,} primary predictions** were aligned to the archive and independently rescored. Every training/evaluation batch was audited: **{summary['query_occurrences']:,} query occurrences**, zero owner-cutoff violations. Original-model held-out output comparisons passed, maximum discrepancy **{summary['maximum_original_output_error']:.3g}**.\n\nFirst-backward nonfinite-gradient counts by seed: **{summary['first_backward_nonfinite']}**. We retained the released computation. Exact source structure and a small local neural gradient/update check passed; they do not certify healthy optimization on real data.\n\nPrimary measured worker-body cost estimate **USD {summary['worker_body_usd']:.6f}** excludes unitemized overhead. Current reservations plus the USD3 overhead allowance total **USD {ceiling:.6f}** under the aggregate USD10 cap. This is a conservative resource allowance, not an itemized invoice. [Evidence](../labs/evidence/l152/summary.json) · [Budget ledger](../labs/_budget_l152.json).\n"
notebook_report=P/'_notebook_l152_results.json'
if notebook_report.exists():
 nr=json.loads(notebook_report.read_text());audit+=f"\nThe archived pre-correction portable full GPU gate executed **{nr['cells']} code cells** and **five additional ten-epoch fits**, independently rescoring **{nr['predictions']} predictions** in an isolated pinned runtime. Those runs are separate from the primary mean. A later reporting-only correction makes the ±0.20 endpoints inclusive within floating-point roundoff; its CHECK and default CPU notebook were rerun. Archived code and source hashes identify the earlier GPU execution; model and trainer are unchanged. Live Colab and deployment remain NOT_CHECKED.\n"
captions={'architecture':'Complete released regression pipeline with a worked scalar neighbor-sum trace. Scalar weights are illustrative; actual learned transformations have 128 coordinates.','loss':'The same skewed outcomes have median 1 and mean 2.75. L1 and squared error ask for different predictions.','calibration':'Measured seed 0 held-out bins, with edges frozen from validation predictions. Strict fractions retain ties; counts accompany the diagnostic.','results':'Five complete primary runs with mean and sample seed SD. The band is descriptive closeness, not equivalence.'}
def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results).replace('[[AUDIT]]',audit)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l152/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l152/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption} Scroll horizontally on narrow screens.</figcaption></figure>')
 fallback='Static tie example: targets0,1,1,9; prediction1; below/equal/above=.25/.50/.25; median violation0; MAE2.25. Predicting2.75 changes MAE to3.125 and violation to.25.'
 s=s.replace('[[CALIBRATION_WIDGET]]',fallback if portable else '<div class="route-widget" id="l152-calibration"></div><noscript>'+fallback+'</noscript>')
 s=s.replace('[[WARMUP]]','Answer the retrieval questions below before continuing.' if portable else '<div id="warmup"></div>')
 s=s.replace('[[PREDICT]]','Predict before reading: does zero empirical median violation guarantee future conditional calibration? Explain why not.' if portable else '<div id="predict"></div><noscript>Zero empirical median violation does not guarantee future conditional calibration.</noscript>')
 s=s.replace('[[TEACHBACK]]','Write your explanation before reviewing the teacher solution. Submit it to the teaching agent.' if portable else '<div id="teachback"></div>')
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s
def doc(title,body,interactive=False):
 body=re.sub(r'<table([^>]*)>',r'<div class="route-scroll" tabindex="0"><table\1>',render(body)).replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','regression-calibration-viz','l152-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 152 — '+title+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','atomic-route','checkpoint','regression-calibration'])+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0151-classification-portfolio.html">Lesson 151</a></nav><header><p class="route-kicker">Year 4 · Quarter 4 · Lesson 152</p><h1>'+title+'</h1></header>'+body+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(doc(TITLE,prose(),True))
reference='''## Regression entry contract
Query population + units → pinned full procedure → validation-only checkpoint → complete five-seed metrics → validation-fixed diagnostic bins → audits/cost → bounded claim.

| Quantity | Definition | Interpretation |
|---|---|---|
| MAE | mean(abs(p-y)) | Primary absolute error; L1 targets median |
| RMSE | sqrt(mean((p-y)^2)) | Secondary diagnostic; large misses matter more |
| Bias | mean(p-y) | Positive means overprediction |
| Median inequalities | P(y<p)≤.5 and P(y>p)≤.5 | Ties remain separate |
| Median violation | max(0,below-.5,above-.5) | Empirical bin diagnostic, not population proof |

Targets0,1,1,9 predicted as1 give below/equal/above=.25/.50/.25, mean(y-p)=1.75 and median violation0. Empty bins have undefined statistics. Freeze unique validation prediction quintiles before test; exact-edge predictions go right. Repeated driver queries are dependent.

## Full procedure
Seeds0–4, ten full epochs, Adam.005, batch512, channels128, two sum-GraphSAGE layers, uniform128/64fanouts, L1, train2nd/98thpercentile clipping. First strict validation minimum selects; final resampling can change validation MAE. No post-test changes or fitted correction.

## Measured result
'''+results+'\n## Evidence\n'+audit+'''
## Boundaries and next action
Historical identity/feature-arrival legality NOT_ESTABLISHED; whole paper/fresh manual-FE NOT_RUN; learner PENDING_WRITTEN_DEFENSE. Prior F1 test exposure disclosed. Mean±seedSD is not future-population uncertainty.

[Lesson](../lessons/0152-regression-portfolio.html) · [Protocol](../labs/l152-reproduction.md) · [Template](../labs/l152-entry-template.md) · [RelBench Table7](https://arxiv.org/html/2407.20060v1#A2.T7) · [Calibration primary source](https://arxiv.org/abs/2108.03210).
'''
(R/'reference/regression-portfolio.html').write_text(doc('Regression portfolio reference',reference))
files={str(p.relative_to(P)):p.read_text() for p in sorted((P/'sources/l117').glob('*')) if p.is_file()}
for name in ['_run_l117.py','_run_l152.py','relkit/rdl_l117.py','relkit/batch_audit_l123.py','relkit/regression_l152.py','sources/l152/protocol.json','requirements-l117-runtime.txt']:files[name]=(P/name).read_text()
payload={}
for seed in range(5):
 raw=(E/f'paper/seed-{seed}/predictions.npz').read_bytes();payload[str(seed)]=dict(data=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest())
packet='''# PROVIDED: author evidence is labeled; hashes check its embedded bytes.
SOURCE_FILES=json.loads(zlib.decompress(base64.b64decode('''+repr(base64.b64encode(zlib.compress(json.dumps(files).encode())).decode())+''')))
P=Path.cwd()
for name,text in SOURCE_FILES.items():
    dest=P/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
AUTHOR_SUMMARY='''+repr(summary)+'''
PAYLOAD='''+repr(payload)+'''
'''
bootstrap='''# @colab-bootstrap: default audit; full training requires the pinned GPU runtime.
import os,sys,subprocess
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable,'-m','pip','install','-q','pytorch-frame==0.2.3','torch-geometric==2.6.1','relbench==1.1.0','sentence-transformers==3.3.1'])
from pathlib import Path
import math,statistics,json,io,base64,hashlib,zlib
import numpy as np,pandas as pd
from IPython.display import display
RUN_FULL_REPRODUCTION=False
'''
core=(P/'relkit/rdl_l117.py').read_text()
prep=(P/'_run_l117.py').read_text().split("if __name__=='__main__':")[0].replace('from relkit.rdl_l117 import make_pkey_fkey_graph,fit_rdl\n','').replace('P=Path(__file__).resolve().parent','P=Path.cwd()').replace('def run(seed,epochs,output):','def prepare_and_fit(seed,epochs,output):').replace('    from relkit.rdl_l117 import foreign_key_edges\n','')
audit_code=(P/'relkit/batch_audit_l123.py').read_text()
runner=(P/'_run_l152.py').read_text().split("if __name__=='__main__':")[0]
for line in ['import relkit.rdl_l117 as model_module\n','import _run_l117 as original_runner\n','from relkit.batch_audit_l123 import audit_batch\n','    from relkit.regression_l152 import keyed_metrics,median_diagnostics\n']:runner=runner.replace(line,'')
runner=runner.replace('def run(seed,epochs,output):','def run(seed,epochs,output):\n    global NeighborLoader,Model').replace('model_module.','').replace('original_runner.run(','prepare_and_fit(').replace('Path(__file__)',"(P/'_run_l152.py')")
scoring='''# CHECK: live functions evaluate every primary author prediction.
records=[];all_bins={};count=0
for r in AUTHOR_SUMMARY['records']:
    item=PAYLOAD[str(r['seed'])];raw=base64.b64decode(item['data']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
    arrays=np.load(io.BytesIO(raw));metrics={}
    for split in ['val','test']:
        q=[dict(entity=int(e),time=int(t),target=float(y)) for e,t,y in zip(arrays[split+'_entity'],arrays[split+'_time'],arrays[split+'_target'])]
        pred=[dict(entity=int(e),time=int(t),prediction=float(v)) for e,t,v in zip(arrays[split+'_entity'],arrays[split+'_time'],arrays[split+'_pred'])]
        metrics[split]=keyed_metrics(q,list(reversed(pred)));count+=metrics[split]['n']
        assert abs(metrics[split]['mae']-r[split+'_mae'])<1e-12
    edges=np.unique(np.quantile(arrays['val_pred'],[.2,.4,.6,.8])).tolist()
    bins={s:median_diagnostics(arrays[s+'_target'],arrays[s+'_pred'],edges) for s in ['val','test']}
    assert bins==AUTHOR_SUMMARY['bins'][str(r['seed'])]
    all_bins[str(r['seed'])]=dict(edges=edges,bins=bins)
    records.append(dict(r,val_mae=metrics['val']['mae'],test_mae=metrics['test']['mae'],test_rmse=metrics['test']['rmse']))
metrics=portfolio_summary(records);assert metrics==AUTHOR_SUMMARY['metrics'];assert count==6295
display(pd.DataFrame(records));display(pd.DataFrame(all_bins['0']['bins']['test']))
entry=dict(evidence_owner='AUTHOR_PACKET_INDEPENDENTLY_RESCORED',task='rel-f1/driver-position',protocol=json.loads(SOURCE_FILES['sources/l152/protocol.json']),records=records,metrics=metrics,diagnostics=all_bins,source_hashes=AUTHOR_SUMMARY['hashes'],written_defense=None)
Path('l152-portfolio-entry.json').write_text(json.dumps(entry,indent=2))
print('AUTHOR_PACKET_INDEPENDENTLY_RESCORED: 6295 predictions; learner PENDING_WRITTEN_DEFENSE')
'''
gate='''# PROVIDED: full five-seed execution, separate from the author-evidence lane.
if RUN_FULL_REPRODUCTION:
    import importlib.metadata as md
    for name,version in {'torch':'2.5.1','torch-geometric':'2.6.1','pytorch-frame':'0.2.3','relbench':'1.1.0','numpy':'1.26.4','pandas':'2.2.3','sentence-transformers':'3.3.1'}.items():
        assert md.version(name).split('+')[0]==version,(name,'Pinned runtime required')
    assert torch.cuda.is_available(),'Use pinned CUDA runtime with native pyg-lib; author used T4 and16GiB host RAM'
    root=Path('l152-own-runs');root.mkdir(exist_ok=False);fresh=[];fresh_bins={}
    for seed in range(5):
        output=root/f'seed-{seed}';r=run(seed,10,output)
        a=np.load(output/'predictions.npz');score={}
        for split in ['val','test']:
            q=[dict(entity=int(e),time=int(t),target=float(y)) for e,t,y in zip(a[split+'_entity'],a[split+'_time'],a[split+'_target'])]
            pred=[dict(entity=int(e),time=int(t),prediction=float(v)) for e,t,v in zip(a[split+'_entity'],a[split+'_time'],a[split+'_pred'])]
            score[split]=keyed_metrics(q,list(reversed(pred)));assert abs(score[split]['mae']-r['scores'][split])<1e-12
        fresh.append(dict(seed=seed,epochs=r['epochs'],complete=True,val_mae=score['val']['mae'],test_mae=score['test']['mae'],test_rmse=score['test']['rmse']))
        fresh_bins[str(seed)]=json.loads((output/'diagnostics.json').read_text())
    own=dict(evidence_owner='OWN_FRESH_RUN',task='rel-f1/driver-position',records=fresh,metrics=portfolio_summary(fresh),diagnostics=fresh_bins,protocol=json.loads(SOURCE_FILES['sources/l152/protocol.json']),written_defense=None)
    Path('l152-own-run-entry.json').write_text(json.dumps(own,indent=2));print(own['metrics'])
else:print('Default lane: author evidence audit; your fresh training NOT_RUN.')
'''
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson 152 · '+TITLE+'\n\n'+('Teacher solution' if solution else 'Student lab')+' · PROVIDED / TODO / CHECK / EXIT. Three live functions score keyed predictions, diagnose median balance and summarize complete seed evidence. The default CPU path audits author evidence; the post-EXIT pinned GPU gate trains five new full-data fits.'),nb.v4.new_code_cell(bootstrap),nb.v4.new_code_cell(packet,metadata={'tags':['data-payload']})]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if not section.strip():continue
  cells.append(nb.v4.new_markdown_cell(section))
  if section.startswith('## 2'):
   for part in re.split(r'(?=^# %% )',core,flags=re.M):
    if part.strip():cells.append(nb.v4.new_code_cell('# PROVIDED: visible released pipeline stage\n'+part))
  for number,name,check in [('3','keyed_metrics','check_metrics'),('4','median_diagnostics','check_diagnostics'),('5','portfolio_summary','check_summary')]:
   if section.startswith('## '+number):
    code=functions[name] if solution else functions[name].split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'
    cells.append(nb.v4.new_code_cell('# TODO: this live function is called by both evidence lanes.\n'+code))
    cells.append(nb.v4.new_code_cell('# CHECK: run unchanged.\n'+checks['rejects']+'\n'+checks[check]+'\n'+check+'('+name+')\nprint("PASS '+name+'")'))
  if section.startswith('## 5'):cells.append(nb.v4.new_code_cell(scoring))
 cells.append(nb.v4.new_markdown_cell('## Post-EXIT · Complete fresh reproduction appendix\n\nThe full visible model/trainer above and preprocessing/audits below execute the released experiment. Source and MIT license files are embedded, then written into this notebook’s working directory. RUN_FULL_REPRODUCTION=False by default. Enable it only in the pinned CUDA runtime with native pyg-lib; install using the included requirements and the operator image recipe. The author uses T4 +2CPU +16GiB host RAM. Local notebook execution has no monetary guard; the supplied Modal operator reserves cost before dispatch. Output directories are single-use. Five complete fits are required; no subsets count as the full experiment.'))
 cells.extend([nb.v4.new_code_cell('# PROVIDED: fresh archive verification, text encoding and graph construction\n'+prep),nb.v4.new_code_cell('# PROVIDED: every sampled occurrence is checked against its original graph\n'+audit_code),nb.v4.new_code_cell('# PROVIDED: instrumented full training invokes your functions\n'+runner),nb.v4.new_code_cell(gate)])
 cells.append(nb.v4.new_code_cell("report=dict(status='PASS',predictions=count,full_training='FIVE_FRESH_FITS' if RUN_FULL_REPRODUCTION else 'NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')\nPath('l152-report.json').write_text(json.dumps(report,indent=2))\nprint(report)"))
 for i,c in enumerate(cells):c.id=f'l152-{i:03d}'
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}});path=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and path.exists():
  old=nb.read(path,4);notebook.metadata=old.metadata
  for new,prev in zip(notebook.cells,old.cells):
   if new.cell_type==prev.cell_type=='code' and new.source==prev.source:new.outputs=prev.outputs;new.execution_count=prev.execution_count;new.metadata=prev.metadata
 nb.write(notebook,path)
print('Built lesson/reference and portable student/solution notebooks')

# Refresh the prepared HTML after prose/figure-only changes, preserving verified code outputs.
from nbconvert import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor
executed=nb.read(P/'solutions'/f'{S}.ipynb',4)
if all(c.execution_count is not None for c in executed.cells if c.cell_type=='code'):
 exporter=HTMLExporter(template_name='lab');exporter.register_preprocessor(TagRemovePreprocessor(remove_input_tags={'data-payload'}),enabled=True)
 html,_=exporter.from_notebook_node(executed);(P/'html'/f'{S}.html').write_text(html)
