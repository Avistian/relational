"""Deterministic HTML, reference and portable learner/solution notebook builder."""
import ast,base64,hashlib,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l181';S='0181-relbench-v2-autocomplete'
r=json.loads((E/'report.json').read_text());pins=json.loads((E/'input-manifest.json').read_text())
def defs(path):
 s=path.read_text();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
fns=defs(P/'relkit/autocomplete_l181.py')
status='**Observed:** complete two-task aggregation-baseline reproduction. Fresh GNN training **NOT_RUN_TRAINING_HEALTH_GATE**; the approved baseline-plus-GNN experiment **INCOMPLETE**. RelGT-AC numerical reproduction **NOT_RUN_SOURCE_GAPS**. New cloud/API spend **$0**. Learner: **PENDING_WRITTEN_DEFENSE**.'
rows=['| Task / test recipe | Measured R² | Measured MAE |','|---|---:|---:|']
for task,d in r['tasks'].items():
 for kind in ['global_mean','global_median','entity_mean']:
  v=d['scores']['test'][kind];rows.append(f"| {task} / {kind.replace('_',' ')} | {v['r2']:.6f} | {v['mae']:.6f} |")
results='\n'.join(rows)+'\n\nEntity median equals entity mean here; both equal global zero. The full validation/test table is in the measured report.'
captions={'masking':'Synthetic query and context rows under two masking policies. Future rows are excluded independently of target masking. Retained context targets change the information set.','gnn':'Complete released GNN path and its source fitting boundaries. Shapes describe the intended model; no full sampled GNN update ran in L181.','relgt-ac':'Paper-described RelGT-AC architecture and an illustrative attention calculation. This is not authenticated implementation or executed model evidence.','baselines':'Measured complete test MAE for all five deterministic recipes. The row-identity entity baselines fall back to zero; no GNN result or seed uncertainty is implied.'}
def prose(portable=False):
 s=(R/'lessons/content'/(S+'.md')).read_text().replace('[[STATUS]]',status).replace('[[RESULTS]]',results)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/'figures/l181'/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l181/'+name+'.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 replacements={'WARMUP':('From memory: explain query cutoff, validation-only selection and complete (entity,time) keys. Distinguish source replay from fresh training.','<div id="warmup"></div>'),'PREDICT':('Predict before continuing: with no shared row IDs between fit and test, does entity mean retrieve driver history or use the zero fallback? Write your reason.','<div id="predict"></div>'),'EXPLORER':('Try the visibility cases in the notebook below: keep query time 10, change context time 9→11, and compare global with seed_only masking.','<div id="autocomplete-visibility"></div><noscript>At query time 10, global masking hides position and points in both query and past context rows. Query-only masking retains past context values 2 and 18. A future row at 11 is excluded in both cases.</noscript>'),'TEACHBACK':('Write your defense in the EXIT fields and ask the teacher to review it.','<div id="teachback"></div>')}
 for name,(plain,widget) in replacements.items():s=s.replace('[['+name+']]',plain if portable else widget)
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def doc(title,body,interactive=False):
 html=re.sub(r'<table([^>]*)>',r'<div class="route-scroll" tabindex="0"><table\1>',render(body)).replace('</table>','</table></div>')
 css=['lesson','atomic-route','checkpoint','lab-access','autocomplete-visibility'];scripts=['retrieval-pool','retrieval-bank','predict','teachback','autocomplete-visibility','l181-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 181 — '+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in css)+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0180-public-encoder-checkpoint.html">Lesson 180</a></nav><header><p class="route-kicker">Year 5 · Quarter 3 · Lesson 181</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Autocomplete without reading the answer',prose(),True))
reference='''**Task contract:** identify table, primary key, query time, target, excluded proxies, context availability, fitting population, selection metric and evaluation split. Autocomplete predicts a hidden existing value; forecasting derives a future target. Neither implies foundation-model transfer.

**Masking:** the released RelBench v2 baseline removes target/proxies throughout the target table. RelGT-AC describes masking only the query row. Temporal filtering is independent: a masked future row can still expose future metadata. Correlation alone does not establish leakage; record availability and the task's declared exclusions.

**F1:** results-position uses resultId; qualifying-position uses qualifyId. These are not driver IDs. Complete keys are (row ID, time). Counts are train/validation/test: results 8,997/1,400/4,798; qualifying 2,228/1,854/5,733. Source intervals exclude the lower bound and include the upper bound; retain the exact boundary details in the protocol.

**Baseline fit:** train for validation; train plus validation for test. Global mean/median are constants. Entity mean/median aggregate by task row identity, then fall back to zero for unseen keys. All validation/test keys are new here. A driver-history join would be another model.

**Metrics:** MAE is mean absolute error; lower is better. R² = 1 − SSE/SST, where SSE is squared prediction error and SST is squared distance from the evaluation truth mean. Negative R² is possible. Align full keys before scoring. Deterministic recipes need no artificial seed error bars.

**GNN source:** typed ResNet row encoders of width 128 → two sum-aggregating GraphSAGE layers → query linear head. Adam learning rate 0.005, batch 512, ten epochs, fanouts 128/64, L1 loss. Validation MAE selects the checkpoint; training percentiles 2/98 clip predictions. Source graph statistics use all dates; later temporal sampling does not undo fitting-scope exposure.

**RelGT-AC paper:** seed masking → typed/text features → five additive row terms → local GraphSAGE → global Transformer → task head. TF-IDF has 64 features; the experiment setup specifies text projection width 32 (the general equation instead writes hidden width). Hidden width 128, two Transformer layers, eight heads. Source not authenticated; keep paper claims separate. Its quoted comparisons are validation-based, not reproduced test results.

**Observed:** all 25,010 labels and 68,925 predictions checked independently; 40 paper baseline metric comparisons within 0.001. The numerical encoder component fails gradients on real missing values under current CPU dependencies. No repaired full model, GPU pilot or GNN fit. The selected experiment remains INCOMPLETE; learner PENDING_WRITTEN_DEFENSE.

'''+results+'\n\n'+status+'''

[Lesson](../lessons/0181-relbench-v2-autocomplete.html) · [Student lab](../labs/0181-relbench-v2-autocomplete.ipynb) · [Protocol](../labs/l181-reproduction.md) · [Measured report](../labs/evidence/l181/report.json) · [RelBench paper](https://arxiv.org/html/2602.12606v1) · [RelGT-AC paper](https://arxiv.org/html/2606.03040v1).
'''
# Avoid compressed prose in the reference by separating adjacent digits and words.
reference=reference.replace('results8997/1400/4798','results 8,997/1,400/4,798').replace('qualifying2228/1854/5733','qualifying 2,228/1,854/5,733').replace('all 25,010 labels and68925predictions','all 25,010 labels and 68,925 predictions').replace('40paper','40 paper').replace('within.001','within 0.001')
(R/'reference/relbench-v2-autocomplete.html').write_text(doc('Autocomplete task contract — quick reference',reference))
(E/'report.md').write_text('# L181 measured baseline evidence\n\n'+status+'\n\n'+results+'\n')
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for name in sorted(pins['files']):
  info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,(E/'packet'/name).read_bytes())
encoded=base64.b64encode(buf.getvalue()).decode();digest=hashlib.sha256(buf.getvalue()).hexdigest()
appendix=['sources/l181/upstream/examples/model.py','sources/l181/upstream/relbench/modeling/nn.py','sources/l181/upstream/examples/gnn_autocomplete.py','sources/l181/upstream/examples/baseline_autocomplete.py','sources/l181/upstream/relbench/base/task_autocomplete.py','sources/l181/LinearEncoder.py','_run_l181.py']

def make(solution):
 cells=[]
 def md(s):cells.append(nb.v4.new_markdown_cell(s))
 def code(s,tags=None):cells.append(nb.v4.new_code_cell(s,metadata={'tags':tags or []}))
 md('# Lesson 181 · RelBench v2 autocomplete\n\nMirror scope: real full F1 task/baseline reproduction and information-access contracts. Three live learner functions; original GNN model/trainer visible after EXIT. Fresh GNN training stopped at a numerical encoder prerequisite. This is author evidence, not your completed work.\n\nTier B: real relational F1 data embedded with hashes. Dependencies: Python3, numpy, pandas, pyarrow, duckdb and scikit-learn. The default lab needs no data downloads or cloud calls; install missing packages in your environment before running. Live Colab is NOT_CHECKED.')
 code('# @colab-bootstrap\nfrom pathlib import Path\nimport json,copy\nimport numpy as np\nimport pandas as pd\nimport pyarrow,duckdb,sklearn\nprint("L181: complete baseline replay; fresh GNN fit NOT_RUN; cloud $0")',['colab-bootstrap'])
 md(prose(True))
 md('## PROVIDED · Authenticate the embedded data\nSHA256 authenticates bytes. The packet contains all nine raw tables, all six task splits, all 20 prediction files, source excerpts and the observed gradient receipt. It does not contain a trained GNN.')
 code('import base64,hashlib,io,zipfile\nPACKET='+repr(encoded)+'\nraw=base64.b64decode(PACKET)\nassert hashlib.sha256(raw).hexdigest()=='+repr(digest)+"\nP=Path('l181-packet');P.mkdir(exist_ok=True)\nwith zipfile.ZipFile(io.BytesIO(raw)) as z:\n    for name in z.namelist():assert not Path(name).is_absolute() and '..' not in Path(name).parts\n    z.extractall(P)\nmanifest="+repr(pins)+"\nprint('Authenticated packet:',len(manifest['files']),'files')",['data-payload'])
 tasks=[('visible_columns','Hide the intended cells','Return columns in original order. For policy global, hide target and proxies in every row. For seed_only, hide them only when seed is true. Reject unknown policies or duplicate column names. This function does not filter rows by timestamp; keep that separate.'),('baseline_predictions','Make fitting scope explicit','fit contains entity,time,y; query contains entity,time. Return a NumPy float prediction array in query order for global_zero, global_mean, global_median, entity_mean or entity_median. Group entity recipes by entity only; unseen entities use zero. Reject unknown recipes and empty/nonfinite fit labels. The caller chooses train versus train+validation.'),('keyed_scores','Score the complete query population','truth has entity,time,y; predictions has entity,time,pred. Reject missing/duplicate identities, mismatched complete key sets, nonfinite values, fewer than two rows, or constant truth. Join on both identity fields; return n,mae,r2. Do not rely on file order or join only on entity.')]
 for name,title,instructions in tasks:
  md('## TODO · '+title+'\n'+instructions)
  node=ast.parse(fns[name]).body[0];signature='def '+name+'('+ast.unparse(node.args)+'):'
  code(fns[name] if solution else signature+'\n    raise NotImplementedError("Implement '+name+'")')
  # Per-function immediate feedback; final complete check still exercises interactions.
  simple={'visible_columns':"assert visible_columns(['position','points','grid'],'position',['points'],False,'global')==['grid']\nassert visible_columns(['position','grid'],'position',[],False,'seed_only')==['position','grid']",
  'baseline_predictions':"fit=pd.DataFrame({'entity':[1,1,2],'time':[1,2,1],'y':[2.,8.,11.]})\nquery=pd.DataFrame({'entity':[2,3,1],'time':[3,3,3]})\nassert np.array_equal(baseline_predictions(fit,query,'entity_mean'),[11,0,5])",
  'keyed_scores':"truth=pd.DataFrame({'entity':[1,1,2],'time':[1,2,2],'y':[1.,3.,5.]})\npred=pd.DataFrame({'entity':[2,1,1],'time':[2,1,2],'pred':[4.,2.,3.]})\nassert abs(keyed_scores(truth,pred)['r2']-.75)<1e-12"}
  md('### CHECK · '+title+'\nA passing toy check is preparation for the full-population audit below.');code(simple[name]+"\nprint('Immediate CHECK passed')")
 md('## CHECK · Adversarial contracts\nRepeated entity IDs at different times, shuffled files, missing rows, incorrect fallbacks and unknown policies must not silently pass.')
 code(defs(P/'_check_l181.py')['checks']);code("assert checks(visible_columns,baseline_predictions,keyed_scores)=='PASS'\nprint('All three live contracts PASS')")
 md('## PROVIDED · Complete selected baseline experiment\nThis audit invokes your three functions, reconstructs all 25,010 labels and reproduces all 68,925 predictions. Reading the saved score table alone does not satisfy it.')
 code(defs(P/'_audit_l181.py')['audit181']);code("report=audit181(P,manifest,visible_columns,baseline_predictions,keyed_scores)\nPath('l181-report.json').write_text(json.dumps(report,indent=2))\nprint(report['baseline_status'],report['predictions'],'predictions')\nprint(report['gnn_status'],report['selected_experiment'])\nprint(pd.DataFrame([{'task':t,'recipe':k,**v} for t,d in report['tasks'].items() for k,v in d['scores']['test'].items()]).to_string(index=False))")
 md('## PROVIDED + CHECK · An independent route\nSQL reconstructs the target populations from raw parquet. Python scalar statistics regenerate predictions; sklearn independently calculates MAE and R². This checks the saved predictions independently of your implementation.')
 code(defs(P/'_verify_l181.py')['independent181']);code("independent=independent181(P,report)\nassert independent['labels']==25010 and independent['predictions']==68925\nassert independent['max_metric_error']<1e-10\nprint(independent)")
 md('## TRY · Change one assumption\nPredict first: when an entity occurs in validation but not training, does source-prescribed test refitting change its entity-mean prediction? Then compare. This synthetic case explains fit scope; the real F1 query IDs do not overlap.')
 code("fit=pd.DataFrame({'entity':[1,1],'time':[1,2],'y':[2.,8.]})\nval=pd.DataFrame({'entity':[2],'time':[3],'y':[7.]})\nq=pd.DataFrame({'entity':[2],'time':[4]})\na=baseline_predictions(fit,q,'entity_mean')[0]\nb=baseline_predictions(pd.concat([fit,val]),q,'entity_mean')[0]\nassert (a,b)==(0,7)\nprint('Train only:',a,'Train+validation:',b)\nassert visible_columns(['position','points','grid'],'position',['points'],False,'global')==['grid']\nassert visible_columns(['position','points','grid'],'position',['points'],False,'seed_only')==['position','points','grid']")
 md('## TRY · Reject incorrect work and corrupted evidence\nOne wrong function predicts zero for every recipe. Another packet has a changed source digest. Explain both failures.')
 code("try:checks(visible_columns,lambda fit,q,k:np.zeros(len(q)),keyed_scores)\nexcept AssertionError:print('Wrong baseline rejected')\nelse:raise AssertionError('Wrong baseline accepted')\nbad=copy.deepcopy(manifest);bad['files']['source/examples/gnn_autocomplete.py']='0'*64\ntry:audit181(P,bad,visible_columns,baseline_predictions,keyed_scores)\nexcept ValueError:print('Changed source rejected')\nelse:raise AssertionError('Corrupted source accepted')")
 md('## EXIT · Defend your task contract\nComplete these in your own words and ask the teaching agent to review them. No automated cell awards personal mastery. The failed GNN prerequisite remains separate from your baseline implementation.')
 code("submission={'learner':'PENDING_WRITTEN_DEFENSE','row_identity':'','mask_scope_and_temporal_cutoff':'','baseline_fit_population':'','negative_r2_meaning':'','observed_gradient_failure_and_limit':'','relgt_ac_comparison_gap':''}\nPath('l181-submission.json').write_text(json.dumps(submission,indent=2))")
 md('## NEXT STEP · Full GNN reproduction gate\nDefault OFF. This is an executable local admission check, not a cloud-training implementation. The full pinned trainer appears below; the post-gate cloud operator is NOT_VALIDATED. A repaired encoder would require a newly declared protocol. Raising the budget alone cannot fix the scientific gate.')
 code("RUN_PAPER_REPRO=False\nif RUN_PAPER_REPRO:\n    fresh=audit181(P,manifest,visible_columns,baseline_predictions,keyed_scores)\n    if fresh['gradient_failures']:\n        raise RuntimeError('BLOCKED: authenticated numerical encoder gradient prerequisite fails; no dispatch')\n    raise RuntimeError('Changed prerequisites require review; post-gate cloud operator NOT_VALIDATED')\nelse:\n    print('Full GNN reproduction gated OFF; NOT_RUN_TRAINING_HEALTH_GATE')")
 md('## Appendix · Original model, trainer and task source\nArchived verbatim for inspection. These source blocks are not executed by this notebook. The lab implements information-access and evaluation mechanisms from scratch; it does not claim a from-scratch GNN or RelGT-AC replica. Library-backed operations remain explicit in the original imports. The full task builder has an expensive per-second date_range; the approved baseline path proves equivalent SQL extrema without constructing it.')
 for name in appendix:md('### '+name+'\n```python\n'+(P/name).read_text()+'\n```')
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(cells):c.id=f'l181-{i:03}'
 return book
for solution in [False,True]:
 book=make(solution);path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
     old=nb.read(path,4);book.metadata=old.metadata
     previous=[c for c in old.cells if c.cell_type=='code'];current=[c for c in book.cells if c.cell_type=='code']
     if [c.source for c in previous]==[c.source for c in current]:
         for prior,c in zip(previous,current):
             c.outputs=prior.outputs;c.execution_count=prior.execution_count;c.metadata=prior.metadata
 nb.write(book,path)
print('Built L181 HTML, reference and both portable notebooks')
