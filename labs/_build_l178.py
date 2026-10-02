"""Deterministic lesson/reference and self-contained full audit notebooks."""
import ast,base64,hashlib,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l178';S='0178-fair-model-comparison'
r=json.loads((E/'report.json').read_text());pins=json.loads((E/'audit-input-manifest.json').read_text())
def defs(path):
 s=path.read_text();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
fns=defs(P/'relkit/comparison_l178.py')
status='**Observed:** published-result replay **COMPLETE**; fresh three-model comparison **INCOMPLETE_TRAINING_HEALTH_GATE**. The local encoder probe has **256 nonfinite gradient entries**. No fresh benchmark model fits or inference and **$0 cloud/API spend**. Whole-paper reproduction and repaired-model training: **NOT_RUN**.'
def table():
 s='| Saved published arm | Replayed mean AUROC | Sample SD, 10 supports | Paper reference |\n|---|---:|---:|---:|\n'
 for a,x in r['replay']['models'].items():s+=f"| {a} | {x['mean']:.6f} | {x['sample_sd']:.6f} | {x['paper']:.4f} |\n"
 return s
captions={'routes':'Three planned pipelines with their actual internal operations. Q denotes the evaluation-query count; F denotes input feature count. All full fresh model runs are NOT_RUN.','gradient':'Actual raw support-history missingness triggers a local numerical-encoder gradient failure. The diagnostic repair is not a new benchmark result.','evidence':'The complete published prediction replay and the stopped fresh comparison support different claims.'}
def prose(portable=False):
 s=(R/'lessons/content'/(S+'.md')).read_text().replace('[[STATUS]]',status).replace('[[RESULTS]]',table())
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/'figures/l178'/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l178/'+name+'.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption} Scroll horizontally on a narrow screen.</figcaption></figure>')
 tags={'WARMUP':('Recall how validation selection differs from test reporting, and why equal support counts may hide extra labels.','<div id="warmup"></div>'),'PREDICT':('Predict: must you complement the labels, probabilities, or both to preserve AUROC under a change of positive class?','<div id="predict"></div>'),'EXPLORER':('Use your live comparison_gate function below to vary one contract assumption. A hypothetical pass never alters the real stopped evidence.','<div id="comparison-explorer"></div><noscript>Observed L178 stopped at a gradient-health failure. Matching the 30-day label horizon, explicit label access and validation selection still requires complete feature/time audits and a healthy training path. READY_FOR_PILOT is not a completed experiment.</noscript>'),'TEACHBACK':('Write the defense at the end of the notebook and ask the teacher for feedback.','<div id="teachback"></div>')}
 for tag,(plain,html) in tags.items():s=s.replace('[['+tag+']]',plain if portable else html)
 s=s.replace('[[CODE]]','Implement the comparison gate in the live TODO below.' if portable else '```python\n'+fns['comparison_gate']+'\n```')
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def doc(title,body,interactive=False):
 html=render(body).replace('<table>','<div class="route-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 css=['lesson','atomic-route','checkpoint','lab-access','comparison-gate'];scripts=['retrieval-pool','retrieval-bank','predict','teachback','comparison-gate','l178-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in css)+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0177-compute-budget-realism.html">Lesson 177</a></nav><header><p class="route-kicker">Year 5 · Quarter 2 · Lesson 178</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('FM vs tuned GNN vs RDBLearn',prose(),True))
reference='''**Compare contracts before scores.** Match the raw database, complete query keys, target window and positive class, permitted task labels, feature/temporal policies and selection procedure. Equal support counts and frozen weights do not establish equal label access.

**Positive-class conversion:** for complemented release labels, use `y_DNF = 1 − y_release` and `p_DNF = 1 − p_release`. Cast saved probabilities to float64 first. Change both or you reverse the interpretation of the score. The real F1 target is 30 days; 60 days changes 82 test labels.

**Tuned GNN:** record every configuration × seed × epoch, choose on validation, then freeze test evaluation. Reject missing/duplicate grid cells. Earliest epoch and declared configuration order break ties.

**Numerical health:** a finite output is insufficient. An affine operation on NaN followed by output masking may leave NaN parameter gradients. The actual local probe has 256. A repaired primitive is not a repaired benchmark model.

**Stop rules:** unmatched information or uncompleted audits block admission. The observed gradient failure stops training. An affordable plan does not override either. READY_FOR_PILOT admits only the next bounded diagnostic, never a scientific conclusion.

'''+status+'\n\n'+table()+'''

The table replays the original RDB-PFN experiment; it is not a RDBLearn-versus-GNN leaderboard. 30 runs, 21,060 predictions, all 702 queries. No new model inference. Historical availability and historical identity remain unestablished.

[Lesson](../lessons/0178-fair-model-comparison.html) · [Student notebook](../labs/0178-fair-model-comparison.ipynb) · [Full report](../labs/evidence/l178/report.json) · [Protocol](../labs/l178-reproduction.md) · [RDBLearn paper](https://arxiv.org/html/2602.18495v1) · [RelGNN paper](https://arxiv.org/html/2502.06784v2) · [RDB-PFN paper](https://arxiv.org/html/2603.03805v5).
'''
(R/'reference/fair-model-comparison.html').write_text(doc('Fair model comparison — quick reference',reference))
(E/'report.md').write_text('# L178 evidence summary\n\n'+reference+'\n')
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for name in sorted(pins['files']):
  info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,(P/name).read_bytes())
packet=base64.b64encode(buf.getvalue()).decode();digest=hashlib.sha256(buf.getvalue()).hexdigest()
def make(solution):
 cells=[]
 def md(s):cells.append(nb.v4.new_markdown_cell(s))
 def code(s,tags=None):cells.append(nb.v4.new_code_cell(s,metadata={'tags':tags or []}))
 md('# Lesson 178 · A fair comparison needs comparable evidence\n\nReconstruct all 30 published runs and all 12,679 raw task labels. Implement three live contracts; use them to justify the stopped fresh experiment. Default execution uses NumPy, pandas and pyarrow only; no network, GPU, model fit or paid job. Ask the teacher for help at any step.')
 code('from pathlib import Path\nimport json, math\nimport numpy as np\nimport pandas as pd\nprint("Complete saved-evidence replay; fresh comparison stopped; cloud/API $0")')
 md(prose(True))
 md('## PROVIDED · Authenticate every original input\nThe packet includes every prediction NPZ and original receipt, complete raw task/database files, model sources and the actual preflight evidence. SHA256 checks byte identity; it cannot prove historical information availability.')
 code('import base64,hashlib,io,zipfile\nPACKET='+repr(packet)+'\nraw=base64.b64decode(PACKET)\nassert hashlib.sha256(raw).hexdigest()=='+repr(digest)+"\nP=Path('l178-packet');P.mkdir(exist_ok=True)\nwith zipfile.ZipFile(io.BytesIO(raw)) as z:\n    for name in z.namelist():assert not Path(name).is_absolute() and '..' not in Path(name).parts\n    z.extractall(P)\npins="+repr(pins)+"\nprint('Authenticated packet:',len(pins['files']),'files')",['data-payload'])
 tasks=[('keyed_auc','Score the exact query population','Require unique two-part keys and identical key sets; align predictions by key, reject invalid probabilities, and compute tie-aware AUROC for binary labels. Different cutoffs for the same driver must remain distinct.',"keys=[[1,10],[1,20],[2,10],[3,10]]\nassert keyed_auc(keys,[0,1,1,0],keys[::-1],[.2,.8,.9,.1])==1.0\nprint('CHECK: reordered complete keys score correctly')"),('comparison_gate','Admit only matched, checked contracts','Require RDBPFN, RelGNN and RDBLearn. Match task, horizon_days, positive, query_digest, database_digest, support_digest and cost_ceiling. Require target_history NONE, temporal/preprocessing audits PASS and selection VALIDATION_ONLY for every arm. Cost ceilings must be finite positive numbers ≤1.5. RelGNN training_health must be PASS. Return status/reasons: missing arms INCOMPLETE_CONTRACT; a FAIL-prefixed health status INCOMPLETE_TRAINING_HEALTH_GATE; other gaps INCOMPLETE_COMPARABILITY_GATE; otherwise READY_FOR_PILOT.',"assert comparison_gate({})['status']=='INCOMPLETE_CONTRACT'\nprint('CHECK: a missing model cannot enter the comparison')"),('select_validation','Keep test out of tuning','Require exactly every expected configuration × seed × epoch, with finite validation_auc in [0,1]. Select the earliest maximum validation epoch per configuration and seed, then the configuration with greatest mean best validation AUROC over seeds. Listed configuration order breaks ties. Return config, epochs (string seed keys), mean_validation_auc. Never read test_auc.',"try:\n    select_validation([],['0.001','0.005'],[0,1,2],10)\nexcept ValueError:\n    print('CHECK: no winner from an empty tuning grid')\nelse:\n    raise AssertionError('Missing fits were accepted')")]
 for name,title,description,check in tasks:
  md('## TODO · '+title+'\n'+description)
  stub=fns[name].split('    """',1)[0]+'    raise NotImplementedError("TODO: '+name+'")'
  code(fns[name] if solution else stub);code(check)
 md('## CHECK · Behavioral and adversarial contracts\nYour actual functions control these checks. Tied scores, reordered identities, missing runs and tempting test metrics must not defeat the contract.')
 code(defs(P/'_check_l178.py')['checks']);code("assert checks(keyed_auc,comparison_gate,select_validation)=='PASS'\nprint('All three learner contracts passed')")
 md('## PROVIDED · Complete experiment and raw-label audit\nThis function calls your AUROC scorer for every original prediction file, your comparison gate for the observed failed contract, and your selector on the empty actual tuning grid. It independently reconstructs every 30-day label from all raw results and checks the numerical failure witness.')
 code(defs(P/'_audit_l178.py')['audit178'])
 code("report=audit178(P,pins,keyed_auc,comparison_gate,select_validation)\nPath('l178-report.json').write_text(json.dumps(report,indent=2))\nprint('Published replay:',report['replay']['status'],report['replay']['predictions'],'predictions')\nprint('Fresh comparison:',report['fresh']['status'])\nprint(json.dumps(report['raw_task'],indent=2))\nprint(json.dumps(report['replay']['models'],indent=2))")
 md('## CHECK · An independent pairwise metric\nThis computes every positive–negative comparison for all 30 runs, independently of your keyed scorer.')
 code("for phase in ['pilot-2','full-1']:\n    folder=P/'evidence/l166'/phase\n    for record in json.loads((folder/'receipt.json').read_text())['records']:\n        a=np.load(folder/(record['arm']+'-'+str(record['seed'])+'.npz'))\n        y=a['label'];p=a['probability'].astype(float);positive=p[y==1,None];negative=p[y==0][None,:]\n        pairwise=float(((positive>negative)+.5*(positive==negative)).mean())\n        assert abs(pairwise-keyed_auc(a['keys'],y,a['keys'],p))<1e-12\nprint('All 21,060 saved predictions independently rescored')")
 md('## TRY · Hypothetical validation selection, not measured model results\nConfiguration B has the seductive test score. The intended answer must still select A using validation. Reverse the test scores and check invariance.')
 code("fixture=[dict(config=c,seed=s,epoch=e,validation_auc=(.8 if c=='A' else .7),test_auc=(.1 if c=='A' else .99)) for c in ['A','B'] for s in [0,1,2] for e in [1,2]]\nfirst=select_validation(fixture,['A','B'],[0,1,2],2)\nfor row in fixture:row['test_auc']=1-row['test_auc']\nassert select_validation(fixture,['A','B'],[0,1,2],2)==first\nprint('Practice only:',first)")
 md('## TRY · A hypothetical repaired contract\nChange only the health flag first: uncompleted feature/time audits must still block admission. Then assume those audits passed. This is an assumption exercise; the saved observed report stays incomplete.')
 code("import copy\ncontracts=copy.deepcopy(report['fresh']['contracts'])\ncontracts['RelGNN']['training_health']='PASS'\nassert comparison_gate(contracts)['status']=='INCOMPLETE_COMPARABILITY_GATE'\nfor c in contracts.values():c['temporal_audit']=c['preprocessing_audit']='PASS'\nassert comparison_gate(contracts)['status']=='READY_FOR_PILOT'\nassert report['fresh']['status']=='INCOMPLETE_TRAINING_HEALTH_GATE'\nprint('Hypothetical pilot admission does not alter the real failure')")
 md('## EXIT · Defend the missing leaderboard\nState which lane is complete, which is stopped, the model/runtime scope of the gradient failure, and the exact next experiment you would freeze. Explain the label complement, the target-history trap and why a finite forward pass is insufficient. Ask the teacher to review your defense.')
 code("submission={'learner':'PENDING_WRITTEN_DEFENSE','complete_lane':'','stopped_lane':'','information_contract':'','gradient_scope':'','next_frozen_experiment':''}\nPath('l178-submission.json').write_text(json.dumps(submission,indent=2))")
 md('## Appendix · Visible source and executed diagnostic programs\nThe original model and featurization implementations below are archived for inspection, not executed in the default replay. No full three-model runner is claimed: the protocol stopped before that stage. To repeat the CPU encoder diagnostic in the repository with its vendored PyTorch Frame source, use a fresh output directory:\n```bash\nL178_DIAGNOSTIC_OUT=/tmp/my-new-l178-probe PYTHONPATH=labs/sources/l178 .venv/bin/python labs/_gradient_preflight_l178.py\n```\nThis repeats the local prerequisite, not the historical CUDA stack or a graph training experiment.')
 for name in ['sources/l166/upstream/model_pretrain/src/models.py','sources/l178/relgnn/examples__relgnn_nn.py','sources/l178/relgnn/examples__relgnn_conv.py','sources/l178/relgnn/examples__relgnn_model.py','sources/l178/rdblearn/rdblearn/estimator.py','sources/l178/rdblearn/rdblearn/preprocessing.py','_gradient_preflight_l178.py','_verify_gradient_l178.py','_preflight_l178.py']:
  md('### '+name+'\n```python\n'+(P/name).read_text()+'\n```')
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(cells):c.id=f'l178-{i:03}'
 return book
for solution in [False,True]:
 book=make(solution);path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
  old=nb.read(path,4)
  if [c.source for c in old.cells if c.cell_type=='code']==[c.source for c in book.cells if c.cell_type=='code']:
   book.metadata=old.metadata
   for a,b in zip([c for c in book.cells if c.cell_type=='code'],[c for c in old.cells if c.cell_type=='code']):a.outputs=b.outputs;a.execution_count=b.execution_count;a.metadata=b.metadata
 nb.write(book,path)
print('Built L178 lesson, reference and complete portable notebooks')
