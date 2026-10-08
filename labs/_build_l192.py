"""Build the complete setup lesson and portable student/solution notebooks."""
import ast,base64,hashlib,html,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l192';Q=E/'packet';S='0192-open-fm-setup-data'
report=json.loads((E/'report.json').read_text());manifest=json.loads((E/'input-manifest.json').read_text())
status='<div class="evidence"><strong>Author status:</strong> setup/audit package prepared. Selected model reproduction: <code>INCOMPLETE_SOURCE_PREPROCESSING_GATE</code>. Model evaluations: 0. New cloud/API spend: $0. Learner: <code>PENDING_WRITTEN_DEFENSE</code>.</div>'
tasks='| Split | Complete rows | Positive labels |\n|---|---:|---:|\n'+''.join(f"| {s} | {v['rows']:,} | {v['positives']:,} |\n" for s,v in report['splits'].items())
results='''| Evidence lane | Actual result | Boundary |
|---|---|---|
| Released task archive | Authenticated; 13,779 complete keys | Raw-label reconstruction not rerun |
| Annual window-end schedule | Zero unfinished past training windows | Not a full feature-availability audit |
| Original full preprocessor | Known codes 0,1,2 → 1,2,3 after unseen a | Synthetic source counterexample |
| Independent original encoder | 100 generated cases confirm shifts | Task AUROC impact unmeasured |
| Complete selected-task search | 27 validation + 3 selected tests planned; 0 run | INCOMPLETE_SOURCE_PREPROCESSING_GATE |
| Published AUROC target | 0.7167 cited; measured score absent | No match or superiority claim |
| Pretraining / whole-paper benchmark | NOT_RUN | Outside this selected-task evidence |
'''
captions={'architecture':'RDBLearn’s complete inference route. The mean/count example is illustrative; backend inference and the full search remain unrun. nₛ and nq count support/query rows; D counts features.','clocks':'Synthetic day-180 counterexample to using prediction time as label availability. The actual official annual schedule has zero unfinished past training windows under the declared window-end policy.','encoding':'Measured original-preprocessor intervention on synthetic categories: support codes0,1,2; after unseen a, known query codes1,2,3. Numeric control unchanged. No model score was measured.'}
plain={'WARMUP':'Recall from memory: complete query identity; validation-only selection; event time versus availability. Then check the explanations below.', 'PREDICT':'Predict before continuing: after inserting a before b,c,d in sorted order, does b retain code0? Record your answer. The measured answer below is1.', 'CLOCK':'Try the clock example: at day180 the historical prediction is earlier but its label window is unfinished. At day365 both conditions pass.', 'ENCODING':'Intervene mentally: inserting z instead of a leaves b,c,d at0,1,2. An illustrative frozen vocabulary preserves known codes for either input; that repair is not a benchmark result.', 'CHECKLIST':'Review source/checkpoint/data identity, invariant checks, complete query keys, availability, validation-only selection and full cost. A checked list does not reverse the source failure.', 'TEACHBACK':'Write a defense of the source gate, actual task checks, remaining uncertainty and next admissible experiment. Do not claim historical AUROC harm from this synthetic failure.'}
def prose(portable=False):
 text=(R/'lessons/content'/(S+'.md')).read_text().replace('[[STATUS]]',status).replace('[[TASK_TABLE]]',tasks).replace('[[RESULTS]]',results)
 for key,caption in captions.items():
  url='data:image/png;base64,'+base64.b64encode((P/'figures/l192'/(key+'.png')).read_bytes()).decode() if portable else '../labs/figures/l192/'+key+'.svg'
  picture=f'<img src="{url}" alt="{caption}">'
  if not portable:picture=f'<picture><source media="(max-width:500px)" srcset="../labs/figures/l192/{key}-mobile.svg">'+picture+'</picture>'
  text=text.replace('[[FIG:'+key+']]',f'<figure class="route-figure">{picture}<figcaption>{caption}</figcaption></figure>')
 for key,desc in plain.items():text=text.replace('[['+key+']]',desc if portable else '<div id="'+key.lower()+'"></div><noscript>'+desc+'</noscript>')
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/')
  text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def doc(title,body,interactive=False):
 markup=re.sub(r'<table([^>]*)>',r'<div class="route-scroll" tabindex="0"><table\1>',render(body)).replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','checklist','reproduction-contract','l192-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson192 — '+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in ['lesson','checkpoint','atomic-route','reproduction-contract'])+'</head><body class="checkpoint l192"><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html">Curriculum</a></nav><header><p class="route-kicker">Year5 · Quarter4 · Lesson192</p><h1>'+title+'</h1><p class="subtitle">A published number begins with stable inputs</p></header>'+markup+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Open FM reproduction: setup & data',prose(),True))
reference='''## Reproduction contract

Name the paper version, task, metric, splits, source revision, data and checkpoint hashes, preprocessing, search space, selection rule, seeds, aggregation, tolerance and total cost. A released default is not proof of a historical setting.

## Three invariants

1. Identity is (entity, cutoff); reject duplicate full keys.
2. A historical label needs both an earlier prediction and an availability time no later than the query. Window-end availability is a declared conservative policy.
3. A known category must retain the same representation as the stored support after any query transform. New-category handling must not silently renumber the support vocabulary.

## Search and stopping rule

RDBLearn toolkit v1: study-outcome target0.7167AUROC. Depths2/3/4 × TabPFNv2/TabPFNv2.5/LimiX;10,000support limit. Three new seeds produce27validation candidates and3selected tests. Validation-only maximum; listed order breaks ties. Historical seeds unknown. Full-search forecast unknown;USD10aggregate cap,USD8planned stop,USD2reserve.

## Actual evidence

'''+results+'''\n## Re-run and interpret

[Complete contract](../labs/l192-reproduction.md) · [Lesson](../lessons/0192-open-fm-setup-data.html) · [Executed lab](../labs/html/0192-open-fm-setup-data.html) · [Primary paper](https://arxiv.org/html/2602.18495v1).

The source gate is failed. Do not replace the paper’s search with a single cheap model or call the synthetic counterexample an AUROC result. A repaired pipeline requires an explicitly frozen continuation. Ask the agent to review your admission record.
'''
(R/'reference/open-fm-setup-data.html').write_text(doc('Open FM reproduction — reference',reference))

def defs(path):return {n.name:ast.get_source_segment(path.read_text(),n) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
functions=defs(P/'relkit/setup_l192.py');audit=defs(P/'_audit_l192.py')['audit'];checks=defs(P/'_test_l192.py')['checks'];srcdefs=defs(Q/'preprocessing.py')
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for name in sorted(manifest['files']):
  info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,(Q/name).read_bytes())
payload=base64.b64encode(buf.getvalue()).decode();digest=hashlib.sha256(buf.getvalue()).hexdigest()
instructions={
 'keyed_rows':'Index rows by (entity,cutoff). Require integer entity/day/binary label values (booleans are not valid integers here), reject duplicate full keys, preserve repeated entities at different cutoffs. Return a dictionary from key to original row.',
 'available_history':'Accept integer query day and history rows with integer available_at no earlier than their own cutoff. Validate the complete keys using your keyed_rows. Return original rows in order only when their prediction cutoff is earlier and their label is available by the query day. Reject missing availability instead of guessing.',
 'select_candidate':'Accept the frozen candidate order and a complete set of validation records. Require finite AUROC in[0,1], unique known IDs and exactly the expected grid. Reject test records, duplicates, unknowns and incomplete grids. Return the maximum-validation candidate; frozen order breaks ties.'}
immediate={
 'keyed_rows':"toy=[dict(entity=7,cutoff=10,label=1),dict(entity=7,cutoff=20,label=0)]\nassert len(keyed_rows(toy))==2\nprint('Two times for one entity retained')",
 'available_history':"history=[dict(entity=7,cutoff=0,label=1,available_at=365)]\nassert available_history(history,180)==[]\nassert available_history(history,365)==history\nprint('Prediction and availability clocks checked')",
 'select_candidate':"example=[dict(id='a',split='val',auc=.62),dict(id='b',split='val',auc=.71)]\nassert select_candidate(example,['a','b'])=='b'\nprint('Authored selection example: b; no model result')"}

def make(solution):
 cells=[]
 def md(s):cells.append(nb.v4.new_markdown_cell(s))
 def code(s,tags=None):cells.append(nb.v4.new_code_cell(s,metadata={'tags':tags or []}))
 md('# Lesson192 · Open FM reproduction: setup & data\n\n**Mirror scope:** setup/data admission for a named RDBLearn published result. The selected full search is blocked by an executed source-preprocessing diagnostic. Default execution authenticates all13,779query records, replays the complete admission audit, and executes the original categorical encoder. It does not fit a backend or pretrain a model.\n\nPython3.12 with pandas, NumPy and scikit-learn. These are standard Colab scientific packages; actual liveColab was NOT_CHECKED. All data, source and figures needed by this notebook are embedded. No network or repository checkout is required. Full original preprocessor execution requires the separately supplied pinned CLI environment.\n\nPROVIDED cells expose the mechanism. Three TODO functions drive the real audit. CHECK cells reject wrong contracts. EXIT requires your written interpretation.')
 code('# @colab-bootstrap\nfrom pathlib import Path\nimport ast,base64,csv,hashlib,io,json,math,zipfile\nimport numpy as np\nimport pandas as pd\nfrom sklearn.base import BaseEstimator,TransformerMixin\nfrom sklearn.preprocessing import LabelEncoder\nprint("L192: audit + original encoder replay; backend inference NOT_RUN")',['colab-bootstrap'])
 # Keep the narrative in digestible notebook sections, with portable diagrams beside mechanisms.
 for block in re.split(r'(?=^## )',prose(True),flags=re.M):
  if block.strip():md(block)
 md('## PROVIDED · Authenticate the embedded evidence\n\nThe archive hash pins every byte. The visible audit independently checks each member against the input manifest. Hash agreement proves input identity, not scientific validity. The complete original preprocessor observations are author evidence; your next cells replay the primitive separately.')
 code('PACKET='+repr(payload)+'\nraw=base64.b64decode(PACKET)\nassert hashlib.sha256(raw).hexdigest()=='+repr(digest)+"\npacket=Path('l192-packet');packet.mkdir(exist_ok=True)\nwith zipfile.ZipFile(io.BytesIO(raw)) as z:\n    for name in z.namelist():\n        assert not Path(name).is_absolute() and '..' not in Path(name).parts\n    z.extractall(packet)\nmanifest="+repr(manifest),['data-payload'])
 for name in functions:
  md('## TODO · '+name+'\n\n**Goal.** '+instructions[name]+'\n\n**Why.** This contract contributes to the complete admission record, so a locally convenient shortcut cannot silently change the experiment.')
  node=ast.parse(functions[name]).body[0]
  code(functions[name] if solution else 'def '+name+'('+ast.unparse(node.args)+'):\n    # TODO: implement the stated contract.\n    raise NotImplementedError('+repr(name)+')',['solution' if solution else 'todo'])
  md('### CHECK · immediate feedback');code(immediate[name])
 md('## CHECK · malformed and incomplete experiments\nThese tests exercise all three live functions, including duplicate keys, missing availability, unknown candidates, ties and test-score contamination.')
 code(checks);code('print(checks(keyed_rows,available_history,select_candidate))')
 md('## PROVIDED · Original released categorical encoder\n\nThis class is unmodified source from the pinned RDBLearn release. Follow fit, unseen detection, dictionary expansion and sorting. It is the same class called by the released full preprocessor, whose complete source is readable after EXIT. We execute the class with real pandas/NumPy/scikit-learn; there is no emulated LabelEncoder.')
 code(srcdefs['SafeLabelEncoderTransformer'],['original-source'])
 md('### Predict, then intervene\nPredict the known-category codes before and after transforming a new category. The stored support matrix remains fixed. The source counterexample is synthetic, while the task audit below uses real released task rows.')
 code("support=pd.DataFrame({'category':pd.Series(['b','c','d']*4,dtype='object')})\nknown=pd.DataFrame({'category':pd.Series(['b','c','d'],dtype='object')})\nenc=SafeLabelEncoderTransformer().fit(support)\nbefore=enc.transform(known).category.tolist()\nenc.transform(pd.DataFrame({'category':pd.Series(['a'],dtype='object')}))\nafter=enc.transform(known).category.tolist()\nassert before==[0,1,2] and after==[1,2,3]\nobserved=json.loads((packet/'preprocessing.json').read_text())\nassert before==observed['before']['category'] and after==observed['after']['category']\nprint('Original encoder:',before,'→',after)\nprint('Full-preprocessor author evidence matches this primitive; no model AUROC measured')")
 md('## PROVIDED · Complete visible admission audit\n\nThe canonical function below uses your three functions. It audits every query key and every distinct cutoff, verifies the recorded transformation against independent sorted-set arithmetic, and separates planned evaluation counts from measured model runs.')
 code(audit)
 code("report=audit(packet,manifest,keyed_rows,available_history,select_candidate)\nPath('l192-report.json').write_text(json.dumps(report,indent=2)+'\\n')\nprint(report['status'])\nprint('Official queries:',report['task_queries'])\nprint('Unfinished earlier training windows:',report['unavailable_past_train_rows'])\nprint('Model evaluations:',report['executed_model_evaluations'])\nassert report=="+repr(report))
 md('## EXIT · your admission record\n\nWrite a defense of the stopped reproduction. Name the target, the complete search, the failed invariant, the passing task checks and their limits. Explain why changing the encoder would require a new declared protocol. State the evidence and complete cost forecast needed before continuation. Leave author preparation separate from your own mastery.')
 code("defense=''  # Write your own reasoning; the solution does not impersonate a learner.\nsubmission=dict(status=report['status'],defense=defense,learner='PENDING_WRITTEN_DEFENSE',model_auc=None)\nPath('l192-submission.json').write_text(json.dumps(submission,indent=2)+'\\n')\nprint(submission['learner'])")
 md('## Full reproduction continuation — source-gated\n\nThe27validation candidates and3selected tests remain NOT_RUN. The benchmark operator is not admitted or claimed ready: checkpoint identities, full feature checks and full cost remain unresolved. See the distributed `l192-reproduction.md` for tested audit/preflight commands. No RUN flag bypasses the failed gate. This preserves the intended experiment rather than quietly replacing its backend or sample size.\n\nThe following complete release source makes the blocked pipeline reviewable. It is displayed, not automatically executed: importing AutoGluon or a model environment is outside this notebook’s default dependency contract.')
 for filename in ['constants.py','config.py','preprocessing.py','estimator.py']:
  text=(Q/filename).read_text();tree=ast.parse(text)
  imports='\n'.join(ast.get_source_segment(text,n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)))
  if imports:md('### Original '+filename+' · imports\n\n```python\n'+imports+'\n```')
  for n in tree.body:
   if isinstance(n,(ast.Import,ast.ImportFrom)):continue
   if isinstance(n,ast.ClassDef) and n.name=='SafeLabelEncoderTransformer':continue
   name=getattr(n,'name','module constants');md('### Original '+filename+' · '+name+'\n\n```python\n'+ast.get_source_segment(text,n)+'\n```')
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(book.cells):c['id']=hashlib.sha256((str(i)+c.source).encode()).hexdigest()[:12]
 dest=P/('solutions' if solution else '')/(S+'.ipynb');dest.parent.mkdir(exist_ok=True)
 if solution and dest.exists():
  old=nb.read(dest,4);book.metadata=old.metadata
  previous=[c for c in old.cells if c.cell_type=='code'];current=[c for c in book.cells if c.cell_type=='code']
  if [c.source for c in previous]==[c.source for c in current]:
   for prior,c in zip(previous,current):
    c.outputs=prior.outputs;c.execution_count=prior.execution_count;c.metadata=prior.metadata
 nb.write(book,dest)
make(False);make(True)
print('Lesson, reference, student and solution built')
