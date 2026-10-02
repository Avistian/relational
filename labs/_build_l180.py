"""Deterministic lesson, quick reference and portable learner/solution notebooks."""
import ast,base64,hashlib,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l180';S='0180-public-encoder-checkpoint'
r=json.loads((E/'report.json').read_text());pins=json.loads((E/'input-manifest.json').read_text())
def defs(path):
 source=path.read_text();return {n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
fns=defs(P/'relkit/checkpoint_l180.py')
status='**Observed:** complete saved-context replay; public RT-v1 fine-tuning **NOT_RUN**; practical Q2 exit **INCOMPLETE**. The temporal gate fails and one reported full run exceeds the $10 ceiling. No new inference, training or cloud/API spend. Learner defense: **PENDING_WRITTEN_DEFENSE**.'
captions={'architecture':'RT-v1 forward and supervised-update paths. Public initialization, datatype encoders, four sequential relation attentions, boolean decoder and validation selection are distinct steps. This is the archived architecture, not an executed L180 model.','temporal':'A real saved schedule cell lies after its query cutoff. The event-time rule fails; absent publication histories prevent an outcome-leakage conclusion.','cost':'One reported full fine-tuning run already exceeds the entire lesson budget before non-GPU costs. These are price scenarios, not observed runtimes or invoices.'}
def prose(portable=False):
 s=(R/'lessons/content'/(S+'.md')).read_text().replace('[[STATUS]]',status)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/'figures/l180'/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l180/'+name+'.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 tags={'WARMUP':('From memory: distinguish public weights from a local course encoder; identify who may select a checkpoint; explain why a finite forward pass does not prove finite gradients.','<div id="warmup"></div>'),'PREDICT':('Predict: does reproducing saved context evidence establish fresh public-encoder training? Explain before inspecting the solution.','<div id="predict"></div>'),'EXPLORER':('Use your live checkpoint_decision function in the scenario cells below. Changing assumptions does not change measured evidence.','<div id="encoder-checkpoint"></div><noscript>The observed verdict is BLOCKED: the saved temporal rule fails, execution prerequisites remain unchecked, and $25.185600 GPU-only exceeds $10. Fresh fine-tuning NOT_RUN; practical exit INCOMPLETE.</noscript>'),'TEACHBACK':('Write the defense at the end of this notebook and ask the teacher to review it.','<div id="teachback"></div>')}
 for tag,(plain,html) in tags.items():s=s.replace('[['+tag+']]',plain if portable else html)
 s=s.replace('[[CODE]]','Implement checkpoint_decision below; it must retain all blockers.' if portable else '```python\n'+fns['checkpoint_decision']+'\n```')
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def doc(title,body,interactive=False):
 html=render(body).replace('<table>','<div class="route-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 css=['lesson','atomic-route','checkpoint','lab-access','encoder-checkpoint'];scripts=['retrieval-pool','retrieval-bank','predict','teachback','encoder-checkpoint','l180-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 180 — '+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in css)+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0178-fair-model-comparison.html">Lesson 178</a></nav><header><p class="route-kicker">Year 5 · Quarter 2 · Checkpoint 180</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Fine-tune a public encoder on one database',prose(),True))
reference='''**Claim ladder:** a public filename → authenticated weight bytes → loaded compatible model → legal task inputs → finite supervised updates → validation-selected retained weights → independently verified complete-key test predictions. Each arrow needs evidence. Fresh pretraining and historical identity are separate claims.

**RT-v1 contract:** `pretrain_rel-f1_driver-dnf.pt`, fixed release; 1,024 cell context; width256, 12blocks, 8heads; frozen MiniLM inputs; trainable RT parameters; boolean BCE for DNF. Source example max_steps32,769, per-rank batch32 × eight ranks = global256, AdamW LR1e-4 / weight decay0. The paper rounds to33k steps. Source example is Amazon churn and must be explicitly adapted to F1.

**Temporal contract:** query target masked; event-dated cells no later than query cutoff; visible 30-day labels have completed windows. Unknown arrival times remain unknown. L175's385future schedule cells in77contexts fail this event-time policy without proving outcome leakage.

**Selection:** source compares validation AUROC, while also logging test metrics. Default checkpoint saving is disabled. Verify retained states and complete split identities before claiming a validation-selected result. Neither a source reading nor passing a local primitive proves RT gradient health.

**Cost:** 8 × 1.5 × 3600 ×0.000583 = USD25.185600 GPU-only; A10080GB gives29.980800. Include CPU, RAM, preparation, evaluation, failures and retries. Both exceed the USD10 lesson ceiling. A saved checkpoint or shortened schedule is not a full fresh fit.

**Decision:** preserve every blocker. Prerequisites passed → eligible for separately authorized execution; executed and independently verified fit → practical evidence; reviewed defense → learner completion. An author replay does not establish the latter two.

'''+status+'''

[Lesson](../lessons/0180-public-encoder-checkpoint.html) · [Student lab](../labs/0180-public-encoder-checkpoint.ipynb) · [Protocol](../labs/l180-reproduction.md) · [Report](../labs/evidence/l180/report.json) · [RT paper](https://arxiv.org/html/2510.06377v1#S4.SS1) · [Pinned model card](https://huggingface.co/stanford-star/rt-v1/blob/299701dedae451f3dfa40717b831d9dc17c0e4e7/README.md).
'''
(R/'reference/public-encoder-checkpoint.html').write_text(doc('Public encoder checkpoint — quick reference',reference))
(E/'report.md').write_text('# L180 observed evidence\n\n'+status+'\n\n'+json.dumps({k:r[k] for k in ['contexts','cell_slots','totals','costs_gpu_only_usd','decision']},indent=2)+'\n')
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for name in sorted(pins['files']):
  info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,(E/'packet'/name).read_bytes())
packet=base64.b64encode(buf.getvalue()).decode();digest=hashlib.sha256(buf.getvalue()).hexdigest()

def make(solution):
 cells=[]
 def md(s):cells.append(nb.v4.new_markdown_cell(s))
 def code(s,tags=None):cells.append(nb.v4.new_code_cell(s,metadata={'tags':tags or []}))
 md('# Lesson 180 · Public encoder checkpoint\n\nThe runnable work is complete saved-context replay and source/cost auditing. Fresh public-encoder fine-tuning is NOT_RUN and the practical exit is INCOMPLETE. Implement three functions, then explain the gap. Ask the teacher for feedback.\n\nDependencies: Python3, NumPy, pandas, pyarrow, ml_dtypes. If absent, install these in your chosen notebook environment before running. Default cells perform no network access, GPU work or cloud calls. The source appendix is displayed, not executed.')
 code('from pathlib import Path\nimport json,copy\nimport numpy as np\nimport pandas as pd\nimport ml_dtypes\nprint("L180: saved evidence only; cloud/API $0; fresh fine-tuning NOT_RUN")')
 md(prose(True))
 md('## PROVIDED · Authenticate the complete evidence packet\nEvery saved context, the raw label oracle, raw result/driver tables, and source example are embedded. SHA256 pins byte identity, not historical legality. The packet needs no download.')
 code('import base64,hashlib,io,zipfile\nPACKET='+repr(packet)+'\nraw=base64.b64decode(PACKET)\nassert hashlib.sha256(raw).hexdigest()=='+repr(digest)+"\nP=Path('l180-packet');P.mkdir(exist_ok=True)\nwith zipfile.ZipFile(io.BytesIO(raw)) as z:\n    for name in z.namelist():assert not Path(name).is_absolute() and '..' not in Path(name).parts\n    z.extractall(P)\nmanifest="+repr(pins)+"\nprint('Packet bytes authenticated:',len(manifest['files']),'files')",['data-payload'])
 tasks=[('temporal_counts','Audit all context cells','Return integer counts future_cells, unmasked_query_targets, unavailable_labels, unknown_time_cells. Ignore padding. The unknown timestamp sentinel is -2147483648. A visible label is an unpadded label cell whose mask is false. Its outcome becomes available at timestamp+horizon (equality is allowed). Known future cells violate the event-time rule even if their value is masked; the sampled metadata itself still belongs to that future row. Reject mismatched/non-1D arrays and nonpositive horizons.'),('full_run_cost','Price the complete protocol','Return a six-decimal USD string using Decimal(str(input)). Multiply GPU count × hours ×3600 × rate × run count, then add overhead once. Require finite positive values, integer GPU/run counts and nonnegative overhead. No rounding through binary floats.'),('checkpoint_decision','Keep every blocker and every completion stage','Require PASS for temporal, training_health, checkpoint_bytes, selection, full_population and source_protocol. Missing/non-PASS fields become uppercase blockers. Validate finite nonnegative gpu_only_usd and positive cap_usd; unknown values add BUDGET_UNKNOWN, excess adds BUDGET. Require a finite nonnegative all_in_upper_usd: absent/invalid adds ALL_IN_COST_UNKNOWN, a bound below GPU-only cost adds ALL_IN_COST_INVALID, and a bound above the cap adds BUDGET once. Admission is BLOCKED or READY_FOR_SEPARATELY_AUTHORIZED_RUN. Practical exit starts INCOMPLETE; with no blockers AND fresh_fit VERIFIED it becomes PENDING_WRITTEN_DEFENSE, or PASS if written_defense is PASS. Scenario fields are assumptions, never authorization.')]
 for name,title,instructions in tasks:
  md('## TODO · '+title+'\n'+instructions)
  node=ast.parse(fns[name]).body[0];signature=f'def {name}('+ast.unparse(node.args)+'):'
  code(fns[name] if solution else signature+'\n    raise NotImplementedError("Implement '+name+'")')
 md('## CHECK · Behavioral contracts\nThese checks include masked targets, horizon boundaries, eight-GPU accounting, missing evidence, and the distinction between admission and completion.')
 code(defs(P/'_check_l180.py')['checks'])
 code('assert checks(temporal_counts,full_run_cost,checkpoint_decision)=="PASS"\nprint("CHECK: three live contracts passed")')
 md('## PROVIDED · Full source and context replay\nThis consumes your three functions, reconstructs all2,106query labels from raw results, checks every complete key and matches every context audit. A zero-count stub cannot produce a valid report.')
 code(defs(P/'_audit_l180.py')['audit180'])
 code("report=audit180(P,manifest,temporal_counts,full_run_cost,checkpoint_decision)\nPath('l180-report.json').write_text(json.dumps(report,indent=2))\nprint(report['audit_status'],report['contexts'],'contexts')\nprint(json.dumps(report['totals'],indent=2))\nprint(json.dumps(report['decision'],indent=2))")
 md('## CHECK · Independent scalar oracle\nThe reference below counts cells one by one instead of using vector masks. It reruns the full packet with an independent temporal implementation, retaining your cost and checkpoint functions.')
 code(defs(P/'_verify_l180.py')['scalar'])
 code("assert audit180(P,manifest,scalar,full_run_cost,checkpoint_decision)==report\nassert int(8*5400*583)==int(__import__('decimal').Decimal(report['costs_gpu_only_usd']['A100_40GB'])*1000000)\nprint('Independent full context and integer micro-dollar checks PASS')")
 md('## TRY · Remove one blocker, then predict the remaining verdict\nThis is a hypothetical scenario. It changes neither the report nor your budget authorization.')
 code("scenario=copy.deepcopy(report['evidence']);scenario['cap_usd']='40'\na=checkpoint_decision(scenario)\nassert 'BUDGET' not in a['blockers'] and 'TEMPORAL' in a['blockers']\nprint('Budget-only change:',a)\nfor k in ['temporal','training_health','checkpoint_bytes','selection','full_population','source_protocol']:scenario[k]='PASS'\nassert 'ALL_IN_COST_UNKNOWN' in checkpoint_decision(scenario)['blockers']\nscenario['all_in_upper_usd']='35' # hypothetical upper bound, not measured or authorized\nassert checkpoint_decision(scenario)['practical_exit']=='INCOMPLETE'\nprint('Prerequisites hypothetically passed; fresh fit is still absent')\nassert report['decision']['practical_exit']=='INCOMPLETE'")
 md('## TRY · A flawed price and a corrupted packet\nExplain why both must be rejected before using a score. The first mistake counts only one of eight GPUs. The second changes the expected digest for source code.')
 code("try:checks(temporal_counts,lambda *a,**kw:'3.148200',checkpoint_decision)\nexcept AssertionError:print('Missing GPU multiplicity rejected')\nelse:raise AssertionError('Wrong cost admitted')\nbad=copy.deepcopy(manifest);bad['files']['example_finetune.py']='0'*64\ntry:audit180(P,bad,temporal_counts,full_run_cost,checkpoint_decision)\nexcept ValueError:print('Source tampering rejected')\nelse:raise AssertionError('Corrupt packet admitted')")
 md('## EXIT · Your written defense\nFill every field in your own words. Name an observation, its limitation and the next required evidence. No automatic cell changes learner status to PASS. Ask the teacher to review it. Fresh public-encoder training remains a separate outstanding practical requirement.')
 code("submission={'learner':'PENDING_WRITTEN_DEFENSE','practical_exit':'INCOMPLETE','public_initialization':'','trainable_path':'','temporal_witness_and_limit':'','full_cost':'','selection_and_coverage':'','next_approved_experiment':''}\nPath('l180-submission.json').write_text(json.dumps(submission,indent=2))")
 md('## Appendix · Original load-bearing implementation\nThe following pinned code is visible for inspection, not executed by this notebook. The source training example is for Amazon churn, has checkpoint saving disabled and evaluates both splits. A future F1 runner must resolve the documented adaptations and gates. The L180 repository entry point is executable only through admission, and deliberately exits before dispatch. Post-gate training remains UNVALIDATED.')
 for name in ['sources/l180/example_finetune.py','sources/l180/upstream/rt/model.py','sources/l180/upstream/rt/main.py','_run_l180.py']:
  md('### '+name+'\n```python\n'+(P/name).read_text()+'\n```')
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(cells):c.id=f'l180-{i:03}'
 return book
for solution in [False,True]:
 book=make(solution);path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
  old=nb.read(path,4)
  if [c.source for c in old.cells if c.cell_type=='code']==[c.source for c in book.cells if c.cell_type=='code']:
   book.metadata=old.metadata
   for a,b in zip([c for c in book.cells if c.cell_type=='code'],[c for c in old.cells if c.cell_type=='code']):a.outputs=b.outputs;a.execution_count=b.execution_count;a.metadata=b.metadata
 nb.write(book,path)
print('Built L180 lesson, reference, student and solution notebooks')
