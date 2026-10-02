"""Build the connected lesson and self-contained full accounting notebooks."""
import ast,base64,hashlib,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l177';S='0177-compute-budget-realism'
r=json.loads((E/'report.json').read_text());pins=json.loads((E/'input-manifest.json').read_text())
def defs(path):
 text=path.read_text();return {n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
fns=defs(P/'relkit/compute_l177.py')
status='**Executed:** complete selected accounting replay: **300 inference receipts, 18 fit records, three cloud attempt reservations**. No new model run or cloud/API spend. Fresh training, fresh inference and whole-paper reproduction are `NOT_RUN`; learner status is `PENDING_WRITTEN_DEFENSE`. Invoice reconciliation and live Colab are `NOT_CHECKED`.'
def table():
 a=r['fit_groups'];v=r['l176'];old=r['l175']
 return f'''| Evidence | Work completed | Recorded time | What it supports |
|---|---|---|---|
| L173 · local CPU | Six small-model pretraining fits | {a['173']['fit_seconds']:.3f} s group; {r['local_commands']['173']['seconds']:.3f} s all local commands | Course masked-cell objective; no per-fit timer |
| L174 · local CPU | Twelve adaptation/control fits | {a['174']['fit_seconds']:.3f} s sum of fits; {r['local_commands']['174']['seconds']:.3f} s all local commands | Four adaptation policies on the course tasks |
| L176 · L4 GPU | All 300 checkpoint evaluations | {v['worker_seconds']:.3f} s worker bodies | Two tasks, three models, five context sizes, ten seeds |
| L175 · cloud CPU | Three reserved audit attempts; stopped before GPU inference | {old['observed_worker_seconds']:.3f} s in two receipts; first attempt missing | Failed/stopped work still consumes reservations |
'''
captions={'routes':'Three actual course routes: small-model pretraining, temporal adaptation, and checkpoint inference. Different scopes prevent a speed ranking.','accounting':'The enclosing worker clock contains the inner timers. Estimated worker cost, planning reservation and an unavailable invoice answer different questions.','context':'Full L176 timing evidence. Solid lines: F1; dashed lines: trial. Time is the ten-seed mean. Memory is the maximum allocated tensor memory across ten runs, not total device usage.'}
def prose(portable=False):
 s=(R/'lessons/content'/(S+'.md')).read_text().replace('[[RESULTS]]',table()).replace('[[STATUS]]',status)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/'figures/l177'/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l177/'+name+'.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption} Scroll horizontally on narrow screens.</figcaption></figure>')
 replacements={'WARMUP':('Before reading, recall: what changes in fine-tuning, what stays fixed in ICL, and why did L175 stop?','<div id="warmup"></div>'),'PREDICT':('Predict before continuing: should you add an enclosing worker timer to its internal evaluation timers?','<div id="predict"></div>'),'EXPLORER':('Use your assess_plan function below to compare baseline and extended scenarios. Missing measurements cannot produce an unconditional admission.','<div id="budget-explorer"></div><noscript>The complete evidence and worked 140.4-minute scenario are printed below. Unknown active work or total memory prevents complete admission. A scientific failure stops the plan regardless of cost.</noscript>'),'TEACHBACK':('Write your own decision and ask the teacher for feedback.','<div id="teachback"></div>')}
 for tag,(plain,html) in replacements.items():s=s.replace('[['+tag+']]',plain if portable else html)
 s=s.replace('[[CODE]]','Implement the conservative forecast in the live TODO below.' if portable else '```python\n'+fns['forecast_seconds']+'\n```')
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def doc(title,body,interactive=False):
 html=render(body).replace('<table>','<div class="route-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 css=['lesson','atomic-route','checkpoint','lab-access','compute-budget'];scripts=['retrieval-pool','retrieval-bank','predict','teachback','compute-budget','l177-evidence','l177-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in css)+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0176-few-shot-icl-evaluation.html">Lesson 176</a></nav><header><p class="route-kicker">Year 5 · Quarter 2 · Lesson 177</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Compute budget realism',prose(),True))
(R/'assets/l177-evidence.js').write_text('window.L177Evidence='+json.dumps({k:r[k] for k in ['l176','l175','rt_price_scenarios']},separators=(',',':'))+';\n')
ref='''**Worker time:** the elapsed interval measured inside a process. **GPU-hours:** sum of GPU count × elapsed hours across jobs. **Researcher time:** active preparation, debugging and review; not inferable from model timers.

**Nested clocks:** use the enclosing timer for its total. Inner timers explain its contents. Deduplicate repeated metadata by the event identity. L176 has 300 evaluation records but 12 distinct task/model/phase load events.

**Three money columns:** worker estimate = measured seconds × stated resource rate; reservation = every attempt's timeout/lifecycle allowance × rate + overhead; invoice = provider-billed amount. Never add a subset estimate to an enclosing reservation. Never release a failed attempt's reservation without reconciliation.

**Forecast:** remaining count × slowest relevant pilot × declared margin + fixed work. This is a heuristic, not a confidence bound. Freeze the full grid before timing; do not drop seeds or change hardware silently.

**Memory:** host RAM, tensor allocations, reserved GPU allocator memory and total device usage are different quantities. L176's 12.052 GiB allocation peak does not prove the total required capacity.

**Admission:** reject known scientific, cost, memory or time violations. If an essential measurement is missing, return INCOMPLETE_MEASUREMENT. Serial-session scenarios conservatively add machine and active time. FEASIBLE_SCENARIO means assumptions fit, not that a future run is guaranteed.

'''+table()+'''

**Worked money:** L176 worker estimate $0.139440; reservation $5.230039; invoice NOT_ITEMIZED. L175 reserves $3.172199 including three attempts, overhead and build, but GPU inference is NOT_RUN after a temporal failure.

**Source-based scenarios:** RT v1 reports about 16 pretraining or 12 fine-tuning GPU-hours per run. At the checked A100 40 GB base rate, GPUs alone cost about $33.58/$25.19. This is a hypothetical rental calculation; it does not reproduce the model or establish its bill.

'''+status+'''

[Lesson](../lessons/0177-compute-budget-realism.html) · [Notebook](../labs/0177-compute-budget-realism.ipynb) · [Full report](../labs/evidence/l177/report.json) · [Protocol](../labs/l177-reproduction.md) · [Modal pricing](https://modal.com/pricing) · [RT v1 §4.1](https://arxiv.org/html/2510.06377v1#S4.SS1) · [PyTorch memory source](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/cuda/memory.py).
'''
(R/'reference/compute-budget-realism.html').write_text(doc('Compute budgets — quick reference',ref))
(E/'report.md').write_text('# L177 Compute Feasibility Ledger\n\n'+ref+'\n')
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for name in sorted(pins['files']):
  info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,(P/name).read_bytes())
packet=base64.b64encode(buf.getvalue()).decode();digest=hashlib.sha256(buf.getvalue()).hexdigest()
def make(solution):
 cells=[]
 def md(t):cells.append(nb.v4.new_markdown_cell(t))
 def code(t,tags=None):cells.append(nb.v4.new_code_cell(t,metadata={'tags':tags or []}))
 md('# Lesson 177 · Compute budget realism\n\nImplement three live accounting contracts and reconstruct the complete approved ledger. Standard Python only; no package installation, network access, GPU or paid job. The saved evidence does not become fresh training when replayed. Learner status remains PENDING_WRITTEN_DEFENSE.')
 code('from pathlib import Path\nimport json, math\nprint("Saved accounting replay; cloud/API spend $0")')
 md(prose(True))
 md('## PROVIDED · Authenticate the complete original accounting packet\nAll 300 L176 per-run JSON receipts and phase receipts, all 18 L173/L174 fit records, local attempt ledgers, L175 failure records, timer implementations and pinned primary sources are included. Prediction arrays/checkpoints are outside this accounting audit. Hashes authenticate bytes, not the original truth of a clock or an invoice.')
 code('import base64,hashlib,io,zipfile\nPACKET='+repr(packet)+'\nraw=base64.b64decode(PACKET)\nassert hashlib.sha256(raw).hexdigest()=='+repr(digest)+'''\nP=Path('l177-packet');P.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    for name in z.namelist():assert not Path(name).is_absolute() and '..' not in Path(name).parts
    z.extractall(P)
'''+ 'pins='+repr(pins)+'\nprint("Authenticated",len(pins["files"]),"accounting inputs")',['data-payload'])
 tasks=[('reservation_total','Keep every attempt reserved','Input: dictionaries with id, seconds, lifecycle_seconds and USD-per-second rate, plus overhead USD. Reject duplicate/empty identities and invalid finite nonnegative numbers, including booleans. Sum each timeout plus lifecycle allowance at its rate, then add overhead once. Use Decimal from string values for money. Failed attempts remain included.',"attempts=[dict(id='failed',seconds=900,lifecycle_seconds=30,rate=.00006172),dict(id='passed',seconds=900,lifecycle_seconds=30,rate=.00006172)]\nassert math.isclose(reservation_total(attempts,3),3.1147992,abs_tol=1e-12)\nprint('CHECK 1: both attempts remain reserved')"),('forecast_seconds','Forecast the entire remaining grid','Input: nonempty pilot timings, nonnegative integer remaining count, margin at least one, and nonnegative fixed seconds. Reject invalid/nonfinite values. Return the slowest pilot × remaining count × margin + fixed seconds. This is a declared serial heuristic.',"assert forecast_seconds([1,2,8],10,3,5)==245\nprint('CHECK 2: conservative forecast 245 s')"),('assess_plan','Keep the admission gates independent','Return a dictionary with status, missing, serial_session_seconds and meaning. Required measurement names are cost, wall_seconds, active_seconds, required_gib; None means unknown. Limits must be known. Reject malformed numbers. Prioritize SCIENTIFIC_STOP, OVER_BUDGET, MEMORY_LIMIT, EXCEEDS_SESSION, then INCOMPLETE_MEASUREMENT, otherwise FEASIBLE_SCENARIO. A known violation can reject even when another measurement is absent. serial_session_seconds is wall+active or None. Use meaning="Scenario admission only; not a measured future guarantee".',"assert assess_plan(5,3000,1200,16,24,3600)['status']=='EXCEEDS_SESSION'\nassert assess_plan(5,3000,1200,16,24,10800)['status']=='FEASIBLE_SCENARIO'\nassert assess_plan(5,300,None,None,24,10800)['status']=='INCOMPLETE_MEASUREMENT'\nprint('CHECK 3: time and missing evidence are separate gates')")]
 for name,title,description,check in tasks:
  md('## TODO · '+title+'\n'+description)
  stub=fns[name].split('    """',1)[0]+'    raise NotImplementedError("TODO: '+name+'")'
  code(fns[name] if solution else stub);code(check)
 md('## CHECK · Reject bad units and unsupported decisions\nThese checks call your live functions. Three plausible wrong implementations are rejected by the separate author verification. Passing a checklist does not replace the written defense.')
 code(defs(P/'_check_l177.py')['checks']);code("assert checks(reservation_total,forecast_seconds,assess_plan)=='PASS'\nprint('All learner contracts passed')")
 md('## PROVIDED · Full accounting reconstruction\nAuthenticate every input, verify the complete Cartesian grid, compare all raw fit records with their saved reports, deduplicate load events, reconstruct both cloud ledgers, and retain unknown measurements. The supplied audit directly calls all three learner functions.')
 code(defs(P/'_audit_l177.py')['audit177'])
 code("report=audit177(P,pins,reservation_total,forecast_seconds,assess_plan)\nPath('l177-report.json').write_text(json.dumps(report,indent=2))\nprint(report['status'])\nprint(json.dumps({k:report[k] for k in ['local_commands','fit_groups','l176','l175','rt_price_scenarios','session_scenarios','boundaries']},indent=2))")
 md('## CHECK · An independent SQL view of the inner clocks\nThe outer body timer is not available from these per-run rows. This view checks the inner totals without calling the reporting implementation.')
 code('''import sqlite3
connection=sqlite3.connect(':memory:')
connection.execute('create table timing(phase text, task text, arm text, k int, seed int, seconds real, load real, primary key(task,arm,k,seed))')
for phase in ['pilot-1','remaining-1']:
    receipt=json.loads((P/f'evidence/l176/{phase}/receipt.json').read_text())
    for row in receipt['records']:
        connection.execute('insert into timing values(?,?,?,?,?,?,?)',(phase,row['database'],row['arm'],row['context'],row['seed'],row['seconds'],row['load_seconds']))
assert connection.execute('select count(*) from timing').fetchone()[0]==300
inner=connection.execute('select sum(seconds) from timing').fetchone()[0]
loads=connection.execute('select sum(load) from (select max(load) load from timing group by phase,task,arm)').fetchone()[0]
assert math.isclose(inner,report['l176']['evaluation_seconds'],abs_tol=1e-9)
assert math.isclose(loads,report['l176']['distinct_load_seconds'],abs_tol=1e-9)
print('Independent SQL: 300 evaluations; inner seconds',inner,'distinct load seconds',loads)
''')
 md('## TRY · Change one planning assumption\nThe cost uses the old reservation; time is the old pilot forecast plus an assumed 120 s of startup. A 16 GiB total requirement is hypothetical, greater than the observed 12.052 GiB tensor peak but not a measured bound. A repeated run is not authorized by this calculation.')
 code("for minutes in [60,180]:\n    print(minutes,'minutes:',assess_plan(report['l176']['reserved_usd'],report['l176']['forecast_seconds']+120,1200,16,24,minutes*60))\nprint('Missing memory:',assess_plan(5,300,1200,None,24,3600))")
 md('## EXIT · Write two feasible plans and one refusal\nState what the run would establish, the full grid, currency/time/memory units, every attempt reservation, missing measurements, explicit active-time assumptions and scientific stop. Explain the three L176 money values and why you cannot compare CPU course-fit seconds to GPU foundation-model inference as a speed claim. Time your own next practice session; update your assumptions from that observation. Ask the teacher for feedback.')
 code("submission={'learner':'PENDING_WRITTEN_DEFENSE','baseline_plan':'','extended_plan':'','refused_plan':'','next_active_minutes':None}\nPath('l177-submission.json').write_text(json.dumps(submission,indent=2))")
 md('## Appendix · Exact timer and reservation source\nRead the clock placement, not only field names. These archived programs are shown for inspection, not executed by this notebook. No checkpoint or fresh GPU invocation is included in the accounting reproduction scope.')
 for name in ['_run_l173.py','_run_l174.py','_run_l176.py','sources/l177/modal-l176.py','_dispatch_l175.py']:
  md('### '+name+'\n```python\n'+(P/name).read_text()+'\n```')
 md('The per-fit L174 timer is defined inside train_adaptation:')
 md('```python\n'+defs(P/'relkit/finetune_l174.py')['train_adaptation']+'\n```')
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(cells):c.id=f'l177-{i:03}'
 return book
for solution in [False,True]:
 book=make(solution);path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
  prior=nb.read(path,4)
  if [c.source for c in prior.cells if c.cell_type=='code']==[c.source for c in book.cells if c.cell_type=='code']:
   book.metadata=prior.metadata
   for a,b in zip([c for c in book.cells if c.cell_type=='code'],[c for c in prior.cells if c.cell_type=='code']):a.outputs=b.outputs;a.execution_count=b.execution_count;a.metadata=b.metadata
 nb.write(book,path)
print('Built lesson, reference, interactive evidence and full portable notebooks')
