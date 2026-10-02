"""Build connected HTML and portable notebooks from the visible experiment."""
import ast,base64,hashlib,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l186';S='0186-production-constraints'
r=json.loads((E/'report.json').read_text());pins=json.loads((E/'input-manifest.json').read_text())
source=(P/'relkit/serving_l186.py').read_text()
def definitions(path):
 text=path.read_text();return {n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
fns=definitions(P/'relkit/serving_l186.py')
status='**Executed:** complete **81-scenario course simulation / 810,000 responses**, plus **300 original batch-receipt replays**. All service times are hypothetical. Live production performance, fresh model inference and paper-result reproduction are `NOT_RUN`; request latency from original batch receipts is `NOT_MEASURED`. Learner status: `PENDING_WRITTEN_DEFENSE`.'
def results():
 s='| Policy · normal arrivals · seed 0 | Requests/s | p99 latency | Deadline misses | Stale responses |\n|---|---:|---:|---:|---:|\n'
 for x in r['results']:
  if x['condition']=='normal' and x['seed']==0:
   s+=f"| {x['policy']} | {x['rate']} | {x['p99_ms']:,} ms | {x['deadline_misses']/100:.2f}% | {x['stale_responses']/100:.2f}% |\n"
 return s+'\nThis worked table shows seed 0; the figure includes all three seeds and the full report retains all 81 cells. No best seed or policy was selected.'
def receipts():
 v=r['receipt_replay']
 return f"**{v['evaluation_seconds']:.3f} s** of inner evaluation time + **{v['distinct_load_seconds']:.3f} s** from 12 distinct loading events sit inside **{v['worker_seconds']:.3f} s** of worker time. The difference is other worker work; it is not measured cloud startup. Do not add inner timers to the outer total."
captions={'serving':'Relational serving routes. All three policies read the same customer-linked dependencies; only their legal read time and assumed request cost differ.','tradeoffs':'Complete normal-arrival subset: three rates, three policies and all three seeds. Hypothetical worker times; these are simulated finite-trace results.','clocks':'A complete saved request trace. Event, refresh and response clocks show why a 1 ms response violates a 2,000 ms source-age limit.'}
def prose(portable=False):
 s=(R/'lessons/content'/(S+'.md')).read_text().replace('[[STATUS]]',status).replace('[[RESULTS]]',results()).replace('[[RECEIPTS]]',receipts())
 for name,caption in captions.items():
  url='data:image/png;base64,'+base64.b64encode((P/'figures/l186'/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l186/'+name+'.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure tabindex="0"><img src="{url}" alt="{caption}"><figcaption>{caption} Scroll horizontally on narrow screens.</figcaption></figure>')
 tags={'WARMUP':('Recall the difference between a temporal cutoff and observed arrival, and between an inner timer and its enclosing timer.','<div id="warmup"></div>'),'PREDICT':('Predict: can a 1 ms response contain a stale dependency? Explain using the two separate limits.','<div id="predict"></div>'),'EXPLORER':('Use the scenario selection cell below to inspect complete saved cells. Predict a change before inspecting the output.','<div id="serving-explorer"></div><noscript>The complete seed-0 baseline table above remains readable. Cached features at 50 requests/s have p99 13 ms but cannot repair an interrupted payment stream.</noscript>'),'TEACHBACK':('Write the six-sentence contract in the EXIT cell below.','<div id="teachback"></div>')}
 for key,(plain,html) in tags.items():s=s.replace('[['+key+']]',plain if portable else html)
 s=s.replace('[[CODE]]','Implement the FIFO contract in the TODO cell below.' if portable else '```python\n'+fns['schedule']+'\n```')
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def doc(title,body,interactive=False):
 html=render(body).replace('<table>','<div class="route-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','serving-contract','l186-evidence','l186-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in ['lesson','checkpoint','lab-access','serving-contract'])+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0177-compute-budget-realism.html">Compute evidence</a></nav><header><p class="subtitle">Year 5 · Quarter 3 · Lesson 186</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Production constraints',prose(),True))
(R/'assets/l186-evidence.js').write_text('window.L186Evidence='+json.dumps({'results':r['results']},separators=(',',':'))+';\n')
ref='''**The contract:** deadline, dependency information set, source-age limit, fallback, and monitor. Query identity and model/feature versions must survive to later scoring.

| Clock / metric | Definition | What it cannot establish |
|---|---|---|
| Event time | When the observed source state applies | When it reached our service |
| Arrival time | When the source snapshot becomes visible | Whether all real transactions arrived |
| Materialization time | When a refresh is ready | How recent its source is |
| Request latency | Response minus request arrival | Information freshness |
| Observed source age | Response minus oldest required source event | Complete event-stream integrity or accuracy |
| Materialization age | Response minus refresh completion | Dependency event age |

**Queue:** start[i] = max(arrival[i], finish[i−1]); finish[i] = start[i] + service. Queue wait belongs in request latency. The simulated single 15 ms worker has nominal capacity about 66.7 requests/s; this excludes real service variability and all unmodeled resources.

**Availability:** event ≤ arrival ≤ read. Cached read occurs at a refresh tick whose tick + 20 ms has completed by service start. An older late arrival never replaces a newer snapshot. Missing dependency → unknown age → freshness failure. Reads at start may include events after request arrival; this is an explicit serving choice.

**Monitor:** last 100 completed responses, strictly more than 10 stale. No complete window → no alert. An already-active alert is not newly detected failure. The 100-response window changes elapsed duration with traffic. Labels mature separately; no scores means no quality claim.

**Simulation limits:** hypothetical fixed durations, one worker, unlimited separate refresh resources, no actual model, no cancellation or network, no fallback executed. Nominal exponential arrivals are rounded to at least 1 ms. All 81 scenarios drain all requests. The frozen 100 ms deadline and 2,000 ms source-age limits are course assumptions.

'''+receipts()+'\n\n'+status+'''

[Lesson](../lessons/0186-production-constraints.html) · [Notebook](../labs/0186-production-constraints.ipynb) · [Protocol](../labs/l186-reproduction.md) · [Report](../labs/evidence/l186/report.json) · [Huyen: real-time ML](https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html) · [Huyen: monitoring](https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html) · [Google SRE](https://sre.google/sre-book/monitoring-distributed-systems/).
'''
(R/'reference/production-constraints.html').write_text(doc('Serving constraints — quick reference',ref))
(E/'report.md').write_text('# L186 Relational Serving Contract\n\n'+status+'\n\n'+results()+'\n\n'+receipts()+'\n')
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for name in sorted(pins['files']):
  info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,(E/name).read_bytes())
packet=base64.b64encode(buf.getvalue()).decode();sha=hashlib.sha256(buf.getvalue()).hexdigest()

def make(solution):
 cells=[]
 def md(t):cells.append(nb.v4.new_markdown_cell(t))
 def code(t,tags=None):cells.append(nb.v4.new_code_cell(t,metadata={'tags':tags or []}))
 md('# Lesson 186 · Production constraints\n\nComplete 81-cell course simulation plus original 300-receipt audit. No package install, network call, cloud dispatch or trained model. Implement three live contracts, run every cell, and defend the missing production evidence.')
 code('from pathlib import Path\nimport bisect, hashlib, itertools, json, math, random\nfrom collections import deque\nprint("Hypothetical serving simulation; new cloud/API spend $0")')
 md(prose(True))
 md('## PROVIDED · Frozen protocol\nThe timing numbers below are hypotheses, not L176 measurements. Use the approved 81-cell grid unchanged for the complete report. Any changed experiment needs a new identity.')
 code('CONFIG = '+repr(r['config']))
 tasks=[('schedule','Schedule every request','Return (start,finish) tuples for one FIFO worker. Validate positive integer service and nonnegative ordered integer arrivals. Include queue wait.',"assert schedule([0,2,30],10)==[(0,10),(10,20),(30,40)]\nprint('CHECK: second request waits 8 ms')"),('source_age','Keep the oldest dependency visible','Return response time minus the minimum dependency event; if any is None, return None. Reject empty lists, future or noninteger events, and invalid response times.',"assert source_age(100,[80,90])==20\nassert source_age(100,[80,None]) is None\nprint('CHECK: missing source does not become fresh')"),('freshness_alert','Wait for the complete monitor window','Require window positive, threshold in [0,1], Boolean flags. Use only the last window responses; strictly greater than threshold. No full window returns False.',"assert not freshness_alert([True]*99)\nassert not freshness_alert([True]*10+[False]*90)\nassert freshness_alert([True]*11+[False]*89)\nprint('CHECK: strict threshold and complete window')")]
 for name,title,desc,check in tasks:
  md('## TODO · '+title+'\n'+desc)
  stub=fns[name].split('    """',1)[0]+'    raise NotImplementedError("TODO: '+name+'")'
  code(fns[name] if solution else stub);code(check)
 md('## CHECK · Behavioral contracts\nThese call your live implementations; the author verification separately rejects three plausible wrong implementations.')
 code(definitions(P/'_check_l186.py')['checks']);code("assert checks(schedule,source_age,freshness_alert)=='PASS'")
 md('## PROVIDED · Visible trace generator, legal reads, metrics and full runner\nThe simulation calls all three learner functions above. Every policy receives the paired trace. There is no hidden API or model. Original source: labs/relkit/serving_l186.py.')
 for name in ['digest','make_trace','summarize','simulate','run_grid']:code(fns[name])
 md('## PROVIDED · Authenticate every original receipt\nThe packet includes all 300 per-run records, phase/cost receipts, original L176 timer code and its inherited artifact manifest. No probability arrays or checkpoint weights are needed for timing replay. Hashes protect identity; they do not establish original clock truth.')
 code('import base64, io, zipfile\nPACKET='+repr(packet)+'\nraw=base64.b64decode(PACKET)\nassert hashlib.sha256(raw).hexdigest()=='+repr(sha)+"\nE=Path('l186-packet'); E.mkdir(exist_ok=True)\nwith zipfile.ZipFile(io.BytesIO(raw)) as z:\n    for name in z.namelist(): assert not Path(name).is_absolute() and '..' not in Path(name).parts\n    z.extractall(E)\npins="+repr(pins),['data-payload'])
 code(definitions(P/'_audit_l186.py')['audit_receipts'])
 md('## RUN · All 81 cells and all 300 batch receipts\nThis computes new simulated responses and replays old timing metadata. It does not perform fresh model inference.')
 code("report=run_grid(CONFIG)\nreport['receipt_replay']=audit_receipts(E,pins)\nPath('l186-report.json').write_text(json.dumps(report,indent=2))\nprint(report['status'], report['cells'], 'cells;', report['requests'], 'responses')\nprint(json.dumps(report['receipt_replay'],indent=2))")
 md('## CHECK · Independent full response reconstruction\nDifferent implementation: indexed prefix maxima for available state, Lindley queue recurrence, integer monitor threshold. All 810,000 response rows must hash identically. This is an independent implementation on the shared deterministic input generator, not an independent real-world replication.')
 code(definitions(P/'_verify_l186.py')['independent'])
 code("lookup={(x['rate'],x['condition'],x['seed'],x['policy']):x for x in report['results']}\nchecked=0\nfor rate,condition,seed in itertools.product(CONFIG['rates'],CONFIG['conditions'],CONFIG['seeds']):\n    trace=make_trace(rate,condition,seed,CONFIG)\n    for policy in CONFIG['policies']:\n        rows=independent(trace,policy,CONFIG)\n        assert digest(rows)==lookup[rate,condition,seed,policy]['response_sha256']\n        checked+=len(rows)\nassert checked==810000\nprint('Independent response reconstruction:',checked)")
 md('## TRY · Inspect one complete cell\nPredict before changing condition or policy. The report contains all seeds and rates. Do not call the lowest observed latency a production guarantee.')
 code("policy,rate,condition,seed='cached',50,'normal',0\nselected=lookup[rate,condition,seed,policy]\nprint(json.dumps(selected,indent=2))")
 md('## EXIT · Write a serving contract\nUse six sentences: identity, read point, deadline, freshness, fallback, monitoring. Explain why all source timings are assumptions, why batch timers cannot establish request p99, and which labels are still unavailable. Ask the teacher for feedback; author execution does not establish mastery.')
 code("submission={'learner':'PENDING_WRITTEN_DEFENSE','contract':'','measurements_before_deployment':[]}\nPath('l186-submission.json').write_text(json.dumps(submission,indent=2))")
 md('## Original batch timer source · audit appendix\nThis is preserved source for interpreting L176 receipts. It is not executed here.\n\n```python\n'+(E/'packet/_run_l176.py').read_text()+'\n```')
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 for i,cell in enumerate(cells):cell.id=f'l186-{i:03}'
 return book
for solution in [False,True]:
 book=make(solution);path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
  old=nb.read(path,4)
  if [c.source for c in old.cells if c.cell_type=='code']==[c.source for c in book.cells if c.cell_type=='code']:
   book.metadata=old.metadata
   for a,b in zip([c for c in book.cells if c.cell_type=='code'],[c for c in old.cells if c.cell_type=='code']):a.outputs=b.outputs;a.execution_count=b.execution_count;a.metadata=b.metadata
 nb.write(book,path)
print('Built HTML, reference, explorer evidence and both portable notebooks')
