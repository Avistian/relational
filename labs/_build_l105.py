"""Build L105 HTML/reference and visible standalone student/solution notebooks."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0105-continuous-time';TITLE='Continuous time: event streams versus snapshots'
CAP={'aggregation':'Synthetic six-event stream: width 10 yields five binary snapshot edges; count weights sum to six. Timestamp ties remain simultaneous.','order':'Synthetic directed contacts, separate from Wikipedia: identical count-weighted snapshots can conceal opposite temporal reachability.','release':'Synthetic query at 6 seconds: three eligible records are withheld by closed-window access. Reading the completed current window exposes the event at 8. Vertically separated dots at 3 represent two tied events.','results':'Measured full Wikipedia census, not a prediction benchmark. Fixed widths from origin zero; weights preserve the event total. Wait assumes availability equals event time.'}
report=json.loads((P/'_analysis_l105_results.json').read_text())
def measured():
 s='**Author execution: PASS, all 157,474 events × all three declared widths.**\n\n'
 s+='| Window | Binary edge records | Binary collapse | Hidden strict pairs | Mean wait |\n|---|---:|---:|---:|---:|\n'
 for label,r in zip(['Hourly','Daily','Weekly'],report['records']):s+=f'| {label} | {r["snapshot_edges"]:,} | {r["collapsed_percent"]:.2f}% | {r["hidden_strict_pairs"]:,} | {r["delay_mean_seconds"]/3600:.2f} h |\n'
 s+='\nWeighted count sums: **157,474 in every condition**. All 4,816 timestamp-tied pairs are excluded from the strict-order count.\n\n| Window | Mean past records withheld / query | Mean nonpast exposure / query | Unreleased events at final query |\n|---|---:|---:|---:|\n'
 for label,r in zip(['Hourly','Daily','Weekly'],report['records']):s+=f'| {label} | {r["mean_withheld_past"]:,.2f} | {r["mean_nonpast_exposure"]:,.2f} | {r["unreleased_events_at_last_query"]:,} |\n'
 return s+'\nThese global counts are deterministic; no error bars or confidence intervals are warranted for this fixed-file census. They are not predictive performance estimates.'
def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',measured())
 for key,cap in CAP.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l105/{key}.png').read_bytes()).decode() if portable else f'../labs/figures/l105/{key}.svg'
  s=s.replace('[[FIG:'+key+']]',f'<figure class="stream-figure" tabindex="0"><img src="{src}" alt="{cap}"><figcaption>{cap}</figcaption></figure>')
 for key,id_,fallback in [('AGGREGATION','l105-aggregation','**Portable comparison:** width 10 → five binary records; width 20 → four. Both have weighted total six.'),('RELEASE','l105-release','**Portable boundary:** query 9 → no released records; query 10 → four. If the time-8 event arrives at 15, the first window waits until 15.'),('PREDICT','l105-predict','**Predict:** do count weights recover event order, or only preserve interaction totals? Give a counterexample before continuing.'),('TEACHBACK','l105-teachback','**Write your defense before opening the solution.**')]:s=s.replace('[['+key+']]',fallback if portable else f'<div id="{id_}"></div>')
 source=(P/'relkit/stream_l105.py').read_text();fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='visible_snapshot_mask');snippet='\n'.join(source.splitlines()[fn.lineno-1:fn.end_lineno])
 s=s.replace('[[CODE:release]]','Implement the release gate yourself in TODO 3 below.' if portable else '```python\n'+snippet+'\n```')
 if portable:
  s=re.sub(r'<details><summary>(.*?)</summary>',r'**\1**\n\n',s).replace('</details>','')
  s=s.replace('href="../','href="https://avistian.github.io/relational/').replace('](../','](https://avistian.github.io/relational/')
  s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s
head='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 105 — '+TITLE+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/event-snapshot.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="0104-information-leakage-in-time.html">Lesson 104</a></nav><header><p class="stream-kicker">Year 3 · Quarter 3 · Lesson 105</p><h1>'+TITLE+'</h1></header>'
body=render(prose()).replace('<table>','<div class="stream-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
(R/'lessons'/f'{S}.html').write_text(head+body+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in ['predict','teachback','event-snapshot-viz','l105-lesson'])+'</body></html>')
fixture="""# PROVIDED — the teaching example, separate from Wikipedia.
def fixture(t,u=None,v=None,a=None):
    t=np.asarray(t,dtype=float); n=len(t)
    return {'t':t,'a':np.asarray(a if a is not None else t,dtype=float),
            'u':np.asarray(u if u is not None else [0]*n,dtype=np.int64),
            'v':np.asarray(v if v is not None else [0]*n,dtype=np.int64),
            'e':np.arange(n,dtype=np.int64)}
worked=fixture([1,3,3,8,12,19],u=[0,0,1,0,1,0],v=[0,0,1,1,0,0])
"""
checks={
'bin_index':"""np.testing.assert_array_equal(bin_index(np.array([0,9.999,10,20]),10),[0,0,1,2])
np.testing.assert_array_equal(bin_index(np.array([7,17]),10,origin=7),[0,1])
for invalid_width in [0,-1,float('nan')]:
    try: bin_index(np.array([1]),invalid_width)
    except ValueError: pass
    else: raise AssertionError('Reject invalid widths before assigning bins')
print('PASS: exact boundary, nonzero origin, invalid widths')""",
'aggregate_events':"""snap=aggregate_events(worked,10)
np.testing.assert_array_equal(snap['bin'],[0,0,0,1,1])
np.testing.assert_array_equal(snap['u'],[0,0,1,0,1])
np.testing.assert_array_equal(snap['v'],[0,1,1,0,0])
np.testing.assert_array_equal(snap['count'],[2,1,1,1,1])
assert snap['count'].sum()==6
late=aggregate_events(fixture([1,2],u=[0,1],a=[1,12]),10)
np.testing.assert_array_equal(late['release'],[12,12])
assert len(aggregate_events(fixture([]),10)['count'])==0
assert len(aggregate_events(fixture([1,2],u=[0,1],v=[1,0]),10)['count'])==2
print('PASS: typed endpoints, count conservation, empty stream, whole-window late arrival')""",
'visible_snapshot_mask':"""snap=aggregate_events(worked,10)
assert snap['count'][visible_snapshot_mask(snap,9)].sum()==0
assert snap['count'][visible_snapshot_mask(snap,10)].sum()==4
assert snap['count'][visible_snapshot_mask(snap,20)].sum()==6
late=aggregate_events(fixture([1,2],u=[0,1],a=[1,12]),10)
assert not visible_snapshot_mask(late,10).any()
assert visible_snapshot_mask(late,12).all()
assert not visible_snapshot_mask(aggregate_events(fixture([21]),10),29).any()
print('PASS: publication boundary, delayed window, final partial bin')"""}
intent={
'bin_index':'Return int64 window indices for a one-dimensional array. Validate finite positive width, finite origin/times, and t ≥ origin. Use the half-open convention; test exact boundaries, not just mid-window examples.',
'aggregate_events':'Return aligned arrays bin/u/v/count/start/end/first/last/release sorted by (bin,u,v). Call validate_events and your bin_index. Use np.unique on structured keys plus reduction operations. first/last are audit metadata. For release, reduce availability over each entire window, then map that window maximum to every edge in it. Handle empty input.',
'visible_snapshot_mask':'Return one boolean per snapshot edge: is its complete window published by the query? Reject a nonfinite query. Explain why equality is allowed here while the event-history rule remains strict.'}
bootstrap="""# @colab-bootstrap — standalone NumPy runtime, no relkit or repository import.
import sys, subprocess, importlib.metadata
try: installed=importlib.metadata.version('numpy')
except importlib.metadata.PackageNotFoundError: installed=None
if installed!='2.5.0':
    subprocess.check_call([sys.executable,'-m','pip','install','numpy==2.5.0'])
print('Python',sys.version.split()[0], '| required NumPy 2.5.0')
"""
full="""# PROVIDED — complete raw-data replay. About 560 MB on first download.
# You may place the authenticated raw file here before running.
RAW_PATH=Path('l105-data/wikipedia.csv')
events=load_wikipedia(RAW_PATH)
results=[audit_stream(events,width) for width in WIDTHS]
EXPECTED=__EXPECTED__
assert results==EXPECTED, 'Full-data results differ: inspect protocol and live TODO functions'
from IPython.display import HTML, display
rows=''.join(f"<tr><td>{r['width_seconds']}</td><td>{r['snapshot_edges']:,}</td><td>{r['collapsed_percent']:.2f}%</td><td>{r['hidden_strict_pairs']:,}</td><td>{r['delay_mean_seconds']/3600:.2f}</td></tr>" for r in results)
display(HTML('<table><tr><th>Window seconds</th><th>Binary edges</th><th>Collapse</th><th>Hidden strict pairs</th><th>Mean wait hours</th></tr>'+rows+'</table>'))
import json
Path('l105-fresh.json').write_text(json.dumps(results,indent=2)+'\\n')
print('PASS: full file, all three conditions, exact author-report equality')
""".replace('__EXPECTED__',repr(report['records']))
oracle="""# PROVIDED — independently group the complete projected stream in SQLite.
import sqlite3
from collections import Counter,defaultdict
conn=sqlite3.connect(':memory:')
conn.execute('CREATE TABLE events(u INTEGER,v INTEGER,t REAL)')
conn.executemany('INSERT INTO events VALUES(?,?,?)',
    ((int(u),int(v),float(t)) for u,v,t in zip(events['u'],events['v'],events['t'])))
for width,measured in zip(WIDTHS,results):
    oracle=conn.execute('SELECT CAST(t / ? AS INTEGER),u,v,COUNT(*),MIN(t),MAX(t) FROM events GROUP BY 1,2,3 ORDER BY 1,2,3',(width,)).fetchall()
    snap=aggregate_events(events,width)
    np.testing.assert_array_equal(np.column_stack([snap[k] for k in ['bin','u','v','count','first','last']]),np.array(oracle))
    buckets=defaultdict(Counter)
    for t in events['t']: buckets[int(t//width)][float(t)]+=1
    past_pairs=nonpast=0
    for times in buckets.values():
        previous=0; total=sum(times.values())
        for t,count in sorted(times.items()):
            past_pairs+=previous*count; nonpast+=(total-previous)*count; previous+=count
    assert measured['hidden_strict_pairs']==measured['withheld_past_sum']==past_pairs
    assert measured['nonpast_exposure_sum']==nonpast
    assert nonpast-past_pairs==len(events['t'])+2*measured['tied_timestamp_pairs']
conn.close()
print('PASS: all snapshot rows and pair totals reconstructed independently')
"""
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 105 lab · '+TITLE+'\n\n**'+('Solution' if solution else 'Student')+'** · Full course representation experiment, not a model benchmark. All implementations are visible. Start with retrieval; leave the solution closed until you attempt the live tasks.'),nbf.v4.new_code_cell(bootstrap)]
 # Short sections rather than one oversized prose cell; retain standalone narrative.
 for part in re.split(r'(?=^## )',prose(True),flags=re.M):
  if part.strip():cells.append(nbf.v4.new_markdown_cell(part.strip()))
 cells.append(nbf.v4.new_markdown_cell('## Implement and test\n\nPredict the expected output before each CHECK. The full-data experiment below calls your functions directly. The reference evidence above is author output, not a claim about your current kernel.'))
 source=(P/'relkit/stream_l105.py').read_text()
 for chunk in re.split(r'^# %% ',source,flags=re.M)[1:]:
  title,code=chunk.split('\n',1);code=code.strip();task=next((name for name in checks if 'def '+name+'(' in code),None)
  cells.append(nbf.v4.new_markdown_cell('### '+title+ ('\n\n'+intent[task]+'\n\n**Prediction:** write the expected check outputs before implementing.' if task else '')))
  if task and not solution:
   tree=ast.parse(code);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef));lines=code.splitlines();signature=lines[fn.lineno-1]
   code=signature+'\n    """'+intent[task]+'"""\n    raise NotImplementedError("Implement '+task+' before the CHECK")'
  cells.append(nbf.v4.new_code_cell(code))
  if title=='PROVIDED: validate the event contract':cells.append(nbf.v4.new_code_cell(fixture))
  if task:cells.extend([nbf.v4.new_markdown_cell('### CHECK · '+task),nbf.v4.new_code_cell(checks[task])])
 cells.extend([nbf.v4.new_markdown_cell('## RUN · the complete experiment\n\nNo hidden training or sampled data. Three widths, every raw record, exact pinned bytes. If a CHECK failed, fix it before downloading. This run does not require the course repository or any saved model.'),nbf.v4.new_code_cell(full),nbf.v4.new_markdown_cell('## CHECK · independent reconstruction\n\nA second algorithm should reach the same answer. SQLite handles endpoint grouping; a timestamp walk counts pairs without the vectorized searchsorted implementation.'),nbf.v4.new_code_cell(oracle),nbf.v4.new_markdown_cell('## EXIT · submit evidence\n\nSubmit your three live functions, the complete result table, and the 150–250 word representation defense described above. Add a new boundary fixture of your own. A matching report alone is not evidence of mastery.\n\n**Source and protocol:** [reproduction ledger](https://avistian.github.io/relational/labs/l105-reproduction.md).\n\n**Published model results:** NOT_APPLICABLE to this course audit. Live Colab: NOT_CHECKED. No model fitting or upstream score reproduction occurred here.')])
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
 for i,c in enumerate(cells):c.id=f'l105-{i:03d}'
 path=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and path.exists():
  old=nbf.read(path,as_version=4);before=[c for c in old.cells if c.cell_type=='code'];after=[c for c in nb.cells if c.cell_type=='code']
  if [c.source for c in before]==[c.source for c in after]:
   nb.metadata=old.metadata
   for a,b in zip(after,before):a.outputs=b.outputs;a.execution_count=b.execution_count;a.metadata=b.metadata
 nbf.write(nb,path)
ref='''# Event streams and snapshots · reference

**CTDG:** timed changes or interactions. **DTDG:** graph sequence on a chosen time grid. Continuous-time representation does not require a point-process model. Ties remain simultaneous unless a reliable secondary clock exists.

| Representation | Retains | Does not generally retain |
|---|---|---|
| Endpoint/time event stream | Repeated interactions and recorded times | Unobserved ingestion history or sub-timestamp ordering |
| Binary window graph | Pair existence within each window | Multiplicity and within-window timing |
| Count-weighted window graph | Pair interaction totals within each window | Exact times and event order |
| State snapshot | Relationships active at a moment | Full history of changes that led to that state |

**Bin:** k = floor((t − origin) / width). Window [origin+k×width, origin+(k+1)×width). At its right boundary, an event belongs to the next window. Declare origin, units, time zone and partial-window policy.

**Before-event history:** event time < query; availability ≤ query.

**Immutable complete-window release:** max(window end, latest constituent availability), over all events in the window. Allow release ≤ query. A live service needs a completeness policy; the maximum arrival seen so far cannot certify that no late events remain.

**Binary collapse:** N − distinct(window,user,page). **Hidden strict pairs:** sum over windows choose(n,2) minus sum over timestamp groups choose(m,2). Counts cover global records, not graph-model dependencies. **Wait:** release − event time. With a=t this becomes window end − event time.

**Choice:** choose event history when within-window sequence or arbitrary-time freshness matters. Choose weighted windows when completed-window totals answer the task and latency is acceptable. Partial incremental windows require historical versions. Keeping first/last times defines a richer summary, not a recovery of every original event.

**Evidence:** L105 executes all 157474 Wikipedia records at hourly/daily/weekly widths. Exact course audit; no model accuracy or full-paper replication is claimed. See the protocol for projected fields and authenticated bytes.

[Lesson](../lessons/0105-continuous-time.html) · [Protocol](../labs/l105-reproduction.md) · [TGN §2](https://arxiv.org/html/2006.10637v3#S2) · [JODIE data](https://snap.stanford.edu/jodie/)
'''
(R/'reference/event-stream-snapshots.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Event streams and snapshots reference</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/event-snapshot.css"></head><body><article>'+render(ref).replace('<table>','<div class="stream-scroll" tabindex="0"><table>').replace('</table>','</table></div>')+'</article></body></html>')
print('Built lesson, student/solution notebooks, reference')
