"""Aligned lesson, standalone visible notebooks and concise timestamp reference."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0109-database-timestamp-contracts';TITLE='Database timestamps: reconstruct what was known'
canonical=(P/'relkit/database_l109.py').read_text();experiment=(P/'relkit/f1_l109.py').read_text()
CAP={'clocks':'Synthetic immutable event: occurred day 2, observed day 7; exclude from the day-5 query.','versions':'Synthetic append-only histories: select both row versions before joining their keys. Day 10 removes the selected tombstone.','maturity':'Synthetic label window (5,9], with final constituent arrival on 12; fit no earlier than 12 under certified completeness.','results':'Author-reference reconstruction: full selected validation/test populations, all five deterministic methods. Ticks show rounded paper targets; no seed uncertainty applies.'}

def result_text():
 r=json.loads((P/'evidence/l109/reproduction.json').read_text());g=json.loads((P/'evidence/l109/graph-census.json').read_text())
 s='**Author-reference evidence, not your current notebook output.** All 8,712 regenerated labels match both the cached targets and the original source SQL exactly.\n\n| Split | Queries | Scheduled / nonempty windows | No prior-year result |\n|---|---:|---:|---:|\n'
 for split in ['train','val','test']:
  a=r['labels'][split];s+=f"| {split} | {a['rows']:,} | {a['scheduled_windows']} / {a['nonempty_windows']} | {a['queries_without_prior_year_result']} |\n"
 s+='\n| Estimator | Validation MAE | Test MAE | Paper rounding verdict |\n|---|---:|---:|---|\n'
 for arm in ['global_zero','global_mean','global_median','entity_mean','entity_median']:
  a=[x for x in r['results'] if x['arm']==arm];s+=f"| {arm.replace('_',' ')} | {a[0]['mae']:.6f} | {a[1]['mae']:.6f} | MATCH / MATCH |\n"
 s+=f"\nAll ten fresh prediction vectors match the original baseline function within 1e-12. The real graph census checks **{g['relation_comparisons']:,} relations across {g['query_cutoffs']} cutoffs**; all ordered edge arrays match independent SQL joins. The synthetic suite adds 520 SQL selector comparisons, 390 SQL graph comparisons and five rejected broken implementations. These are correctness counts, not accuracy confidence intervals. Paid compute: USD0."
 return s

def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',result_text())
 for name,cap in CAP.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l109/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l109/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0"><img src="{src}" alt="{cap}"><figcaption>{cap}</figcaption></figure>')
 for name in ['asof_versions','legal_history','label_ready']:
  node=next(n for n in ast.parse(canonical).body if isinstance(n,ast.FunctionDef) and n.name==name)
  text='Implement `'+name+'` in the live TODO below.' if portable else '```python\n'+ast.get_source_segment(canonical,node)+'\n```'
  s=s.replace('[[CODE:'+name+']]',text)
 for tag,id_,fallback in [('WIDGET','l109-history','**Portable prediction task:** write the selected edge, parent value and readiness at days 6, 8 and 10. Use the worked answer only after committing.'),('PREDICT','l109-predict','**Predict before results:** does matching all published scores prove that the historical feature values were available? Explain before reading.'),('TEACHBACK','l109-teachback','**Written defense:** distinguish the two historical clocks, version selection, graph visibility, label maturity and what the reproduction proves.')]:
  s=s.replace('[['+tag+']]',fallback if portable else '<div id="'+id_+'"></div>')
 if portable:
  s=s.replace('<div id="warmup"></div>','').replace('](../','](https://avistian.github.io/relational/')
  s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def document(title,body,scripts=()):
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/event-snapshot.css"><link rel="stylesheet" href="../assets/database-history.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0108-temporal-neighbor-sampling.html">Lesson 108</a></nav><header><p class="stream-kicker">Year 3 · Quarter 3 · Lesson 109</p><h1>'+title+'</h1></header>'+render(body).replace('<table>','<div class="stream-scroll" tabindex="0"><table>').replace('</table>','</table></div>')+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),['retrieval-pool','retrieval-bank','predict','teachback','database-history-viz','l109-lesson']))
bootstrap="""# @colab-bootstrap — CPU-only; downloads less than 1 MB of public input archives.
import sys,subprocess,importlib.metadata
required={'numpy':'2.5.0','pandas':'3.0.3','pyarrow':'24.0.0'}
missing=[]
for package,version in required.items():
    try:current=importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:current=None
    if current!=version:missing.append(package+'=='+version)
if missing:subprocess.check_call([sys.executable,'-m','pip','install',*missing])
print('Python',sys.version.split()[0], 'runtime pins',required)
"""
fixture="""# PROVIDED — stored histories are held fixed throughout the temporal intervention.
people=[dict(id='A',valid_from=0,observed_at=0,revision=0,deleted=False,value=1),
        dict(id='A',valid_from=0,observed_at=9,revision=1,deleted=False,value=9),
        dict(id='B',valid_from=0,observed_at=7,revision=0,deleted=False,value=2)]
facts=[dict(id='r',valid_from=2,observed_at=3,revision=0,deleted=False,person='A',value=5),
       dict(id='r',valid_from=2,observed_at=8,revision=1,deleted=False,person='B',value=6),
       dict(id='r',valid_from=2,observed_at=10,revision=2,deleted=True,person='B',value=6)]
"""
checks={
'asof_versions':"""assert asof_versions(people,6)['A']['value']==1, 'A late correction cannot rewrite a day-6 feature'
assert 'B' not in asof_versions(people,6), 'B is not observed until day 7'
assert asof_versions(facts,8)['r']['person']=='B', 'Cutoff ties are inclusive here'
assert not asof_versions(facts,10), 'Select the tombstone before removing the deleted row'
print('PASS: corrections, availability, ties and deletion')""",
'legal_history':"""events=[{'event_time':2,'observed_at':7},{'event_time':5,'observed_at':5},{'event_time':9,'observed_at':3}]
assert legal_history(events,5)==[1], 'Both clocks must satisfy this inclusive cutoff'
assert legal_history(events,7)==[0,1]
print('PASS: both clocks and inclusive boundary')""",
'label_ready':"""assert label_ready(5,4,[6,12])==12, 'A late constituent delays fitting'
assert label_ready(5,4,[])==9, 'Zero outcomes still require a closed future window'
assert label_ready(5,4,[6,7])==9, 'Early arrivals do not shorten the target horizon'
print('PASS: window closure and constituent arrival; completeness is assumed')"""}
goals={
'asof_versions':'**Goal:** select one eligible historical version per ID. **Why:** a later value or key must not enter an earlier graph. **Hint:** eligibility, ordered selection, and deletion are three distinct steps; filter tombstones last.',
'legal_history':'**Goal:** return eligible immutable-event indices. **Why:** event time alone admits late arrivals. **Hint:** the query must satisfy both clock comparisons under the stated boundary.',
'label_ready':'**Goal:** calculate the earliest fit time under certified completeness. **Why:** an ended window can still be missing outcomes. **Hint:** neither the horizon nor the constituent arrivals may be ignored.'}
graph_run="""# CHECK/RUN — the graph calls YOUR historical selector, with no fallback implementation.
for day,expected in [(6,[('r','A',3)]),(8,[('r','B',8)]),(10,[])]:
    graph=graph_at(people,facts,day)
    assert graph['edges']==expected
    print('day',day,':',graph['edges'])
# Three questions share the same stored immutable events; YOUR eligibility function executes.
events=[{'event_time':2,'observed_at':7},{'event_time':5,'observed_at':5}]
for day in [5,6,7]:print('legal event indices at',day,legal_history(events,day))
assert label_ready(5,4,[6,12])>9
print('Do not fit the day-5 label at day 9: the source is incomplete.')
"""
real_run="""# RUN — all nine database tables; all 8,712 labels; all ten selected paper cells.
# Visible implementation above executes; no hidden course import or cached score fallback.
report=run_reproduction(Path('l109-data'),Path('l109-output'))
assert all(row['verdict']=='MATCH' for row in report['results'])
display(pd.DataFrame(report['labels']).T)
display(pd.DataFrame(report['results'])[['split','arm','mae','paper_mae','gap','verdict']])
print('Full-paper parity:',report['full_paper_reproduction'])
Path('l109-fresh.json').write_text(json.dumps(report,indent=2))
"""
schema_run="""# RUN — inspect the timestamped REG spec, preserving stable node identities.
tables,tasks,metadata=load_inputs(Path('l109-data'))
display(pd.DataFrame([{'table':name,'rows':len(tables[name]),'primary_key':meta['pkey_col'],
                      'event_column':meta['time_col'],'observed_column':'NOT_RECORDED',
                      'relations':str(meta['fkey_col_to_pkey_table'])} for name,meta in metadata.items()]))
graph=graph_snapshot(tables,metadata,VAL)
print('At validation boundary:',sum(map(len,graph['nodes'].values())),'nodes;',
      sum(map(len,graph['edges'].values())),'edges;',graph['dropped_dangling'],'omitted dangling edges')
# Check a full real relation against a direct key join at exactly the same cutoff.
result_rows=tables['results'].query('date <= @VAL')
driver_rows=tables['drivers']
expected=result_rows[['resultId','driverId']].merge(driver_rows[['driverId']],on='driverId').sort_values(['resultId','driverId']).to_numpy(dtype=np.int64)
np.testing.assert_array_equal(graph['edges'][('results','driverId','drivers')],expected)
print('PASS: real result→driver relation. This is release event-time visibility, not ingestion evidence.')
"""
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 109 · '+TITLE+'\n\n'+('Solution' if solution else 'Student lab')+' · Three live TODOs, immediate CHECKs, and a written EXIT. Default CPU execution reconstructs the entire selected F1 label/heuristic experiment. The separate source and SQL graph audits have exact repository commands in the protocol. No learner mastery is inferred.'),nbf.v4.new_code_cell(bootstrap)]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if section.strip():cells.append(nbf.v4.new_markdown_cell(section))
 cells.extend([nbf.v4.new_markdown_cell('## Implement the historical contract\n\nThe following synthetic fixtures have the two clocks and version histories absent from the F1 archive. Their tests exercise your code directly.'),nbf.v4.new_code_cell(fixture)])
 for chunk in re.split(r'^# %% ',canonical,flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);task=next((name for name in checks if 'def '+name+'(' in body),None)
  cells.append(nbf.v4.new_markdown_cell('### '+heading+ ('\n\n'+goals[task] if task else '')))
  if task and not solution:
   node=next(n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name==task)
   body=body.splitlines()[node.lineno-1]+'\n    raise NotImplementedError("TODO: '+task+'")\n'
  cells.append(nbf.v4.new_code_cell(body.strip()))
  if task:cells.extend([nbf.v4.new_markdown_cell('### CHECK · '+task),nbf.v4.new_code_cell(checks[task])])
 cells.extend([nbf.v4.new_markdown_cell('## Trace the historical graph and label boundary'),nbf.v4.new_code_cell(graph_run)])
 cells.append(nbf.v4.new_markdown_cell('## Complete selected reproduction implementation\n\nThese PROVIDED functions expose the full archive verification, source-derived query schedule, independent target construction and five deterministic estimators. Read `regenerate` alongside `DriverPositionTask.make_table` in the pinned source. No neural model or training loop applies to these heuristic estimators. Targets are regenerated before fitting; predictions cannot inspect evaluation targets.'))
 for chunk in re.split(r'^# %% ',experiment,flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);cells.extend([nbf.v4.new_markdown_cell('### '+heading),nbf.v4.new_code_cell(body.strip())])
 cells.extend([nbf.v4.new_markdown_cell('## Execute and inspect the complete evidence\n\nThe cell authenticates less than 1 MB of public archives. It compares full query populations and targets before calculating the scores. Its fresh results are distinct from the author-reference table above.'),nbf.v4.new_code_cell(real_run),nbf.v4.new_markdown_cell('## Database-to-graph handoff\n\nInspect the declared time column for every table. The source assumes undated tables are eligible at every cutoff; do not reinterpret that assumption as measured availability. The independent all-310-cutoff SQLite census is a separate author check; this cell constructs a complete real snapshot and validates its result-to-driver relation.'),nbf.v4.new_code_cell(schema_run)])
 cells.append(nbf.v4.new_markdown_cell('## EXIT · defend the contract\n\nSubmit your three TODOs and a new late-parent or corrected-key counterexample. Include your fresh label report and a timestamped graph specification. Explain why a matched paper score cannot establish unrecorded availability, why no-future-race drivers are absent from this benchmark task, and what a label completeness certificate supplies.\n\nFor the independent source/SQL audits, run the commands in [the reproduction contract](https://avistian.github.io/relational/labs/l109-reproduction.md). Ask the teaching agent for feedback. Status: **PENDING_WRITTEN_DEFENSE**.'))
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}})
 for i,c in enumerate(cells):c.id=f'l109-{i:03d}'
 path=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and path.exists():
  old=nbf.read(path,as_version=4);a=[c for c in cells if c.cell_type=='code'];b=[c for c in old.cells if c.cell_type=='code']
  if [c.source for c in a]==[c.source for c in b]:
   nb.metadata=old.metadata
   for x,y in zip(a,b):x.outputs=y.outputs;x.execution_count=y.execution_count;x.metadata=y.metadata
 nbf.write(nb,path)
 if solution and all(c.execution_count is not None for c in cells if c.cell_type=='code'):
  html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
ref='''## Clock contract

Event time: when an immutable occurrence happened. Valid time: when a state applies. Observed time: when the predictor could use the value. System time: when a particular store recorded it; downstream availability may be later. Name the timezone, precision, system boundary and tie order.

## Historical state

For the course effective-step representation, filter valid_from ≤ t and observed_at ≤ t. Select the greatest (valid_from, observed_at, revision) per ID, then remove selected tombstones. Reject duplicate keys and missing clocks. Do not join current values and filter only their creation times.

## Historical graph

Select both endpoint histories before following the selected child FK. Keep stable typed IDs. An edge requires both endpoints; count dangling references. Every hop in a snapshot uses the root cutoff. TGAT recursive event-time cutoffs answer a different question.

## Feature and label boundaries

Immutable features: event ≤ t AND observed ≤ t. Future target window: (t,t+h]. Fit no earlier than max(t+h, required arrivals), under an explicit completeness certificate. Observed arrivals alone cannot prove absence of unseen records. Fit preprocessing only on permitted data.

## Release assumptions versus measured knowledge

The F1 release uses event dates, and treats undated tables as always eligible. It supplies no actual ingestion/version history. Its driver-position population requires a future-window result; its lower-only activity filter is not a prior-history eligibility rule. Exact label/score reproduction establishes the named benchmark slice, not deployment validity or full-paper parity.

## Contract template

For each table/field: identity; event/valid column; observed column and system; revision/deletion rule; tie convention; missing-time policy. For each relation: historical FK; endpoint rule; missing-parent policy; multi-hop cutoff. For each label: horizon; constituent availability; completeness; version; permitted fit boundary.

[Lesson 109](../lessons/0109-database-timestamp-contracts.html) · [Protocol](../labs/l109-reproduction.md) · [RelBench](https://arxiv.org/html/2407.20060v1) · [Bitemporal history](https://martinfowler.com/articles/bitemporal-history.html)
'''
(R/'reference/database-timestamp-contracts.html').write_text(document('Database timestamp contracts · quick reference',ref))
print('Built L109 lesson, reference, student and solution notebooks')
