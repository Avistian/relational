"""Generate lesson, standalone notebooks and reference from canonical sources."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0101-static-vs-temporal';T='Static vs temporal: a graph query cannot read its future'
paper=json.loads((P/'_paper_l101_results.json').read_text());course=json.loads((P/'_experiment_l101_results.json').read_text())
def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text()
 for name in ['query-path','label-window']:
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l101/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l101/{name}.svg'
  alt={'query-path':'At query day 5, two-hop inputs include A, B and E; future C and late-arriving D are excluded.','label-window':'A label window ending at day 8 can train a day-10 model; a window ending at day 11 cannot.'}[name]
  s=s.replace('[[FIG:'+name+']]',f'<figure class="temporal-figure" tabindex="0"><img src="{src}" alt="{alt}"><figcaption>Illustrative computation. {alt}</figcaption></figure>')
 course_table='| Seed | Legal accuracy | Event-only accuracy | Static accuracy |\n|---|---:|---:|---:|\n'
 for seed in range(3):
  course_table+=f'| {seed} | '+' | '.join(f"{100*next(x for x in course['records'] if x['seed']==seed and x['arm']==arm)['test_accuracy']:.2f}%" for arm in ['legal','event_only','static'])+' |\n'
 course_table+='\n**Measured author-reference results:** all nine fits completed; synthetic diagnostic, not a paper score.'
 s=s.replace('[[COURSE_RESULTS]]',course_table)
 tab='| Method | Validation MAE / paper | Test MAE / paper | Verdict |\n|---|---:|---:|---|\n'
 for arm in ['global_zero','global_mean','global_median','entity_mean','entity_median']:
  rows=[next(x for x in paper['results'] if x['split']==split and x['arm']==arm) for split in ['val','test']]
  tab+='| '+arm+' | '+' | '.join(f"{x['mae']:.6f} / {x['paper_mae']:.3f}" for x in rows)+' | MATCH |\n'
 s=s.replace('[[PAPER_RESULTS]]',tab)
 for key,mount in [('WARMUP','l101-warmup'),('PREDICT','l101-predict'),('VISIBILITY','temporal-visibility')]:
  replacement={'WARMUP':'**Retrieval first:** answer the questions below without notes.','PREDICT':'**Commit before reading:** may a day-5 query use a day-4 record that arrived on day 7? Explain your rule.','VISIBILITY':'| Record | Event day | Arrival day | Value | Legal at day 5? |\n|---|---:|---:|---:|---|\n| Past purchase | 3 | 3 | 2 | Yes |\n| Late correction | 4 | 7 | 8 | No |\n| Boundary purchase | 5 | 5 | 4 | Yes |\n| Future outcome | 8 | 8 | 10 | No |'}[key] if portable else f'<div id="{mount}"></div>'
  s=s.replace('[['+key+']]',replacement)
 if portable:
  s=s.replace('href="../','href="https://avistian.github.io/relational/').replace('](../','](https://avistian.github.io/relational/')
  s=re.sub(r'\]\((0100-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s
head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 101 — {T}</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/temporal-lesson.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="0100-heterogeneous-gnn-checkpoint.html">Lesson 100</a></nav><header><p>Year 3 · Quarter 3 · Lesson 101</p><h1>{T}</h1></header>'
(R/'lessons'/f'{S}.html').write_text(head+render(prose()).replace('<table>', '<div class="temporal-scroll" tabindex="0" role="region" aria-label="Scrollable evidence table"><table>').replace('</table>', '</table></div>')+'<div id="l101-teachback"></div></article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','teachback','temporal-visibility','l101-lesson'])+'</body></html>')
checks={
 'available':"assert available(4,7,5) is False\nassert available(5,5,5) is True\nassert available(6,4,5) is False\nprint('PASS: arrival, event and equality boundaries')",
 'split_queries':"queries=[{'time':t,'label_ready':t+3} for t in [5,8,10,17,20]]\nassert split_queries(queries,10,20)==['train','purged','val','val','test']\nprint('PASS: maturity and split boundary placement')",
 'incoming_subgraph':"nodes={'a':(0,0),'b':(1,1),'c':(8,8),'d':(2,9),'e':(5,5)}\nedges=[('b','a',1),('c','b',8),('d','b',2),('e','b',5)]\nassert incoming_subgraph(nodes,edges,'a',5,2)==({'a','b','e'},{0,3})\nassert incoming_subgraph(nodes,edges,'a',9,2)==(set(nodes),set(range(4)))\nassert incoming_subgraph(nodes,edges,'a',5,0)==({'a'},set())\nprint('PASS: all hops and distinct query times')"}
source=(P/'relkit/temporal_l101.py').read_text();tree=ast.parse(source)
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 101 — '+T+'\n\n'+('Teacher solution: executed evidence does not certify learner mastery.' if solution else 'Student edition: complete the three live TODOs before running the experiment.')),
 nbf.v4.new_code_cell("# @colab-bootstrap — standalone; no course clone or hidden model import.\n# If dependencies are missing, install the pinned versions listed below first.\nimport sys\nprint('Python', sys.version.split()[0])"),
 nbf.v4.new_markdown_cell('## Runtime — PROVIDED\n\nTested Python 3.12.3. In a fresh environment install:\n```bash\npython -m pip install '+ ' '.join(f'{k}=={v}' for k,v in json.loads((P/'_verify_l101_results.json').read_text())['versions'].items())+'\n```\nLive Colab and a fresh installation are NOT_CHECKED. Code below is entirely inline. The small real task archive is downloaded and verified by SHA-256. The current kernel needs internet only on its first download.'),nbf.v4.new_markdown_cell(prose(True)),nbf.v4.new_markdown_cell('## Implement the information contract\n\nThe three TODO functions below are the live functions called by the experiment and traversal. A raised error means the task is unfinished; do not skip its CHECK.')]
 prelude='\n\n'.join(ast.get_source_segment(source,n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign)))
 cells.append(nbf.v4.new_code_cell(prelude))
 for n in tree.body:
  if not isinstance(n,ast.FunctionDef):continue
  task=n.name in checks;label='TODO' if task and not solution else 'PROVIDED'
  cells.append(nbf.v4.new_markdown_cell(f'### {label} · `{n.name}`\n\n'+(ast.get_docstring(n) or 'Visible implementation.')))
  code=ast.get_source_segment(source,n)
  if task and not solution:
   code=code[:code.index('\n')]+f"\n    # TODO: implement the stated contract; preserve the function signature.\n    raise NotImplementedError('{n.name}')"
  cells.append(nbf.v4.new_code_cell(code))
  if task:
   cells.append(nbf.v4.new_markdown_cell('### CHECK — run without editing'))
   cells.append(nbf.v4.new_code_cell(checks[n.name]))
 cells += [nbf.v4.new_markdown_cell('## Run the complete course diagnostic — PROVIDED\n\nPredict all three outcomes before running. These are new kernel results, separate from the author-reference table.'),nbf.v4.new_code_cell("fresh_course=course_experiment()\ndisplay(pd.DataFrame(fresh_course['records']).drop(columns=['test_predictions','test_labels']))\ndisplay(pd.DataFrame(fresh_course['split_audit']))"),
 nbf.v4.new_markdown_cell('## Full selected paper slice — PROVIDED\n\nAll five heuristic baselines, complete validation and test tables. No subsampling. This is not the full GNN benchmark. `MATCH` applies only to the ten printed MAE cells.'),
 nbf.v4.new_code_cell("fresh_paper=paper_replay(Path('l101-cache'),Path('l101-predictions.npz'))\ndisplay(pd.DataFrame(fresh_paper['results']))\nassert all(r['verdict']=='MATCH' for r in fresh_paper['results'])\nPath('l101-fresh-evidence.json').write_text(json.dumps({'paper':fresh_paper,'course':fresh_course},indent=2))\nprint('Full selected slice:',fresh_paper['status'],'| Full paper:',fresh_paper['full_paper_reproduction'])"),
 nbf.v4.new_markdown_cell('## EXIT TICKET\n\nSubmit the implementations, fresh JSON and the five-part defense from section 8. Explain one future path, one late record, one immature label, the train+validation test fit and the exact scope of MATCH.\n\n**PENDING_WRITTEN_DEFENSE** — running prepared code alone does not establish mastery.')]
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
 path=P/('solutions' if solution else '')/f'{S}.ipynb';path.parent.mkdir(exist_ok=True)
 for i,cell in enumerate(nb.cells):cell.id=f'l101-cell-{i:03d}'
 if solution and path.exists():
  old=nbf.read(path,as_version=4)
  old_code=[c for c in old.cells if c.cell_type=='code']
  new_code=[c for c in nb.cells if c.cell_type=='code']
  if [c.source for c in old_code]==[c.source for c in new_code]:
   for a,b in zip(new_code,old_code):a.outputs=b.outputs;a.execution_count=b.execution_count;a.metadata=b.metadata
 nbf.write(nb,path)
ref='''# Temporal visibility: a reference card

A query is **(typed entity, prediction time t, horizon Δ)**. Inputs stop at t; labels describe (t,t+Δ]. All boundaries must be declared.

| Gate | Check |
|---|---|
| Rows | Event time ≤ t and availability time ≤ t |
| Features | Retrieve the version available at t; creation time is insufficient |
| Edges | Relationship and both endpoints were available at t |
| Multi-hop | Apply the original query cutoff at every hop |
| Sampling | Filter legal neighbors before finite-fanout selection |
| Fitting | Every training label has matured by the model-fit time |
| Selection | Validation decides; test scores never decide |
| Preprocessing | Vocabulary, imputation and other fitted state use legal fit data |
| Caches | Include query time and model state, not just entity ID |

**Two different clocks:** a historical training query hides its own future inputs, even though its future label is known by the later model-fit time.

**Counterexample:** an event on day 4 arriving day 7 cannot serve a day-5 query. A query on day 8 with horizon 3 cannot train a model fitted day 10.

**Reproduction:** the complete five-baseline RelBench driver-position slice matches ten rounded MAE cells. Full-paper and historical execution identity remain NOT_ESTABLISHED.

[Lesson and worked trace](../lessons/0101-static-vs-temporal.html) · [Contract](../labs/l101-reproduction.md) · [Fey et al.](https://proceedings.mlr.press/v235/fey24a.html)
'''
(R/'reference/temporal-visibility.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Temporal visibility</title><link rel="stylesheet" href="../assets/lesson.css"></head><body><article>'+render(ref)+'</article></body></html>')
print('Built lesson, both notebooks and reference')
