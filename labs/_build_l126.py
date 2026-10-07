"""Build aligned HTML/reference and segmented standalone student/solution labs."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0126-relbench-beta';TITLE='RelBench beta: from database to evaluated predictions'
canonical=(P/'relkit/beta_l126.py').read_text();tree=ast.parse(canonical)
checks=(P/'_check_l126.py').read_text();nodes={n.name:ast.get_source_segment(checks,n) for n in ast.parse(checks).body if isinstance(n,ast.FunctionDef)}
functions={n.name:ast.get_source_segment(canonical,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
summary=json.loads((P/'evidence/l126/summary.json').read_text())
results='**Author-reference execution · complete F1 API tour · COURSE_ONLY.**\n\n| Table | Rows exposed after the test cap |\n|---|---:|\n'
for name,item in summary['schema']['tables'].items():results+=f"| {name} | {item['rows']:,} |\n"
results+=f"\n**{summary['schema']['total_rows']:,} rows**, nine tables, {len(summary['schema']['relations'])} declared foreign-key relations. All **7,453 train / 499 validation / 760 test** task rows are loaded. The training median is **{summary['training_median']:.6f}**; validation MAE **{summary['validation_mae']:.6f}**, test MAE **{summary['test_mae']:.6f}**. All test predictions were independently rescored in SQL. [Measured report](../labs/evidence/l126/summary.json) · [Independent audit](../labs/_audit_l126_results.json).\n"
captions={'boundaries':'Synthetic timeline: the dataset cap at day 30 admits five events, but the query at day 10 may use only days 5 and 10. Labels have a separate future window.','identity':'Worked keyed-score example: positions 1,2,0 restore evaluator order; entity A alone cannot distinguish two query times.','average-precision':'Synthetic five-row example: three tied-score thresholds contribute 1/3, 2/9 and 1/5 to AP 0.755556.'}
fallback='Static AP trace: labels[1,0,1,0,1], scores[0.9,0.5,0.5,0.1,0.1]. Grouped AP=0.755556. Reordering within ties leaves it unchanged. Giving every row score0.5 yields prevalence3/5=0.6.'
def prose(portable=False):
    text=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results)
    for name,cap in captions.items():
        src='data:image/png;base64,'+base64.b64encode((P/f'figures/l126/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l126/{name}.svg'
        text=text.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0"><img src="{src}" alt="{cap}"><figcaption>{cap}</figcaption></figure>')
    text=text.replace('[[WARMUP]]','Recall the difference between entity identity, prediction time and label horizon.' if portable else '<div id="warmup"></div><noscript><p>Recall the difference between entity identity, prediction time and label horizon.</p></noscript>')
    text=text.replace('[[AP_WIDGET]]',fallback if portable else '<div id="l126-ap"></div><noscript><p>'+fallback+'</p></noscript>')
    text=text.replace('[[TEACHBACK]]','Explain which historical identity claim remains unestablished after all source checks pass.' if portable else '<div id="l126-teachback"></div>')
    if portable:
        text=text.replace('](../','](https://avistian.github.io/relational/');text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
    return text

def document(title,text,interactive=False):
    body=re.sub(r'<table([^>]*)>',r'<div class="stream-scroll" tabindex="0"><table\1>',render(text)).replace('</table>','</table></div>')
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','event-snapshot','reproduction','task-table'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0125-pytorch-frame-deep-dive.html">Lesson 125</a></nav><header><p class="stream-kicker">Year4 · Quarter1 · Lesson126</p><h1>'+title+'</h1></header>'+body+'</article>'+(''.join('<script src="../assets/'+x+'.js"></script>' for x in ['retrieval-pool','retrieval-bank','teachback','average-precision-viz','l126-lesson']) if interactive else '')+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
bootstrap='''# @colab-bootstrap: proposed Colab installer, live Colab NOT_CHECKED.
import sys,subprocess,platform
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable,'-m','pip','install','-q',
        'relbench==1.1.0','numpy==1.26.4','pandas==2.2.3','pyarrow==18.1.0','scikit-learn==1.5.2'])
import relbench
assert relbench.__version__=='1.1.0', 'Use the pinned RelBench1.1.0 API'
print({'python':platform.python_version(),'relbench':relbench.__version__})
'''
preamble='\n'.join(canonical.splitlines()[:next(n for n in tree.body if isinstance(n,ast.FunctionDef)).lineno-1])+'\nfrom types import SimpleNamespace\n'
portable=prose(True);sections=re.split(r'(?=^## )',portable,flags=re.M)
# Section index0 is introduction; integer1..10 correspond to lesson sections.
payload={n:base64.b64encode((P/'evidence/l126'/n).read_bytes()).decode() for n in ['f1-db.zip','f1-task.zip']}
full_run='''# RUN: real, complete database/task through your live schema/alignment functions.
import base64,tempfile
payload = '''+repr(payload)+'''
with tempfile.TemporaryDirectory(prefix='l126-notebook-db-') as folder:
    report,predictions=f1_tour(base64.b64decode(payload['f1-db.zip']),base64.b64decode(payload['f1-task.zip']),folder)
assert report['schema']['total_rows']==74063 and report['splits']=={'train':7453,'val':499,'test':760}
Path('l126-report.json').write_text(json.dumps(report,indent=2))
predictions.to_csv('l126-predictions.csv',index=False)
print('Complete database:', report['schema']['total_rows'], 'rows')
print('Task splits:', report['splits'])
print('Train median:',report['training_median'],'Test MAE:',report['test_mae'])
print('Historical beta full reconstruction: NOT_RUN; course tour: PASS')
'''
def task_cells(name,check,hint,solution):
    code=functions[name]
    if not solution:code=code.splitlines()[0]+'\n    raise NotImplementedError("TODO: '+name+'")'
    return [nbf.v4.new_markdown_cell('### TODO · `'+name+'`\n\n'+hint),nbf.v4.new_code_cell(code),nbf.v4.new_markdown_cell('### CHECK\n\nPredict the result; then run. Diagnose failures before editing the check.'),nbf.v4.new_code_cell(nodes[check]+'\n'+check+'('+name+')\nprint("PASS: '+name+'")')]
for solution in [False,True]:
    cells=[nbf.v4.new_markdown_cell('# Lesson126 · '+TITLE+'\n\n'+('Executed solution/reference.' if solution else 'Student notebook: four live functions.')+'\n\nThis TierB complete F1 API tour is separate from TierC beta source fixtures. No historical model-score reproduction is claimed. All teaching code and available-data archives are inline; no repository import is needed.'),nbf.v4.new_code_cell(bootstrap),nbf.v4.new_code_cell(preamble)]
    cells.extend(nbf.v4.new_markdown_cell(x) for x in sections[:4])
    cells+=task_cells('schema_audit','check_schema','Return table counts plus resolved/null/dangling FK counts. Validate primary-key uniqueness.',solution)
    cells.extend(nbf.v4.new_markdown_cell(x) for x in sections[4:6])
    cells.append(nbf.v4.new_code_cell(nodes['fixture']))
    cells+=task_cells('engagement_table','check_engagement','Separate prior eligibility from future labels; preserve cutoff and right endpoint rules.',solution)
    cells.append(nbf.v4.new_markdown_cell(sections[6]));cells+=task_cells('align_predictions','check_alignment','Require exact entity/time keys and return evaluator order. Reject ambiguous/missing outputs.',solution)
    cells.append(nbf.v4.new_markdown_cell(sections[7]));cells+=task_cells('average_precision','check_ap','Compute recall increments only at the ends of equal-score groups.',solution)
    for name in ['beta_events','beta_cutoffs','unpack_verified','f1_tour']:
        cells.append(nbf.v4.new_markdown_cell('### PROVIDED · `'+name+'`\n\n'+{'beta_events':'Normalize activity sources without assuming their event IDs share one namespace.','beta_cutoffs':'Read the released backward fixed-duration split grid.','unpack_verified':'Verify bytes before loading the embedded archives.','f1_tour':'Follow your schema and alignment functions through actual RelBench objects. The fixed training median is the complete predictor.'}[name]));cells.append(nbf.v4.new_code_cell(functions[name]))
    cells.append(nbf.v4.new_markdown_cell('### RUN · your beta contracts composed on a fixture\n\nSynthetic only. These hand-declared scores are not trained predictions. Change a boundary event and explain what moves.'))
    cells.append(nbf.v4.new_code_cell('''users,events,t=fixture()
queries=engagement_table(users,events,[t])
keyed=queries[['OwnerUserId','timestamp']].copy()
keyed['score']=[.9,.5,.1]
scores=align_predictions(queries,keyed.iloc[::-1],['OwnerUserId','timestamp'])
assert average_precision(queries.contribution,scores)==1.0
print(queries.to_string(index=False))
print('Synthetic fixture AP:',average_precision(queries.contribution,scores))
print('Fixed-duration train cutoffs:', beta_cutoffs('2014-01-01')['train'])'''))
    cells.append(nbf.v4.new_markdown_cell(sections[8]));cells.append(nbf.v4.new_code_cell(full_run))
    cells.extend(nbf.v4.new_markdown_cell(x) for x in sections[9:])
    cells.append(nbf.v4.new_markdown_cell('## EXIT files\n\nSubmit `l126-report.json`, `l126-predictions.csv`, four implementations and the five written answers. Source and author execution do not establish learner mastery. Learner remains PENDING_WRITTEN_DEFENSE.\n\n## NEXT STEP · historical recovery\n\nThe checkout command `python labs/_recover_l126.py --archive-dir /path/to/archives --report /tmp/l126-recovered.json` verifies both original hashes and reconstructs all split labels. Missing bytes return exit2. Full historical execution is NOT_RUN; synthetic recovery-path checks pass. See the linked protocol for exact environment and source gaps.'))
    cells.append(nbf.v4.new_markdown_cell('## Source appendix · pinned beta implementation\n\nRead-only original code, **not executed by this notebook**. The task source is independently executed by the checkout audit. The training example has stale `rtb` imports and is not a validated runnable trainer. We retain it to make the historical architecture/training boundary inspectable; no numerical target exists in the beta paper. MIT license follows.'))
    for name in ['LICENSE','relbench/data/dataset.py','relbench/data/task.py','relbench/tasks/stackex.py','relbench/metrics.py','relbench/external/nn.py','examples/train.py']:
        text=(P/'sources/l126/beta'/name).read_text();cells.append(nbf.v4.new_markdown_cell('### '+name+'\n\n```'+('python' if name.endswith('.py') else '')+'\n'+text+'\n```'))
    for i,c in enumerate(cells):c.id=f'l126-{i:03d}'
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python3','language':'python'},'language_info':{'name':'python'}})
    path=P/('solutions' if solution else '')/f'{S}.ipynb'
    if solution and path.exists():
        old=nbf.read(path,as_version=4);a=[c for c in nb.cells if c.cell_type=='code'];b=[c for c in old.cells if c.cell_type=='code']
        if [c.source for c in a]==[c.source for c in b]:
            nb.metadata=old.metadata
            for x,y in zip(a,b):x.outputs=y.outputs;x.execution_count=y.execution_count;x.metadata=y.metadata
    nbf.write(nb,path)
    if solution and all(c.execution_count is not None for c in nb.cells if c.cell_type=='code'):
        html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
reference='''## Objects and identity

| Object | Carries | Inspect |
|---|---|---|
| Table | Dataframe + key/time metadata | PK uniqueness, FK destinations |
| Database | Named Tables | All rows and FK coverage |
| Dataset | Database + split timestamps | Final test cap versus earlier query cutoffs |
| Task | Entity, target, horizon, tables, metrics | Query key, eligibility, label window |
| Prediction file | Entity + time + score | Exact key match, finite scores |
| Evaluator | Hidden labels + declared metric | Expected ordering, tie handling |

## Beta engagement

Paper: December 2023 arXiv:2312.04615v1 §4.4.1. Code 0433616e (0.1.1).
Existing real users with prior contributions through cutoff. Positive if any
post/comment/vote occurs in `(t,t+730days]`. Validation 2019-01-01; test 2021-01-01.
Walk training cutoffs backward in 730-day steps. Two calendar years differ across leap days.

## Metric

`AP = sum(recall_increment * precision_at_threshold)`.
Process all equal scores together. Labels[1,0,1,0,1], scores[.9,.5,.5,.1,.1]
produce AP 1/3+2/9+1/5=.755556. Constant scores give prevalence. AP is neither
ROC-AUC nor trapezoidal PR area. No positives: explicit 0 convention.

## Evidence

82 synthetic task cases agree with original code and SQL; 200 AP cases agree with
released metric. Complete modern F1 tour: 74,063 rows / 9 tables; 7,453 train / 499 validation / 760 test.
Training-median predictor test MAE 4.444671 is COURSE_ONLY.
Historical beta archives unavailable: full contract NOT_RUN. Beta predictive-score
table absent: numerical target NOT_APPLICABLE. Whole-paper identity NOT_ESTABLISHED.
Learner PENDING_WRITTEN_DEFENSE; live Colab/deployment NOT_CHECKED.

[Lesson](../lessons/0126-relbench-beta.html) · [Notebook](../labs/0126-relbench-beta.ipynb) · [Protocol](../labs/l126-reproduction.md) · [Source manifest](../labs/_sources_l126.json).
'''
(R/'reference/relbench-beta.html').write_text(document('RelBench beta · reference',reference))
print('Built L126 lesson, reference and standalone student/solution notebooks')
