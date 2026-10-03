"""Build connected teaching artifacts from one source and an authenticated packet."""
import ast,base64,hashlib,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l193';F=P/'figures/l193';S='0193-open-fm-full-task-set'
r=json.loads((E/'report.json').read_text());tasks=json.loads((E/'packet/tasks.json').read_text());probe=json.loads((E/'packet/preprocessing.json').read_text())
source=(P/'relkit/multitask_l193.py').read_text();functions={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
replay_source=(P/'_replay_l193.py').read_text();replay_code=ast.get_source_segment(replay_source,next(n for n in ast.parse(replay_source).body if isinstance(n,ast.FunctionDef)))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l193'})
colors=dict(ink='#183f36',green='#236d5b',blue='#355d89',amber='#a26027',paper='#f3f7f3')
def save(fig,name):
 for ext in ['png','svg']:fig.savefig(F/(name+'.'+ext),dpi=160,metadata={'Date':None} if ext=='svg' else {'Software':'L193'},facecolor=fig.get_facecolor())
 plt.close(fig)
def box(ax,y,title,body,color):
 ax.add_patch(FancyBboxPatch((.035,y),.93,.132,boxstyle='round,pad=.01',linewidth=1,edgecolor=color,facecolor='white'))
 ax.text(.07,y+.095,title,weight='bold',fontsize=12,color=color,va='center');ax.text(.07,y+.042,body,fontsize=10.5,color=colors['ink'],va='center',linespacing=1.45)
fig,ax=plt.subplots(figsize=(6.6,8.2));fig.patch.set_facecolor(colors['paper']);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off');fig.subplots_adjust(left=.03,right=.97,top=.96,bottom=.02)
ax.text(.035,.975,'RDBLearn · one query through the system',fontsize=17,weight='bold',color=colors['ink'],va='top')
for y,title,body,col in [(.76,'1  Relational input','Tables + foreign keys + (entity, cutoff)\nSupport labels have their own availability boundary.',colors['green']),(.58,'2  Deterministic DFS at depth H','Allowed history → join paths → aggregate columns\nX support: n × p; X query: q × p',colors['green']),(.40,'3  Fit preprocessing on support','Same fitted map for support and query features\nMeasured code instability stops admission here.',colors['amber']),(.22,'4  Frozen tabular foundation model','TabPFN-v2 / TabPFN-v2.5 / LimiX-16M\nLabeled support + query → q predictions',colors['blue']),(.04,'5  Validation chooses; test evaluates','9 candidates per task and seed → 1 selected model\nComplete query keys → score → seed/task ledger',colors['blue'])]:
 box(ax,y,title,body,col)
 if y>.05:ax.annotate('',xy=(.5,y-.04),xytext=(.5,y-.005),arrowprops=dict(arrowstyle='->',color=colors['ink'],lw=1.3))
save(fig,'architecture')
fig,ax=plt.subplots(figsize=(4.3,8.4));fig.patch.set_facecolor(colors['paper']);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off');fig.subplots_adjust(left=.025,right=.975,top=.98,bottom=.015)
ax.text(.04,.97,'RDBLearn\nOne query through the system',fontsize=15,weight='bold',color=colors['ink'],va='top',linespacing=1.4)
mobile=[(.73,'1  Relational records','Tables, foreign keys, query cutoff\nSupport labels must be available.',colors['green']),(.56,'2  DFS at selected depth','Filter history → join → aggregate\nSupport: n × p; query: q × p',colors['green']),(.39,'3  Fitted preprocessing','One map for support and queries\nCode instability stops this run.',colors['amber']),(.22,'4  Frozen tabular predictor','TabPFN-v2 / v2.5 / LimiX-16M\nSupport + queries → predictions',colors['blue']),(.05,'5  Select, evaluate, record','Nine validation choices → one\nFull test keys → score → ledger',colors['blue'])]
for y,title,body,col in mobile:
 ax.add_patch(FancyBboxPatch((.04,y),.92,.135,boxstyle='round,pad=.01',linewidth=1,edgecolor=col,facecolor='white'))
 ax.text(.08,y+.101,title,fontsize=12,weight='bold',color=col,va='center');ax.text(.08,y+.043,body,fontsize=10.8,color=colors['ink'],va='center',linespacing=1.65)
 if y>.05:ax.annotate('',xy=(.5,y-.032),xytext=(.5,y-.008),arrowprops=dict(arrowstyle='->',color=colors['ink']))
save(fig,'architecture-mobile')
fig,ax=plt.subplots(figsize=(6.6,4.4),layout='constrained');fig.patch.set_facecolor(colors['paper']);ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
ax.text(.02,.98,'Same categories. Different fitted meaning.',va='top',fontsize=17,weight='bold',color=colors['ink'])
for x,title,lines,col in [(.03,'SUPPORT FIT',['b → 0','c → 1','d → 2'],colors['green']),(.54,'AFTER QUERY a',['b → 1','c → 2','d → 3'],colors['amber'])]:
 ax.text(x,.75,title,fontsize=11,weight='bold',color=col)
 for i,line in enumerate(lines):ax.text(x,.59-i*.15,line,fontsize=22,color=col)
ax.annotate('sort expanded\nclass list',xy=(.5,.49),xytext=(.28,.49),ha='center',fontsize=10,color=colors['ink'],arrowprops=dict(arrowstyle='->',color=colors['ink']))
ax.text(.03,.07,'Fresh original-source diagnostic · numeric controls unchanged\nBenchmark occurrence and score impact remain unestablished.',fontsize=10.5,color=colors['ink'])
save(fig,'codes')
fig,axes=plt.subplots(2,1,figsize=(6.6,6.7),layout='constrained');fig.patch.set_facecolor(colors['paper'])
axes[0].bar(range(1,9),[.9,.9,.9,.5,.5,.5,.5,.5],color=[colors['green']]*3+['#adc0b6']*5)
axes[0].axhline(.65,color=colors['blue'],linestyle='--',label='All 8 tasks: .650');axes[0].axhline(.9,color=colors['amber'],linestyle=':',label='Only first 3: .900');axes[0].set(ylim=(0,1.15),xticks=range(1,9),ylabel='Synthetic task AUROC',xlabel='Task');axes[0].set_title('Missing tasks can flatter the mean',loc='left',weight='bold');axes[0].legend(fontsize=10,loc='lower left',frameon=False)
axes[1].axis('off');axes[1].text(0,.94,'WITHIN A TASK',fontsize=11,weight='bold',color=colors['green']);axes[1].text(0,.74,'3 complete seeds → mean + sample SD',fontsize=15,color=colors['ink']);axes[1].text(0,.48,'ACROSS COMPARABLE TASKS',fontsize=11,weight='bold',color=colors['blue']);axes[1].text(0,.28,'All declared tasks → equal-task mean',fontsize=15,color=colors['ink']);axes[1].text(0,.04,'Synthetic teaching example. Actual L193 coverage: 0 / 21.',fontsize=10.5,color=colors['amber']);axes[0].spines[['top','right']].set_visible(False)
save(fig,'aggregation')
status='<div class="repro-status"><strong>Full model reproduction: INCOMPLETE.</strong> Fresh source diagnostic and complete ledger audit are finished. Model inference: NOT_RUN, 0/21 tasks. Cloud/API spend: $0. Learner: PENDING_WRITTEN_DEFENSE.</div>'
results='| Task | Metric | Paper target | Fresh mean ± SD | Coverage |\n|---|---|---:|---|---|\n'+'\n'.join(f"| {t['id']} | {t['metric']} | {t['paper_score']:.4f} | — NOT_RUN | 0/3 seeds |" for t in tasks)
captions={'architecture':'RDBLearn computation and admission boundaries. Intended per-query temporal filtering requires its own audit. The shared preprocessing gate stopped the declared experiment before backend inference.','codes':'Measured code changes for the same known categories after an unseen category is transformed. The diagram reports a synthetic source diagnostic, not task-level model accuracy.','aggregation':'Authored missingness example and the two levels of aggregation. Support-seed variation and variation across tasks answer different questions.'}
def fill(text,portable=False):
 text=text.replace('[[STATUS]]',status).replace('[[RESULTS]]',results).replace('[[SELECTION_CODE]]','```python\n'+functions['choose_config']+'\n```')
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l193/'+name+'.svg'
  img=f'<img src="{src}" alt="{caption}">'
  if name=='architecture' and not portable:img='<picture><source media="(max-width:520px)" srcset="../labs/figures/l193/architecture-mobile.svg">'+img+'</picture>'
  text=text.replace('[[FIG:'+name+']]',f'<figure class="repro-figure">{img}<figcaption>{caption}</figcaption></figure>')
 if portable:
  text=re.sub(r'<div id="(?:category-explorer|coverage-explorer|multitask-quiz)"></div>','',text)
  text=text.replace('<noscript>','<p>').replace('</noscript>','</p>')
  text=text.replace('](../','](https://avistian.github.io/relational/')
  text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def doc(title,body,interactive=False):
 scripts='<script id="multitask-data" type="application/json">'+json.dumps(probe)+'</script><script src="../assets/quiz.js"></script><script src="../assets/multitask-reproduction.js"></script>' if interactive else ''
 body=body.replace('<table>','<div class="repro-table" tabindex="0" role="region" aria-label="Scrollable results table"><table>').replace('</table>','</table></div>')
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/multitask-reproduction.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html">Curriculum</a></nav>'+body+'</article>'+scripts+'</body></html>'
prose=(R/'lessons/content'/(S+'.md')).read_text()
(R/'lessons'/(S+'.html')).write_text(doc('Lesson 193 — Open FM reproduction: full task set','<h1>A result table that keeps every task</h1>'+render(fill(prose)),True))
ref='''# Full-task reproduction · field guide

**Freeze first:** paper version, task IDs, model/source/checkpoint/data hashes, support policy, splits, candidate order, seeds, metrics, budget and stop rule.

| Stage | Contract | Failure rule |
|---|---|---|
| Source admission | Same support/query preprocessing meaning; valid temporal access | Stop and identify the defect |
| Validation selection | Every candidate; validation only; fixed tie order | Missing candidate: no selection |
| Seed summary | Every seed has a completed or explicitly missing record | Missing score: full-seed mean withheld |
| Task aggregation | Every declared task; one weight per task | Incomplete coverage: full-suite mean withheld |
| Regression scaling | Verified positive baseline denominator per task | Unknown denominator: no normalized aggregate |
| Claim | Fresh inference, reference arithmetic and source audit distinguished | No substitution between evidence types |

**Equations:** task mean = sum of seed scores / S; sample variance = sum of squared deviations / (S−1). Macro score = sum of task means / T. Normalized MAE_t = MAE_t / verified baseline MAE_t. Never average AUROC with MAE or raw MAEs in different units.

**Worked trace:** seed scores .6/.7/.8 → mean .7, sample SD .1. Missing one seed → incomplete, not a two-seed full result. Synthetic errors5/.2 with baseline errors10/.1 → .5/2 → mean1.25.

**L193:** 21 tasks × 3 seeds × 9 validation candidates = 567; 63 selected-test slots. All model slots NOT_RUN. Original preprocessing diagnostic fails under the recorded environment; all 21 task rows retain null fresh results. Published classification reference means .7452875 and .7879 are arithmetic on rounded table scalars. Regression aggregate identity NOT_ESTABLISHED. No backend executor was validated after the source stop.

**Continue only after:** resolve source correctness and version identity; audit raw features/query keys/temporal availability; pin checkpoints and normalizers; measure the full-grid cost; validate the backend runner. A repaired pipeline needs a separately identified experiment. Complete the learner defense independently of engineering checks.

[Lesson](../lessons/0193-open-fm-full-task-set.html) · [Student notebook](../labs/0193-open-fm-full-task-set.ipynb) · [Protocol](../labs/l193-reproduction.md) · [Full report](../labs/evidence/l193/report.json) · [Primary source](https://arxiv.org/html/2602.18495v1).
'''
(R/'reference/open-fm-full-task-set.html').write_text(doc('Full-task reproduction field guide',render(ref)))
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted((E/'packet').iterdir())+[E/'input-manifest.json']:
  info=zipfile.ZipInfo(str(p.relative_to(E)),date_time=(2026,10,2,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,p.read_bytes())
payload=base64.b64encode(buf.getvalue()).decode();digest=hashlib.sha256(buf.getvalue()).hexdigest();expected=json.dumps(r,sort_keys=True)
def notebook(solution):
 cells=[]
 def md(s):cells.append(nb.v4.new_markdown_cell(s))
 def code(s,tags=None):cells.append(nb.v4.new_code_cell(s,metadata={'tags':tags or []}))
 md('# Lesson 193 · Open FM reproduction — full task set\n\n'+fill(prose,True))
 md('## PROVIDED · Offline audit packet\nPython3 standard library only. The entire frozen task ledger, paper reference and original-source diagnostic observations are embedded. This notebook audits the packet; it does not execute the upstream preprocessor or run GPU models. No repository checkout, account, downloads or installation required.')
 code('import base64, hashlib, io, json, math, statistics, tempfile, zipfile\nfrom pathlib import Path\nworkspace=Path(tempfile.mkdtemp(prefix="l193-audit-"))')
 code('payload='+repr(payload)+'\nraw=base64.b64decode(payload)\nassert hashlib.sha256(raw).hexdigest()=='+repr(digest)+'\nwith zipfile.ZipFile(io.BytesIO(raw)) as archive:\n    for name in archive.namelist():\n        if not (workspace/name).resolve().is_relative_to(workspace.resolve()):\n            raise ValueError("Unsafe archive path")\n    archive.extractall(workspace)\npacket=workspace/"packet"\nmanifest=json.loads((workspace/"input-manifest.json").read_text())\ntasks=json.loads((packet/"tasks.json").read_text())', ['data-payload'])
 contracts=[('choose_config','rows,order,metric','Require all unique candidate IDs from order. Every split must be val; scores finite and in the metric domain. Maximize AUROC, minimize MAE; ties use order.',"rows=[dict(id='b',split='val',score=.7),dict(id='a',split='val',score=.7)]\nassert choose_config(rows,['a','b'],'AUROC')=='a'\ntry:\n    choose_config([dict(rows[0],split='test'),rows[1]],['a','b'],'AUROC')\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('Test contamination accepted')"),('summarize_task','task,runs,seeds','Require exact task/seed identities. COMPLETE rows have finite scores; NOT_RUN/FAILED rows have null scores. Return task, status, expected_seeds, completed_seeds, mean, sample_sd. Mean/SD stay null unless all seeds complete (single-seed SD is null).',"toy_task=dict(id='toy',metric='AUROC',group='classification',normalizer=None)\ntoy_runs=[dict(task='toy',seed=s,status='COMPLETE',score=v) for s,v in enumerate([.6,.7,.8])]\nsummary=summarize_task(toy_task,toy_runs,[0,1,2])\nassert abs(summary['mean']-.7)<1e-12 and abs(summary['sample_sd']-.1)<1e-12\ntoy_runs[2].update(status='NOT_RUN',score=None)\nassert summarize_task(toy_task,toy_runs,[0,1,2])['mean'] is None"),('aggregate_suite','tasks,summaries','Require exactly one summary per declared task. Group by benchmark/metric; equal task weights. Return status, expected_tasks, complete_tasks, metric, mean per group. Only complete groups get a mean; MAE requires positive task normalizer values and reports NORMALIZED_MAE. Unknown normalizers yield NORMALIZER_NOT_ESTABLISHED.',"toy_tasks=[dict(id='a',metric='MAE',group='regression',normalizer=10),dict(id='b',metric='MAE',group='regression',normalizer=.1)]\ntoy_summaries=[dict(task='a',status='COMPLETE',mean=5),dict(task='b',status='COMPLETE',mean=.2)]\nassert aggregate_suite(toy_tasks,toy_summaries)['regression']['mean']==1.25\ntoy_tasks[1]['normalizer']=None\nassert aggregate_suite(toy_tasks,toy_summaries)['regression']['mean'] is None")]
 for name,args,task,check in contracts:
  md('## TODO · '+name+'\n'+task+'\n\nPredict a failure case before writing code. The exact reference contract is visible in the executed solution; try your own first.')
  code(functions[name] if solution else 'def '+name+'('+args+'):\n    # TODO: implement the contract above.\n    raise NotImplementedError('+repr(name)+')')
  md('### CHECK · Synthetic contract examples');code(check+'\nprint("Contract check PASS")')
 md('## PROVIDED · Full authenticated replay\nThis operator calls all three learner functions. Paper scalars and synthetic exercises are clearly separated from the empty fresh-model ledger. Read the completeness and source checks before executing.')
 code(replay_code)
 md('## CHECK · Every task and every planned run\nCompare the complete report, not a selected scalar.')
 code('report=replay193(packet,manifest,choose_config,summarize_task,aggregate_suite)\nassert report==json.loads('+repr(expected)+')\nPath("l193-report.json").write_text(json.dumps(report,indent=2))\nprint(report["status"])\nprint("Tasks:",report["task_count"],"| Validation slots:",report["validation_slots"],"| Test slots:",report["test_slots"])\nprint("Executed model evaluations:",report["executed_model_evaluations"])\nprint("Source diagnostic:",report["diagnostic"])\nprint("Published reference arithmetic:",report["published_reference"])')
 md('## EXIT · Defend a result table\nWrite the named experiment, completed evidence, exact stop reason, aggregation denominator, uncertainty interpretation and prerequisites for resuming. Explain why a repaired pipeline is a separate variant. Ask the tutor to review your answer; author checks cannot complete your exit.')
 code('submission=dict(learner="PENDING_WRITTEN_DEFENSE",named_experiment="",evidence="",stop_reason="",denominator="",uncertainty="",continuation="",repair_boundary="")\nPath("l193-submission.json").write_text(json.dumps(submission,indent=2))')
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(cells):c.id=f'l193-{i:03}'
 return book
for solution in [False,True]:
 path=P/('solutions' if solution else '')/(S+'.ipynb');book=notebook(solution)
 if solution and path.exists():
  old=nb.read(path,4)
  if [c.source for c in old.cells]==[c.source for c in book.cells]:book=old
 nb.write(book,path)
print('Built Lesson193, field guide, 3 portable figures, student and solution notebooks')
