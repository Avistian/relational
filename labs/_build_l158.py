"""Generate lesson, reference, figures and self-contained CPU replay notebooks."""
import ast,base64,hashlib,json,re,zlib
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from _replay_l158 import replay,render_report
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l158';S='0158-year-4-synthesis';TITLE='Year 4 synthesis: a verdict that could change'
m=json.loads((E/'input-manifest.json').read_text());r=replay(P,m)
(E/'report.json').write_text(json.dumps(r,indent=2)+'\n');(E/'report.md').write_text(render_report(r))
F=P/'figures/l158';F.mkdir(exist_ok=True,parents=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l158'})
fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained');fig.patch.set_facecolor('#f3f6f3')
rows=[x for x in r['matched']['rows'] if x['split']=='test']
ax=axes[0]
for i,arm in enumerate(['fe_mae','rdl_mae']):
 ax.scatter([i+(j-2)*.04 for j in range(5)],[x[arm] for x in rows],s=45,color=['#315d55','#aa6239'][i],zorder=3)
 mean=r['matched']['metrics']['test'][arm]['mean'];ax.plot([i-.22,i+.22],[mean,mean],color='#183c33',lw=2)
ax.set(xticks=[0,1],xticklabels=['Manual FE','Basic RDL'],ylabel='Test MAE (positions) ↓',title='Five seeds per method · same F1 task')
ax=axes[1];b=r['matched']['metrics']['test']['benefit_mae']['mean'];lo,hi=r['matched']['test_benefit_driver_bootstrap_95']
ax.axvline(0,color='#999',ls='--');ax.errorbar([b],[0],xerr=[[b-lo],[hi-b]],fmt='o',color='#315d55',capsize=6,lw=2)
ax.set(xlim=(-.4,.25),ylim=(-.5,.5),yticks=[],xlabel='FE − RDL MAE (positions)',title='Conditional driver-cluster 95% interval')
ax.text(-.37,-.34,'← Favors FE',fontsize=10);ax.text(.08,-.34,'Favors RDL →',fontsize=10)
for ax in axes:ax.set_facecolor('#f3f6f3');ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.12)
fig.suptitle('The mean favors FE; the interval leaves direction unresolved.',fontsize=14)
for ext in ['svg','png']:fig.savefig(F/('comparison.'+ext),dpi=150,metadata={'Date':None} if ext=='svg' else {'Software':'L158'})
plt.close(fig)
fig,ax=plt.subplots(figsize=(10,3.7),layout='constrained');fig.patch.set_facecolor('#f3f6f3');ax.axis('off')
boxes=[(.02,.60,'L151 predictions\nseed 0'),(.37,.60,'L154 report\nsame predictions'),(.72,.60,'L158 essay\nsame predictions')]
for x,y,text in boxes:ax.text(x,y,text,transform=ax.transAxes,fontsize=14,va='center',bbox=dict(boxstyle='round,pad=.7',fc='white',ec='#a8c3b6'))
for x in [.27,.62]:ax.annotate('',xy=(x+.08,.60),xytext=(x,.60),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#315d55',lw=2))
ax.text(.02,.20,'3 artifact views → 1 prediction packet\nFresh F1 fits can add runs, but still add 0 new databases.',transform=ax.transAxes,fontsize=15,color='#234c40',linespacing=1.8)
ax.set_title('Trace the evidence unit before counting support.',loc='left',fontsize=16,pad=20)
for ext in ['svg','png']:fig.savefig(F/('lineage.'+ext),dpi=150,metadata={'Date':None} if ext=='svg' else {'Software':'L158'})
plt.close(fig)
for svg in F.glob('*.svg'):
 svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
table='| Evidence |'+render_report(r).split('| Evidence |',1)[1].split('\n\n',1)[0]
captions={'comparison':'Actual L155 test scores. Dots are training seeds; horizontal bars are means. The interval resamples drivers, conditional on fitted models, and crosses zero.',
          'lineage':'L154 and L158 reuse the L151 predictions. New F1 runs still evaluate the same task and database; file counts are not independent-domain counts.'}

def prose(portable=False):
 text=(R/'lessons/content'/f'{S}.md').read_text().replace('[[TABLE]]',table)
 text=text.replace('[[REPLAY]]',f"**Executed author evidence:** {r['prediction_rows_rescored']:,} rows rescored, {r['selection_checks']} validation-selection checks and {r['frozen_inputs']} frozen inputs. No fresh training. These counts describe audit work, not independent scientific replications.")
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l158/'+name+'.svg'
  text=text.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 for key,fallback,html in [('WARMUP','Retrieve: what identifies a query; which split selects a checkpoint; what does seed SD measure?','<div id="warmup"></div>'),('PREDICT','Predict before reading: does a small mean difference prove equivalence? Write your reason.','<div id="predict"></div>'),('WIDGET','Hold the F1 scores fixed. Ask for a local quality claim, a portfolio claim, an effort claim and a legality claim. Explain why those require different evidence. Repeating the report five times adds no new task.','<div id="thesis-claim"></div><noscript>One F1 comparison does not establish a portfolio win. Human effort is unobserved; historical availability remains unestablished. Repeating a report adds no new evidence.</noscript>'),('TEACHBACK','Explain the gap between computational replay and a broad thesis claim in your own words. Ask the teaching agent to review it.','<div id="teachback"></div>')]:
  text=text.replace('[['+key+']]',fallback if portable else html)
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/')
  text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def doc(title,body,interactive=False):
 html=render(body).replace('<table>','<div class="route-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','thesis-claim','l158-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in ['lesson','atomic-route','checkpoint'])+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0157-open-source-contribution.html">Lesson 157</a></nav><header><p class="route-kicker">Year 4 · Quarter 4 · Lesson 158</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(doc(TITLE,prose(),True))
ref='''## The claim ladder

| Claim | Required evidence | Common invalid substitute |
|---|---|---|
| Computation reproduced | Pinned inputs, complete lanes, independent metric checks | A successful script exit alone |
| Local quality advantage | Matched comparator plus appropriate uncertainty | Published comparator under another protocol |
| Human effort reduced | Comparable prospectively recorded human work | GPU runtime or code length |
| Broad advantage | Declared multi-domain design and fair baselines | More F1 seeds or report copies |
| Historical legality | Availability and fitting-policy audit | Timestamps alone |

## Sentence formula
On **population**, method A versus **baseline** changed **metric** by **effect**, with **uncertainty and its conditioning**. This supports **bounded claim**, while **missing observation** prevents **broader claim**.

## Current evidence
'''+table+'''

## Evidence units
Count complete task and database identities once. Exact prediction hashes expose reuse; distinct hashes do not prove independence. Validation-only pilots never count as test tasks. No average across AUROC, MAE and MAP.

## Falsifier
Declare the next task, information policy, comparator, practical threshold, seeds, selection rule, evaluation split, interval method and disconfirming outcome before test access. A threshold is a planning choice, not a result. Measure effort separately.

## Review
Five axes, 0–2 each: accuracy/provenance, comparator fairness, uncertainty/dependence, counter-evidence, falsifiability. Target8/10 with no zero after teacher review; code PASS does not establish mastery.

[Lesson](../lessons/0158-year-4-synthesis.html) · [Essay template](../labs/l158-essay-template.md) · [Replay report](../labs/evidence/l158/report.md) · [Protocol](../labs/l158-reproduction.md) · [Primary reading](https://arxiv.org/html/2407.20060v1#S6).
'''
(R/'reference/year-4-synthesis.html').write_text(doc('Year 4 synthesis — quick reference',ref))

def defs(path):
 source=path.read_text();return [(node.name,ast.get_source_segment(source,node)) for node in ast.parse(source).body if isinstance(node,ast.FunctionDef)]
packet={name:base64.b64encode((P/name).read_bytes()).decode() for name in m['files']}
packed=base64.b64encode(zlib.compress(json.dumps(packet,sort_keys=True).encode(),9)).decode()
bootstrap='''# @colab-bootstrap — PROVIDED: Python3.10+ and numpy. No downloads or training.
import base64,gzip,hashlib,json,math,statistics,zlib
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path('l158-packet')
ROOT.mkdir(exist_ok=True)
'''
payload='''# PROVIDED: compressed, hash-pinned author evidence; not a model or task solution.
PACKED = '''+repr(packed)+'''
manifest = '''+repr(m)+'''
packet=json.loads(zlib.decompress(base64.b64decode(PACKED)))
for name,encoded in packet.items():
    path=ROOT/name
    if not path.resolve().is_relative_to(ROOT.resolve()): raise ValueError('Unsafe packet path')
    raw=base64.b64decode(encoded)
    if hashlib.sha256(raw).hexdigest()!=manifest['files'][name]: raise ValueError('Changed payload')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
print('Extracted',len(packet),'frozen inputs')
'''
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson 158 · Year 4 synthesis\n\nStandalone CPU audit. PROVIDED cells expose the computation; TODO cells implement three claim guards; CHECK cells give immediate feedback; EXIT exports your evidence and written defense. Author solution is not learner mastery. Links point to the course site and may remain unavailable until publication; all replay inputs are embedded.'),nb.v4.new_markdown_cell(prose(True)),nb.v4.new_code_cell(bootstrap),nb.v4.new_code_cell(payload,metadata={'tags':['data-payload']})]
 for path,heading in [('relkit/portfolio_l154.py','PROVIDED · Complete seeds and comparable metrics'),('relkit/effort_l155.py','PROVIDED · Keyed paired losses and observed effort'),('_replay_l154.py','PROVIDED · Original portfolio evidence replay'),('_report_l155.py','PROVIDED · Matched F1 report and conditional driver bootstrap')]:
  cells.append(nb.v4.new_markdown_cell('## '+heading+'\nThese established functions are visible here so the notebook runs without importing the course repository. Read the function docstrings and trace the query/seed checks.'))
  for name,code in defs(P/path):
   cells.append(nb.v4.new_code_cell(code))
   if path=='_replay_l154.py' and name=='replay':cells.append(nb.v4.new_code_cell('replay_portfolio = replay  # Preserve upstream replay before defining the synthesis replay.'))
 tasks={
 'evidence_coverage':('Count evidence without counting the same task twice.','Return completed_tasks, completed_databases, unique_prediction_packets, reused_packets, missing_tasks and independence. Group hashes; reject duplicate IDs and non-test/unknown statuses.',"fixture=[dict(id='first',task='f1/position',database='f1',prediction_hash='abc',status='COMPLETE',split='test'),dict(id='copy',task='f1/position',database='f1',prediction_hash='abc',status='COMPLETE',split='test')]\nc=evidence_coverage(fixture)\nassert c['completed_tasks']==1 and c['unique_prediction_packets']==1 and c['reused_packets']==[['copy','first']]\nprint('CHECK: copied evidence adds no task')"),
 'claim_verdict':('Keep claim scope within the actual evidence.','Handle local_quality, portfolio_superiority, human_effort and leak_free; reject unknown claims and invalid effect/interval values. Portfolio superiority remains NOT_ESTABLISHED because no cross-database inference design exists. See check strings for the output contract.',"example=dict(matched_tasks=1,required_tasks=3,benefit=-.06,interval=[-.31,.17],human_effort='NOT_OBSERVED',historical_availability='NOT_ESTABLISHED')\nassert claim_verdict('local_quality',example)=='POINT_ESTIMATE_FAVORS_FE; SUPERIORITY_NOT_ESTABLISHED'\nassert claim_verdict('human_effort',example)=='NOT_OBSERVED'\nprint('CHECK: quality and human effort remain separate')"),
 'validate_falsifier':('Turn a future claim into a checkable experiment specification.','Require nonempty task/metric/direction/baseline/information_policy/disconfirm_if; correct metric direction; nonnegative finite minimum_benefit; val selection and test decision. Return the specification plus SPECIFIED_NOT_EXECUTED and PENDING_SCIENTIFIC_REVIEW.',"future=dict(task='new-db/task',metric='MAE',direction='lower',minimum_benefit=.1,selection_split='val',decision_split='test',baseline='SQL + LightGBM',information_policy='same legal information',disconfirm_if='upper bound below threshold')\nassert validate_falsifier(future)['status']=='SPECIFIED_NOT_EXECUTED'\nprint('CHECK: a specified experiment has not yet run')")}
 for name,code in defs(P/'relkit/synthesis_l158.py'):
  goal,hint,check=tasks[name];cells.append(nb.v4.new_markdown_cell('## TODO · '+name+'\n\n**Goal:** '+goal+'\n\n**Why:** this function is passed directly into the full evidence replay below.\n\n**Contract:** '+hint))
  signature=code.split('\n',1)[0]
  cells.append(nb.v4.new_code_cell(code if solution else signature+'\n    raise NotImplementedError("TODO: '+name+'")'))
  cells.append(nb.v4.new_code_cell('# CHECK\n'+check))
 cells.append(nb.v4.new_markdown_cell('## PROVIDED · Synthesis adapter\nThe adapter verifies every frozen input before using your functions. It recomputes scores, audits the complete validation histories and checks the saved reports. Temporal policy verdicts are explicitly inherited.'))
 for name,code in defs(P/'_replay_l158.py'):cells.append(nb.v4.new_code_cell(code))
 for name,code in defs(P/'_check_l158.py'):cells.append(nb.v4.new_code_cell(code))
 cells.append(nb.v4.new_code_cell("# CHECK: adversarial fixtures plus real-data replay\ncheck(evidence_coverage,claim_verdict,validate_falsifier)\nreport=replay(ROOT,manifest,coverage=evidence_coverage,verdict=claim_verdict,falsifier=validate_falsifier)\nassert report['prediction_rows_rescored']==98918\nprint(render_report(report))\nPath('l158-report.json').write_text(json.dumps(report,indent=2)+'\\n')"))
 cells.append(nb.v4.new_markdown_cell('## EXIT · Write the essay\nUse the five-section template in the lesson. Cite at least one numerical result, one counter-result or gap, and one reused evidence chain. Define a falsifier. The solution below is a short author example, not a completed learner essay. Ask the teaching agent to apply the 0–2 rubric; passing code does not grade prose.'))
 essay='On the one matched F1 task, the FE point estimate is better by 0.0642 MAE, but the conditional driver interval crosses zero. Local effort was not observed. The portfolio supports auditable execution on two completed tasks, not broad RDL superiority. Repeated F1 fits and report views do not add databases. I would test a predeclared practical benefit on an unseen database under matched information and fair budgets, with prospective effort logs and a frozen uncertainty rule. Historical availability and the incomplete recommendation experiment remain unresolved.' if solution else ''
 cells.append(nb.v4.new_code_cell('# EXIT: replace with your 700–1,000-word defense; submit for teacher review.\nwritten_defense = '+repr(essay)+"\nsubmission=dict(report=report,written_defense=written_defense,learner='PENDING_WRITTEN_DEFENSE')\nPath('l158-essay-evidence.json').write_text(json.dumps(submission,indent=2)+'\\n')\nprint('Exported evidence; learner PENDING_WRITTEN_DEFENSE')"))
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(notebook.cells):c.id=f'l158-{i:03d}'
 path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
  prior=nb.read(path,4)
  if [(c.cell_type,c.source) for c in prior.cells]==[(c.cell_type,c.source) for c in notebook.cells]:
   for cell,old in zip(notebook.cells,prior.cells):
    if cell.cell_type=='code':cell.outputs=old.outputs;cell.execution_count=old.execution_count;cell.metadata=old.metadata
 nb.write(notebook,path)
print('Built lesson, reference, two figures, student and solution notebooks')
