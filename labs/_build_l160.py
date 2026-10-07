"""Build the Year 4 exam from canonical prose, visible mechanisms and frozen inputs."""
import ast,base64,hashlib,json,re,zlib
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from _replay_l160 import replay160,render160
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l160';S='0160-year-4-exit-exam';TITLE='Year 4 exit exam: defend your RelBench portfolio'
m=json.loads((E/'input-manifest.json').read_text());r=replay160(P,m)
(E/'report.json').write_text(json.dumps(r,indent=2)+'\n');(E/'report.md').write_text(render160(r))
F=P/'figures/l160';F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l160'})
def save(fig,name):
 for ext in ['png','svg']:fig.savefig(F/(name+'.'+ext),dpi=150,metadata={'Date':None} if ext=='svg' else {'Software':'L160'})
 plt.close(fig)
fig,ax=plt.subplots(figsize=(10,4.1),layout='constrained');fig.patch.set_facecolor('#f3f6f3');ax.axis('off')
labels=['Study outcome','Driver position','Site sponsor']
values=[['COMPLETE','NOT RUN','UNOBSERVED','UNKNOWN'],['COMPLETE','COMPLETE','UNOBSERVED','UNKNOWN'],['INCOMPLETE','NOT RUN','UNOBSERVED','UNKNOWN']]
t=ax.table(cellText=values,rowLabels=labels,colLabels=['Test experiment','Matched FE','Human effort','Temporal sign-off'],cellLoc='center',bbox=[.16,.24,.83,.59]);t.auto_set_font_size(False);t.set_fontsize(10)
for (i,j),cell in t.get_celld().items():
 cell.set_edgecolor('#d5ded9');cell.set_facecolor('#e2ede5' if cell.get_text().get_text()=='COMPLETE' else '#fff7eb' if i>0 else '#e3eae7')
ax.set_title('Count evidence by task; require every column.',loc='left',fontsize=17,pad=12)
ax.text(.16,.10,'2/3 tasks    ·    1/3 FE comparisons    ·    0/3 effort ratios    ·    0/3 sign-offs',transform=ax.transAxes,fontsize=11)
ax.text(.16,.015,'Replay PASS  →  evidence above verified  →  Year 4 exit INCOMPLETE',transform=ax.transAxes,fontsize=12,color='#713e28');save(fig,'coverage')
fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained');fig.patch.set_facecolor('#f3f6f3')
for ax in axes:ax.axis('off')
a=axes[0].table(cellText=[['(8,10)',7],['(7,20)',4],['(7,10)',1]],colLabels=['Stored FE key','Prediction'],cellLoc='center',bbox=[.06,.27,.86,.49]);a.auto_set_font_size(False);a.set_fontsize(12)
b=axes[1].table(cellText=[['(7,10)',1,1,2,0,1,-1],['(7,20)',3,4,5,1,2,-1],['(8,10)',5,7,5,2,0,2]],colLabels=['Truth key','y','FE','RDL','|eFE|','|eRDL|','Δ'],cellLoc='center',bbox=[0,.27,1,.49]);b.auto_set_font_size(False);b.set_fontsize(10)
for tab in [a,b]:
 for (i,j),cell in tab.get_celld().items():cell.set_edgecolor('#d5ded9');cell.set_facecolor('#e3eae7' if i==0 else 'white')
axes[0].set_title('1 · Storage order is arbitrary',loc='left',fontsize=14)
axes[1].set_title('2 · Join onto complete truth keys',loc='left',fontsize=14)
axes[0].text(.07,.10,'Same driver, two cutoffs.\nEntity alone is not the query key.',transform=axes[0].transAxes,fontsize=12)
axes[1].text(.02,.10,'Δ = FE error − RDL error\nMean benefit = (−1 −1 +2) / 3 = 0',transform=axes[1].transAxes,fontsize=12)
fig.suptitle('Synthetic worked trace · pair errors only after alignment',fontsize=16);save(fig,'alignment')
for svg in F.glob('*.svg'):svg.write_text('\n'.join(l.rstrip() for l in svg.read_text().splitlines())+'\n')
captions={'coverage':'Measured author evidence: two completed tasks, one matched FE comparison, unobserved human effort and unestablished temporal sign-offs. Each missing column blocks the exit.',
'alignment':'Synthetic trace, not a benchmark result. Reverse-stored FE predictions are joined by entity and cutoff before computing absolute errors and benefits.'}
table='| Task |'+render160(r).split('| Task |',1)[1].split('\n\n',1)[0]
def prose(portable=False):
 text=(R/'lessons/content'/(S+'.md')).read_text().replace('[[TABLE]]',table).replace('[[REPLAY]]',f"**Executed author evidence:** {r['prediction_rows_rescored']:,} saved prediction rows rescored; {r['selection_checks']} validation decisions checked; {r['frozen_inputs']} frozen inputs. **Replay PASS · Year 4 exit INCOMPLETE · learner PENDING_WRITTEN_DEFENSE.**")
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l160/'+name+'.svg'
  note='' if portable else ' On a narrow screen, focus this figure and scroll horizontally to read every column.'
  text=text.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}{note}</figcaption></figure>')
 for key,fallback,html in [('WARMUP','Recall query keys, validation selection and feature arrival before reading further.','<div id="warmup"></div>'),('PREDICT','Predict: does completing the third test experiment also supply its FE comparison, effort logs and temporal audit? Explain before reading further.','<div id="predict"></div>'),('WIDGET','Counterfactual: add a third test experiment, then matched FE, effort logs, temporal sign-offs and actual review. At which step can every gate pass? These additions are hypothetical.','<div id="exit-evidence"></div><noscript>The measured exit is INCOMPLETE: two tasks, one FE comparison, no effort ratios or temporal sign-offs. Every gate needs its own evidence.</noscript>'),('TEACHBACK','Explain replay PASS versus exit INCOMPLETE; ask the teaching agent to review your reasoning.','<div id="teachback"></div>')]:text=text.replace('[['+key+']]',fallback if portable else html)
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/').replace('href="../','href="https://avistian.github.io/relational/')
  text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def doc(title,body,interactive=False):
 html=re.sub(r'<table([^>]*)>',r'<div class="route-scroll" tabindex="0"><table\1>',render(body)).replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','exit-evidence','l160-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in ['lesson','atomic-route','checkpoint','lab-access'])+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0159-foundation-model-preview.html">Lesson 159</a></nav><header><p class="route-kicker">Year 4 · Quarter 4 · Lesson 160</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc(TITLE,prose(),True))
ref='''## Year 4 exit checklist

Every declared task (at least three) needs a complete test experiment, a matched manual-FE comparison, a prospectively observed effort ratio and a reviewed temporal sign-off. Preserve negative results. Teacher review needs at least 8/10 and no zero on the five-axis rubric. More seeds do not add tasks. An incomplete experiment remains a documented gap.

## Query and metric contract
Join by (entity, cutoff), rejecting missing/duplicate/nonfinite entries. MAE benefit = FE − RDL; AUROC/MAP benefit = RDL − FE. Positive favors RDL. Never average these raw metric families. A conditional driver bootstrap is not uncertainty across databases or training runs.

## Human clock
FE active-human minutes / RDL active-human minutes, same task and scope. Missing logs → NOT_OBSERVED; missing arm → INCOMPLETE; zero RDL minutes → UNDEFINED_ZERO_DENOMINATOR. Record shared infrastructure and unattended compute separately. Numerical checks cannot authenticate observations.

## Status meanings
Replay PASS verifies saved computation. Exit INCOMPLETE means experimental requirements remain unmet. PENDING_WRITTEN_DEFENSE requires actual reviewer assessment. REVISION_REQUIRED means the rubric threshold was missed. Neither reported scores nor code completion proves learner mastery.

## Current packet
'''+table+'''

## Closing the gaps
Freeze the task inclusion rule and fair information policy. Resolve arrival-history evidence, plan prospective effort logs, then budget missing FE/third-task experiments including every seed and check. Preserve L153 STOP until a new scope and budget are approved. A negative complete result is valid evidence; RDL need not win.

[Lesson](../lessons/0160-year-4-exit-exam.html) · [Submission rubric](../labs/l160-submission.md) · [Effort template](../labs/l160-effort-template.json) · [Protocol](../labs/l160-reproduction.md) · [Report](../labs/evidence/l160/report.md) · [Primary reading](https://arxiv.org/html/2407.20060v1#S6).
'''
(R/'reference/year-4-exit-exam.html').write_text(doc('Year 4 exit exam — quick reference',ref))
def defs(path):
 s=path.read_text();return [(n.name,ast.get_source_segment(s,n)) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)]
packet={name:base64.b64encode((P/name).read_bytes()).decode() for name in m['files']}
packed=base64.b64encode(zlib.compress(json.dumps(packet,sort_keys=True).encode(),9)).decode()
bootstrap='''# @colab-bootstrap — PROVIDED: Python3.10+ and numpy; no downloads or training.
import base64,copy,gzip,hashlib,json,math,statistics,zlib
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path('l160-packet')
ROOT.mkdir(exist_ok=True)
'''
payload='''# PROVIDED: hash-pinned evidence bytes, not model weights or task solutions.
PACKED = '''+repr(packed)+'''
manifest = '''+repr(m)+'''
packet=json.loads(zlib.decompress(base64.b64decode(PACKED)))
for name,encoded in packet.items():
    path=ROOT/name
    if not path.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Unsafe path')
    raw=base64.b64decode(encoded)
    if hashlib.sha256(raw).hexdigest()!=manifest['files'][name]:raise ValueError('Changed evidence')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
print('Extracted',len(packet),'frozen inputs')
'''
tasks={
'aligned_losses':('Pair errors by complete query identity.','Accept truth keys/targets and both prediction key/value lists. Reject empty, duplicate, missing or nonfinite data. Return fe_loss, rdl_loss, benefit and mean_benefit in truth order. Positive benefit favors RDL.',"keys=[(7,10),(7,20),(8,10)]\na=aligned_losses(keys,[1,3,5],keys[::-1],[7,4,1],keys,[2,5,5])\nassert a['benefit']==[-1.,-1.,2.] and a['mean_benefit']==0\nprint('CHECK: correct losses despite reversed storage')"),
'observed_effort':('Keep unknown effort unknown.','Input is a list of unique event records with id, arm (FE/RDL), task, scope, minutes, kind=human_active and prospective=True. Require finite nonnegative minutes and one task/scope. Return status, ratio_fe_over_rdl and minutes. Preserve NOT_OBSERVED, INCOMPLETE and UNDEFINED_ZERO_DENOMINATOR; only a valid two-arm observation has OBSERVED status.',"assert observed_effort([])==dict(status='NOT_OBSERVED',ratio_fe_over_rdl=None,minutes={})\nlogs=[dict(id='f',arm='FE',task='x',scope='build+debug+validate',minutes=90,kind='human_active',prospective=True),dict(id='r',arm='RDL',task='x',scope='build+debug+validate',minutes=30,kind='human_active',prospective=True)]\nassert observed_effort(logs)['ratio_fe_over_rdl']==3\nprint('CHECK: synthetic logs give 3; absent real logs stay unknown')"),
'exit_gates':('Make every requirement independently necessary.','Accept one entry per task, linked failure_cases, and optional review with reviewer and five integer scores. Require at least 3 tasks and complete test/matched-FE/observed-effort/temporal-PASS for every declared task. Gate keys: task_coverage, matched_fe, human_effort, temporal_audit, honest_failures. Count keys: declared_tasks, completed_tasks, matched_fe_tasks, observed_effort_tasks, temporal_pass_tasks. Return counts, gates, written_defense, exit and a boundary string. Review is pending until supplied, passes at 8/10 with no zero, otherwise REVISION_REQUIRED. Experimental failure has priority as INCOMPLETE. Reject duplicate task IDs or unknown statuses.',"rows=[dict(task=t,status='COMPLETE',split='test',matched_fe='COMPLETE',effort='OBSERVED',temporal='PASS') for t in ['a/x','a/y','b/z']]\nfailures=[dict(evidence='saved/result.json',limitation='A model lost on a complete task')]\nassert exit_gates(rows,failures)['exit']=='PENDING_WRITTEN_DEFENSE'\nrows[2]['status']='INCOMPLETE'\nassert exit_gates(rows,failures,dict(reviewer='fixture',scores=[2]*5))['exit']=='INCOMPLETE'\nprint('CHECK: perfect review cannot replace the third experiment')")}
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson 160 · Year 4 exit exam\n\nStandalone CPU evidence audit. Tier B saved real RelBench predictions; synthetic fixtures are explicitly labeled. Three TODO functions drive the actual assessment. No fresh training or learner completion is inferred. Course links may remain unavailable until publication; all computation inputs are embedded.'),nb.v4.new_markdown_cell(prose(True)),nb.v4.new_code_cell(bootstrap),nb.v4.new_code_cell(payload,metadata={'tags':['data-payload']})]
 for path,heading in [('relkit/portfolio_l154.py','Complete seed sets and comparable scores'),('relkit/effort_l155.py','Paired errors and human-effort scope'),('_replay_l154.py','Original portfolio replay'),('_report_l155.py','Matched comparison and conditional bootstrap'),('relkit/synthesis_l158.py','Evidence lineage and bounded claims'),('_replay_l158.py','Complete declared L158 evidence replay')]:
  cells.append(nb.v4.new_markdown_cell('## PROVIDED · '+heading+'\nEstablished replay code is visible here. Trace the keys, seed and selection checks. These functions perform saved-evidence analysis; the original model/trainer is linked in the reproduction contract.'))
  for name,code in defs(P/path):
   cells.append(nb.v4.new_code_cell(code))
   if path=='_replay_l154.py' and name=='replay':cells.append(nb.v4.new_code_cell('replay_portfolio = replay'))
   if path=='_replay_l158.py' and name=='replay':cells.append(nb.v4.new_code_cell('replay_synthesis = replay'))
 for name,code in defs(P/'relkit/exam_l160.py'):
  goal,contract,check=tasks[name];cells.append(nb.v4.new_markdown_cell('## TODO · '+name+'\n\n**Goal:** '+goal+'\n\n**Contract:** '+contract+'\n\nPredict one failure before running CHECK. Your function is invoked by the full real-data replay below.'))
  cells.append(nb.v4.new_code_cell(code if solution else code.split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'))
  cells.append(nb.v4.new_code_cell('# CHECK\n'+check))
 cells.append(nb.v4.new_markdown_cell('## PROVIDED · Actual exit adapter\nThis verifies frozen inputs before it invokes your functions. Check the original empty effort log and the inherited temporal verdicts; neither can be promoted to an observation.'))
 for file in ['_replay_l160.py','_check_l160.py']:
  for name,code in defs(P/file):cells.append(nb.v4.new_code_cell(code))
 cells.append(nb.v4.new_code_cell("# CHECK: adversarial cases and full real evidence\ncheck160(aligned_losses,observed_effort,exit_gates)\nreport=replay160(ROOT,manifest,align=aligned_losses,effort=observed_effort,gates=exit_gates)\nassert report['prediction_rows_rescored']==98918 and report['selection_checks']==45\nassert report['assessment']['exit']=='INCOMPLETE'\nprint(render160(report))\nPath('l160-report.json').write_text(json.dumps(report,indent=2)+'\\n')"))
 cells.append(nb.v4.new_markdown_cell('## EXIT · Submit a defended decision\nWrite 700–1,000 words using the rubric above. Include provenance, a matched numerical result, uncertainty, adverse evidence, the human/temporal gaps and a costed falsifiable next step. The author example is deliberately short and receives no mastery credit. Leave review null until a real teacher assessment.'))
 essay='The frozen replay passes, but the declared Year 4 portfolio remains incomplete. It contains two completed tasks, one matched FE comparison and no observed effort ratio or temporal sign-off. The F1 point estimate favors FE by 0.064225 MAE positions; its conditional driver interval crosses zero and does not establish equivalence. Extra seeds or report copies add no task. Recommendation remains budget-stopped. I would resolve the information policy, plan prospective effort observation and obtain a complete costed protocol before filling the missing experiments. Synthetic reconstruction in L159 supplies no extra real benchmark task.' if solution else ''
 cells.append(nb.v4.new_code_cell('# EXIT — author example is not a learner defense; replace with your submission.\nwritten_defense = '+repr(essay)+"\nsubmission=dict(experiment=report['experiment'],input_manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),report=report,written_defense=written_defense,review=None,learner='PENDING_WRITTEN_DEFENSE')\nPath('l160-submission.json').write_text(json.dumps(submission,indent=2)+'\\n')\nprint('Exported l160-submission.json; learner PENDING_WRITTEN_DEFENSE')"))
 cells.append(nb.v4.new_markdown_cell('## NEXT STEP · Complete the missing experiments under a new protocol\nFull training remains in the source-pinned L151–L157 notebooks and reproduction contracts linked above. L153 retains its budget STOP; the recorded safety-adjusted forecast was $51.85 against a $10 cap. L160 authorizes only this CPU replay. Do not silently reduce seeds, epochs or candidate catalog. Discuss the missing information policy, prospective effort observations and aggregate cost with the teaching agent before new paid execution. Whole-paper and original human-study reproduction remain NOT_RUN.'))
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(book.cells):c.id=f'l160-{i:03d}'
 path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
  old=nb.read(path,4);book.metadata=old.metadata
  previous=[c for c in old.cells if c.cell_type=='code'];current=[c for c in book.cells if c.cell_type=='code']
  if [c.source for c in previous]==[c.source for c in current]:
   for prior,cell in zip(previous,current):
    cell.outputs=prior.outputs;cell.execution_count=prior.execution_count;cell.metadata=prior.metadata
 nb.write(book,path)
print('Built lesson, reference, two figures and portable student/solution notebooks')
