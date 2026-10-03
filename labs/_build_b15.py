"""Build standalone lesson/reference and portable notebooks from visible code."""
import ast,base64,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;R=P.parent;S='b15-parameter-free-encoders';E=P/'evidence/b15';F=P/'figures/b15';F.mkdir(parents=True,exist_ok=True)
report=json.loads((E/'diagnostic.json').read_text());paper=json.loads((E/'paper-status.json').read_text())
# Portable scientific figure: exact same values as the interactive tables.
f,axes=plt.subplots(1,2,figsize=(10,4.7));f.patch.set_facecolor('#fcfcf8')
for ax in axes:ax.axis('off')
axes[0].set_title('One local label, two task rules',fontsize=15,pad=22)
t=axes[0].table(cellText=[['Copy','1','0','1'],['Flip','1','1','0']],colLabels=['World','Local b','Support y\n(input 0)','Query y'],loc='center',cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(11);t.scale(1,2.4)
axes[0].text(.5,.08,'Same b → local P(y=1)=0.5\nExternal support identifies Copy or Flip',ha='center',transform=axes[0].transAxes,fontsize=11)
axes[1].set_title('Two columns, identical local evidence',fontsize=15,pad=22)
t=axes[1].table(cellText=[['Local 1','0','0','0'],['Local 2','1','1','1'],['Query','0','1','?']],colLabels=['Row','A','B','Label'],loc='center',cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(11);t.scale(1,2)
axes[1].text(.5,.06,'Both A and B explain local labels\nExternal (0,1) label selects a column',ha='center',transform=axes[1].transAxes,fontsize=11)
f.suptitle('B15 · information before parameters',fontsize=19,y=1.02);f.tight_layout();f.savefig(F/'information.png',dpi=150,bbox_inches='tight');plt.close(f)
visibility='''<section class="lv-board" data-label-visibility><h3>Trace which labels can cross the boundary</h3><div class="lv-path"><div class="lv-node"><strong>Remote support</strong>event 2 → available 4<small>Input 0, task label theta</small></div><div class="lv-node"><strong>Late support</strong>event 3 → available 11<small>Old event; unfinished label</small></div><div class="lv-node lv-hidden"><strong>Query Q</strong>anchor day 10<small>Own label always hidden</small></div></div><label>Declared support boundary <select name="scope"><option value="expanded">Include external support</option><option value="local">Local neighborhood only</option></select></label><label>Information cutoff <select name="cutoff"><option value="3">Day 3</option><option value="10" selected>Day 10: baseline</option><option value="11">Day 11: retrospective access</option></select></label><button type="button">Reset visibility</button><p class="lv-baseline">Baseline day 10: remote visible, late hidden, query hidden. Day 11 is retrospective access and is invalid for a day-10 prediction.</p><output aria-live="polite">Day 10: remote label visible; late label and query label hidden.</output><noscript><p>At day 3 neither support label is available. At day 11 both are available, but this changes the prediction-time contract.</p></noscript></section>'''
rules='''<section class="lv-board" data-label-rules><h3>Can you identify the world from the visible input?</h3><div class="lv-path"><div class="lv-node"><strong>Local observation</strong>Neighbor label b = 1<small>Identical in both worlds</small></div><div class="lv-node"><strong>Head context</strong><span data-support>External support hidden</span><small>Encoding of b stays fixed</small></div><div class="lv-node"><strong>Compatible rules</strong>Copy: 1 → 1<br>Flip: 1 → 0<small>Discard only with legal evidence</small></div></div><label>World generating the truth <select name="world"><option value="0">Copy: theta = 0</option><option value="1">Flip: theta = 1</option></select></label><label>Head information <select name="access"><option value="local">Local observation only</option><option value="expanded">Add permitted support</option></select></label><button type="button">Reset worlds</button><p class="lv-baseline">Baseline: both worlds equally likely, P(query label = 1) = 0.5. The world control is shown to you, not passed to the local predictor.</p><output aria-live="polite">Copy world, local-only input: probability 0.5.</output><noscript><p>External input 0 has label 0 in Copy and 1 in Flip; it identifies the rule. Local input alone does not.</p></noscript></section>'''
columns='''<section class="lv-board" data-label-columns><h3>Which observation distinguishes A from B?</h3><table><thead><tr><th>Observation</th><th>A</th><th>B</th><th>Label</th></tr></thead><tbody><tr><td>Local 1</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Local 2</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Query</td><td>0</td><td>1</td><td>Hidden</td></tr></tbody></table><label>True relevant column <select name="column"><option value="0">Column A</option><option value="1">Column B</option></select></label><label>Visible labeled examples <select name="examples"><option value="local">Two local examples</option><option value="expanded">Add external (0,1)</option></select></label><button type="button">Reset columns</button><p class="lv-baseline">Baseline: both columns remain compatible; 0 bits about S. The true column is used only to generate labels and score predictions.</p><output aria-live="polite">Two local examples: query probability 0.5; information about S = 0 bits.</output><noscript><p>External (0,1) with label 0 identifies A; with label 1 identifies B. Either observation supplies one bit.</p></noscript></section>'''
table='| Fixture / access | Worlds | Accuracy | Brier score |\n|---|---:|---:|---:|\n'
for group,items in [('Rule',report['rule_scores']),('Column',report['column_scores'])]:
 for arm,v in items.items():table+=f"| {group} / {arm} | {4 if group=='Rule' else 8} | {v['accuracy']:.2f} | {v['brier']:.3f} |\n"
status='**Selected paper status: `'+paper['status']+'`.** No fresh paper inference ran; cloud spend is **$0**. Exact candidate/seed/checkpoint/refit mapping remains unavailable in the pinned release. A runnable generic example is not an authenticated experiment receipt.'
source=(R/'lessons/content'/f'{S}.md').read_text()
# Normalize prose number boundaries without changing code, identifiers or URLs.
for old,new in [('for20minutes','for 20 minutes'),('for25–40minutes','for 25–40 minutes'),('after1,7and30days','after 1, 7 and 30 days'),('day10','day 10'),('day2','day 2'),('day4','day 4'),('day3','day 3'),('day11','day 11'),('b=1','b = 1'),('0for','0 for'),('1for','1 for'),('returns1when','returns 1 when'),('probability0.5to','probability 0.5 to'),('label1','label 1'),('input0','input 0'),('theta:0in','theta: 0 in'),('and1in','and 1 in'),('Proposition3.1','Proposition 3.1'),('AppendixD.1','Appendix D.1'),('support1','support 1'),('support2','support 2'),('label0identifies','label 0 identifies'),('label1it','label 1 it'),('**0bits**','**0 bits**'),('**1bit**','**1 bit**'),('Proposition4.1','Proposition 4.1'),('AppendixD.2','Appendix D.2'),('achieves75%accuracy','achieves 75% accuracy'),('least0.5is','least 0.5 is'),('all16interventions','all 16 interventions'),('§§3–4','§§3–4'),('AppendixA','Appendix A'),('Table 5','Table 5'),('AUROC0.7271','AUROC 0.7271'),('commit`','commit `'),('The25-file','The 25-file'),('all24tracked','all 24 tracked'),('a150-word','a 150-word'),('in1day','in 1 day'),('In7days','In 7 days'),('In30days','In 30 days')]:source=source.replace(old,new)
(R/'lessons/content'/f'{S}.md').write_text(source)
body=source
for key,value in dict(VISIBILITY=visibility,RULES=rules,COLUMNS=columns,RESULTS=table,PAPER=status).items():body=body.replace('{{'+key+'}}',value)
def document(title,text,interactive=False):
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','label-visibility'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/label-visibility.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html#research-bridge">Research bridge</a></nav>'+render(text)+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document('B15 · Parameter-free encoders',body,True))
ref='''# B15 · Information boundary reference

**Ask first:** what can the encoder see, and what extra examples can the head see?

| Term | Meaning |
|---|---|
| Parameter-free encoder | Fixed feature operations; the head may still be pretrained |
| Local label feature | Known label of another row, supplied inside the encoder view |
| Labeled head support | Permitted examples supplied to an in-context prediction head |
| Hidden query label | The answer being predicted; never a legal input |
| Identifiability | Whether different candidate rules can be distinguished from observations |
| Existence counterexample | A distribution where a universal guarantee fails; not all distributions |

**Copy/flip:** same neighboring b=1 permits y=1 or y=0. Equal prior gives p=.5. An external pair (0,theta) identifies y=b XOR theta.

**Columns:** local (0,0)→0 and (1,1)→1 cannot distinguish y=A from y=B. Query (0,1) is ambiguous. External (0,1) with an observed label resolves the column.

**Visibility:** full(entity,time) keys; exclude query identity; event<cutoff; available≤cutoff; declared context membership. The timestamp alone does not establish label availability.

**Proposition 3.1:** existence of distributions where a fixed local encoder/head cannot beat its chance divergence bound, although a task-specific predictor exists. **Proposition 4.1:** existence of distributions where local labels provide no information about relevant columns. Neither says learned encoders can never help. Finite lab fixtures illustrate, not prove, the general claims.

'''+table+'\n\n'+status+'''

**Source discrepancy:** v1.1 README says history ON; pinned config defaults OFF and estimator guards the path. Default is not the executed Table 5 configuration. Full suite/pretraining NOT_RUN. Learner defense pending.

[Lesson](../lessons/b15-parameter-free-encoders.html) · [Lab](../labs/b15-parameter-free-encoders.ipynb) · [Contract](../labs/b15-reproduction.md) · [Paper v2](https://arxiv.org/html/2607.05476v2)
'''
(R/'reference'/f'{S}.html').write_text(document('B15 · Information boundary reference',ref))
code=(P/'relkit/labels_b15.py').read_text();tree=ast.parse(code);tasks=['visible_labels','predict_rules','predict_columns']
provided='\n'.join(code.splitlines()[:7])+'\n\n'+ '\n\n'.join(ast.get_source_segment(code,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name not in tasks)
# Import preamble by AST avoids splitting an arbitrary line.
provided='\n'.join(ast.get_source_segment(code,n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)))+'\n\n'+'\n\n'.join(ast.get_source_segment(code,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name not in tasks)
figure='![Exact finite information examples](data:image/png;base64,'+base64.b64encode((F/'information.png').read_bytes()).decode()+')'
archive=io.BytesIO()
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted((P/'sources/b15').rglob('*')):
  if p.is_file():z.write(p,str(p.relative_to(P)))
 for p in [P/'_reproduce_b15.py',E/'paper-status.json',P/'b15-reproduction.md']:z.write(p,str(p.relative_to(P)))
blob=base64.b64encode(archive.getvalue()).decode()
checks={
'visible_labels':'''rows=[dict(key=('Q',10),available=9,label=1),dict(key=('A',2),available=4,label=0),dict(key=('B',3),available=11,label=1)]
assert visible_labels(rows,('Q',10),10,{r['key'] for r in rows})==[(('A',2),0)]
rows[0]['label']=0;rows[2]['label']=0
assert visible_labels(rows,('Q',10),10,{r['key'] for r in rows})==[(('A',2),0)]
print('Visibility CHECK passed')''',
'predict_rules':'''assert predict_rules(1,[]) == .5
assert predict_rules(1,[(0,1)]) == 0
assert predict_rules(0,[(1,1)]) == 0
try: predict_rules(0,[(0,0),(0,1)])
except ValueError: pass
else: raise AssertionError('Contradictory support accepted')
print('Rule CHECK passed')''',
'predict_columns':'''assert predict_columns((1,0),[((0,0),0),((1,1),1)]) == .5
assert predict_columns((1,0),[((0,1),0)]) == 1
assert predict_columns((0,0),[]) == 0
print('Column CHECK passed')'''}
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('''# B15 · Parameter-free encoders: limits and assumptions

**Skill:** identify what a predictor can know before comparing encoder parameters. Read [the lesson](https://avistian.github.io/relational/lessons/b15-parameter-free-encoders.html). This notebook is portable: Python standard library only; no downloads, repository imports or cloud calls. PROVIDED cells are complete; implement three TODO cells and run each CHECK. Submit the EXIT and written defense. Author execution is not learner mastery.

**Mirror scope:** finite illustrations of the information barriers discussed in Propositions 3.1/4.1 of [paper v2](https://arxiv.org/html/2607.05476v2). The general parity-graph proofs are not reproduced. Real-data Table 5 inference remains source-gated. Tier C synthetic data is appropriate here because the question is identifiability, not empirical model ranking.

## Concept recap

An encoder maps a local database view to features. A parameter-free encoder has no learned weights, but its head can be pretrained. Support labels are permitted examples; query labels are answers we must hide. Event time and label availability can differ.

If a neighbor has label b=0, both Copy (query 0) and Flip (query 1) remain possible. XOR returns 1 when its inputs differ, so y=b XOR theta names these two rules. An external example with input 1 and observed label 0 identifies Flip. Without it, equally likely rules give probability .5.

A second ambiguity concerns columns: (0,0)→0 and (1,1)→1 fit both y=A and y=B. Seeing labels does not identify a column when both explain them. A new labeled (0,1) row can distinguish them. These are existence examples, not claims about all tasks.

We enumerate all four rule worlds and eight column worlds, equally weighted, plus all 16 hidden-label interventions. Accuracy thresholds probability at .5; Brier is mean squared probability error. Mutual information measures reduction in uncertainty in bits: resolving two equally likely columns takes one bit.

'''),nb.v4.new_markdown_cell(figure+'\n\nThe first panel holds the local input fixed across worlds. The second holds local labeled examples fixed across possible relevant columns. External support changes the information boundary.'),nb.v4.new_markdown_cell('## PROVIDED · finite distribution and reporting\n\nNo optimization or pretrained models are hidden here. The full generator, information calculation and scoring are visible below; task functions are defined in your next cells.'),nb.v4.new_code_cell(provided)]
 goals={'visible_labels':'Filter labels using declared context membership, full query identity, strict event cutoff and inclusive availability. Reject duplicate keys and invalid inputs. Goal: changing hidden truth must not change legal support. Hint: a row can be old but its label unavailable.', 'predict_rules':'Retain every binary Copy/Flip rule consistent with the permitted support pairs; average their query predictions. Reject inconsistent support. Goal: expose uncertainty instead of guessing the world. Hint: the world is not an input to this function.', 'predict_columns':'Retain each of the two columns that matches all permitted labeled examples; average its query values. Reject inconsistent support. Goal: distinguish predictive agreement from identifying the relevant feature. Hint: do not choose a column simply because both fit.'}
 for name in tasks:
  node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
  func=ast.get_source_segment(code,node)
  cells.extend([nb.v4.new_markdown_cell('## '+('SOLUTION' if solution else 'TODO')+' · '+name+'\n\n'+goals[name]),nb.v4.new_code_cell(func if solution else func.split('\n')[0]+'\n    raise NotImplementedError("Complete '+name+'")'),nb.v4.new_markdown_cell('### CHECK · '+name+'\n\nRun immediately. A correct implementation must pass before the next task.'),nb.v4.new_code_cell(checks[name])])
 cells.extend([nb.v4.new_markdown_cell('## Complete finite run and independent expected values\n\nThe full distribution is enumerated. Local column accuracy exceeds chance when A=B, even though S stays unidentified. Explain this before running.'),nb.v4.new_code_cell("import json\nreport=run_experiment()\nassert report['rule_scores']['local']=={'accuracy':.5,'brier':.25}\nassert report['column_scores']['local']=={'accuracy':.75,'brier':.125}\nassert report['information_bits']=={'local':0.0,'expanded':1.0}\nassert len(report['interventions'])==16\nfrom pathlib import Path\nPath('b15-diagnostic.json').write_text(json.dumps(report,indent=2)+'\\n')\nprint(json.dumps({k:v for k,v in report.items() if k.endswith('scores') or k=='information_bits'},indent=2))"),nb.v4.new_markdown_cell('## EXIT · separate evidence from mastery\n\nWrite 150 words explaining the existence quantifier, the information-access change, why perfect leakage fails, and why a parameter-free encoder may have a trained head. Return after 1/7/30 days. Ask the teacher about any unclear proof or code step.'),nb.v4.new_code_cell("print('B15-LABEL-VISIBILITY: COMPLETE_FINITE_DIAGNOSTIC')\nprint('Paper inference: NOT_RUN; full suite and pretraining: NOT_RUN')\nprint('Learner: PENDING_WRITTEN_DEFENSE')"),nb.v4.new_markdown_cell('## NEXT STEP · selected Table 5 source audit\n\n'+status+'\n\nThe complete pinned upstream source and paper are embedded below, together with a hash-checking operator. Running this archive extraction is local and free. The guarded paper attempt authenticates sources then refuses an undefined benchmark; it does not launch cloud work. Source commit 78561f0 has history OFF in code, ON in README. Inspect both and distinguish defaults from run receipts. Required missing details: full candidate grid, seeds/subsamples, checkpoint identities, validation/refit/snapshot mapping and data/environment provenance. The generic F1 example cannot substitute for the trial task. Budget USD10 aggregate, stop USD8/reserve USD2, local 3600 seconds; current cloud USD0.'),nb.v4.new_code_cell("import base64, io, zipfile, subprocess, sys\nfrom pathlib import Path\npacket=Path('b15-source-packet'); packet.mkdir(exist_ok=True)\narchive=zipfile.ZipFile(io.BytesIO(base64.b64decode("+repr(blob)+")))\narchive.extractall(packet)\nresult=subprocess.run([sys.executable,str(packet/'_reproduce_b15.py'),'--phase','audit'],capture_output=True,text=True,check=True)\nprint(result.stdout)"),nb.v4.new_code_cell("RUN_PAPER_REPRO=False\nif RUN_PAPER_REPRO:\n    subprocess.run([sys.executable,str(packet/'_reproduce_b15.py'),'--phase','paper'],check=True)\nelse:\n    print('Paper run remains OFF; original Table 5 configuration must be recovered first.')")])
 n=nb.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),language_info=dict(name='python',version='3.12')))
 target=P/'solutions'/f'{S}.ipynb' if solution else P/f'{S}.ipynb';nb.write(n,target)
print('B15 lesson, reference, figure and two portable notebooks built')
