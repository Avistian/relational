"""Deterministically build the approved concept lesson and standalone scope lab."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from _run_l161 import audit161,render161
from relkit.scope_l161 import adaptation_route,database_boundary,scope_verdict
P=Path(__file__).resolve().parent;R=P.parent;S='0161-what-is-a-foundation-model';E=P/'evidence/l161';F=P/'figures/l161'
F.mkdir(parents=True,exist_ok=True)
raw=(E/'fixtures.json').read_bytes();manifest=json.loads((E/'input-manifest.json').read_text())
assert hashlib.sha256(raw).hexdigest()==manifest['fixtures_sha256']
packet=json.loads(raw);report=audit161(packet,adaptation_route,database_boundary,scope_verdict)
assert report==json.loads((E/'report.json').read_text()),'Run the declared audit before rebuilding'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l161'})

def save(fig,name):
 fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox
 for ax in fig.axes:
  for label in ax.texts:
   b=label.get_window_extent(renderer)
   assert b.x0>=bounds.x0 and b.y0>=bounds.y0 and b.x1<=bounds.x1 and b.y1<=bounds.y1,(name,label.get_text())
 for ext in ['png','svg']:fig.savefig(F/(name+'.'+ext),dpi=150,metadata={'Date':None} if ext=='svg' else {'Software':'L161'})
 plt.close(fig)
 for svg in F.glob('*.svg'):svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')

def box(ax,x,y,w,h,title,body,color='#e5f0eb'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.014',facecolor=color,edgecolor='#97b3a5',lw=1))
 ax.text(x+.025,y+h-.08,title,va='top',fontsize=12,weight='bold',color='#174e40')
 ax.text(x+.025,y+h-.17,body,va='top',fontsize=10.5,linespacing=1.55,color='#263d35')

def arrow(ax,start,end):ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color='#287b68',lw=2))
fig,ax=plt.subplots(figsize=(10,4.9));fig.subplots_adjust(left=.025,right=.975,top=.87,bottom=.03);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');fig.patch.set_facecolor('#f5f8f5')
fig.suptitle('Same starting checkpoint; different adaptation operations',fontsize=17,x=.035,ha='left',y=.96)
box(ax,.02,.48,.25,.41,'Pretraining','Databases A + B\nLearning objective\nSaved weights θ₀')
box(ax,.37,.59,.59,.30,'In-context · eight labels','Weights θ₀ unchanged; context = 8 labeled examples\nprediction = f(θ₀; context, query from C)')
box(ax,.37,.15,.59,.30,'Fine-tuning · eight labels','Adaptation updates θ₀ → θC using the 8 examples\nprediction = f(θC; query from C)')
arrow(ax,(.28,.73),(.35,.73));arrow(ax,(.28,.60),(.35,.33))
ax.text(.025,.29,'The query target\nis never supplied.',fontsize=11,color='#79522b')
ax.text(.025,.025,'Schematic protocol trace • changing context can change a frozen model’s output • no measured scores',fontsize=10,color='#50645d');save(fig,'adaptation')
fig,ax=plt.subplots(figsize=(10,4.8));fig.subplots_adjust(left=.025,right=.975,top=.86,bottom=.03);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');fig.patch.set_facecolor('#f5f8f5')
fig.suptitle('Two boundaries: pretraining databases, then target queries',fontsize=17,x=.035,ha='left',y=.96)
box(ax,.02,.32,.25,.52,'Pretraining','A + B\n\nC is absent from\nthe declared inventory')
ax.add_patch(FancyBboxPatch((.37,.21),.59,.69,boxstyle='round,pad=.018',facecolor='#f0f2fa',edgecolor='#859cb3',linestyle='--'))
ax.text(.395,.81,'Target database C',fontsize=13,weight='bold',color='#234d6f')
box(ax,.40,.35,.23,.33,'Adaptation','8 labeled examples\navailable by cutoff','#ffffff')
box(ax,.70,.35,.23,.33,'Test','Disjoint queries\ntargets withheld','#ffffff')
arrow(ax,(.28,.55),(.38,.55));arrow(ax,(.645,.52),(.685,.52))
ax.text(.40,.255,'C context is permitted; C pretraining exposure changes the claim.',fontsize=9.5)
ax.text(.025,.075,'If inventory includes C → SEEN. If inventory is undocumented → UNKNOWN. Neither is held-out evidence.',fontsize=10)
ax.text(.025,.015,'Schematic only • canonical database lineage and temporal availability need independent review',fontsize=10,color='#50645d');save(fig,'boundary')
captions={'adaptation':'Schematic: both routes start from θ₀ and use eight target labels. In-context prediction changes inputs; fine-tuning changes weights. No model performance is shown.',
'boundary':'Schematic: target C is absent from pretraining A/B but supplies eight legal adaptation labels. Its test queries remain separate. Declared disjointness still needs a provenance audit.'}

def prose(portable=False):
 text=(R/'lessons/content'/(S+'.md')).read_text()
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l161/'+name+'.svg'
  caption += '' if portable else ' On a narrow screen, focus the figure and scroll horizontally for the full trace.'
  text=text.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 variants={
 'WARMUP':('Recall frozen weights versus changing inputs, and reconstruction versus transfer.','<div id="warmup"></div>'),
 'PREDICT':('Predict before proceeding: do eight context labels become zero-shot merely because weights stay frozen? Explain.','<div id="predict"></div>'),
 'ADAPTATION':('Static trace: no weight updates + 8 target labels = IN_CONTEXT. Add 20 backbone updates = FINE_TUNING. Remove all labels and updates = ZERO_SHOT.','<div id="adaptation-explorer"></div><noscript>No updates plus 8 target labels is in-context prediction. Weight updates change the route to fine-tuning.</noscript>'),
 'BOUNDARY':('Counterfactual: include C in pretraining A/B and HELD_OUT becomes SEEN. Hide the inventory and it becomes UNKNOWN. Adding legal C adaptation labels alone changes neither membership.','<div id="boundary-explorer"></div><noscript>A/B pretraining excludes C. A/B/C includes C. An undocumented inventory cannot establish either fact. Target adaptation is separate.</noscript>'),
 'TEACHBACK':('Explain which evidence would justify unseen-database transfer and which unknowns remain. Ask the teaching agent to assess your 400–600-word scope.','<div id="teachback"></div>')}
 for key,(fallback,html) in variants.items():text=text.replace('[['+key+']]',fallback if portable else html)
 text=text.replace('[[RESULTS]]','**Executed author audit:** all 15 synthetic records are included: 4 READY_TO_RUN, 1 READY_FOR_REVIEW and 10 REVISE. These counts describe fixtures, not models or databases. [Complete output](../labs/evidence/l161/report.md).')
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/').replace('href="../','href="https://avistian.github.io/relational/')
  text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def doc(title,body,interactive=False):
 html=render(body).replace('<table>','<div class="route-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','foundation-scope','l161-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in ['lesson','atomic-route','checkpoint','lab-access','foundation-scope'])+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0160-year-4-exit-exam.html">Lesson 160</a></nav><header><p class="route-kicker">Year 5 · Quarter 1 · Lesson 161</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('What is a foundation model?',prose(),True))
reference='''## The definition and the claim
A reusable model trained on broad data and adaptable to a range of tasks. Explain corpus diversity and task coverage; neither parameter count nor the number of database files supplies a universal threshold. “Any database in one forward pass” is a stronger ambition.

## Adaptation checklist
| Mode | Backbone updates | Head updates | Target labels |
|---|---|---|---|
| Zero-shot in this lesson | No | No | Zero |
| In-context | No | No | Supplied as context |
| Frozen encoder + head | No | Yes | Often used to train head |
| Fine-tuning | Yes | Optional | Often used; can also adapt without labels |

Few-shot describes the declared small label count, not a unique algorithm. Our classifier prioritizes updates for hybrid routes. Frozen weights can produce changed outputs from changed context.

## Nested evaluation boundaries
Pretraining databases → target database → legal adaptation and validation → disjoint test queries. Target adaptation is allowed under a declared protocol. Pretraining overlap changes an unseen-database claim. Unknown corpus membership is neither disjointness nor contamination proof. Resolve aliases/copies before counting databases.

## Shared benefits and risks
Emergence concerns learned capabilities not individually programmed. Homogenization concerns reuse of a common foundation across applications; limitations can be shared too. These concepts motivate measurement; they are not empirical results about this lesson's fixtures.

## Six lines for a proposal
Inputs and time policy; pretraining corpus and checkpoint; adaptation weights/label budget; database and query splits; matched baseline; per-task criterion and disconfirming result. Keep human source review separate from executable consistency checks.

## Status discipline
READY_TO_RUN: supplied plan is consistent. READY_FOR_REVIEW: record also declares completion; inspect actual evidence. REVISE: contradictions or unknowns remain. None certifies an FM, proves transfer or grades your written defense. Training NOT_RUN; transfer NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/0161-what-is-a-foundation-model.html) · [Scope template](../labs/l161-scope-template.md) · [Reproduction boundary](../labs/l161-reproduction.md) · [Primary source](https://arxiv.org/abs/2108.07258v1) · [Adaptation overview](https://crfm.stanford.edu/report.html).
'''
(R/'reference/foundation-model-scope.html').write_text(doc('Foundation-model scope — quick reference',reference))

def defs(path):
 s=path.read_text();return [(n.name,ast.get_source_segment(s,n)) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)]
contracts={
 'adaptation_route':('Trace changed weights and supplied labels.','Accept nonnegative integer backbone/head update counts and target label count; reject booleans and invalid counts. Return one of the four uppercase route names. Backbone updates take priority over head updates, then distinguish context examples from zero-shot. Counts describe target adaptation, not pretraining.','assert adaptation_route(0,0,8)=="IN_CONTEXT"\nassert adaptation_route(20,0,8)=="FINE_TUNING"\nprint("CHECK: same eight labels, different update route")'),
 'database_boundary':('Protect database identity, not just row identity.','Accept a canonical pretraining list (or None) and a nonempty target database ID. Missing or empty inventory is UNKNOWN. Known membership is SEEN, known absence is HELD_OUT. Reject duplicates, whitespace aliases and non-string IDs; reject a string used as a list. Do not infer provenance from labels.','assert database_boundary(["A","B"],"C")=="HELD_OUT"\nassert database_boundary(None,"C")=="UNKNOWN"\nprint("CHECK: undocumented is not disjoint")'),
 'scope_verdict':('Match the declaration to the proposed cross-database experiment.','Validate required fields and types as specified below. Call route then boundary exactly once using supplied callbacks. Collect all issues, never only the first. REVISE if any issue exists; otherwise READY_FOR_REVIEW if completed, READY_TO_RUN if not. Return mode, pretraining_boundary, status, issues, universal_claim=NOT_ESTABLISHED and learner=PENDING_WRITTEN_DEFENSE. Do not mutate input.','r=example_record()\nassert scope_verdict(r)["status"]=="READY_TO_RUN"\nr["test_labels_used"]=True\nassert "TEST_LABEL_ACCESS" in scope_verdict(r)["issues"]\nprint("CHECK: planned completion does not erase leaked labels")')}
issue_contract='''Required record fields: pretraining_databases, evaluation_database, backbone_updates, head_updates, target_labels, declared_mode, selection_split, evaluation_split, test_labels_used, temporal_audit, baseline_matched, completed, checkpoint_shared. The four flags are strict Booleans; splits are train/validation/test; temporal_audit is PASS/FAIL/NOT_ESTABLISHED; declared_mode is a known route. Raise ValueError for malformed records. Callback validation handles counts and database IDs.

Diagnostic mapping: unknown corpus → PRETRAINING_UNKNOWN; target in corpus → PRETRAINING_OVERLAP; test selection → TEST_SELECTION; evaluation other than test → NO_HELD_OUT_TEST; test labels used → TEST_LABEL_ACCESS; temporal status other than PASS → TEMPORAL_UNVERIFIED; unmatched baseline → BASELINE_UNMATCHED; separately initialized task models → NO_SHARED_CHECKPOINT; declared versus computed mode disagreement → MODE_MISMATCH. Emit in this order. Training-only prespecified selection is allowed; test selection is not. PASS is a declaration that needs external evidence.
'''
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson 161 · What is a foundation model?\n\nStandalone Tier C protocol/design exercise. Three live TODOs feed the audit; no model is fitted. Python3.10+ standard library; no downloads. PROVIDED cells supply scaffolding, TODO cells are your functions, CHECK gives immediate feedback, EXIT exports your scoping note. This notebook contains all computation inputs; course links become public only after publication.'),nb.v4.new_markdown_cell(prose(True)),nb.v4.new_code_cell('# @colab-bootstrap — PROVIDED: standard library only, no network or package installs.\nfrom pathlib import Path\nfrom copy import deepcopy\nimport hashlib,json\n')]
 cells += [nb.v4.new_markdown_cell('## PROVIDED · Immutable synthetic inputs\nA/B/C are fictional canonical database identities. Every record is declared metadata, including the completed example; there are no measured prediction files.'),nb.v4.new_code_cell('fixture_text = '+repr(raw.decode())+'\nassert hashlib.sha256(fixture_text.encode()).hexdigest()=='+repr(manifest['fixtures_sha256'])+'\npacket=json.loads(fixture_text)\nprint("Loaded",len(packet["cases"]),"synthetic protocol records")')]
 cells.append(nb.v4.new_code_cell(dict(defs(P/'_check_l161.py'))['example_record']))
 for name,code in defs(P/'relkit/scope_l161.py'):
  goal,contract,check=contracts[name]
  cells.append(nb.v4.new_markdown_cell('## TODO · '+name+'\n\n**Goal:** '+goal+'\n\n**Why:** '+contract+'\n\nPredict a contradictory record before implementing. '+(issue_contract if name=='scope_verdict' else '')))
  cells.append(nb.v4.new_code_cell(code if solution else code.split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'))
  cells.append(nb.v4.new_code_cell('# CHECK\n'+check))
 cells.append(nb.v4.new_markdown_cell('## PROVIDED · Full audit and adversarial feedback\nThese functions call your three implementations. READY_FOR_REVIEW is a conditional design verdict. It is never a measured score or a foundation-model certification.'))
 for file in ['_run_l161.py','_check_l161.py']:
  for name,code in defs(P/file):
   if name!='example_record':cells.append(nb.v4.new_code_cell(code))
 cells.append(nb.v4.new_code_cell('# CHECK — execute every declared case through the live learner functions\nassert check161(adaptation_route,database_boundary,scope_verdict)=="PASS"\nreport=audit161(packet,adaptation_route,database_boundary,scope_verdict)\nassert all(row["status"]==case["expected"] for row,case in zip(report["rows"],packet["cases"]))\nprint(render161(report))\nPath("l161-report.json").write_text(json.dumps(report,indent=2)+"\\n")'))
 cells.append(nb.v4.new_markdown_cell('## EXIT · Your 400–600-word proposal\nUse the six scope fields above. A teacher assesses definition, provenance, adaptation, evaluation and boundaries (0–2 each; ≥8/10 and no zero). Code checks cannot validate the prose or authenticate the metadata. The short author example below is only an illustration; replace it with your own defended scope.'))
 example='I propose reusing one pretrained checkpoint across churn and delay tasks in a declared held-out database. The pretraining inventory and schema coverage must be audited before claiming transfer. Eight legal context labels define the adaptation budget; no task weight updates are planned. Separate test queries and validation-only selection preserve the evaluation boundary. Compare paired task metrics against the same-label baseline and preregister useful-effect thresholds. At present these are metadata fixtures. No trained model, transfer gain, universal schema support or completed learner defense is established.' if solution else ''
 cells.append(nb.v4.new_code_cell('written_scope = '+repr(example)+'\nmy_protocol = example_record()  # Replace with your proposal; do not invent provenance.\nmy_audit = scope_verdict(my_protocol)\nsubmission=dict(written_scope=written_scope,declared_protocol=my_protocol,audit=my_audit,review=None,learner="PENDING_WRITTEN_DEFENSE")\nPath("l161-submission.json").write_text(json.dumps(submission,indent=2)+"\\n")\nprint("Exported scope; learner PENDING_WRITTEN_DEFENSE")'))
 cells.append(nb.v4.new_markdown_cell('## NEXT STEP · A performance reproduction needs a different contract\nThe Bommasani report supplies this lesson’s concepts, not a selected relational training recipe. Name an exact published model/table, pin its code, corpus, splits, preprocessing, adaptation, selection, all seeds and metric aggregation; budget the complete run plus validation and retries before requesting authorization. Do not relabel these 15 synthetic metadata records as model or paper performance. Ask the teaching agent to review your note.'))
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(book.cells):c.id=f'l161-{i:03d}'
 path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
  old=nb.read(path,4)
  if [(c.cell_type,c.source) for c in old.cells]==[(c.cell_type,c.source) for c in book.cells]:
   book.metadata=old.metadata
   for c,prior in zip(book.cells,old.cells):
    if c.cell_type=='code':c.outputs=prior.outputs;c.execution_count=prior.execution_count;c.metadata=prior.metadata
 nb.write(book,path)
print('Built lesson, reference, two diagrams and portable student/solution notebooks')
