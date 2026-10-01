"""Build the approved lesson, figures and portable lab from canonical sources."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from _run_l162 import audit162
from relkit.vision_l162 import reachable_rows,token_budget,evidence_verdict
P=Path(__file__).resolve().parent;R=P.parent;S='0162-the-relational-fm-vision';F=P/'figures/l162';E=P/'evidence/l162'
raw=(E/'fixtures.json').read_bytes();manifest=json.loads((E/'input-manifest.json').read_text())
assert hashlib.sha256(raw).hexdigest()==manifest['fixtures_sha256']
packet=json.loads(raw);report=audit162(packet,reachable_rows,token_budget,evidence_verdict)
assert report==json.loads((E/'report.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l162'})
def canvas(title,h=5.8):
 fig,ax=plt.subplots(figsize=(11,h));fig.subplots_adjust(left=.025,right=.975,bottom=.025,top=.89);fig.patch.set_facecolor('#f5f8f5');ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');fig.suptitle(title,x=.04,ha='left',fontsize=17,y=.97);return fig,ax

def box(ax,x,y,w,h,title,body,color='#e3eee8'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008',facecolor=color,edgecolor='#97b3a5'))
 ax.text(x+.015,y+h-.035,title,va='top',fontsize=12,weight='bold',color='#174e40')
 ax.text(x+.015,y+h-.105,body,va='top',fontsize=10.5,linespacing=1.5,color='#263d35')

def arrow(ax,a,b,dashed=False):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=2,color='#287b68',linestyle='--' if dashed else '-'))

def save(fig,name):
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 for ax in fig.axes:
  boxes=[b for b in ax.patches if isinstance(b,FancyBboxPatch)]
  for b in boxes:
   assert b.get_x()>=0 and b.get_y()>=0 and b.get_x()+b.get_width()<=1 and b.get_y()+b.get_height()<=1
  for i,a in enumerate(boxes):
   for b in boxes[i+1:]:
    assert not (a.get_x()<b.get_x()+b.get_width() and b.get_x()<a.get_x()+a.get_width() and a.get_y()<b.get_y()+b.get_height() and b.get_y()<a.get_y()+a.get_height()), 'Overlapping figure boxes'
  for text in ax.texts:
   b=text.get_window_extent(renderer)
   assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,(name,text.get_text())
 for ext in ['svg','png']:fig.savefig(F/(name+'.'+ext),dpi=150,metadata={'Date':None} if ext=='svg' else {'Software':'L162'})
 plt.close(fig)
 for svg in F.glob('*.svg'):svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
fig,ax=canvas('The relational vision: encode rows, exchange context, decode a target',6.5)
box(ax,.01,.67,.26,.27,'Serialized row','moons: name=[MASK]\nplanet=7\nℓᵢ tokens, including schema')
box(ax,.34,.67,.25,.27,'BART encoder','Token sequence → features\nConceptual row vector: d\nAcross n rows: n × d')
box(ax,.66,.67,.32,.27,'Graph context','m1 ↔ p1 ↔ s1\nNeighbor aggregation\nRefined representations')
arrow(ax,(.28,.81),(.33,.81));arrow(ax,(.60,.81),(.65,.81))
box(ax,.66,.25,.32,.25,'BART decoder + loss','Predict masked text\nCorrect tokens → cross entropy\nGradient flows back to GNN','#f8eddb')
arrow(ax,(.82,.66),(.82,.51))
ax.text(.02,.54,'TRAINING SCHEDULE',fontsize=11,weight='bold',color='#587169')
box(ax,.01,.25,.28,.23,'Stage 1 · adapt rows','Update BART weights\nMasked row reconstruction')
box(ax,.35,.25,.25,.23,'Stage 2 · add graph','Freeze BART weights\nUpdate GNN weights')
arrow(ax,(.30,.37),(.34,.37));arrow(ax,(.66,.21),(.47,.21),True)
ax.text(.52,.14,'Backward signal; decoder weights stay fixed',ha='center',fontsize=10,color='#79522b')
ax.text(.02,.065,'Abstraction: the exact graph-to-decoder tensor interface is unresolved in the source protocol.',fontsize=10)
ax.text(.02,.012,'Loss and backward arrows are training-only. Local lab audits structure; it does not implement or train BART.',fontsize=10);save(fig,'architecture')
fig,ax=canvas('A missing bridge changes the receptive field',5.3)
positions={'m1':(.11,.68),'p1':(.44,.68),'s1':(.78,.68),'m2':(.44,.29),'z1':(.78,.29)}
for a,b in [('m1','p1'),('p1','s1'),('p1','m2')]:
 x,y=positions[a];u,v=positions[b];ax.plot([x,u],[y,v],color='#287b68',lw=3,zorder=1)
for n,(x,y) in positions.items():
 ax.scatter([x],[y],s=4200,c='#f8eddb' if n=='m1' else '#e3eee8' if n!='z1' else '#ececec',edgecolors='#6b897d',zorder=2)
 ax.text(x,y,n,ha='center',va='center',weight='bold',fontsize=14,zorder=3)
 ax.text(x,y-.135,{'m1':'moon · root','p1':'planet · bridge','s1':'star · 2 hops','m2':'moon · 2 hops','z1':'unrelated'}[n],ha='center',fontsize=11,bbox=dict(facecolor='#f5f8f5',edgecolor='none',pad=2))
ax.text(.01,.92,'Course graph · explicit undirected key-like edges · no automatic same-table connections',fontsize=11)
ax.text(.015,.34,'0 hops: m1\n1 hop: + p1\n2 hops: + m2, s1',linespacing=1.7,fontsize=11)
ax.text(.02,.04,'Remove planet p1 → both star s1 and moon m2 become unreachable from m1.',fontsize=11,color='#79522b');save(fig,'context')
captions={'architecture':'Source-inspired conceptual architecture, not a released tensor specification. The decoder loss trains graph weights through frozen LM operations. n counts rows, d counts representation features; the historical graph/decoder interface is unresolved.', 'context':'Course graph at two hops with all three edges: m1, p1, m2 and s1 are reachable; z1 is not. Remove planet p1 before traversal and only m1 remains. Reachability is potential access, not a prediction.'}

def prose(portable=False):
 text=(R/'lessons/content'/(S+'.md')).read_text()
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l162/'+name+'.svg'
  text=text.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}'+('' if portable else ' On narrow screens, focus and scroll the diagram horizontally.')+'</figcaption></figure>')
 variants={'WARMUP':('Recall why masking a value is not proof of transfer; distinguish frozen weights from frozen inputs.','<div id="warmup"></div>'), 'PREDICT':('Predict: remove p1 while retaining m1 and s1. Is a two-hop route still available? Explain before reading the trace.','<div id="predict"></div>'), 'CONTEXT':('Static intervention: two hops with all edges reaches m1/m2/p1/s1; deleting p1 leaves m1 only.','<div id="context-explorer"></div><noscript>At two hops m1 reaches p1, m2 and s1. Removing p1 leaves only m1.</noscript>'), 'TOKENS':('Static trace: row lengths 3/5 give 64 whole-table pairs and 34 row pairs. Limit 4 drops 1 token and leaves 25 row pairs.','<div id="token-explorer"></div><noscript>Lengths 3/5: 64 whole-table pairs, 34 row pairs. Limit 4 drops one token and retains 25 row pairs.</noscript>'), 'TEACHBACK':('Explain why an LM+GNN can improve reconstruction without establishing transfer. Include architecture, training stages and evidence gaps.','<div id="teachback"></div>')}
 for key,(fallback,html) in variants.items():text=text.replace('[['+key+']]',fallback if portable else html)
 text=text.replace('[[RESULTS]]','**Executed author audit:** all 16 graph interventions, 8 token cases and 64 evidence configurations completed. Of the evidence fixtures, 32 have a reconstruction-only scope ceiling, 31 a multi-table-only ceiling and 1 is eligible for transfer review. All 64 retain transfer **NOT_ESTABLISHED**. These are synthetic declarations, not benchmark populations. [Complete output](../labs/evidence/l162/report.md).')
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/').replace('href="../','href="https://avistian.github.io/relational/')
  text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def doc(title,body,interactive=False):
 html=render(body).replace('<table>','<div class="route-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','relational-vision','l162-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in ['lesson','atomic-route','checkpoint','lab-access','foundation-scope','relational-vision'])+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0161-what-is-a-foundation-model.html">Lesson 161</a></nav><header><p class="route-kicker">Year 5 · Quarter 1 · Lesson 162</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('The relational FM vision',prose(),True))
reference='''## Trace the architecture
Serialize a row with schema → tokenize → LM encoder → graph context → decoder → masked text. The conceptual n×d row array is a teaching abstraction, not a complete historical tensor specification. Stage 1 adapts BART to rows. Stage 2 freezes BART parameters and trains the GNN; gradients through the frozen decoder must still reach the graph.

## Two different scale questions
Many rows: independent row inputs limit LM sequence size, but total encoding and graph work still grow. Wide rows: an individual serialized row can exceed the source's 1024-token BART limit. Truncation deletes evidence; splitting changes context.

Dense attention-pair proxies: whole table `(sum lengths)^2`; independent rows `sum(length^2)`; after explicit truncation `sum(min(length,limit)^2)`. For 3/5 tokens: 64 versus 34; limit 4 reduces 34 to 25 by deleting one token. Excludes decoder, graph, padding and other operations; not measured runtime or total memory.

## Context is a path
Course graph m1—p1—s1 and m2—p1: at most two hops from m1 reaches m1,p1,m2,s1. Removing p1 leaves m1 only. Induced-subgraph filtering occurs before traversal. Structural reachability is not measured predictive usefulness.

## Claim boundaries
Single-table reconstruction → multi-table evaluation → held-out-database transfer are different claims. The six-declaration audit can make a record eligible for review; it never certifies transfer. Inspect corpus provenance, legal adaptation, matched baselines and predictions.

## Reproduction status
Historical target: Table 1 wikiTables BART_table versus+GNN, three tasks/three runs, 10,000 tables, 70/20/10 split, validation-selected checkpoints. NOT_RUN; fidelity NOT_ESTABLISHED because matching code/subsets/splits/configuration remain unresolved. Local 16+8+64-case audit is synthetic and complete, with no model training. Learner PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/0162-the-relational-fm-vision.html) · [Vision map](../labs/l162-vision-map-template.md) · [Protocol](../labs/l162-reproduction.md) · [Primary source §§2–4](https://arxiv.org/html/2305.15321v1).
'''
(R/'reference/relational-fm-vision.html').write_text(doc('Relational FM vision — quick reference',reference))

def defs(path):
 text=path.read_text();return [(n.name,ast.get_source_segment(text,n)) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)]
contracts={
'reachable_rows':('Implement at-most-hop reachability on the induced graph.','nodes maps canonical nonempty row IDs to canonical nonempty table IDs; edges is a list of unique undirected pairs of distinct known nodes. hops is a nonnegative integer, excluding bool. allowed_tables is None or a nonempty list of unique known tables including the root table. Reject invalid inputs with ValueError. Filter nodes/edges before traversal; include root, return sorted unique IDs; do not mutate inputs.','assert reachable_rows({"m1":"moons","p1":"planets","s1":"stars"},[["m1","p1"],["p1","s1"]],"m1",2,["moons","stars"])==["m1"]\nprint("CHECK: removing the intermediate table breaks the path")'),
 'token_budget':('Expose the information price of truncation.','lengths is a nonempty list of nonnegative integers; limit is a positive integer. Reject booleans, floats and invalid inputs with ValueError. Return rows, total_tokens, retained_tokens, dropped_tokens, overflow_rows (zero-based indices), whole_table_pairs, row_pairs, retained_row_pairs. Whole-table and row pairs use original lengths; retained pairs use min(length,limit). No tokenization or total-compute claim.','r=token_budget([3,5],4)\nassert (r["whole_table_pairs"],r["row_pairs"],r["retained_row_pairs"],r["dropped_tokens"])==(64,34,25,1)\nprint("CHECK: lower retained pair count deletes one token")'),
 'evidence_verdict':('Require evidence for the claim being made.','Accept exactly six Boolean fields in this order: multitable, heldout_database, pretraining_audited, adaptation_legal, matched_baseline, predictions_verified. Reject missing/extra fields or non-Booleans with ValueError. Collect false fields as MULTITABLE_EVALUATION, HELD_OUT_DATABASE, PRETRAINING_PROVENANCE, LEGAL_ADAPTATION, MATCHED_BASELINE, VERIFIED_PREDICTIONS in that order. Return supported=RECONSTRUCTION_ONLY if multitable is false, otherwise MULTITABLE_ONLY; if no missing items return TRANSFER_REVIEW_ELIGIBLE. Always return transfer=NOT_ESTABLISHED and missing=list. Scope ceilings are conditional, not authenticated evidence.','r=evidence_verdict({k:True for k in ["multitable","heldout_database","pretraining_audited","adaptation_legal","matched_baseline","predictions_verified"]})\nassert r["supported"]=="TRANSFER_REVIEW_ELIGIBLE" and r["transfer"]=="NOT_ESTABLISHED"\nprint("CHECK: eligible for review is not established transfer")')}
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson 162 · The relational FM vision\n\nStandalone Tier C mechanism/design audit. Three live TODOs; standard-library computation, no downloads or model training. All fixture inputs and functions are embedded. Course links become public only after publication. Complete the map yourself; author preparation is not learner mastery.'),nb.v4.new_markdown_cell(prose(True)),nb.v4.new_code_cell('# @colab-bootstrap — PROVIDED: Python 3.10+ standard library, no installs/downloads.\nfrom pathlib import Path\nimport json,hashlib\n')]
 cells += [nb.v4.new_markdown_cell('## PROVIDED · All experiment inputs\nSixteen graph interventions, eight length/budget cases, all 64 evidence configurations. No random seed, fitted model or measured benchmark population.'),nb.v4.new_code_cell('fixture_text='+repr(raw.decode())+'\nassert hashlib.sha256(fixture_text.encode()).hexdigest()=='+repr(manifest['fixtures_sha256'])+'\npacket=json.loads(fixture_text)')]
 for name,code in defs(P/'relkit/vision_l162.py'):
  goal,contract,check=contracts[name];cells.append(nb.v4.new_markdown_cell('## TODO · '+name+'\n\n**Goal:** '+goal+'\n\n**Contract:** '+contract+'\n\nPredict an edge case before implementing.'))
  cells.append(nb.v4.new_code_cell(code if solution else code.split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'))
  cells.append(nb.v4.new_code_cell('# CHECK\n'+check))
 cells.append(nb.v4.new_markdown_cell('## PROVIDED · Full audit and adversarial feedback\nEvery function below calls your live implementations. The evidence classifier authenticates no files; inspect provenance and predictions separately.'))
 for path in ['_check_l162.py','_run_l162.py']:
  for name,code in defs(P/path):cells.append(nb.v4.new_code_cell(code))
 cells.append(nb.v4.new_code_cell('assert check162(reachable_rows,token_budget,evidence_verdict)=="PASS"\nreport=audit162(packet,reachable_rows,token_budget,evidence_verdict)\nassert report["counts"]==dict(graph=16,token=8,evidence=64)\nprint(render162(report))\nPath("l162-report.json").write_text(json.dumps(report,indent=2)+"\\n")'))
 cells.append(nb.v4.new_markdown_cell('## EXIT · Your opportunities/obstacles map\nWrite 400–600 words plus four table rows: opportunity → mechanism → obstacle → falsifiable test → evidence boundary. Use the template in the lesson. Review architecture, arithmetic, causality, sources and evaluation (0–2 each; ≥ 8/10, no zero). Code cannot assess your prose. The short author illustration below is not a completed learner submission.'))
 example='Row-wise text encoding supplies local semantics; graph paths supply relational context. Our m1 example reaches star s1 only through p1, so removing an intermediate table loses access even when both endpoints remain. Separate rows reduce an attention-pair proxy, but graph cost and wide rows remain obstacles. I would test transfer using a provenance-audited held-out database, legal adaptation examples and a matched baseline. This is a proposed test: the historical single-table reconstruction table does not establish it, and the synthetic audit trains no model.' if solution else ''
 cells.append(nb.v4.new_code_cell('vision_map='+repr(example)+'\nsubmission=dict(vision_map=vision_map,review=None,learner="PENDING_WRITTEN_DEFENSE")\nPath("l162-submission.json").write_text(json.dumps(submission,indent=2)+"\\n")\nprint("Exported map; learner PENDING_WRITTEN_DEFENSE")'))
 cells.append(nb.v4.new_markdown_cell('## Full-reproduction gate\nHistorical Table 1 wikiTables training is NOT_RUN; protocol fidelity NOT_ESTABLISHED. No matching code, exact subset/split or adequate complete training recipe was located. Do not reinterpret this synthetic runner as a historical trainer. Resolving the source gaps and budgeting all runs precede any training authorization. Ask the teaching agent to assess your map.'))
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(book.cells):c.id=f'l162-{i:03d}'
 path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
  old=nb.read(path,4)
  if [(c.cell_type,c.source) for c in old.cells]==[(c.cell_type,c.source) for c in book.cells]:
   book.metadata=old.metadata
   for c,prior in zip(book.cells,old.cells):
    if c.cell_type=='code':c.outputs=prior.outputs;c.execution_count=prior.execution_count;c.metadata=prior.metadata
 nb.write(book,path)
print('Built L162 HTML, reference, two figures, student and solution')
