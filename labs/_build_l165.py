"""Canonical L165 lesson, portable notebooks and model-specific diagrams."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;R=P.parent;S='0165-kumorfm-in-context-relational-learning';F=P/'figures/l165';E=P/'evidence/l165'
raw=(E/'fixtures.json').read_bytes();manifest=json.loads((E/'input-manifest.json').read_text());assert hashlib.sha256(raw).hexdigest()==manifest['fixtures_sha256']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l165'})
def canvas(title,height):
 fig,ax=plt.subplots(figsize=(12,height));fig.subplots_adjust(left=.025,right=.975,bottom=.02,top=.9);fig.patch.set_facecolor('#f5f8f5');ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');fig.suptitle(title,x=.04,ha='left',fontsize=18,y=.97);return fig,ax

def box(ax,x,y,w,h,title,body,color='#e3eee8'):
 patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008',facecolor=color,edgecolor='#91ad9f');ax.add_patch(patch)
 ax.text(x+.013,y+h-.027,title,va='top',fontsize=11.5,weight='bold',color='#174e40')
 ax.text(x+.013,y+h-.082,body,va='top',fontsize=10,linespacing=1.45,color='#263d35')

def arrow(ax,a,b):
 ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=2,color='#287b68'))

def save(fig,name):
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 for ax in fig.axes:
  for t in ax.texts:
   b=t.get_window_extent(renderer)
   assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,(name,t.get_text())
 for ext in ['svg','png']:fig.savefig(F/(name+'.'+ext),dpi=150,metadata={'Date':None} if ext=='svg' else {'Software':'L165'})
 plt.close(fig)
 for svg in F.glob('*.svg'):svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
fig,ax=canvas('KumoRFM v1: two places where labeled context can enter',8)
box(ax,.01,.72,.28,.24,'Database + predictive task','Rows, key links, event times\nTask defines the outcome\nQuery target stays hidden')
box(ax,.36,.72,.28,.24,'Owner-specific input graphs','Query graph: cutoff q\nContext graphs: cutoff c < q\nEach example keeps its own clock')
box(ax,.71,.72,.28,.24,'Historical label generator','Outcome ends at c + horizon\nMust be known by query time\nLabels supplied separately','#fff0d9')
arrow(ax,(.30,.85),(.35,.85))
ax.annotate('',xy=(.85,.975),xytext=(.15,.975),arrowprops=dict(arrowstyle='->',lw=1.6,color='#bb8140',connectionstyle='arc3,rad=-.045'))
box(ax,.01,.37,.28,.25,'1 · Row encoding','Cells, including attached labels\nN_T × C_T × F → N_T × F\nShared-width row representations')
box(ax,.36,.37,.28,.25,'2 · Relational transformer','Rows + eligible attached labels\nType / hop / time / structure codes\nRoot readout: F features')
box(ax,.71,.37,.28,.25,'3 · Across-subgraph ICL','m context vectors + m labels\nk query vectors + hidden targets\nOutputs: k class probabilities')
arrow(ax,(.49,.70),(.10,.64));arrow(ax,(.85,.70),(.27,.64));arrow(ax,(.85,.70),(.85,.64))
arrow(ax,(.30,.49),(.35,.49));arrow(ax,(.65,.49),(.70,.49))
ax.text(.04,.28,'WITHIN A ROOTED GRAPH',fontsize=10,weight='bold',color='#79522b')
ax.text(.04,.225,'Attached labels obey that graph’s time boundary.',fontsize=10)
ax.text(.60,.28,'ACROSS ROOTED GRAPHS',fontsize=10,weight='bold',color='#79522b')
ax.text(.60,.225,'Each historical graph keeps its own cutoff.',fontsize=10)
box(ax,.01,.02,.98,.13,'Training boundary','Pretraining is reported; exact recipe / weights unresolved. In-context inference freezes weights. Fine-tuning is a separate mode.','#e9edf3')
save(fig,'architecture')
fig,ax=canvas('Two clocks: historical features stop before their outcome window',5.6)
xs={8:.13,10:.39,12:.65,13:.78,14:.91}
for y in [.65,.31]:ax.plot([.06,.94],[y,y],color='#9bb4a9',lw=2)
for day,x in xs.items():
 ax.plot([x,x],[.26,.70],color='#d7e0da',lw=1);ax.text(x,.77,str(day),ha='center',fontsize=13,weight='bold')
ax.text(.02,.89,'DAY',fontsize=10,color='#49675a');ax.text(.02,.97,'Course example · anchor 8 · four-day horizon · one-day arrival delay',fontsize=11)
ax.plot([xs[8],xs[12]],[.65,.65],color='#bb8140',lw=8,solid_capstyle='round')
ax.scatter([xs[8],xs[12],xs[13]],[.65]*3,s=90,c=['#226a57','#bb8140','#226a57'],zorder=4)
ax.text(xs[8],.54,'Graph stops\nat day 8',ha='center',fontsize=11)
ax.text(xs[12],.54,'Window\nends',ha='center',fontsize=11)
ax.text(xs[13]+.035,.54,'Label\narrives',ha='center',fontsize=11)
ax.scatter([xs[10],xs[13]],[.31]*2,s=120,c=['#ac533b','#226a57'],zorder=4)
ax.text(xs[10],.16,'Query 10: EXCLUDE\nOutcome still unfinished',ha='center',fontsize=11)
ax.text(xs[13],.16,'Query 13: INCLUDE\nContext graph stays at 8',ha='center',fontsize=11)
ax.text(.02,.015,'A later query may unlock a label; it must not advance the historical example’s feature cutoff.',fontsize=11)
save(fig,'clocks')
captions={'architecture':'Source-reported interfaces and information paths, not a checkpoint-compatible implementation. F is symbolic; complete masks and training configuration remain unresolved. Attached labels inside a historical graph obey that graph’s cutoff.','clocks':'The day-8 example is unavailable at query 10 and eligible at query 13. Its input graph remains anchored at day 8. Integer days and inclusive arrival are course conventions.'}

def prose(portable=False):
 text=(R/'lessons/content'/(S+'.md')).read_text()
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l165/'+name+'.svg'
  text=text.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}'+('' if portable else ' On a narrow screen, focus and scroll the figure horizontally.')+'</figcaption></figure>')
 variants={'WARMUP':('Recall: how does Griffin fine-tuning change weights, and what stays fixed in in-context inference?','<div id="warmup"></div>'),
 'PREDICT':('Predict before reading on: at day12, should the example whose label arrives at13 be included? Explain.','<div id="predict"></div>'),
 'CONTEXT':('Static trace: anchor8, horizon4, delay1. At query10 or12 exclude; at13 include. Context graph cutoff stays8. Changing hidden query label cannot change inputs.','<div id="context-explorer"></div><noscript>Anchor8, horizon4, delay1: exclude at query10 and12; include at13. Context graph cutoff stays8. Query labels never enter input.</noscript>'),
 'TEACHBACK':('Explain fixed weights, the two context paths, both time boundaries, and the missing v1 reproduction evidence.','<div id="teachback"></div>')}
 for key,(static,html) in variants.items():text=text.replace('[['+key+']]',static if portable else html)
 report=json.loads((E/'report.json').read_text());assert report['counts']==dict(context=144,graph=96,scoring=24,query_label_interventions=2)
 text=text.replace('[[RESULTS]]','**Executed author audit:** all 144 context cases, 96 graph cases and 24 score orderings completed. Two hidden-label interventions leave inputs identical. Independent checks cover 1,134 tie-aware AUROC cases. These are deterministic synthetic contracts, not measured KumoRFM performance. [Full results](../labs/evidence/l165/report.json).')
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/').replace('href="../','href="https://avistian.github.io/relational/')
  text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def doc(title,body,interactive=False):
 html=re.sub(r'<table([^>]*)>',r'<div class="route-scroll" tabindex="0"><table\1>',render(body)).replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','relational-context','l165-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in ['lesson','atomic-route','checkpoint','lab-access','foundation-scope','relational-context'])+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0164-griffin-graph-centric-rdb-fm.html">Lesson 164</a></nav><header><p class="route-kicker">Year 5 · Quarter 1 · Lesson 165</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('KumoRFM: in-context relational learning',prose(),True))
reference='''## Fixed weights, changing context
Prediction = f(weights, query graph, context graphs and labels). In-context adaptation changes inputs. Fine-tuning changes weights. Neither implies the absence of earlier pretraining. A labeled-context model is not label-free simply because it is described as zero-shot.

## Information paths
Row/cell representations → relational transformer and attached context labels → root vectors → across-subgraph ICL with explicit labels → query probabilities. The diagram in the lesson follows the report; the course does not provide matching KumoRFM encoders or weights.

## Two clocks, three checks
Context anchor c < query cutoff q; outcome end ≤ q; label arrival ≤ q. Context input features stop at c, query features at q. For c=8, end12, arrival13: exclude at q10 or12, include at13; context graph stays at8. Attached labels in a historical graph must also be known by that graph's cutoff. Explicit arrival data is essential; event time alone cannot prove availability.

## Keyed AUROC
Align by complete (entity, cutoff) keys. A positive score above a negative contributes1, tied contributes0.5, below contributes0. Average all positive–negative pairs. Scores [.8,.9] versus [.8,.1] yield .875. Reject absent/extra/duplicate keys, nonfinite scores and single-class populations.

## Evidence boundary
Named local audit:144context,96graph,24scoring cases and2hidden-label interventions. Synthetic contracts, not model predictions. Historical target: v1 Table2 rel-f1/driver-dnf in-context82.41AUROC, or.8241. Exact weights/data/context recipe unresolved; historical NOT_RUN, fidelity NOT_ESTABLISHED. A modern client call cannot authenticate v1. Reported comparator ≠ an upper performance bound.

[Lesson](../lessons/0165-kumorfm-in-context-relational-learning.html) · [Protocol](../labs/l165-reproduction.md) · [Capability note](../labs/l165-capability-template.md) · [Original report §§2–2.3/Table2](../labs/sources/l165/paper.pdf).
'''
(R/'reference/relational-in-context-learning.html').write_text(doc('Relational in-context learning — quick reference',reference))

def defs(path):
 text=path.read_text();return [(n.name,ast.get_source_segment(text,n)) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)]
contracts={
'eligible_context':('Select historical labels that are both mature and available.','Input: list of records with exactly entity (nonempty trimmed string), cutoff, label_end, available_at (integer times, excluding bool), label (integer0/1); query exactly entity/cutoff. Require label_end≥cutoff and available_at≥label_end; reject duplicate full keys. Select cutoff<query.cutoff, label_end≤query.cutoff and available_at≤query.cutoff. Return copies sorted by (cutoff,entity), without mutating inputs. Reject invalid inputs with ValueError.','r=dict(entity="u",cutoff=8,label_end=12,available_at=13,label=1)\nassert eligible_context([r],dict(entity="q",cutoff=12))==[]\nassert eligible_context([r],dict(entity="q",cutoff=13))==[r]\nprint("CHECK: window end is not label arrival")'),
'visible_graph':('Keep the owner cutoff at every hop.','rows: exact fields id (unique nonempty trimmed string), event_at (integer or None for timeless), available_at (integer). Require available_at≥event_at when not None. edges: unique undirected lists of two distinct known IDs; static links inherit endpoint availability. root must exist and be available. cutoff is integer; hops is nonnegative integer; bool is invalid. Filter rows by event_at≤cutoff (or None) and available_at≤cutoff before traversal. Return sorted unique IDs within at most hops, including root. Reject extra fields, invalid clocks/edges/duplicates with ValueError; do not mutate.','rows=[dict(id="u",event_at=None,available_at=0),dict(id="o",event_at=5,available_at=8)]\nassert visible_graph(rows,[["u","o"]],"u",7,2)==["u"]\nprint("CHECK: late arrival is excluded even after early event")'),
'keyed_auc':('Score complete query identities and handle ties.','truth and predictions are nonempty lists with exactly entity (nonempty trimmed string), cutoff (integer, no bool) and label (integer0/1) or score (finite int/float, no bool). Reject duplicate keys, unequal key sets and single-class truth with ValueError. Return binary AUROC in0–1; ties earn half credit. Input order must not matter; do not mutate.','truth=[dict(entity="u",cutoff=1,label=1),dict(entity="u",cutoff=2,label=0)]\npred=[dict(entity="u",cutoff=2,score=.5),dict(entity="u",cutoff=1,score=.5)]\nassert keyed_auc(truth,pred)==.5\nprint("CHECK: same entity at two cutoffs remains two records")')}
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson165 · KumoRFM in-context relational learning\n\nStandalone standard-library course audit. No download, API key, model training or cloud spending. Full prose and portable diagrams are embedded. Public course links work only after publication. Three live learner functions feed the complete experiment.'),nb.v4.new_markdown_cell(prose(True)),nb.v4.new_code_cell('# @colab-bootstrap — Python3.10+ standard library only.\nfrom pathlib import Path\nimport json,hashlib')]
 cells += [nb.v4.new_markdown_cell('## PROVIDED · Complete deterministic input packet\nAll fixtures are invented course inputs, not F1 records. Exact bytes are checked before use.'),nb.v4.new_code_cell('fixture_text='+repr(raw.decode())+'\nassert hashlib.sha256(fixture_text.encode()).hexdigest()=='+repr(manifest['fixtures_sha256'])+'\npacket=json.loads(fixture_text)')]
 for name,code in defs(P/'relkit/context_l165.py'):
  goal,contract,check=contracts[name]
  cells += [nb.v4.new_markdown_cell('## TODO · '+name+'\n\n**Goal:** '+goal+'\n\n**Contract:** '+contract+'\n\nPredict one boundary case before writing code.'),nb.v4.new_code_cell(code if solution else code.split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'),nb.v4.new_code_cell('# CHECK\n'+check)]
 cells.append(nb.v4.new_markdown_cell('## PROVIDED · Integrated feedback and complete experiment\nThe following checks and runner use your live functions. There is no hidden reference implementation. The metric checks target known ties, shuffled predictions and repeated entities at different cutoffs.'))
 for path in ['_check_l165.py','_run_l165.py']:
  for name,code in defs(P/path):cells.append(nb.v4.new_code_cell(code))
 cells.append(nb.v4.new_code_cell('assert check165(eligible_context,visible_graph,keyed_auc)=="PASS"\nreport=audit165(packet,eligible_context,visible_graph,keyed_auc)\nprint(render165(report))\nPath("l165-report.json").write_text(json.dumps(report,indent=2)+"\\n")'))
 cells.append(nb.v4.new_markdown_cell('## EXIT · Capability and reproduction note\nWrite400–600words plus four claim/evidence rows. Explain the day8/day13 trace, two ICL locations, exact Table2 target and missing artifacts. Score0–2each for mechanism, clocks, keyed scoring, source identity and bounded conclusion; target≥8/10,no zero. A teacher must review your prose. The short illustration below is not a completed defense.'))
 example='A frozen model can adapt because labeled inputs change. At query13, the day8 context label can be available while its feature graph must still end at8. The report describes within-graph and across-graph context. Our contract audit does not authenticate its private masks or reproduce the historical .8241AUROC. Matching weights, historical task keys and context policy must precede that claim.' if solution else ''
 cells.append(nb.v4.new_code_cell('capability_note='+repr(example)+'\nsubmission=dict(note=capability_note,review=None,learner="PENDING_WRITTEN_DEFENSE")\nPath("l165-submission.json").write_text(json.dumps(submission,indent=2)+"\\n")\nprint("Learner PENDING_WRITTEN_DEFENSE")'))
 cells.append(nb.v4.new_markdown_cell('## Full-reproduction boundary\nHistorical KumoRFM-v1 Table2 driver-dnf: NOT_RUN. Matching inference/weights, data identity and context protocol remain unresolved. This notebook is the complete approved local audit, not an API demo or reconstructed historical model. Resolve missing artifacts and obtain a costed protocol before inference. LiveColab and deployment are separate checks. Ask the agent to review your defense.'))
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(book.cells):c.id=f'l165-{i:03d}'
 path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
  old=nb.read(path,4);book.metadata=old.metadata
  previous=[c for c in old.cells if c.cell_type=='code'];current=[c for c in book.cells if c.cell_type=='code']
  if [c.source for c in previous]==[c.source for c in current]:
   for prior,c in zip(previous,current):
    c.outputs=prior.outputs;c.execution_count=prior.execution_count;c.metadata=prior.metadata
 nb.write(book,path)
print('Built L165 lesson/reference, two figures and portable notebooks')
