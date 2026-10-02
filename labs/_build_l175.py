"""Deterministic lesson, reference and portable student/solution notebook builder."""
import ast,base64,hashlib,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l175';S='0175-zero-shot-evaluation'
r=json.loads((E/'verified-audit.json').read_text())
captions={'architecture':'Actual RT-v1 architecture: typed cell encoders, twelve four-attention blocks, and a frozen boolean head. The sampler determines information access.','availability':'Synthetic 30-day task illustration. A completed outcome window is necessary, but does not establish ingestion time. No unfinished-window labels were observed in the measured contexts.','audit':'Measured original-sampler audit: full702queries for each of three context seeds. Event-time audit fails; no model scores were computed.'}
def results():
 return '''| Check | Complete measured result |
|---|---:|
| Contexts / cell slots | 2,106 / 2,156,544 |
| Independently reconstructed query labels | 2,106 |
| Exposed query targets | 0 |
| Same-time visible outcome labels | 0 |
| Visible labels with unfinished outcome windows | 0 |
| Future-dated cells / affected contexts | **385 / 77** |
| Visible historical label cells | 29,988 |
| Historical test-period label cells | 12,351 |
| Cells without a timestamp | 432,050 |
| Checkpoint evaluations | **0 of 6 — NOT_RUN** |

Counts include repeated cells across query contexts. They are not counts of unique database rows. Context seeds0/1/2 contain26/26/25affected queries respectively.'''
finding='''All future-dated cells belong to **race schedule rows**: year, round, name, date and time. One query at **2011-03-27 00:00 UTC** includes a race dated **06:00 UTC** that day. The sampler reaches it through an unfiltered foreign-key-to-parent step. These attributes may have been known in advance; the snapshot contains no arrival history to prove that. We establish failure of the declared event-time filter, **not demonstrated exposure to future race outcomes**. The complete audit found no unmasked query target and no label with an unfinished outcome window.'''
def prose(portable=False):
 s=(R/'lessons/content'/(S+'.md')).read_text().replace('[[RESULTS]]',results()).replace('[[FINDING]]',finding)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/'figures/l175'/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l175/'+name+'.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption} Scroll horizontally on narrow screens.</figcaption></figure>')
 replacements={'WARMUP':('Recall: what does L171 exclude when holding out a database? Does L174 head-only tuning change parameters?','<div id="warmup"></div>'),'EXPLORER':('Trace each access path: gradients, validation selection, historical context labels, unavailable answers. Frozen weights only close the first path.','<div id="access-explorer"></div><noscript>Frozen weights do not prevent validation or context-label access. Unavailable answers invalidate the clean evaluation.</noscript>'),'PREDICT':('Predict first: does freezing weights make a same-time neighboring 30-day outcome safe? Explain before reading the answer.','<div id="predict"></div>'),'TEACHBACK':('Write your evaluation card before consulting teacher feedback.','<div id="teachback"></div>')}
 for tag,(plain,html) in replacements.items():s=s.replace('[['+tag+']]',plain if portable else html)
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/')
  s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def doc(title,body,interactive=False):
 html=render(body).replace('<table>','<div class="route-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 css=['lesson','atomic-route','checkpoint','lab-access','zero-shot-evaluation'];scripts=['retrieval-pool','retrieval-bank','predict','teachback','zero-shot-evaluation','l175-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in css)+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0174-fine-tuning-protocol.html">Lesson174</a></nav><header><p class="route-kicker">Year5 · Quarter2 · Lesson175</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Zero-shot evaluation',prose(),True))
reference='''**Four access paths:** target gradients; target validation selection; historical context labels; unavailable answers. RT-v1 uses “zero-shot” for a database excluded from gradient pretraining, with target validation and context-label access still possible.

**Three clocks:** event timestamp; label-window end; arrival time. For driver-dnf, a necessary label condition is `label_cutoff +30days <= query_cutoff`. A scheduled future event might be known earlier; without arrival evidence, report the uncertainty.

**Complete keys:** `(driverId, cutoff)`. Reject missing/duplicate keys; align predictions before computing AUROC. AUROC counts correctly ordered positive-negative pairs, with half credit for ties. It is not classification accuracy.

**Order of work:** pin model/source/data → audit exposure and every context → project aggregate cost → run frozen evaluations → independently score → defend the claim. A failed gate stops the downstream claim.

'''+results()+'\n\n'+finding+'''

**Current boundary:** complete native-context audit and local replay; checkpoint inference NOT_RUN; original historical identity/availability NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE. Scheduled rows failing the event-time bound do not prove outcome leakage.

[Lesson](../lessons/0175-zero-shot-evaluation.html) · [Notebook](../labs/0175-zero-shot-evaluation.ipynb) · [Protocol](../labs/l175-reproduction.md) · [RT paper](https://arxiv.org/html/2510.06377v1#S4).
'''
(R/'reference/zero-shot-evaluation.html').write_text(doc('Zero-shot evaluation — quick reference',reference))
# Packet includes all real contexts and raw oracle. Stable ZIP timestamps and file order.
files={f'contexts-{seed}.npz':E/f'audit-3/contexts-{seed}.npz' for seed in range(3)}
files.update({'context-audit.json':E/'audit-3/context-audit.json','label-oracle.npz':E/'label-oracle.npz','table_info.json':P/'sources/l175/table_info.json','column_index.json':P/'sources/l175/column_index.json'})
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for name,path in sorted(files.items()):
  info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,path.read_bytes())
packet=base64.b64encode(buf.getvalue()).decode();hashes={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in files.items()}
source=(P/'relkit/zero_shot_l175.py').read_text();tree=ast.parse(source);definitions={n.name:ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
checks=(P/'_check_l175.py').read_text();checktree=ast.parse(checks);checkdefs='\n\n'.join(ast.get_source_segment(checks,n) for n in checktree.body if isinstance(n,ast.FunctionDef))

def make(solution):
 cells=[]
 def md(t):cells.append(nb.v4.new_markdown_cell(t))
 def code(t,tags=None):cells.append(nb.v4.new_code_cell(t,metadata={'tags':tags or []}))
 md('# Lesson175 · Zero-shot evaluation\n\nImplement an information-access audit before trusting a model score. Complete three TODOs, then replay all2,106real saved contexts. No paid jobs or fresh model inference are launched.\n\nAuthor evidence: native-context audit COMPLETE; six checkpoint evaluations NOT_RUN after temporal gate failure. Your mastery remains PENDING_WRITTEN_DEFENSE.')
 code('''# @colab-bootstrap
import importlib.util, subprocess, sys
needed=[name for name,module in [('numpy','numpy'),('ml_dtypes==0.5.3','ml_dtypes'),('einops','einops'),('torch','torch')] if importlib.util.find_spec(module) is None]
if needed: subprocess.check_call([sys.executable,'-m','pip','install',*needed])
import numpy as np
from pathlib import Path
import json
''')
 md(prose(True))
 md('## PROVIDED · Authenticate the complete portable evidence\nThe compressed bytes contain all contexts, original node metadata and raw F1 results used to reconstruct labels. Hashes catch accidental corruption. This is saved evidence, not output from a model trained in this kernel.')
 code('import base64, hashlib, io, zipfile\nPACKET='+repr(packet)+'\nEXPECTED='+repr(hashes)+'''\nroot=Path('l175-packet');root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PACKET))) as z:
    assert set(z.namelist())==set(EXPECTED)
    for name,digest in EXPECTED.items():
        data=z.read(name);assert hashlib.sha256(data).hexdigest()==digest,name
        (root/name).write_bytes(data)
print('Authenticated',len(EXPECTED),'files. No network required for evidence.')''',['data-payload'])
 exercises=[('exposure_claim','TODO1 · Classify information access','Return the exact claim string. Unavailable answers take precedence; then target gradients; then validation/context labels. Freezing parameters alone is insufficient.'),('audit_context','TODO2 · Audit available information','Timestamps are Unix seconds. Padding is excluded; unknown time is int32 minimum. Count each requested category, including labels whose outcome window has not finished. A known future schedule fails this strict event-time policy even when its real arrival time is unknown.'),('aligned_auc','TODO3 · Align full identities, then score','Reject duplicate/missing (entity,cutoff) keys and nonfinite scores. Align scores to labels before computing pairwise AUROC. Give ties half credit. Do not collapse repeated entities.')]
 for name,title,description in exercises:
  md('## '+title+'\n'+description)
  if solution:code(definitions[name])
  else:
   fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);sig=definitions[name].split('\n',1)[0]
   code(sig+'\n    raise NotImplementedError("'+title+'")')
  if name=='exposure_claim':code("assert exposure_claim(False,True,True,False)=='NO_TARGET_GRADIENTS_WITH_LABEL_ACCESS'\nassert exposure_claim(False,False,False,True)=='INVALID_TEST_EXPOSURE'\nprint('CHECK1 passed: frozen weights are only one information boundary.')")
  elif name=='audit_context':code("probe=audit_context([0,0,-40,10],[False]*4,[True,False,False,False],[True,True,True,False],[True,False,False,False],0,30)\nassert probe['unmasked_query_targets']==0 and probe['unavailable_labels']==1 and probe['future_cells']==1\nprint('CHECK2 passed: task-row time differs from outcome availability.')")
  else:code("keys=np.array([[7,100],[7,200],[8,100],[8,200]]);y=np.array([0,1,1,0]);scores=np.array([.2,.9,.7,.7]);perm=[2,0,3,1]\nassert aligned_auc(keys,y,keys[perm],scores[perm])==.875\nprint('CHECK3 passed: keyed synthetic AUROC is 0.875; this is not a model score.')")
 md('## PROVIDED / CHECK · Reject plausible wrong policies\nThe checks exercise your live functions, including missing keys, tie handling and unfinished outcome windows. Mutation checks temporarily substitute three wrong policies and must reject each.')
 code(definitions['reserve_cost']);code(checkdefs);code("print(checks());print('Rejected wrong policies:',mutation_checks())")
 md('## PROVIDED · Full replay and independent label reconstruction\nFor each of702queries×3seeds, invoke your context audit, validate full identities, and independently derive DNF from raw race results in the following30days. Compare against the native-sampler receipt. This runs your TODO2 over every context, not a canned answer.')
 code(definitions['replay_packet']);code("report=replay_packet(root,audit_context)\nassert report['contexts']==2106\nassert report['totals']['future_cells']==385\nprint(json.dumps(report,indent=2))\nPath('l175-replay.json').write_text(json.dumps(report,indent=2))")
 md('## EXIT · Written evaluation card\nExplain which database was excluded, who selected the checkpoint, which labels the context may expose, why the event-time check failed, and why future scheduled fields do not prove outcome leakage. State which model result remains unmeasured. Paste this defense to your teacher for feedback.')
 code("submission={'learner':'PENDING_WRITTEN_DEFENSE','defense':''}\nPath('l175-submission.json').write_text(json.dumps(submission,indent=2))\nassert not submission['defense']  # Fill in your own explanation; no author answer is submitted.")
 md('## Appendix A · Original RT-v1 model, visible end to end\nThese are the archived source definitions used by the proposed reproduction. They are provided for inspection. Defining them does not load weights, reproduce training, or execute GPU inference. Follow name/value/mask encoding → attention masks →12blocks → boolean head. The main practice above concerns evaluation, so we do not replace this model with a toy.')
 model=(P/'sources/l175/upstream/rt/model.py').read_text();mt=ast.parse(model);first=min(n.lineno for n in mt.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)))
 code('\n'.join(model.splitlines()[:first-1]))
 descriptions={'MaskedAttention':'Project Q/K/V, apply a relation-specific attention mask, then project back.','FFN':'SwiGLU uses one branch to gate another before the return projection.','RelationalBlock':'Four residual attention stages followed by a residual feed-forward stage.','_make_block_mask':'Convert explicit allowed cell pairs into the native attention representation.','RelationalTransformer':'All semantic encoders, mask vectors, relation masks, twelve blocks, and typed heads. Label tensors enter the loss; masked label values are excluded from token encoding.'}
 for node in mt.body:
  if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
   md('### '+node.name+'\n'+descriptions.get(node.name,''));code(ast.get_source_segment(model,node))
 md('## Appendix B · Original native sampler\nUnmodified Rust source follows. Notice where PK→FK timestamps are checked and where FK→PK parents are enqueued without that condition. The completed author run used this source; notebook replay uses its saved arrays.\n\n```rust\n'+(P/'sources/l175/upstream/rustler/src/fly.rs').read_text()+'\n```')
 md('## Appendix C · Complete gated six-run inference operator\nThe current receipt blocks this operator before model import or checkpoint loading. Its post-gate GPU path is UNVALIDATED. The complete original trainer remains [archived source](https://avistian.github.io/relational/labs/sources/l175/upstream/rt/main.py); no training is claimed here.')
 operator=(P/'_run_l175.py').read_text();ot=ast.parse(operator);code('import time\n'+ast.get_source_segment(operator,next(n for n in ot.body if isinstance(n,ast.FunctionDef))))
 code("try:\n    run(Path('unused-source'),Path('unused-checkpoints'),root,Path('must-not-be-created'))\nexcept RuntimeError as error:\n    assert 'BLOCKED_TEMPORAL_AUDIT' in str(error)\n    print('Expected stop:',error)\nelse:\n    raise AssertionError('Failed temporal audit did not block inference')\nassert not Path('must-not-be-created').exists()")
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(book.cells):c.id=f'l175-{i:03}'
 return book
student=make(False);solution=make(True);P.joinpath('solutions').mkdir(exist_ok=True)
nb.write(student,P/(S+'.ipynb'))
solpath=P/'solutions'/(S+'.ipynb')
if solpath.exists():
 old=nb.read(solpath,4)
 if [c.source for c in old.cells if c.cell_type=='code']==[c.source for c in solution.cells if c.cell_type=='code']:
  for newcell,oldcell in zip([c for c in solution.cells if c.cell_type=='code'],[c for c in old.cells if c.cell_type=='code']):
   newcell.outputs=oldcell.outputs;newcell.execution_count=oldcell.execution_count
nb.write(solution,solpath)
(E/'report.md').write_text('# L175 measured audit\n\n'+results()+'\n\n'+finding+'\n')
print('Built lesson, reference and notebooks:',len(student.cells),'cells')
