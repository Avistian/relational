"""Deterministic lesson, reference and portable student/solution notebook builder."""
import ast,base64,hashlib,io,json,re,zlib
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l145';SLUG='0145-relational-graph-transformer';TITLE='RelGT: five elements, one relational token'
audit=json.loads((E/'prepared/audit.json').read_text());pilots=[json.loads(p.read_text()) for p in sorted(E.glob('pilot-*/result.json'))]
results='| Evidence | Training queries | Validation queries | Future token occurrences |\n|---|---:|---:|---:|\n'
for s,r in audit['split_audits'].items():results+=f"| Full {s} token audit | {r['queries'] if s=='train' else '—'} | {r['queries'] if s=='val' else '—'} | {r['future_token_occurrences']:,} |\n"
results+='\n**Full selected reproduction: INCOMPLETE.** The source token cache fails the temporal contract. The full test cache contains 760 queries; zero future-token violations there does not repair leaked training/validation contexts. All **8,712 labels** were independently rebuilt. Graph preprocessing is **reused under verified hashes from L143**, not freshly regenerated here.\n'
for r in pilots:results+=f"\nDepth {r['layers']} timing pilot: **{r['history'][0]['queries']} training queries**, **256 validation queries**, partial validation MAE **{r['scores']['val']:.6f}**. Test **NOT_RUN**. This is invalid-context source replay for timing, not benchmark evidence. Real-batch original-model maximum output difference: **{r['real_batch_original_max_error']:.3g}**.\n"
if (E/'cost-decision.json').exists():
 d=json.loads((E/'cost-decision.json').read_text());results+='\n**Cost decision: '+d['decision']+'.** '+d['explanation']+'\n'
results+='\nHistorical identity **NOT_ESTABLISHED**; whole paper **NOT_RUN**; live Colab/deployment **NOT_CHECKED**; learner **PENDING_WRITTEN_DEFENSE**.\n'
captions={'architecture':'Trace a query through five encoders, local and global attention, and the prediction head. Source variant; all dimensions shown.','token':'Fixed-weight illustrative scalar projection; removing the fifth component changes 4 to 9.','ownership':'Two queries share an entity but require different temporal contexts. A positive stored age can hide a violation.','attention':'Local all-pairs computation and global centroid state have different information and update boundaries.'}
def prose(portable=False):
 s=(R/'lessons/content'/f'{SLUG}.md').read_text().replace('[[RESULTS]]',results)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l145/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l145/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="relgt-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 placeholders={'WARMUP':('warmup','Answer the retrieval questions before proceeding.'),'TOKEN_WIDGET':('relgt-token','Predict the fixed-weight projection when structure changes from 5 to 0. Then compute it.'),'OWNER_WIDGET':('relgt-owner','Keep event time 9 and cached age 1 fixed. Which actual cutoffs from 5 through 10 permit the event?'),'PREDICT':('relgt-predict','Predict first: does a positive cached age prove that the row is legal for its actual query?'),'TEACHBACK':('relgt-teachback','Write your explanation before checking the reference solution. Send it to the teaching agent for feedback.')}
 for tag,(id,fallback) in placeholders.items():s=s.replace('[['+tag+']]',fallback if portable else f'<div id="{id}"></div><noscript>{fallback}</noscript>')
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def document(title,body,interactive=False):
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','relgt-viz','relgt-lesson'] if interactive else []
 html=re.sub(r'<table([^>]*)>',r'<div class="relgt-scroll" tabindex="0"><table\1>',render(body)).replace('</table>','</table></div>')
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/relgt.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0144-contextgnn.html">Lesson 144</a></nav><header><p class="relgt-kicker">Year 4 · Quarter 3 · Lesson 145</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join(f'<script src="../assets/{s}.js"></script>' for s in scripts)+'</body></html>'
(R/'lessons'/f'{SLUG}.html').write_text(document(TITLE,prose(),True))
ref='''## Forward trace
Query=(entity, cutoff). K=300 slots including root. Type/hop/time/row/structure → five normalized 512-vectors → concatenate → 2560→1024→512 mixer. Local Transformer: four heads, all pairs within each query; root plus weighted neighbor readout. Global branch: root attends 4096 EMA centroids with log occupancy bias. Concatenate local/global → FFN → scalar head.

## Source contracts
Age=(cutoff−time)/86400; negative ages are masked, not filtered. Time encoder: sinusoid then linear. Structure: random scalar and four residual GIN layers, including random draws in evaluation. Local attention dropout remains active during evaluation. EMA buffers update only during training. Last tied validation minimum selects the checkpoint. MAE uses train-percentile clipping at evaluation.

## Temporal audit
Cache by entity alone can overwrite earlier queries with later contexts. Always compare raw row timestamps with each actual query cutoff; positive cached ages are insufficient. Global fallback is unfiltered. Source parity cannot prove temporal validity. A corrected sampler is a separate experiment.

## Reproduction verdict
'''+results+'''\n## Recall
Why concatenate rather than sum the five elements? Which centroids enter global attention? What should a checkpoint contain besides gradient-trained parameters? Why can eval() remain stochastic?

[Lesson](../lessons/0145-relational-graph-transformer.html) · [Protocol](../labs/l145-reproduction.md) · [Paper](https://arxiv.org/html/2505.10960v1). Ask the teaching agent for feedback on your explanation.
'''
(R/'reference/relgt.html').write_text(document('RelGT reference',ref))
files={p.name:p.read_text() for p in (P/'sources/l145').iterdir() if p.is_file()}
# Portable full-population audit stores only raw timestamps, targets and query identity.
pack={}
for split in ['train','val','test']:
 z=__import__('numpy').load(E/f'prepared/{split}-audit.npz');buf=io.BytesIO();__import__('numpy').savez_compressed(buf,**{k:z[k] for k in ['entity','cutoff','target','token_times']});pack[split+'-audit.npz']=base64.b64encode(buf.getvalue()).decode()
for p in E.glob('pilot-*/predictions.npz'):pack[str(p.relative_to(E))]=base64.b64encode(p.read_bytes()).decode()
packet="SOURCE_FILES=json.loads(zlib.decompress(base64.b64decode("+repr(base64.b64encode(zlib.compress(json.dumps(files).encode())).decode())+")))\nSOURCE_ROOT=Path('sources/l145').resolve();SOURCE_ROOT.mkdir(parents=True,exist_ok=True)\nfor name,text in SOURCE_FILES.items():(SOURCE_ROOT/name).write_text(text)\nAUTHOR_FILES=json.loads(zlib.decompress(base64.b64decode("+repr(base64.b64encode(zlib.compress(json.dumps(pack).encode())).decode())+")))\nAUTHOR_AUDIT="+repr(audit)+"\nAUTHOR_PILOTS="+repr(pilots)
bootstrap='''# @colab-bootstrap — pinned installs on Colab; local kernel uses its existing stack.
import os,sys,subprocess
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable,'-m','pip','install','-q','torch==2.5.1','--index-url','https://download.pytorch.org/whl/cu124'])
    subprocess.check_call([sys.executable,'-m','pip','install','-q','numpy==1.26.4','pandas==2.2.3','pyarrow==18.1.0','scipy==1.14.1','scikit-learn==1.5.2','pytorch-frame==0.2.3','torch-geometric==2.6.1','relbench==1.1.0','h5py==3.12.1','einops==0.8.0'])
    if 'torch' in sys.modules and sys.modules['torch'].__version__.split('+')[0]!='2.5.1':
        raise RuntimeError('Packages installed; restart runtime and run from the top.')
from pathlib import Path
import ast,base64,copy,hashlib,io,json,math,tempfile,types,zlib,contextlib
import numpy as np
import torch
from IPython.display import display
RUN_FULL_REPRODUCTION=False
'''
contracts=(P/'relkit/relgt_contracts_l145.py').read_text();defs={n.name:ast.get_source_segment(contracts,n) for n in ast.parse(contracts).body if isinstance(n,ast.FunctionDef)}
checks=(P/'_check_l145.py').read_text();checkdefs={n.name:ast.get_source_segment(checks,n) for n in ast.parse(checks).body if isinstance(n,ast.FunctionDef)}
mechanism=(P/'_mechanism_l145.py').read_text();mechanism=mechanism[:mechanism.index("if __name__=='__main__':")];mechanism=mechanism.replace('    if visible_namespace is None:\n        import relkit.relgt_l145 as visible\n        visible_namespace=vars(visible)','    if visible_namespace is None:visible_namespace=globals()')
trainer=(P/'_full_l145.py').read_text();trainer=re.sub(r'^from relkit[^\n]*\n','',trainer,flags=re.M).replace('    from _mechanism_l145 import original_modules\n','')
model=(P/'relkit/relgt_l145.py').read_text().replace('from relkit.relgt_contracts_l145 import mix_five\n','')
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson 145 · '+TITLE+'\n\n'+('Reference solution' if solution else 'Student lab')+' · PROVIDED / TODO / CHECK / EXIT. Three learner functions are called by the full model, temporal audit and checkpoint selector. Default execution checks a synthetic full network and saved author evidence. It does not train the nine-configuration benchmark. Author execution does not establish learner mastery.'),nb.v4.new_code_cell(bootstrap)]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if section.strip():cells.append(nb.v4.new_markdown_cell(section))
 payload=nb.v4.new_code_cell('# PROVIDED: pinned source and saved author evidence, not current learner results.\n'+packet);payload.metadata['tags']=['data-payload'];cells.append(payload)
 for name,check,title,hint in [('mix_five','check_mix','Preserve five separate elements','Join the last dimensions, keeping the source order and gradient flow.'),('audit_tokens','check_audit','Retain query ownership','Use actual timestamps and each query cutoff; never trust only the cached age.'),('last_validation_min','check_selection','Preserve the released tie rule','Reject invalid histories and let a tied minimum replace the earlier epoch.')]:
  cells.append(nb.v4.new_markdown_cell('## TODO · '+title+'\n\n**Goal:** implement `'+name+'`. **Why:** this contract lies on the model/audit/trainer path. **Hint boundary:** '+hint))
  signature=defs[name].splitlines()[0];cells.append(nb.v4.new_code_cell(defs[name] if solution else signature+'\n    raise NotImplementedError("TODO: '+name+'")'))
  cells.append(nb.v4.new_code_cell(checkdefs[check]+'\n'+check+'('+name+')\nprint("PASS: '+name+'")'))
 cells.append(nb.v4.new_code_cell('# PROVIDED: independently keyed metric\n'+defs['keyed_mae']))
 cells.append(nb.v4.new_markdown_cell('## PROVIDED · complete released model\n\nRead the typed encoders, stochastic structure computation, attention and centroid update. Only tensor/graph/column primitives are imported. These MIT-licensed source-adapted classes remain visible; the five-element mixer calls your live function.'))
 for block in model.split('# ---- '):
  if block.strip():cells.append(nb.v4.new_code_cell(('# ---- '+block) if not block.startswith('# Source') else block))
 cells.append(nb.v4.new_markdown_cell('## CHECK · source parity and original failure probes\n\nThe fixture uses width16, K3 and eight centroids to test the full computation cheaply. It is a synthetic mechanism test, not a downscaled benchmark result. Stochastic draws are aligned for the original-model comparison.'))
 cells.append(nb.v4.new_code_cell(mechanism+'\nmechanism=mechanism_check(SOURCE_ROOT,globals())\nprint(mechanism)'))
 audit_code='''# CHECK: audit ALL saved author token timestamps with YOUR function.
counts={}
for split in ['train','val','test']:
    z=np.load(io.BytesIO(base64.b64decode(AUTHOR_FILES[split+'-audit.npz'])))
    keys=list(zip(z['entity'].tolist(),z['cutoff'].tolist()))
    stamps=[[None if v==-1 else int(v) for v in row] for row in z['token_times']]
    violations=audit_tokens(keys,stamps)
    independent=int(((z['token_times']!=-1)&(z['token_times']>z['cutoff'][:,None])).sum())
    assert len(violations)==independent==AUTHOR_AUDIT['split_audits'][split]['future_token_occurrences']
    counts[split]=len(violations)
print('Future-token occurrences:',counts)
scored=0
for r in AUTHOR_PILOTS:
    z=np.load(io.BytesIO(base64.b64decode(AUTHOR_FILES['pilot-'+str(r['layers'])+'/predictions.npz'])))
    keys=list(zip(z['val_entity'].tolist(),z['val_cutoff'].tolist()));order=np.arange(len(keys))[::-1]
    score=keyed_mae(keys,z['val_target'],[keys[i] for i in order],z['val_pred'][order])
    assert abs(score-r['scores']['val'])<1e-9
    scored+=len(keys)
print('Independently rescored pilot predictions:',scored)
'''
 cells.append(nb.v4.new_code_cell(audit_code))
 cells.append(nb.v4.new_markdown_cell('## EXIT · write before revealing\n\nExplain why source parity and positive ages cannot repair the observed temporal failure. Trace the B×K×d tensors and identify checkpoint state. Describe a corrected experiment without claiming it reproduces historical results. Submit this defense for feedback; checks alone do not establish mastery.'))
 cells.append(nb.v4.new_markdown_cell('## PROVIDED · full selected-task trainer\n\nNine 100-epoch schedules, complete data, validation selection and separate test scoring. The default gate is off. It expects the hash-verified archived graph and full task materialization described in the protocol. A temporal failure blocks clean reproduction before training; repaired tokens require a separately declared experiment.'))
 cells.append(nb.v4.new_code_cell(trainer))
 cells.append(nb.v4.new_code_cell("if RUN_FULL_REPRODUCTION:\n    raise RuntimeError('Use the budgeted modal/l145_repro.py::full entrypoint. Current temporal audit FAIL blocks dispatch.')\nelse:\n    print('Full selected reproduction INCOMPLETE; nine full fits NOT_RUN.')"))
 cells.append(nb.v4.new_code_cell("report=dict(status='PASS',mechanism=mechanism,future_token_occurrences=counts,independently_rescored_pilot_predictions=scored,full_selected_reproduction='INCOMPLETE',learner='PENDING_WRITTEN_DEFENSE')\nPath('l145-report.json').write_text(json.dumps(report,indent=2))\nprint(report)"))
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
 for i,c in enumerate(notebook.cells):c.id=hashlib.sha256((str(i)+c.source).encode()).hexdigest()[:12]
 path=P/('solutions' if solution else '')/(SLUG+'.ipynb');path.parent.mkdir(exist_ok=True)
 if solution and path.exists():
  old=nb.read(path,4);notebook.metadata=old.metadata
  previous=[c for c in old.cells if c.cell_type=='code'];current=[c for c in cells if c.cell_type=='code']
  if [c.source for c in previous]==[c.source for c in current]:
   for before,after in zip(previous,current):
    after.outputs=before.outputs;after.execution_count=before.execution_count;after.metadata=before.metadata
 nb.write(notebook,path)
print('Built lesson, reference and portable notebooks')
