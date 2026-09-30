"""Canonical lesson/reference and portable live-function notebooks for L146."""
import ast,base64,hashlib,io,json,re,zlib
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l146';S='0146-gnn-vs-graph-transformer';TITLE='GNN vs graph transformer: defend the comparison'
audit=json.loads((E/'prepared/audit.json').read_text());table=json.loads((E/'paper-table.json').read_text());summary=json.loads((E/'summary.json').read_text())
def functions(path):
 s=path.read_text();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
contracts=functions(P/'relkit/comparison_l146.py');checks=functions(P/'_check_l146.py')
results='| Full-data course arm | Validation MAE, mean ± seed SD | Test MAE, mean ± seed SD | Parameters |\n|---|---:|---:|---:|\n'
for arm,label in [('gnn','Typed-mean GNN'),('relgt','Reduced corrected RelGT')]:
 a=summary['arms'][arm];results+=f"| {label} | {a['val']['mean']:.6f} ± {a['val']['sample_sd']:.6f} | {a['test']['mean']:.6f} ± {a['test']['sample_sd']:.6f} | {a['parameters']:,} |\n"
d=summary['paired']['test'];results+=f"\n**Paired GNN − RelGT test difference: {d['mean']:+.6f} ± {d['sample_sd']:.6f} MAE** (sample seed SD). Positive favors RelGT. Per-seed differences: "+', '.join(f'{v:+.6f}' for v in d['differences'])+'.\n'
results+=f"\nAll **{summary['verified_predictions']:,} primary held-out predictions** independently aligned and rescored. Six fresh fits each covered all **7,453 training queries for ten epochs**. Both arms used the same **499 validation / 760 test** keys. The sampler rebuilt all **8,712 labels** and passed its event-time audit. Repeated legal token slots: **43,000 train / 1,228 validation / 10,623 test**. These are occurrences, not distinct rows.\n"
results+='\n**Full selected paper reproduction: INCOMPLETE.** Nine full RelGT fits and a fresh canonical RDL comparator: **NOT_RUN**. Historical identity/selection: **NOT_ESTABLISHED**. Whole paper: **NOT_RUN**. Course runs do not change these verdicts.\n'
captions={'paths':'Trace original row attributes under an explicit sampling, propagation and root-readout contract. Illustrative chain; not a benchmark result.','selection':'Table 6 arithmetic under two declared selectors. Matching a test minimum does not establish historical selection intent.','architecture':'The actual two reduced course variants, including input shapes, internal operations, readouts and training boundaries.','results':'Three fresh seed pairs on the full task. Sample seed SD is not a confidence interval. The two models have distinct mechanisms and runtime.'}
def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l146/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l146/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="cmp-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption} Scroll horizontally for detail on a narrow screen.</figcaption></figure>')
 for tag,id,fallback in [('PREDICT','cmp-predict','Predict before reading the author scores: does lower validation error guarantee lower test error?'),('WARMUP','warmup','Recall before reading: what makes a temporal query unique? What does validation select?'),('PATH_WIDGET','cmp-path','Baseline: circuit sampled, bridge legal, two GNN layers. Root-readout GNN cannot yet use circuit attributes; all-pairs attention can.'),('SELECTION_WIDGET','cmp-selection','Validation minimum: L4/.3 → test4.6316. Test minimum: L1/.5 → test3.917. Historical selection remains unresolved.'),('TEACHBACK','cmp-teachback','Write your defense before viewing the reference answer; ask the teaching agent for feedback.')]:
  s=s.replace('[['+tag+']]',fallback if portable else f'<div id="{id}"></div><noscript>{fallback}</noscript>')
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def document(title,body,interactive=False):
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','graph-comparison','graph-comparison-lesson'] if interactive else []
 html=render(body).replace('<table>','<div class="cmp-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/graph-comparison.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0145-relational-graph-transformer.html">Lesson 145</a></nav><header><p class="cmp-kicker">Year 4 · Quarter 3 · Lesson 146</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join(f'<script src="../assets/{s}.js"></script>' for s in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
ref='''## Five comparison contracts
**Identity:** exact task, unique(entity,cutoff) keys, same targets. **Information:** query-specific legal context, cutoff at each hop, preprocessing scope. **Selection:** separate epoch checkpoint and configuration selection; freeze both before test. **Resources:** disclose search count, parameters, total time, preprocessing and seeds. **Claim:** distinguish published table, source replay, corrected experiment and historical identity.

## Information trace
Root-readout message passing needs enough layers for the selected path. All-pairs attention shortens communication inside a sample; it cannot retrieve excluded rows. A future bridge must be removed before expansion. Padding legal rows changes their multiplicity; report it.

## Paired metric
For each keyed query: |GNN−y|−|RelGT−y|. Positive favors RelGT. Reject duplicate, missing and extra keys before subtracting. Average queries within each seed, then summarize paired seeds. Sample seed SD is not a confidence interval; correlated queries do not justify an independent-row test.

## Reported-table selection audit
Table6 F1 minimum displayed validation3.1046: L4/.3, test4.6316. Minimum displayed test3.917: L1/.5, validation3.3257; matches headline. Printed final metrics and stochastic reevaluation do not recover historical selection-time scores. Do not infer intent.

## Actual measured scope
'''+results+'''\n## Reproduction boundaries
Course: K32,width64,2layers,128centroids,3seeds,10epochs, corrected temporal contexts and evaluation. Source full lane: K300,width512,4096centroids,9configs×100epochs, retained but blocked. Event-time PASS does not certify historical feature availability. No learner mastery inferred.

[Lesson](../lessons/0146-gnn-vs-graph-transformer.html) · [Protocol](../labs/l146-reproduction.md) · [Paper](https://arxiv.org/html/2505.10960v1#A4). Ask the teaching agent to review your comparison claim.
'''
(R/'reference/gnn-vs-graph-transformer.html').write_text(document('GNN–transformer comparison reference',ref))
# Only evidence and original source archives are compressed; executable models stay visible.
pack={}
for p in sorted(E.glob('fit-*/predictions.npz')):pack[str(p.relative_to(E))]=base64.b64encode(p.read_bytes()).decode()
for s in ['train','val','test']:
 z=dict(__import__('numpy').load(E/f'prepared/{s}.npz'));buf=io.BytesIO();__import__('numpy').savez_compressed(buf,**z);pack[f'prepared/{s}.npz']=base64.b64encode(buf.getvalue()).decode()
sources={p.name:p.read_text() for p in (P/'sources/l145').iterdir() if p.is_file()}
payload="AUTHOR_FILES=json.loads(zlib.decompress(base64.b64decode("+repr(base64.b64encode(zlib.compress(json.dumps(pack).encode())).decode())+")))\nSOURCE_FILES=json.loads(zlib.decompress(base64.b64decode("+repr(base64.b64encode(zlib.compress(json.dumps(sources).encode())).decode())+")))\nSOURCE_ROOT=Path('sources/l145');SOURCE_ROOT.mkdir(parents=True,exist_ok=True)\nfor name,source in SOURCE_FILES.items():(SOURCE_ROOT/name).write_text(source)\nAUTHOR_SUMMARY="+repr(summary)+"\nAUTHOR_TABLE="+repr(table)+"\nAUTHOR_AUDIT="+repr(audit)
bootstrap='''# @colab-bootstrap: pinned packages on Colab; local kernel uses its environment.
import os,sys,subprocess
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable,'-m','pip','install','-q','torch==2.5.1','--index-url','https://download.pytorch.org/whl/cu124'])
    subprocess.check_call([sys.executable,'-m','pip','install','-q','numpy==1.26.4','pandas==2.2.3','pyarrow==18.1.0','scipy==1.14.1','scikit-learn==1.5.2','pytorch-frame==0.2.3','torch-geometric==2.6.1','relbench==1.1.0','h5py==3.12.1','einops==0.8.0'])
    if 'torch' in sys.modules and sys.modules['torch'].__version__.split('+')[0]!='2.5.1':
        raise RuntimeError('Packages installed; restart runtime and run from the top.')
from pathlib import Path
import ast,base64,copy,hashlib,io,json,math,random,tempfile,types,zlib,contextlib
import numpy as np
import torch
from IPython.display import display
RUN_FULL_COURSE=False
RUN_FULL_REPRODUCTION=False
RAW_GRAPH_ROOT=None  # L143 hash-verified materialization: graph.pt, prepared.json, cache/, unpacked/
COURSE_CONTEXT_ROOT=None
'''
model=(P/'relkit/relgt_course_l146.py').read_text().replace('from relkit.relgt_contracts_l145 import mix_five\n','')
mix=functions(P/'relkit/relgt_contracts_l145.py')['mix_five']
gnn=re.sub(r'^from relkit[^\n]*\n','',(P/'relkit/gnn_l146.py').read_text(),flags=re.M)
helpers=functions(P/'_full_l145.py');trainer=re.sub(r'^from (relkit|_full_l145)[^\n]*\n','',(P/'_full_l146.py').read_text(),flags=re.M)
original=functions(P/'_mechanism_l145.py')['original_modules'];original='@contextlib.contextmanager\n'+original
fixture=functions(P/'_mechanism_l146.py')['full_fixture']
start=fixture.index('    if namespace is None:');end=fixture.index('    torch.set_num_threads',start)
fixture=fixture[:start]+"    RelGT=namespace['RelGT'];EncoderLayer=namespace['EncoderLayer'];CourseGNN=namespace['CourseGNN']\n"+fixture[end:]
meancheck=functions(P/'_mechanism_l146.py')['check_mean']
scoring='''# CHECK: reference scores are AUTHOR evidence, not fresh learner training.
selected=select_config(AUTHOR_TABLE['configs'],AUTHOR_TABLE['validation'])
assert selected=='L4-d0.3'
rows=[];verified=0
for seed in [0,1,2]:
    arrays={arm:np.load(io.BytesIO(base64.b64decode(AUTHOR_FILES[f'fit-{arm}-{seed}/predictions.npz']))) for arm in ['gnn','relgt']}
    for split in ['val','test']:
        ref=np.load(io.BytesIO(base64.b64decode(AUTHOR_FILES[f'prepared/{split}.npz'])))
        keys=list(zip(ref['entity'],ref['cutoff']));n=len(keys)
        a,b=arrays['gnn'],arrays['relgt'];order=np.random.default_rng(seed).permutation(n)
        ak=list(zip(a[split+'_entity'],a[split+'_cutoff']));bk=list(zip(b[split+'_entity'],b[split+'_cutoff']))
        diff=paired_errors(keys,ref['target'],ak,a[split+'_pred'],[bk[j] for j in order],b[split+'_pred'][order])
        expected=AUTHOR_SUMMARY['paired'][split]['differences'][seed]
        assert abs(float(diff.mean())-expected)<1e-10
        rows.append(dict(seed=seed,split=split,gnn_minus_relgt=float(diff.mean())));verified+=2*n
for run in AUTHOR_SUMMARY['runs']:
    history=run['history'];assert select_config([h['epoch'] for h in history],[h['val_mae'] for h in history])==run['selected_epoch']
for split in ['train','val','test']:
    z=np.load(io.BytesIO(base64.b64decode(AUTHOR_FILES[f'prepared/{split}.npz'])))
    assert not ((z['token_times']!=-1)&(z['token_times']>z['cutoff'][:,None])).any()
import pandas as pd
display(pd.DataFrame(rows))
assert verified==7554
print('Independent paired author predictions:',verified)
'''
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson 146 · '+TITLE+'\n\n'+('Reference solution' if solution else 'Student lab')+' · PROVIDED / TODO / CHECK / EXIT. Skill: defend a comparison through information, selection and pairing contracts. Default execution performs full-network fixtures and rescoring of saved author predictions. Six fresh full-data runs were executed separately. A passing notebook does not establish learner mastery.'),nb.v4.new_code_cell(bootstrap)]
 # Main conceptual progression stays before the implementation appendix.
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if section.strip():cells.append(nb.v4.new_markdown_cell(section))
 c=nb.v4.new_code_cell('# PROVIDED: immutable author evidence plus pinned MIT original sources.\n'+payload);c.metadata['tags']=['data-payload'];cells.append(c)
 for name,test,title,hint in [('temporal_context','check_context','Keep the query owner through expansion','Filter neighbors before entering the frontier. Pad only with already selected legal rows.'),('select_config','check_selection','Make selection independent of test scores','Require one finite validation value for every configuration. Freeze the first-tie rule.'),('paired_errors','check_pairs','Align before subtracting errors','Check each complete key set. Return GNN absolute loss minus RelGT absolute loss in reference order.')]:
  cells.append(nb.v4.new_markdown_cell('## TODO · '+title+'\n\n**Goal:** implement `'+name+'`. **Why:** the full preparation/trainer/scorer calls this contract. **Hint boundary:** '+hint))
  cells.append(nb.v4.new_code_cell(contracts[name] if solution else contracts[name].splitlines()[0]+'\n    raise NotImplementedError("TODO: '+name+'")'))
  cells.append(nb.v4.new_code_cell(checks[test]+'\n'+test+'('+name+')\nprint("PASS: '+name+'")'))
 cells.append(nb.v4.new_code_cell('# PROVIDED: concatenate the five encodings without merging coordinates.\n'+mix))
 cells.append(nb.v4.new_markdown_cell('## PROVIDED · reduced RelGT, fully visible\n\nThe following source-derived classes implement row encoders, GIN structure, attention, EMA updates and the scalar head. The original MIT implementation is embedded for independent training-output/gradient checks. Only two evaluation expressions differ: dropout becomes zero, and structural draws use a fixed generator. Model sizes are selected by the course trainer below.'))
 for block in model.split('# ---- '):
  if block.strip():cells.append(nb.v4.new_code_cell(('# ---- '+block) if not block.startswith('# Course') else block))
 cells.append(nb.v4.new_markdown_cell('## PROVIDED · explicit GNN control\n\nTrace a directed relation mean before the linear map. A destination self term is added once per type, then relation contributions are summed. This is a custom control with the same typed row encoder; it is not the paper\'s canonical RDL implementation.'))
 cells.append(nb.v4.new_code_cell(gnn))
 cells.append(nb.v4.new_code_cell(meancheck+'\ncheck_mean(relation_mean)\nprint("PASS: relation mean and exact derivatives")'))
 cells.append(nb.v4.new_markdown_cell('## CHECK · both complete networks\n\nA six-row numerical fixture exercises forward/backward paths. The corrected RelGT training path must match original source outputs and all gradient tensors under the same random state. The CPU fixture repeats exactly and leaves centroid/normalization buffers unchanged; full CUDA fits use 0.00001 absolute tolerance for aggregation ordering. These checks are mechanism evidence, not task accuracy.'))
 cells.append(nb.v4.new_code_cell(original+'\n'+fixture+'\nmechanism=full_fixture(SOURCE_ROOT,globals())\nprint(mechanism)'))
 cells.append(nb.v4.new_markdown_cell('## CHECK · full-population author evidence\n\nUse your live selector on the printed paper table and on each fresh fit\'s validation history. Use your paired-error function on every saved held-out prediction. One arm is shuffled deliberately. The saved contexts are audited against their own cutoff timestamps. This rechecks author evidence, not a new fit.'))
 cells.append(nb.v4.new_code_cell(scoring))
 cells.append(nb.v4.new_markdown_cell('## EXIT · written defense\n\nExplain why the corrected small experiment and the published-protocol reproduction remain separate. Name an unisolated architectural difference, an unresolved historical selection fact, and a meaningful next experiment. Submit your defense to the teaching agent.'))
 cells.append(nb.v4.new_markdown_cell('## PROVIDED · complete full-data course preparation and trainer\n\nThis includes raw-label reconstruction, sampled-context construction, grouped TensorFrame batching, both model constructors, every epoch, checkpoint restoration, keyed output artifacts and repeat-evaluation checks. It calls your three functions where applicable. The default gate is off; enabling it requires the independently prepared L143 graph described in the protocol and enough compute for all six fits.'))
 cells.append(nb.v4.new_code_cell('from relbench.datasets import get_dataset\nfrom relbench.tasks import get_task\n'+helpers['sha']+'\n'+helpers['task_data']))
 cells.append(nb.v4.new_code_cell(trainer))
 cells.append(nb.v4.new_code_cell("if RUN_FULL_COURSE:\n    if RAW_GRAPH_ROOT is None or COURSE_CONTEXT_ROOT is None:\n        raise ValueError('Supply the hash-verified raw graph and a new course context directory per protocol.')\n    prepare(COURSE_CONTEXT_ROOT,RAW_GRAPH_ROOT)\n    for seed in CONFIG['seeds']:\n        for arm in ['gnn','relgt']:\n            fit(COURSE_CONTEXT_ROOT,RAW_GRAPH_ROOT,Path('fresh-l146')/f'fit-{arm}-{seed}',arm,seed)\nelse:\n    print('Course training not rerun by default; saved author evidence was independently scored.')"))
 cells.append(nb.v4.new_markdown_cell('## Paper-results lane · complete released schedules remain separate\n\nThe original full model differs from the visible course model only in the two disclosed evaluation expressions; architecture dimensions are constructor settings. The full source trainer is reproduced below as inspectable source text, preserving nine 100-epoch fits. The canonical RDL full implementation and five-seed protocol remain in [L117](https://avistian.github.io/relational/labs/l117-reproduction.md). The executable [L146 preflight](https://avistian.github.io/relational/labs/_reproduce_l146.py) blocks a clean source run because temporal and cost contracts fail. This notebook does not convert a corrected run into historical evidence.'))
 # A visible literal preserves the complete released trainer without shadowing the live course functions.
 cells.append(nb.v4.new_code_cell('RELEASED_TRAINER_SOURCE = '+repr((P/'_full_l145.py').read_text())+'\nprint("Released trainer retained:",len(RELEASED_TRAINER_SOURCE.splitlines()),"lines; nine complete schedules.")'))
 # Format the literal as real readable lines, never a one-line escaped code dump.
 cells[-1].source='RELEASED_TRAINER_SOURCE = r\'\'\'\n'+(P/'_full_l145.py').read_text()+"\n'''\nprint('Released trainer retained; nine complete schedules. See separate preflight.')"
 cells.append(nb.v4.new_code_cell("if RUN_FULL_REPRODUCTION:\n    raise RuntimeError('STOP: source temporal FAIL, complete search over budget, historical cross-config selection NOT_ESTABLISHED. Use the separate audited runner only after resolving these contracts.')\nelse:\n    print('Full selected paper reproduction INCOMPLETE; original full fits NOT_RUN.')"))
 cells.append(nb.v4.new_code_cell("report=dict(status='PASS',mechanism=mechanism,verified_author_predictions=verified,course_reference='COMPLETE',full_selected_reproduction='INCOMPLETE',learner='PENDING_WRITTEN_DEFENSE')\nPath('l146-report.json').write_text(json.dumps(report,indent=2))\nprint(report)"))
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
 for i,c in enumerate(cells):c.id=hashlib.sha256((str(i)+c.source).encode()).hexdigest()[:12]
 path=P/('solutions' if solution else '')/(S+'.ipynb');path.parent.mkdir(exist_ok=True)
 if solution and path.exists():
  old=nb.read(path,4)
  if [c.source for c in old.cells]==[c.source for c in cells]:notebook=old
 nb.write(notebook,path)
print('Built lesson, reference, student and solution')
