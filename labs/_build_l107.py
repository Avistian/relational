"""Deterministic HTML + standalone notebook builder, retaining identical executed cells."""
import ast,base64,hashlib,json,re
import black
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;SLUG='0107-snapshot-methods';TITLE='Snapshot methods: decide what remembers'
CAP={
'architecture':'Two complete prediction paths: GCN-GRU carries node-indexed state; EvolveGCN carries feature-to-feature weights. Each supplies node vectors to a pair head.',
'normalization':'Synthetic path A–B–C. Degree normalization with self-loops sends node B to 1.2247; row averaging would give 1.0000.',
'summary':'Synthetic H top-k calculation: score four node rows, select B and D, weight by tanh, then transpose to a 2-by-2 summary matching W.',
'recurrence':'Synthetic matrix interpolation with replacement gate 0.25. The top-left weight changes from 0.20 to 0.35 before it is used in graph convolution.',
'history':'Measured deterministic mechanism witness: changing the first snapshot changes H, but not released O. Fixed mean-slope activations isolate content dependence.',
'results':'Author evidence: all three Wikipedia seeds and sample SD; full selected SBM released runs versus paper targets. Different tasks and candidate universes are not directly comparable.'}

def results():
 path=P/'_analysis_l107_results.json'
 if not path.exists():return '**Experiment execution is in progress. No final paper-result claim is available yet.**'
 r=json.loads(path.read_text());text='**Author-run reference evidence**, not your current notebook output.\n\n| Wikipedia arm | Pooled AP, mean ± sample SD | Pooled AUROC, mean ± sample SD | Complete fits |\n|---|---:|---:|---:|\n'
 for arm,label in [('3600','Hourly GCN-GRU'),('86400','Daily GCN-GRU'),('tgn','Compact TGN')]:
  v=r['wiki'][arm];text+=f"| {label} | {v['mean']['ap']:.4f} ± {v['sd']['ap']:.4f} | {v['mean']['auc']:.4f} ± {v['sd']['auc']:.4f} | 3 × 10 epochs |\n"
 text+='\nEach fit covers all 81,029 training, 23,621 validation and 23,621 test positive events. Predictions are matched by raw event ID and class before comparing methods. SD measures variation across these seeds, not uncertainty over datasets or future deployments.\n\n| SBM variant | Epochs / selected epoch (0-based) | MAP / paper | MRR / paper | Numeric verdict |\n|---|---:|---:|---:|---|\n'
 for v in ['H','O']:
  z=r.get('sbm',{}).get(v)
  if z is None:text+=f'| Released {v} | RUNNING | — | — | PENDING |\n';continue
  text+=f"| Released {v} | {z['epochs']} / {z['selected_epoch']} | {z['test']['map']:.4f} / {z['target']['map']:.4f} | {z['test']['mrr']:.4f} / {z['target']['mrr']:.4f} | MAP {z['verdict']['map']}; MRR {z['verdict']['mrr']} |\n"
 for variant,z in r.get('sbm',{}).items():
  for metric,verdict in z['verdict'].items():
   if verdict=='FAIL':
    gap=abs(z['test'][metric]-z['target'][metric]);tolerance=.015 if metric=='map' else .003
    text+=f"\n**{variant} {metric.upper()} misses the predeclared CLOSE threshold:** absolute gap {gap:.6f} exceeds {tolerance:.3f}. The full declared run completed, but this published number was not numerically reproduced within that tolerance. Source agreement does not erase the gap; no test-driven retuning was performed.\n"
 text+='\n**Interpretation:** Wikipedia ranks these configured systems on one frozen candidate population. It does not isolate the effect of discretization: encoder, update frequency and gradient horizon also change. The SBM cells test a named released configuration, with source/prose and runtime deviations retained. Neither lane establishes full-paper or historical identity.\n'
 return text

def prose(portable=False):
 s=(R/'lessons/content'/f'{SLUG}.md').read_text().replace('[[RESULTS]]',results())
 for key,cap in CAP.items():
  file=P/f'figures/l107/{key}.png'
  if not file.exists():s=s.replace('[[FIG:'+key+']]','');continue
  src='data:image/png;base64,'+base64.b64encode(file.read_bytes()).decode() if portable else f'../labs/figures/l107/{key}.svg'
  s=s.replace('[[FIG:'+key+']]',f'<figure class="stream-figure" tabindex="0"><img src="{src}" alt="{cap}"><figcaption>{cap}</figcaption></figure>')
 for key,id_,fallback in [('NODE_WIDGET','l107-node','**Portable node-gate trace:** initial 0.20; candidates [0.80, 0.10, 0.60]; retention gate 0.25 gives states [0.6500, 0.2375, 0.5094]. Predict first, then calculate.'),('WEIGHT_WIDGET','l107-weight','**Portable weight-gate trace:** same initial state and candidates; replacement gate 0.25 gives [0.3500, 0.2875, 0.3656]. Explain the changed convention.'),('PREDICT','l107-predict','**Predict before the witness:** under fixed mean-slope activations, can H, released O, or both respond to an early-snapshot intervention?'),('TEACHBACK','l107-teachback','**Write the EXIT defense and send it to the teacher for feedback.**')]:s=s.replace('[['+key+']]',fallback if portable else f'<div id="{id_}"></div>')
 core=(P/'relkit/snapshot_l107.py').read_text();node=next(n for n in ast.parse(core).body if isinstance(n,ast.FunctionDef) and n.name=='normalized_adjacency');snippet='\n'.join(core.splitlines()[node.lineno-1:node.end_lineno]);s=s.replace('[[CODE:normalize]]','Implement the normalization in TODO 1 below; the later graph computations call your function.' if portable else '```python\n'+snippet+'\n```')
 if portable:
  s=s.replace('<div id="warmup"></div>','')
  s=s.replace('](../','](https://avistian.github.io/relational/')
  s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def document(title,body,ref=False):
 scripts=[] if ref else ['retrieval-pool','retrieval-bank','predict','teachback','snapshot-state-viz','l107-lesson']
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/event-snapshot.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0106-temporal-link-prediction.html">Lesson 106</a></nav><header><p class="stream-kicker">Year 3 · Quarter 3 · Lesson 107</p><h1>'+title+'</h1></header>'+render(body).replace('<table>','<div class="stream-scroll" tabindex="0"><table>').replace('</table>','</table></div>')+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{SLUG}.html').write_text(document(TITLE,prose()))
ref='''## Forecast contract

A window `[kΔ,(k+1)Δ)` becomes available at `(k+1)Δ` under immediate arrival. A query q may use only windows with close ≤ q. A partially observed last window is not complete just because the file ended. Snapshot interaction graphs do not imply persistent relationships.

## State and update

| Model | Recurrent object | Shape | Identity constraint |
|---|---|---|---|
| GCN-GRU | Node state S | N × d | Align the same node across slices |
| EvolveGCN-H | GCN weights W | F × d | Summarize current node features before the matrix update |
| Released O | GCN weights W | F × d | Observation-free recurrence, reset per fixed-length query |

GCN: Z = σ(D^(-1/2)(A+I)D^(-1/2)XW), with D from row sums after self-loops. Pair head: concatenate endpoint representations → MLP → class logits.

PyTorch GRU's z retains old state. EvolveGCN matrix z selects the new candidate. Check the equation, not the letter.

## Protocol traps

Six inputs arise from t−5…t, despite `num_hist_steps=5`. Source O is GRU-style, while the paper describes an LSTM. Released degree-schema fitting scans later snapshots. RReLU remains stochastic during validation. All-pair evaluation includes missing diagonal pairs. Source “MRR” averages over all relevant ranks, not just the first.

## Minimum evidence

Pin raw data and source; audit available history, candidates, selection and metric aggregation; inspect state reset and checkpoint state; save every selected prediction; recompute scores independently. CLOSE is a predeclared numeric diagnostic, not historical identity.

Read [the lesson](../lessons/0107-snapshot-methods.html), [full protocol](../labs/l107-reproduction.md) and [primary paper](https://arxiv.org/html/1902.10191v3).'''
(R/'reference/snapshot-state-contracts.html').write_text(document('Snapshot state contracts',ref,True))

CHECKS={
'normalized_adjacency':'''p=np.array([[0,1],[1,0],[1,2],[2,1],[0,1]])
a=normalized_adjacency(p,np.ones(5),4).to_dense().numpy()
raw=np.eye(4)
for i,j in p: raw[i,j]+=1
d=raw.sum(1)
np.testing.assert_allclose(a,raw/np.sqrt(d[:,None]*d[None,:]),rtol=1e-6,err_msg="Keep repeats, add self loops, normalize both endpoints")
assert a[3,3]==1, "An isolated node retains its self loop"
print("CHECK 1 PASS: weighted edges and isolated nodes")''',
'matrix_update':'''previous=torch.tensor([[.2,.4],[.6,.8]],requires_grad=True)
q=torch.zeros_like(previous)
gates=[lambda x,h:torch.full_like(h,.25),lambda x,h:torch.ones_like(h),lambda x,h:torch.full_like(h,.8)]
y=matrix_update(previous,q,gates)
torch.testing.assert_close(y,torch.tensor([[.35,.50],[.65,.80]]))
y.sum().backward();torch.testing.assert_close(previous.grad,torch.full_like(previous,.75))
q=torch.ones(2,2)*.1
p=torch.ones(2,2)*.4
gates=[lambda x,h:torch.full_like(h,.7),lambda x,h:torch.full_like(h,.5),lambda x,h:torch.tanh(x+h)]
torch.testing.assert_close(matrix_update(p,q,gates),.3*p+.7*torch.tanh(q+.5*p))
print("CHECK 2 PASS: values, reset conditioning and old-state gradient")''',
'completed_bins':'''times=np.array([0.,9.9,10.,10.,20.1])
np.testing.assert_array_equal(completed_bins(times,10,10),[0,0])
np.testing.assert_array_equal(completed_bins(times,10,20),[0,0,1,1])
assert len(completed_bins(times,10,9.99))==0
assert len(completed_bins([],10,99))==0
print("CHECK 3 PASS: exact close, ties, empty input")'''}
HINTS={'normalized_adjacency':'Preserve edge multiplicity, add one loop per node, and normalize with the two endpoint degrees. Return a coalesced sparse tensor.','matrix_update':'Compute update/reset gates, a reset-conditioned candidate and the paper-convention interpolation. Do not detach tensors.','completed_bins':'Keep the bin index for each eligible event, including duplicates. Eligibility depends on the scheduled bin end, not the event time alone.'}

def inline_chunks(path):
 source=path.read_text();tree=ast.parse(source);remove=set()
 for n in tree.body:
  if isinstance(n,ast.Try) and any('from '+m in ast.get_source_segment(source,n) for m in ['snapshot_l107','relkit.snapshot_l107']):remove.update(range(n.lineno,n.end_lineno+1))
 source='\n'.join(line for i,line in enumerate(source.splitlines(),1) if i not in remove)
 return re.split(r'^# %% ',source,flags=re.M)

def build(solution):
 cells=[]
 def md(x):cells.append(nbf.v4.new_markdown_cell(x))
 def code(x,tag='provided'):
  formatted=black.format_str(x,mode=black.Mode(line_length=88)).rstrip()
  assert ast.dump(ast.parse(x),include_attributes=False)==ast.dump(ast.parse(formatted),include_attributes=False)
  cells.append(nbf.v4.new_code_cell(formatted,metadata={'tags':[tag]}))
 md('# '+TITLE+'\n\n**Mirror scope:** visible GCN-GRU, released EvolveGCN H/O, complete trainers and authenticated data. Short execution is a full-data daily course fit plus a full-data SBM pilot; named full runs are separately gated. Neither pilot nor source parity is paper-result reproduction. Learner: PENDING_WRITTEN_DEFENSE. Raw Wikipedia download is about 560 MB; the hourly full-data path uses several GB of RAM.')
 code('''# @colab-bootstrap — install the pinned runtime only in Colab or if dependencies are missing.
import importlib.util,subprocess,sys
try:
    IN_COLAB=importlib.util.find_spec("google.colab") is not None
except ModuleNotFoundError:
    IN_COLAB=False
if IN_COLAB or any(importlib.util.find_spec(x) is None for x in ["torch","numpy","pandas","sklearn","yaml"]):
    subprocess.check_call([sys.executable,"-m","pip","install","-q","torch==2.8.0","numpy==2.2.6","pandas==2.3.2","scikit-learn==1.7.1","PyYAML==6.0.2"])
import torch,numpy as np
from pathlib import Path
torch.set_num_threads(1)
print("Actual runtime:",sys.version.split()[0],torch.__version__,np.__version__)
print("Author comparisons used Python 3.12, Torch 2.8.0 and NumPy 2.2.6 on T4.")''')
 # Chunk the standalone explanation at headings so figures sit next to their derivations.
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if section.strip():md(section)
 md('## Implement the three load-bearing operations\n\nWork in order. Definitions below remain live throughout the model and data computations. The solution notebook fills the same functions; there is no hidden replacement cell.')
 for part in inline_chunks(P/'relkit/snapshot_l107.py'):
  if not part.strip():continue
  if '\n' in part:title,body=part.split('\n',1)
  else:title,body='Imports',part
  if title.startswith('"""'):title='Module provenance';body=part
  tree=ast.parse(body);todo=next((n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in CHECKS),None)
  if todo:
   md('### '+title+'\n\n'+HINTS[todo.name])
   if not solution:
    lines=body.splitlines();signature=lines[todo.lineno-1];body=signature+'\n    raise NotImplementedError("TODO: '+todo.name+'")'
   code(body,'solution' if solution else 'todo');code(CHECKS[todo.name],'check')
  else:md('### PROVIDED · '+title+'\n\nTrace its inputs and the state it returns; the next section will call this code.');code(body)
 md('## PROVIDED · TGN baseline and raw Wikipedia loader\n\nThis is the visible L102 model port. Its original paper-target trainer is retained for inspection, but the matched course experiment uses the separate trainer below: smaller memory, no dropout, fixed candidates and validation-best checkpointing. No archived L102 score is substituted for a new fit.')
 for part in inline_chunks(P/'relkit/tgn_l102.py'):
  if not part.strip():continue
  if part.startswith('"""'):code(part);continue
  title,body=part.split('\n',1);md('### PROVIDED · '+title);code(body)
 for filename,heading in [('sbm_l107.py','SBM: every step of the released replay'),('wiki_snapshot_l107.py','Wikipedia: the matched course trainer')]:
  md('## PROVIDED · '+heading+'\n\nInspect the splitter, available history, candidate sampler, loss reduction, validation selection and saved outputs. These are part of the experiment, not neutral boilerplate.')
  for part in inline_chunks(P/'relkit'/filename):
   if not part.strip():continue
   if part.startswith('"""'):code(part);continue
   title,body=part.split('\n',1);md('### PROVIDED · '+title);code(body)
 md('## CHECK · a history intervention\n\nPredict which variant responds when the first snapshot changes but the final graph/features are held fixed. Replace random activations by their mean slope only for this structural check. The source-score replay preserves stochastic activations.')
 code('''from unittest.mock import patch
witness={}
for variant in ["H","O"]:
    torch.manual_seed(22);model=EvolveGCN(4,3,variant)
    graphs=[torch.eye(7)]*6;features=[torch.randn(7,4) for _ in range(6)];masks=[torch.zeros(7,1)]*6
    altered=[x.clone() for x in features];altered[0]=altered[0]*7+5
    with patch("torch.nn.functional.rrelu",side_effect=lambda x,**kw:torch.nn.functional.leaky_relu(x,11/48)):
        before=model(graphs,features,masks);after=model(graphs,altered,masks)
    witness[variant]=float((before-after).abs().max().detach())
assert witness["H"]>1e-7 and witness["O"]==0,witness
print(witness)''','check')
 code((P/'relkit/auth_l107.py').read_text())
 md('## Fresh full-data daily fit\n\nThe next cells authenticate the full raw Wikipedia release and run **one daily GCN-GRU fit, seed 7, all ten epochs**. This is a learning run using the same visible implementation, not the three-seed GPU comparison above. Predict its test AP before executing; device/runtime differences prevent bitwise equivalence. Candidate audit below actively calls TODO 3 at every daily closing boundary.')
 code('''authenticate_wiki_cache("l107-data/wiki")
nodes,edge_features,events,data_audit=wiki_data("l107-data/wiki",seed=7)
authenticate_wiki_cache("l107-data/wiki")
for query in np.arange(0,events["t"].max()+86400,86400):
    bins=completed_bins(events["t"],86400,query)
    # Independent endpoint condition under integer elapsed-day boundaries.
    expected=np.flatnonzero(events["t"]<query)
    assert len(bins)==len(expected), "A completed daily window leaked or hid a past event"
print(data_audit["counts"])
lab=train_snapshot(nodes,edge_features,events,86400,7,"l107-runs/daily-lab",device="cpu",epochs=10)
assert lab["status"]=="COMPLETE" and lab["epochs"]==10
print("Fresh daily learning run:",lab["test"])''')
 md('## Fresh SBM pilot on the complete released data\n\nAuthenticate all 50 snapshots, then execute **one training window, one validation snapshot and one test snapshot** for H. TODOs 1 and 2 are in the actual model path. This checks the full-size operators and candidate universe; it is explicitly a pilot, not a complete paper result. The named 100-epoch lane is below.')
 code('''sbm=load_sbm("l107-data/sbm")
assert sbm["rows"]==4870863 and sbm["n"]==1000 and sbm["last"]==49
pilot=train_sbm(sbm,"H","l107-runs/sbm-pilot",device="cpu",epochs=1,max_seconds=600,pilot=True)
assert pilot["status"]=="PILOT"
print("Full-size pilot only:",pilot["selected"]["test"])''')
 md('## EXIT TICKET\n\nWrite the defense from the lesson in your own words. Attach the trace, selected epoch, runtime, input hash and the gate/history checks. Explain why the daily fit and SBM pilot have different evidence labels from the full author runs. Send the ticket and prose to the teacher; successful execution does not grade your explanation.')
 code('''ticket={"learner_status":"PENDING_WRITTEN_DEFENSE","daily_lab":{k:lab[k] for k in ["status","seed","epochs","selected_epoch","test"]},"sbm_pilot":pilot["status"],"history_witness":witness,"data":{k:data_audit[k] for k in ["raw_sha256","cutoffs","counts","split_sha256"]},"runtime":{"python":sys.version,"torch":torch.__version__,"numpy":np.__version__}}
Path("l107-fresh.json").write_text(json.dumps(ticket,indent=2))
print(json.dumps(ticket,indent=2))''','exit')
 md('## NEXT STEP · complete named SBM replay\n\nSet the gate only when ready for the full 100-epoch/early-stopping experiment. Both variants use the code you have just read. This may take about an hour per variant on a T4; CPU time varies. Use fresh directories, inspect status, and retain partial evidence if a runtime ends. This does not launch paid Modal work. The independent Modal commands and total budget live in the reproduction contract.')
 code('''RUN_PAPER_REPRO=False
if RUN_PAPER_REPRO:
    device="cuda" if torch.cuda.is_available() else "cpu"
    for variant in ["H","O"]:
        paper=train_sbm(sbm,variant,"l107-runs/paper-"+variant,device=device,epochs=100,max_seconds=10000)
        print(variant,paper["status"],paper.get("selected",{}))
else:
    print("Named paper lane NOT_RUN in this notebook execution; author-run evidence is labeled separately.")''')
 md('## NEXT STEP · all nine matched course fits\n\nThe full comparison has three seeds and three arms, all with ten epochs. Run in one pinned environment. Do not mix your local fit with the published author seeds or tune on their test scores.')
 code('''RUN_MATCHED_COMPARISON=False
if RUN_MATCHED_COMPARISON:
    device="cuda" if torch.cuda.is_available() else "cpu"
    for seed in range(3):
        n,e,questions,audit=wiki_data("l107-data/wiki",seed)
        for width in [3600,86400]:
            train_snapshot(n,e,questions,width,seed,f"l107-runs/full/{width}/seed-{seed}",device=device,epochs=10)
        train_tgn_course(n,e,questions,seed,f"l107-runs/full/tgn/seed-{seed}",device=device,epochs=10)
else:
    print("Full nine-fit course lane NOT_RUN in this notebook execution.")''')
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(nb.cells):c.id=f'l107-{i:03d}'
 path=P/('solutions' if solution else '')/f'{SLUG}.ipynb';path.parent.mkdir(exist_ok=True)
 if solution and path.exists():
  old=nbf.read(path,as_version=4)
  nb.metadata=old.metadata
  for new,previous in zip(nb.cells,old.cells):
   if new.cell_type=='code' and previous.cell_type=='code' and new.source==previous.source:new.outputs=previous.outputs;new.execution_count=previous.execution_count;new.metadata=previous.metadata
 nbf.write(nb,path)
 return nb
student=build(False);solution=build(True)
html,_=HTMLExporter(template_name='lab').from_notebook_node(solution);(P/'html'/f'{SLUG}.html').write_text(html)
print('Built lesson, reference, student and solution:',len(solution.cells),'cells')
