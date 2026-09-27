"""Deterministic lesson/reference and portable notebooks; live student tasks drive the neural fixture."""
import ast,base64,hashlib,json,re,textwrap,zlib
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0132-identity-aware-message-passing';TITLE='Identity-aware message passing: who is asking?'
def functions(path):
 s=path.read_text();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
f=functions(P/'relkit/identity_l132.py');checks=functions(P/'_check_l132.py');fixture=functions(P/'_fixture_l132.py')['neural_fixture'];runner=functions(P/'_run_l132.py')
r=json.loads((P/'evidence/l132/fixture.json').read_text())
captions={'architecture':'Two full predictors: shared two-tower vectors versus query-conditioned sponsor readout. B is query count, D is all sponsors, Ncand is sampled sponsor occurrences. Training objectives and ID embeddings differ.', 'collision':'Constant-feature sums collide. Three marked propagation steps return zero walks on the six-cycle and two on a triangle. This is a synthetic counting witness.', 'ownership':'Four sponsor occurrences belong to two queries. The global-ID shortcut creates two false positives; pair membership preserves ownership.', 'reach':'The four-hop condition-to-sponsor path crosses two association tables and a study. Hop count permits reachability; sampling and temporal eligibility still matter.', 'fixture':'Measured tiny neural ablation with identical starting weights, paired random draws and BCE. This is not the published two-tower-versus-ID-GNN experiment.'}
def status_text():
 p=P/'evidence/l132/summary.json'
 if not p.exists():return '**Selected experiment: preparation/pilot in progress.** No completed five-seed benchmark comparison is claimed.'
 s=json.loads(p.read_text());text=f"**Selected full experiment: {s['status']}.** {s['explanation']}\n\n"
 if s.get('pilots'):
  text+='| Pilot | Epochs | Validation MAP % | Test MAP % | Runtime seconds |\n|---|---:|---:|---:|---:|\n'
  for row in s['pilots']:text+=f"| {row['variant']} | {row['epochs']} | {100*row['scores']['val']:.4f} | {100*row['scores']['test']:.4f} | {row['seconds']:.1f} |\n"
  text+='\nPilot scores are feasibility evidence, not the five-run, 20-epoch paper result.\n'
 if s.get('metrics'):
  text+='\n| Variant | Split | Mean MAP % | Sample SD pp | Paper MAP % | Verdict |\n|---|---|---:|---:|---:|---|\n'
  for v,ss in s['metrics'].items():
   for split,m in ss.items():text+=f"| {v} | {split} | {m['mean']*100:.4f} | {m['sample_sd']*100:.4f} | {m['target']*100:.2f} | {m['verdict']} |\n"
 text+='\n[Machine-readable execution summary](../labs/evidence/l132/summary.json). Historical/whole-paper parity remains **NOT_ESTABLISHED**.'
 return text

def prose(portable=False):
 text=(R/'lessons/content'/f'{S}.md').read_text()
 # Expose the operation without handing the notebook TODO a complete implementation.
 text=text.replace('[[MARKER_CODE]]','```text\nencoded root occurrences [B,d] + shared marker [1,d]\nencoded context occurrences [N−B,d] remain unchanged\n```')
 text=text.replace('[[FIXTURE_RESULTS]]',f"Measured final training BCE: **{r['course_fits']['unmarked']['loss']:.6f} without the marker**, **{r['course_fits']['marked']['loss']:.6f} with the marker**. The parity fixture’s maximum output and gradient discrepancies are both **0**. These are author-reference outputs; your notebook executes the experiment again.")
 text=text.replace('[[REPRO_RESULTS]]',status_text())
 text=text.replace('[[FORWARD_CODE]]','```python\n'+f['identity_forward']+'\n```')
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l132/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l132/{name}.svg'
  text=text.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 widgets={'WARMUP':('warmup','Recall: when can 1-WL fail? Which index identifies a query owner? Where does the prediction head read?'),'WALK_WIDGET':('l132-walk','Static check: at depths 1 and 2, both marked roots return 0 and 2 respectively. At depth 3 they diverge: six-cycle 0, triangle 2.'),'OWNER_WIDGET':('l132-owner','Static check: correct labels [1,0,0,1]; global-ID labels [1,1,1,1].'),'TEACHBACK':('l132-teachback','Teach back: explain why the full model comparison changes more than root marking.')}
 for token,(id,fallback) in widgets.items():text=text.replace('[['+token+']]',fallback if portable else '<div class="stream-widget" id="'+id+'"></div><noscript>'+fallback+'</noscript>')
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/');text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def document(title,text,interactive=False):
 body=render(text).replace('<table>','<div class="stream-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','teachback','identity-message-viz','l132-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','event-snapshot','reproduction'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0131-gnn-tabular-stack.html">Lesson 131</a></nav><header><p class="stream-kicker">Year 4 · Quarter 2 · Lesson 132</p><h1>'+title+'</h1></header>'+body+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
previous=nbf.read(P/'solutions/0131-gnn-tabular-stack.ipynb',as_version=4)
bootstrap=next(c.source for c in previous.cells if c.cell_type=='code' and '@colab-bootstrap' in c.source)
imports='''import copy, json, sys, time, hashlib, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_frame import stype
from torch_frame.data import Dataset
from torch_geometric.data import HeteroData
'''
# Store upstream files visibly in named variables; build a standalone runtime only when requested.
source_files={str(p.relative_to(P/'sources')):p.read_text() for p in (P/'sources/l132').glob('*.py')}
source_files['l117/text_model.json']=(P/'sources/l117/text_model.json').read_text()
full_gate='''# Full named experiment. This can require hours and substantial RAM/GPU memory.
# The notebook gate has no dollar enforcement; use the Modal operator for the USD10 guard.
RUN_FULL_REPRODUCTION = False
if RUN_FULL_REPRODUCTION:
    import importlib.metadata as md, types
    assert sys.version_info[:2]==(3,11), 'Use the pinned Python 3.11 runtime'
    for pkg,expected in {'torch':'2.5.1','torch-geometric':'2.6.1','pytorch-frame':'0.2.3','relbench':'1.1.0','numpy':'1.26.4','pandas':'2.2.3','sentence-transformers':'3.3.1'}.items():
        assert md.version(pkg).split('+')[0]==expected,(pkg,md.version(pkg),expected)
    assert torch.cuda.is_available() and md.version('pyg-lib').startswith('0.4.0')
    # The official runs use the immutable released Model; the teaching ablation above uses YOUR marker.
    P=Path('l132-full-runtime').resolve();P.mkdir(exist_ok=True)
    for rel,source in full_sources.items():
        path=P/'sources'/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source)
    import relbench.modeling.nn as runtime_nn
    for name in ['nn.py','loader.py','graph.py']:
        module=__import__('relbench.modeling.'+name[:-3],fromlist=['x'])
        assert Path(module.__file__).read_text()==full_sources['l132/'+name],name
    # Make the independent scorer used by the runner be YOUR live function.
    package=types.ModuleType('relkit');module=types.ModuleType('relkit.identity_l132')
    module.mean_average_precision=mean_average_precision
    sys.modules['relkit']=package;sys.modules['relkit.identity_l132']=module
    prepared=P/'prepared';prepare(prepared)
    fresh=[]
    for variant in ['sage','idgnn']:
        for seed in range(5):
            fresh.append(run(variant,seed,20,prepared,P/'runs'/f'{variant}-{seed}'))
    (P/'all-results.json').write_text(json.dumps(fresh,indent=2))
    print('Ten fresh fits completed. Audit source/deviations before a paper-parity claim.')
else:
    print('Full training NOT_RUN in this kernel. Default output is the synthetic fixture.')
'''
# Portable real-data pilot evidence; the live learner MAP function rescores every query.
pilot_payload={}
for variant in ['sage','idgnn']:
 root=P/f'evidence/l132/pilot/{variant}-100'
 if (root/'result.json').exists():
  raw=(root/'predictions.npz').read_bytes()
  pilot_payload[variant]={'base64':base64.b64encode(raw).decode(),'sha256':__import__('hashlib').sha256(raw).hexdigest(),'result':json.loads((root/'result.json').read_text())}
truth_payload={}
if pilot_payload:
 import pandas as pd
 for split in ['val','test']:
  df=pd.read_parquet(P/f'evidence/l132/prepared/{split}.parquet')
  truth_payload[split]={'source':[int(x) for x in df.condition_id],'time':[int(x) for x in df.timestamp.astype('int64')],'truth':[[int(v) for v in x] for x in df.sponsor_id]}
payload_bytes=json.dumps({'pilot':pilot_payload,'truth':truth_payload},separators=(',',':')).encode()
payload_encoded='\n'.join(textwrap.wrap(base64.b64encode(zlib.compress(payload_bytes)).decode(),96))
pilot_replay="""# Author-reference predictions: independent rescoring, not a fresh benchmark fit.
import base64,io,hashlib,zlib
# Compressed JSON keeps notebook parsing bounded; verify before decoding.
payload_bytes=zlib.decompress(base64.b64decode(\"\"\"
"""+payload_encoded+"""
\"\"\"))
assert hashlib.sha256(payload_bytes).hexdigest()=="""+repr(hashlib.sha256(payload_bytes).hexdigest())+"""
payload=json.loads(payload_bytes)
pilot_payload,truth_payload=payload['pilot'],payload['truth']
pilot_rows=[];rankings=0
for variant,item in pilot_payload.items():
    raw=base64.b64decode(item['base64']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
    saved=np.load(io.BytesIO(raw))
    for split,truth in truth_payload.items():
        np.testing.assert_array_equal(saved[split+'_source'],truth['source'])
        np.testing.assert_array_equal(saved[split+'_time'],truth['time'])
        assert len(set(zip(truth['source'],truth['time'])))==len(truth['source'])
        value=mean_average_precision(saved[split+'_pred'],truth['truth'],10)
        assert abs(value-item['result']['scores'][split])<1e-12
        pilot_rows.append({'variant':variant,'split':split,'pilot_MAP_percent':100*value,'queries':len(truth['source'])})
        rankings+=len(truth['source'])
assert rankings==8276
report['pilot_rankings_rescored']=rankings
Path('l132-report.json').write_text(json.dumps(report,indent=2))
print(pd.DataFrame(pilot_rows).to_string(index=False))
print('8276 author pilot rankings checked with YOUR MAP function. Full training was not run here.')
"""
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 132 · '+TITLE+'\n\n'+('Teacher reference' if solution else 'Student lab')+' · PROVIDED: inspect; TODO: implement; CHECK: execute; EXIT: defend. Default execution trains two tiny controlled neural variants. It does not run the full RelBench benchmark. No repository files are required.'),nbf.v4.new_code_cell(bootstrap),nbf.v4.new_code_cell(imports+checks['rejected'])]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if not section.strip():continue
  cells.append(nbf.v4.new_markdown_cell(section))
  for prefix,name,check in [('## 2','mark_roots','check_marker'),('## 4','candidate_targets','check_targets'),('## 7','mean_average_precision','check_map')]:
   if section.startswith(prefix):
    src=f[name] if solution else f[name].splitlines()[0]+'\n    raise NotImplementedError("TODO: '+name+'")'
    cells.extend([nbf.v4.new_markdown_cell('### TODO · '+name+'\n\nImplement the contract above; keep the CHECK unchanged. Diagnose failures before reading the reference.'),nbf.v4.new_code_cell(src),nbf.v4.new_code_cell(checks[check]+'\n'+check+'('+name+')\nprint("PASS: '+name+'")')])
 cells.append(nbf.v4.new_markdown_cell('## PROVIDED · the released neural stack\n\nThese are the pinned MIT-licensed RelBench modules. Table-specific ResNets, temporal encoders, typed GraphSAGE and both prediction paths are visible. The small experiment uses your root marker and label function.'))
 nn=(P/'sources/l132/nn.py').read_text();model=(P/'sources/l132/model.py').read_text().replace('from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder','')
 for name,code in [('Typed encoders and graph processor',nn),('Two full prediction paths',model),('Your marker inside the explicit forward pass',f['identity_forward']),('A rooted-walk witness',f['cycle_witness']),('Paired fixture and exact source parity',fixture)]:
  cells.extend([nbf.v4.new_markdown_cell('### PROVIDED · '+name),nbf.v4.new_code_cell(code)])
 cells.extend([nbf.v4.new_markdown_cell('## CHECK · execute the controlled experiment\n\nOutputs and gradients must match the original path before fitting. The two fitted arms share weights at initialization and random draws. They differ only in marker use.'),nbf.v4.new_code_cell("report=neural_fixture()\nPath('l132-report.json').write_text(json.dumps(report,indent=2))\nprint('Source output/gradient max error:',report['source_output_max_error'],report['source_gradient_max_error'])\nprint(pd.DataFrame({k:{'BCE':v['loss'],'probabilities':v['probabilities']} for k,v in report['course_fits'].items()}).T)\nprint('Root returns:',[(x['graph'],x['root_return']) for x in report['cycle_witness']])")])
 if pilot_payload:cells.extend([nbf.v4.new_markdown_cell('## CHECK · rescore both real-data pilots\n\nThese keyed predictions came from the author’s full-data one-epoch GPU pilots. Your metric scores every validation/test query for both systems. This is evidence replay, not fresh training; it cannot establish seed uncertainty or the five-seed paper result.'),nbf.v4.new_code_cell(pilot_replay)])
 cells.append(nbf.v4.new_markdown_cell('## PROVIDED · full reproduction sources and trainer\n\nThe following named strings contain complete executable upstream files, not model pseudocode. The gate writes them to an isolated directory. Read `train`, `test`, and the checkpoint-selection block in both runners; they implement different objectives. Source/data checksums and deviations are in the linked reproduction contract.'))
 cells.append(nbf.v4.new_markdown_cell('### RelBench source license\n\n```text\n'+(P/'sources/l132/LICENSE').read_text()+'\n```'))
 cells.append(nbf.v4.new_code_cell('full_sources = {}'))
 for rel,source in source_files.items():
  cells.extend([nbf.v4.new_markdown_cell('### Pinned source · '+rel),nbf.v4.new_code_cell('full_sources['+repr(rel)+'] = '+repr(source))])
 # Show readable train/test routines as separate reference blocks, avoiding opaque strings as the sole implementation view.
 for name in ['gnn_link.py','idgnn_link.py','graph.py','loader.py']:
  cells.append(nbf.v4.new_markdown_cell('### Readable full source · '+name+'\n\n```python\n'+source_files['l132/'+name]+'\n```'))
 for name in ['prepare','instrument_source','run']:
  cells.extend([nbf.v4.new_markdown_cell('### Reproduction operator · '+name),nbf.v4.new_code_cell(runner[name])])
 cells.extend([nbf.v4.new_markdown_cell('## Full-data execution gate\n\nOFF by default. Requires the pinned GPU runtime, full archives and sufficient resources. Five seeds per model, 20 epochs each. This gate has no cloud spending authority or automatic budget enforcement; the provided Modal runner reserves the approved aggregate budget.'),nbf.v4.new_code_cell(full_gate),nbf.v4.new_markdown_cell('## EXIT\n\nSubmit the five defenses and your variant comparison. Include which cells you actually ran. Author evidence, your synthetic run and fresh full-data training are separate. **PENDING_WRITTEN_DEFENSE**.')])
 for i,c in enumerate(cells):c.id=f'l132-{i:03d}'
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}})
 path=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and path.exists():
  old=nbf.read(path,as_version=4);a=[c for c in nb.cells if c.cell_type=='code'];b=[c for c in old.cells if c.cell_type=='code']
  if [c.source for c in a]==[c.source for c in b]:
   nb.metadata=old.metadata
   for x,y in zip(a,b):x.outputs=y.outputs;x.execution_count=y.execution_count;x.metadata=y.metadata
 nbf.write(nb,path)
 if solution and all(c.execution_count is not None for c in nb.cells if c.cell_type=='code'):
  html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
reference='''## Three identities

| Signal | Meaning | Parameters |
|---|---|---|
| Table type | What kind of row? | Separate encoder/relations |
| Global entity ID | Which known database row? | One embedding per ID |
| Root marker | Who is asking this query? | One shared role vector |

## Compute and train

Two-tower: encode source and sponsor separately → dot product → BPR. Released defaults include sponsor-ID embeddings. ID-aware: mark the condition root → four graph layers → sampled sponsor head → BCE. Original ID-GNN uses distinct root/nonroot message functions; the pinned RelBench variant uses an additive marker.

## Query ownership

Label `(owner, sponsor_id)`, never sponsor ID alone. For B queries, `owner + B * sponsor_id` is an injective integer code within int64 range. Cutoffs stay attached to owners at every hop. Missing candidates cannot be repaired by a larger marker.

## MAP@k

Sum precision at relevant ranks, divide by min(k, true-set size), then average equally over nonempty query truth sets. Duplicate predictions are invalid. Preserve query ordering or align by source + cutoff.

## Evidence

Same-state marker intervention isolates a computation. Paired BCE fits isolate a toy learning mechanism. Full two-tower versus ID-GNN comparison changes multiple design choices. None establishes historical paper identity. Learner: PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/0132-identity-aware-message-passing.html) · [Notebook](../labs/0132-identity-aware-message-passing.ipynb) · [Protocol](../labs/l132-reproduction.md) · [Original ID-GNN](https://snap.stanford.edu/idgnn/) · [RelBench paper](https://arxiv.org/html/2407.20060v1).
'''
(R/'reference/identity-aware-message-passing.html').write_text(document('Identity-aware message passing · reference',reference))
print('Built lesson, reference and portable notebooks')
