"""Build B10 lesson/reference and source-visible portable notebooks."""
import ast,base64,hashlib,io,json,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='b10-relational-transformer';E=P/'evidence/b10'
mechanism=json.loads((E/'mechanism.json').read_text());audit=json.loads((E/'temporal-audit.json').read_text())
def fig(name,caption):return f'<figure class="b10-figure"><img src="../labs/figures/b10/{name}.png" alt="{caption}"><figcaption>{caption}</figcaption></figure>'
arch=fig('architecture','RT-v1 end-to-end path. The measured lab uses reduced dimensions and an explicit CPU backend; no pretrained benchmark prediction is shown.')
mobile='''<div class="b10-mobile-architecture"><ol><li><strong>Question:</strong> customer10 → task row100 → hidden y at day10.</li><li><strong>Context:</strong> sample linked rows, enforce visibility, replace the target value.</li><li><strong>Tokens:</strong> type-specific value or mask vector + name vector → S×D states.</li><li><strong>Block:</strong> column → feature → neighbor → full attention, with normalization and residual additions; then SwiGLU. Paper:12blocks.</li><li><strong>Output:</strong> final normalization → typed decoder → target probability.</li><li><strong>Objective:</strong> masked BCE/Huber for training; fixed-weight, keyed scoring for evaluation.</li></ol></div>'''
maskwidget='''<div class="b10-board" data-b10="masks"><h3>Who can this cell read?</h3><p>Choose a reader and a permission. Blue outline marks the reader; “Allowed” marks readable sources.</p><label>Attention <select name="kind"><option value="col">Column</option><option value="feat" selected>Feature</option><option value="nbr">Neighbor</option><option value="full">Full</option></select></label><label>Reader <select name="reader">'''+''.join(f'<option value="{i}">{s}</option>' for i,s in enumerate(['T0.y','T0.time','C0.age','C0.name','O0.amount','O0.time','T1.y','T1.time','C1.age']))+'''</select></label><button type="button">Reset cell example</button><div class="b10-cells"></div><output aria-live="polite">Default feature attention: T0.y reads its own row and parent C0. Equal-score coordinate: (0+10+42+1)/4 = 13.25. Neighbor attention at T0.y has no children and returns zero.</output></div>'''
timewidget='''<div class="b10-board" data-b10="time"><h3>Three clocks, one context</h3><label>Prediction cutoff: day <strong data-value>10</strong><input type="range" min="8" max="14" step="1" value="10"></label><button type="button">Reset cutoff</button><output aria-live="polite">At day10: the day7 order is eligible; the day8 label needs day11; the day12 schedule remains outside the strict event-time policy even if announced on day9.</output></div>'''
results='| Seed | Max output error | Max parameter-gradient error | Permutation error | Hidden-target prediction change |\n|---|---:|---:|---:|---:|\n'
for x in mechanism['seeds']:results+=f'| {x["seed"]} | {x["output_max_error"]:.2e} | {x["parameter_gradient_max_error"]:.2e} | {x["permutation_max_error"]:.2e} | {x["hidden_target_max_error"]:.1f} |\n'
results+='\nAll three pass the predeclared **1e-9 absolute tolerance**. Loss and floating-input gradients also pass. The 8,400 pair checks include padding; three wrong learner functions are rejected. These are numerical verification errors, not benchmark metric improvements.'
body=(R/'lessons/content'/f'{S}.md').read_text()
for key,value in {'ARCHITECTURE':'<div class="b10-desktop-architecture">'+arch+'</div>'+mobile,'MASK_WIDGET':maskwidget,'TIME_WIDGET':timewidget,'MASK_FIGURE':fig('masks','Same context, four directed permission matrices. Readers are rows, sources are columns. Padding is omitted from the picture and forbidden in code.'),'TIME_FIGURE':fig('visibility','Illustrative days: event, arrival and completed-outcome clocks can disagree. This figure is not measured F1 evidence.'),'RESULTS':results}.items():body=body.replace('{{'+key+'}}',value)
def document(title,content,scripts=''):
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/relational-cell-attention.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html#research-bridge">Research bridge</a></nav>'+render(content)+'</article>'+scripts+'</body></html>'
scripts=''.join(f'<script src="../assets/{f}.js"></script>' for f in ['retrieval-pool','retrieval-bank','predict','teachback','relational-cell-attention'])
(R/'lessons'/f'{S}.html').write_text(document('B10 · Relational Transformer',body,scripts))
reference='''# B10 · Cell attention field guide

**Question:** Which information can reach a masked task cell?

| Sublayer | Permission from reader i to source j |
|---|---|
| Column | table_i = table_j AND column_i = column_j |
| Feature | row_i = row_j OR row_j is a referenced parent of row_i |
| Neighbor | row_i is a referenced parent of row_j |
| Full | both cells are in the admitted context |

Apply non-padding to both ends. An empty allowed set has zero update. IDs express relations, not ordinal magnitudes. Cell-order permutation must move values and metadata together.

**Forward path:** eligible row context → typed value or mask vector + name vector → [column, feature, neighbor, full attention, SwiGLU] × blocks → final RMSNorm → datatype decoder. Each sublayer uses a residual addition. BCE supervises booleans; Huber supervises numbers/datetimes. Average over masked cells. Original release does not support masked-text loss.

**Three boundaries:** context selection decides which facts enter; attention decides which admitted cells mix; the loss reads target truth that inference may not encode. Full attention cannot repair unsafe context selection.

**Three clocks:** event timestamp, recorded availability, label-window completion. A scheduled event can be future-dated yet known. Unknown arrival prevents an availability proof. The course policy requires all three conditions; it is not the released RT sampler.

**Three zero-shot questions:** Was the task trained on? Was the database trained on? Can labeled task rows enter the context? Also declare validation selection and historical test-label policy.

**Evidence:** source-shaped CPU fixture checks COMPLETE; full saved-context replay COMPLETE; RT-v1 Table1 driver-dnf inference NOT_RUN because INCOMPLETE_TEMPORAL_GATE. Paper82.0% AUROC is cited, not measured. Pretraining, whole-paper reproduction, CUDA/bfloat16 parity and historical identity are not established by these checks. RT-J is a distinct release.

[Lesson](../lessons/b10-relational-transformer.html) · [Student lab](../labs/b10-relational-transformer.ipynb) · [Protocol](../labs/b10-reproduction.md) · [Primary paper](https://arxiv.org/html/2510.06377v1)
'''
(R/'reference'/f'{S}.html').write_text(document('B10 · Cell attention field guide',reference))
# Portable packet contains authenticated full saved contexts plus pinned source.
ledger=json.loads((P/'sources/b10/source-ledger.json').read_text());buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for name in ledger['inherited_contexts']:z.write(P/name,name)
 for f in (P/'sources/b10').rglob('*'):
  if f.is_file() and '__pycache__' not in str(f):z.write(f,str(f.relative_to(P)))
 z.writestr('evidence/b10/mechanism.json',json.dumps(mechanism))
packet=buf.getvalue();module=(P/'relkit/rt_b10.py').read_text();tree=ast.parse(module)
segments={n.name:ast.get_source_segment(module,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
tests=(P/'_test_b10.py').read_text();testsegments={n.name:ast.get_source_segment(tests,n) for n in ast.parse(tests).body if isinstance(n,ast.FunctionDef)}
verify=(P/'_verify_b10.py').read_text();verifysegments={n.name:ast.get_source_segment(verify,n) for n in ast.parse(verify).body if isinstance(n,ast.FunctionDef)}
def png(name):return '!['+name+'](data:image/png;base64,'+base64.b64encode((P/f'figures/b10/{name}.png').read_bytes()).decode()+')'
for solution in (False,True):
 cells=[]
 def md(s):cells.append(nb.v4.new_markdown_cell(s))
 def code(s):cells.append(nb.v4.new_code_cell(s))
 md('# B10 · Let database cells talk\n\nImplement and defend cell-level relational attention. **PROVIDED** supplies plumbing, **TODO** is your live implementation, **CHECK** gives immediate feedback, and **EXIT** requires your explanation.\n\nThe notebook runs fresh CPU mechanism checks and replays complete saved native contexts. It does not train RT, load its released weights or reproduce a paper AUROC. Tier C synthetic fixture isolates operations; Tier B real F1 contexts audit temporal evidence.\n\nRead [RT-v1 §§3–5](https://arxiv.org/html/2510.06377v1).')
 code('''# @colab-bootstrap — self-contained; no repository imports.
import sys, subprocess, importlib.util
for package in ['numpy','torch','einops','ml_dtypes']:
    if importlib.util.find_spec(package) is None:
        subprocess.check_call([sys.executable,'-m','pip','install',package])
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import base64, hashlib, io, json, zipfile, tempfile, types
from pathlib import Path
from IPython.display import display, Markdown
torch.set_num_threads(1)
TYPES=('number','text','datetime','boolean')
print('Runtime:',torch.__version__, 'Author verification: torch2.13.0+cpu, float64')''')
 md('## Concept recap and worked example\n\nA primary key identifies a row; a foreign key points to a parent row. A task table attaches a prediction target to an entity and a cutoff. A cell token represents one feature value. A mask replaces a hidden value before encoding.\n\nThree tables: taskT0(row100) and old taskT1(row101) both reference customerC0(row10); orderO0(row20) also references C0. CustomerC1(row11) is unrelated. Our nine cells are T0.y, T0.time, C0.age, C0.name, O0.amount, O0.time, T1.y, T1.time, C1.age; slot9 is padding.\n\nFeature attention reads the same row and parents; neighbor attention reads children. Column attention requires BOTH table and column identity. Full attention reads all admitted non-padding cells. Reader is the matrix row; source is its column. Empty neighbor sets return zero.\n\n**Predict:** Can T0.y read O0.amount through feature attention? What changes under full attention?')
 md('## Model architecture\n\n'+png('architecture')+'\n\nValue/type projection or learned mask + projected name → four attention sublayers with residual additions → SwiGLU → typed decoder. RMSNorm rescales each vector by its root mean square. SwiGLU multiplies a smooth-activated projected stream by a second projected stream. Boolean logits use BCE; numeric/datetime predictions use Huber loss. The loss averages only masked positions.\n\nOriginal:12blocks,width256,8heads,FF1024,MiniLM384,bfloat16. Fixture:2blocks,width16,4heads,FF32,illustrative text/name width6,float64. These explicit deviations make this a mechanism check. No real pretrained semantic encoding is claimed.')
 md('## Permissions before weights\n\n'+png('masks')+'\n\nBlue allows a source; it does not fix the learned weight. A query–key score is a dot product divided by the square root of per-head width. Softmax normalizes permitted scores; their weighted values make the update. An empty set must avoid softmax of all negative infinities.')
 code('# PROVIDED · relationship metadata; independent scalar checker\n'+testsegments['fixture']+'\n\n'+testsegments['oracle'])
 goals={'relational_masks':'Construct all four B×S×S permissions from row, parent, table, column and padding metadata. Neither an equal column ID across tables nor a reversed foreign-key direction is sufficient.', 'safe_attention':'Compute B×H×S×d attention from scores and permissions. Forbidden keys get no weight; empty sets produce zero with finite gradients.', 'eligible_rows':'Filter evidence by known event and arrival times, a cutoff and completed label horizons. This is a separate strict course policy; unknown times fail closed.'}
 shortchecks={
 'relational_masks':"b=fixture()\nfor kind,expected in oracle(b).items():\n    assert torch.equal(relational_masks(b)[kind],expected),kind\nprint('CHECK: four directed masks and padding')",
 'safe_attention':"q=torch.zeros(1,1,2,2,dtype=torch.float64)\nv=torch.tensor([[[[2.,0.],[6.,0.]]]],dtype=torch.float64)\na=torch.tensor([[[True,True],[False,False]]])\nassert torch.equal(safe_attention(q,q,v,a),torch.tensor([[[[4.,0.],[0.,0.]]]],dtype=torch.float64))\nprint('CHECK: equal-score mean4 and empty-set zero')",
 'eligible_rows':"t=torch.tensor([7.,8.,12.]);arrival=torch.tensor([7.,11.,9.]);label=torch.tensor([False,True,False])\nassert eligible_rows(t,arrival,10.,label,3.).tolist()==[True,False,False]\nprint('CHECK: event, arrival and outcome completion')"}
 for name in goals:
  md('## TODO · '+name+'\n\n'+goals[name]+'\n\nWrite this function, then run the CHECK. It will be used in later computations; do not edit the check.')
  code(segments[name] if solution else segments[name].split('\n')[0]+'\n    raise NotImplementedError("Complete '+name+'")')
  code('# CHECK\n'+shortchecks[name])
 code('# CHECK · additional adversarial examples\n'+testsegments['checks']+'\nprint(checks())')
 md('## PROVIDED · visible attention projections and residual block\n\nYour `safe_attention` is called by the projection module; your `relational_masks` is called by the complete model. The four sublayers run column→feature→neighbor→full. This sequence is part of the architecture, even though the full layer has broad permissions.')
 for name in ['MaskedAttention','FFN','RelationalBlock']:
  md('### '+name);code(segments[name])
 md('### Typed encoder, masked loss and decoder\n\nThe hidden target is absent from the value representation but remains the truth used by the loss. Therefore a target intervention should preserve predictions and change loss. Text masking is unsupported in this release. Values must be finite preprocessed tensors; missing and padded cells are managed by metadata.')
 code(segments['RelationalTransformer'])
 md('### Visible optimization step (not executed as a training experiment)\n\nTraining would zero gradients, run the model, differentiate the masked loss, clip gradients and update weights. Defining this function does not claim fresh pretraining. The exact original distributed trainer is in the appendix.')
 code(segments['train_step'])
 md('## PROVIDED · authenticated original source and real-context packet\n\nThis self-contained ZIP retains original source, metadata and all three saved native-context seeds. Hash checking authenticates the packet; it does not establish historical information availability. It extracts only packaged relative paths into a temporary directory. No weight download occurs.')
 code('payload=base64.b64decode('+repr(base64.b64encode(packet).decode())+')\nassert hashlib.sha256(payload).hexdigest()=='+repr(hashlib.sha256(packet).hexdigest())+'\nwork=tempfile.TemporaryDirectory(prefix="b10-portable-")\nP=Path(work.name)\nwith zipfile.ZipFile(io.BytesIO(payload)) as z:\n    for name in z.namelist():\n        dest=(P/name).resolve()\n        assert dest.is_relative_to(P.resolve())\n        dest.parent.mkdir(parents=True,exist_ok=True)\n        dest.write_bytes(z.read(name))\n(P/"evidence/b10").mkdir(parents=True,exist_ok=True)\nprint("Authenticated portable packet; saved contexts, not fresh sampling")')
 md('## CHECK · copied-weight source parity\n\nThe original model/masks/loss stay unchanged. Only its compiled sparse attention builder/kernel are replaced with CPU PyTorch SDPA. Our explicit softmax is independently exercised against that backend. This does not validate sparse CUDA/bfloat16 execution. Three seeds, fixed fixture, absolute tolerance1e-9.')
 code(verifysegments['typed_fixture']+'\n\n'+verifysegments['source_module'])
 code('''up=source_module();parity=[]
for seed in (0,1,2):
    b=typed_fixture(seed);torch.manual_seed(seed+10)
    ref=up.RelationalTransformer(2,16,6,4,32).double()
    model=RelationalTransformer(2,16,6,4,32).double()
    model.load_state_dict(ref.state_dict(),strict=True)
    x={k:v.clone().requires_grad_(v.is_floating_point()) for k,v in b.items()}
    r={k:v.clone().requires_grad_(v.is_floating_point()) for k,v in b.items()}
    loss,out=model(x);rloss,rout=ref(r)
    err=max(float((out[t]-rout[t]).abs().max().detach()) for t in TYPES)
    assert err<1e-9 and abs(float(loss.detach()-rloss.detach()))<1e-9
    loss.backward();rloss.backward()
    ge=max(float((p.grad-dict(ref.named_parameters())[n].grad).abs().max()) for n,p in model.named_parameters())
    ie=max(float((x[k].grad-r[k].grad).abs().max()) for k in x if x[k].is_floating_point())
    assert ge<1e-9 and ie<1e-9
    changed={k:v.clone() for k,v in b.items()};changed['number_values'][0,4]=1000
    cl,cp=model(changed)
    assert all(torch.equal(cp[t],out[t]) for t in TYPES) and abs(float(cl.detach()-loss.detach()))>1
    order=torch.tensor([7,3,9,0,5,8,1,4,6,2]);_,permuted=model({k:v[:,order] for k,v in b.items()})
    assert all(torch.allclose(permuted[t],out[t][:,order],atol=1e-9,rtol=0) for t in TYPES)
    parity.append((seed,err,ge,ie))
display(Markdown(chr(10).join(['|Seed|Output max error|Parameter gradient|Input gradient|','|---|---:|---:|---:|']+['|'+str(s)+'|'+f'{a:.2e}|{b:.2e}|{c:.2e}|' for s,a,b,c in parity])))
print('CHECK: copied-weight source, gradients, hidden targets and permutations')''')
 md('## CHECK · a future value must not reach the prediction\n\n'+png('visibility')+'\n\nThis is the stricter course policy, not the released sampler. Predict whether changing a future amount can change the query output. The constructed query remains in the context; candidate evidence is filtered.')
 code('''def prediction_with_future(amount):
    event=torch.tensor([2.,5.,12.]);arrival=event.clone();is_label=torch.zeros(3,dtype=torch.bool)
    keep=eligible_rows(event,arrival,10.,is_label,3.)
    values=torch.tensor([2.,6.,amount])
    x=typed_fixture(0);x['number_values'][0,4]=values[keep].mean().double()
    return model(x)[1]['boolean'][0,0].detach()
assert torch.equal(prediction_with_future(30.),prediction_with_future(30000.))
print('CHECK: excluded future evidence cannot alter the prediction')''')
 md('## PROVIDED / CHECK · complete real saved-context replay\n\nThe next visible auditor authenticates every input, reconstructs all2106labels from raw results, checks702unique driver/cutoff keys per seed and rechecks all2,156,544cell slots. It is fresh analysis of saved data, not fresh native sampling or checkpoint inference.')
 auditcode=(P/'_audit_b10.py').read_text();auditcode=auditcode[:auditcode.index("if __name__=='__main__'")].replace("P=Path(__file__).resolve().parent;S=P/'sources/b10';E=P/'evidence/b10'","S=P/'sources/b10';E=P/'evidence/b10'")
 code(auditcode+'\nreport=audit()\nassert report["totals"]["future_cells"]==385\nassert report["paper_gate"]=="INCOMPLETE_TEMPORAL_GATE"')
 md('## Interpret author and live evidence\n\n'+results+'\n\nThe table above is author-reference evidence. The CHECK outputs above it are from your current kernel. Both are finite-fixture mechanism evidence. In the real saved contexts,385future-dated cells occur in77contexts while query targets remain hidden. Race schedules might have been known earlier, but no arrival logs prove that.432,050untimed cells also lack an availability proof.\n\nNo target gradients does not mean no contextual labels. An unseen task is not necessarily an unseen database. Changing schema names changes inputs and is not a row-permutation test.')
 md('## EXIT · written defense\n\nSubmit your three functions, CHECK outputs and150words tracing T0.y through context selection, typed masking and the four attention operations. Explain why changing hidden truth affects loss but not predictions; why scheduled future fields need a declared policy; and why82.0%AUROC remains a paper claim. Learner status **PENDING_WRITTEN_DEFENSE**. Repeat the mask trace after1/7/30days. Ask the teaching agent about unclear steps.\n\nB11 compares RelGNN/RelGT under matched relational evidence. Identify which units are rows and which are cells before comparing architectures.')
 md('## NEXT STEP · named paper target and full source\n\nRT-v1 Table1, rel-f1/driver-dnf, reported82.0%AUROC. Original12×256model,1024cells,BFS256, full702queries, seed0 plus robustness1/2; original data and weight revisions pinned in the packet. No inference or fresh training has run. Checkpoint bytes remain unauthenticated. The GPU path after the gate remains unvalidated. Full pretraining and whole-paper reproduction are NOT_RUN.\n\nRepository command: `.venv/bin/python labs/_budget_b10.py .venv/bin/python labs/_reproduce_b10.py`. Explicit `--run` recomputes the gate and refuses. The3600second aggregate local budget includes preparation/retries/checks; USD0paid. Live Colab and deployment are not verified. Do not replace RT-v1 with the current RT-J quickstart.')
 code('''RUN_PAPER_REPRO=False
if RUN_PAPER_REPRO:
    # Recheck actual packaged data; no editable PASS flag can bypass this gate.
    current=audit()
    if current['paper_gate']!='PASS':
        raise RuntimeError('BLOCKED_TEMPORAL_AUDIT: checkpoint inference NOT_RUN')
    raise RuntimeError('Use the isolated original-runtime operator with authenticated weights')
print('Paper lane: INCOMPLETE_TEMPORAL_GATE; inference NOT_RUN')''')
 md('## Source appendix · original model and trainer\n\nUnmodified pinned source is readable below; it is not executed again by these appendix cells. Preserve source/release attribution. The backend adapter above is the only source-parity execution substitution. Original model and trainer are separate from the reduced teaching fixture.')
 for name in ['upstream/rt/model.py','upstream/rt/main.py','upstream/rt/data.py','example_pretrain.py','example_finetune.py']:
  md('### '+name+'\n\n```python\n'+(P/'sources/b10'/name).read_text()+'\n```')
 md('### Selected native inference operator (post-gate path unvalidated)\n\n```python\n'+(P/'_infer_b10.py').read_text()+'\n```')
 for c in cells:
  if c.cell_type=='code':compile(c.source,'b10-generated-cell','exec')
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
 target=P/('solutions' if solution else '')/(S+'.ipynb');nb.write(notebook,target)
print('Built B10 lesson/reference/portable notebooks; packet bytes',len(packet))
