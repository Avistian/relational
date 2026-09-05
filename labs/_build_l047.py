"""Build student + solution notebook. All paper model code visible; TODOs stay live."""
import sys
from pathlib import Path
import nbformat as nbf
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from _colab import bootstrap_cells
from relkit.paper_repro import inline_source


def build(solution=False):
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('''# Lab 047 — SAINT: make rows attend, then audit the context
[Lesson](../lessons/0047-saint.html) · [Primary paper](https://arxiv.org/abs/2106.01342) §3.2, Algorithm 1; §4, Eq.5.

**Skill:** implement intersample attention and measure how evaluation companions affect a prediction.
PROVIDED = read/run; TODO = your implementation; CHECK = immediate feedback; EXIT = paste your evidence.
CPU is sufficient for the learning lab (about a minute on the authoring machine; allow several minutes elsewhere).

**Mirror scope:** full supervised released-code path, plus the contrastive loss as a key-parts exercise.
The supervised model does not pretrain: the paper’s semi-supervised results remain untested.
The three local tables are substitutes, not paper Table 2 datasets. Bank/Table 2 has a required scale-up after EXIT.
The paper-results run is GPU recommended; use persistent storage and Modal for unattended long runs.''')
    for c in bootstrap_cells():
        cells.append(nbf.v4.new_markdown_cell(c['source']) if c['cell_type']=='markdown' else nbf.v4.new_code_cell(c['source']))
    md('''## Concept recap — a batch becomes part of the input
A token is a learned vector for one feature. Let **B** be rows, **T** tokens per row including CLS, and **d** coordinates per token. Feature attention operates on `[B,T,d]`: rows remain independent. Intersample attention flattens each row into length `T*d`, then regards the **B rows as one sequence**, `[1,B,T*d]`. Restore the original shape afterward. CLS is a learned summary slot, never the target label.

For each head, `A = softmax(QKᵀ / sqrt(head_dim))`, normalized over keys. Queries ask what to retrieve; keys determine match strength; values carry the information. A weighted sum changes a query even when only a companion changes. For scores `[1,0]` and values `[2,8]`, the weights are approximately `[0.731,0.269]`, and the output is `3.614`. With only the query present the sole weight is 1 and the value is 2.

The contrastive term asks each clean row to identify its own augmented view. Its positive is the matching row index, not a different row with the same class. With B equally likely candidates, mean loss is `log(B)`. Denoising separately reconstructs original features from corrupted views; our small loss exercise does not implement that whole pretraining experiment.

**Code/prose audit:** the released block computes `LN(x) + F(LN(x))`, uses GEGLU feed-forward gates and per-numeric `1→100→d` MLPs. These details differ from the PDF’s equations/prose. We match the released stage, with exact transplanted-weight forward and input-gradient checks; we do not silently call the different descriptions identical.''')
    code('''# PROVIDED — reproducible environment; run locally from labs/ or labs/solutions/
import os, sys, json
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1')
for candidate in (Path.cwd(), Path.cwd().parent, Path.cwd()/'labs'):
    if (candidate/'relkit').is_dir():
        sys.path.insert(0,str(candidate)); LABS=candidate.resolve(); break
import numpy as np
import torch
from torch.nn import functional as F
torch.set_num_threads(1)
from relkit.saint_experiment import prepare, environment
print(json.dumps(environment(),indent=2))''')
    md('''## Task 1 — pack and restore entire rows (Algorithm 1)
**Goal:** convert feature-token batches into a single sequence of flattened rows and back.
**Why:** a transpose that attends down each column implements a different model. Preserve every feature coordinate and include CLS.
Write the two reshape operations. The checks use distinct coordinates so an accidental permutation cannot pass.''')
    code('''# TODO — two focused shape operations
''' + ('''def pack_rows(tokens):
    b,t,d=tokens.shape
    return tokens.reshape(1,b,t*d)

def unpack_rows(rows,n_tokens):
    return rows.reshape(rows.shape[1],n_tokens,rows.shape[-1]//n_tokens)
''' if solution else '''def pack_rows(tokens):
    b,t,d=tokens.shape
    return ____

def unpack_rows(rows,n_tokens):
    return ____
'''))
    code('''# CHECK — values and row ownership, not just shape
x=torch.arange(60.).reshape(3,4,5)
packed=pack_rows(x)
assert packed.shape==(1,3,20)
assert torch.equal(packed[0,1],x[1].flatten())
assert torch.equal(unpack_rows(packed,4),x)
print('PASS: entire-row packing is invertible')''')
    md('''## Task 2 — normalize the attention over candidate rows
**Goal:** implement score scaling and softmax over keys.
**Why:** query normalization has the right tensor shape but fails to form a distribution over companions.
The last axis of Q/K is the head width, not the number of rows or feature tokens.''')
    code('''# TODO — attention probabilities
''' + ('''def attention_weights(q,k):
    scores=q @ k.transpose(-2,-1) / q.shape[-1]**.5
    return scores.softmax(dim=-1)
''' if solution else '''def attention_weights(q,k):
    scores=____
    return ____
'''))
    code('''# CHECK — an independent PyTorch kernel validates your calculation
q,k,v=[torch.randn(2,3,5,7,dtype=torch.float64) for _ in range(3)]
a=attention_weights(q,k)
assert torch.allclose(a.sum(-1),torch.ones_like(a.sum(-1)))
assert torch.allclose(a@v,F.scaled_dot_product_attention(q,k,v),atol=1e-12)
print('PASS: attention agrees with independent reference')''')
    md('''## Task 3 — the contrastive pairing (Eq.5, first term)
**Goal:** use each original row’s index as the correct augmented-view class.
**Why:** class labels are unnecessary for this task. A shuffled positive pairing trains the wrong invariance.
Use cross-entropy on the similarity matrix scaled by temperature. Mean loss is a batch-size-independent rescaling of the paper’s sum. These input vectors are not implicitly normalized; normalization is a separate design choice.''')
    code('''# TODO — contrastive loss; no dataset labels enter this function
''' + ('''def info_nce(z,z_view,temperature=.7):
    logits=z @ z_view.T / temperature
    return F.cross_entropy(logits,torch.arange(len(z),device=z.device))
''' if solution else '''def info_nce(z,z_view,temperature=.7):
    logits=____
    return ____
'''))
    code('''# CHECK — correct pairing beats a mismatched view; equal logits imply log(B)
z=torch.eye(4)
assert info_nce(z,z)<info_nce(z,z.roll(1,0))
assert abs(float(info_nce(torch.zeros(4,3),torch.zeros(4,3)))-np.log(4))<1e-6
print('PASS: identity pairing and uniform-loss baseline')''')
    md('''## PROVIDED — the full supervised implementation
Read the numeric embedding, both attention blocks, normalization wrapper, missing-value token, CLS head, and training loop below.
Your three functions remain in use: this source copy removes their canonical definitions. The model you train calls your implementations.
The exact reference-stage check is regenerable with `_check_l047.py --reference PATH`; see the reproduction reference for the pinned source command.''')
    code(inline_source(str(HERE/'relkit/saint.py'),skip_defs={'pack_rows','unpack_rows','attention_weights','info_nce'}))
    md('''## Task 4 — intervene on a companion, holding the query fixed
**Goal:** distinguish permutation equivariance from independence of batch membership.
**Why:** reordering an unchanged batch should reorder outputs; changing its members can change them.
Both models are in evaluation mode with dropout disabled for this probe. The tensors are synthetic only to isolate the operator; the next experiment trains on real data.''')
    code('''# PROVIDED + CHECK — causal wiring probe, not an accuracy claim
torch.manual_seed(47)
x=torch.randn(5,4,8)
row=SaintStage(4,8,ff_dropout=0.).eval()
col=SaintStage(4,8,ff_dropout=0.,variant='col').eval()
changed=x.clone(); changed[1]=torch.randn_like(x[1])*3
with torch.no_grad():
    delta=(row(x)[0]-row(changed)[0]).abs().max().item()
    assert delta>1e-6
    assert torch.allclose(col(x)[0],col(changed)[0])
    perm=torch.tensor([3,1,4,0,2])
    assert torch.allclose(row(x)[perm],row(x[perm]),atol=1e-5)
print('PASS: different companions move query; row permutations preserve correspondence.',delta)''')
    md('''## Real-data ablation — run your model on three datasets × three seeds
**Held fixed:** split seed 5; 65/15/20 stratified split; preprocessing fitted on training rows; d=8, one stage, four heads, FF dropout .1; 20 epochs; AdamW .001; batch 64; validation AUC selects the checkpoint. Model seeds are 0,1,2 and set before construction.
**Varied:** feature-only vs feature+row attention. This also changes parameter count: it is an ablation of adding the row block, not a parameter-matched proof about attention alone. The feature-only control is depth-matched, not the paper’s six-stage SAINT-s.
**External baseline:** CatBoost, 300 iterations/depth 6/lr .05, same splits, validation AUC. It is an untuned budgeted baseline; no claim about the best achievable tree score.
**Measured:** AUROC (ranking quality for binary classes), seed SD and conditional seed CIs, per-dataset ranks, Friedman/Nemenyi. Three datasets provide little power. No test labels enter forward, and test metrics never choose checkpoints.

Tier A substitutes: credit_g, diabetes, blood_transfusion. Credit_g is not the paper’s Credit card-fraud table. The paper’s Bank, Blastchar, Arrhythmia, Arcene, Forest, Shoppers, Income, HTRU2, KDD99, Philippine, QSAR Bio, Shrutime, Spambase, Credit, Volkert and MNIST are not this local benchmark.''')
    code('''# PROVIDED — harness imports are allowed; your inlined model/train/predict functions are injected
from _verify_l047 import run
result=run(model_cls=SAINT,train_fn=train_saint,predict_fn=predict_saint)
print(json.dumps(result['results'],indent=2))
Path('l047_student_results.json').write_text(json.dumps(result,indent=2))''')
    code('''# CHECK — completeness and context audit, with no enforced winning model
assert len(result['results']['per_dataset'])==3
for dataset,arms in result['results']['per_dataset'].items():
    assert all(len(a['scores'])==3 for a in arms.values())
for name,probe in result['context_probe'].items():
    if name.endswith('/col'):
        assert probe['max_abs_probability_change']<1e-5
assert all(set(p['train']).isdisjoint(p['test']) for p in result['protocols'].values())
print('PASS: three datasets, three seeds, split boundaries, independent-row control')''')
    md('''## EXIT TICKET
Explain the reshape, one companion intervention, and one limitation of the local comparison.
Report the actual winner/tie you measured; no CHECK requires SAINT to win. Distinguish failure to reject a rank-test null from proof of equivalence. Paste the output plus your explanation to the teacher; follow-up questions are welcome.''')
    code('''# EXIT — evidence, not a prewritten conclusion
print('Mean ranks:',result['results']['mean_ranks'])
print('Friedman:',result['results']['friedman'])
print('Nemenyi CD:',result['results']['nemenyi_cd'])
print('Batch-context probe:',json.dumps(result['context_probe']['diabetes/0/colrow'],indent=2))
print('Paper reproduction status: NOT_RUN until the required next step executes.')''')
    md('''## REQUIRED NEXT STEP — attempt the paper’s supervised Bank result
The paper’s **Table 2 Bank AUROC is 0.9330**, five-trial mean. An absolute tolerance of ±0.01 would be a reproduction target only after agreeing the protocol. Our honest status remains INCOMPARABLE when unresolved split/code differences remain, even if the number happens to match.

The same from-scratch implementation and loop run below. `smoke` = 600 rows/1 epoch/1 seed (pipeline check); `closer` = full Bank/20 epochs/3 seeds/d=16; `paper` = full Bank/100 epochs/5 seeds/d=32/8 heads. The last preset adopts paper budgets, not a certificate of exact reproduction. No full pretraining benchmark is included.

Read `GAPS` and the three-bucket ledger before interpreting results. Set `RUN_PAPER_REPRO=True` for a deliberate run. In Colab, attach a GPU and mount Drive; set `OUTPUT_DIR` to a persistent Drive directory. Completed seeds can be resumed; interrupted seeds restart. The best checkpoint is saved for inference, not full optimizer resume. For multi-hour unattended runs use:

```bash
modal run --detach modal/l047_paper_repro.py --preset closer
modal volume get relational-artifacts l047/closer ./l047-bank-results
```

Runtime of the full presets is unmeasured; do not assume it fits a free Colab session. Start with smoke and budget from the measured time. Test predictions, split IDs, preprocessing, versions, seeds, and best weights are saved. Until the larger run finishes, the Table 2 claim stays cited, not reproduced.''')
    code(inline_source(str(HERE/'_paper_repro_l047.py'),skip_imports={'relkit.saint'}))
    code('''# PROVIDED — explicit compute gate
RUN_PAPER_REPRO=False
PRESET='closer'
OUTPUT_DIR='l047_artifacts'  # Colab: change to a mounted persistent Drive path
if RUN_PAPER_REPRO:
    paper_result=main(['--preset',PRESET,'--output-dir',OUTPUT_DIR])
else:
    print('NOT_RUN: paper-scale Bank. Run smoke first; then use Colab GPU or the Modal command above.')''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
    out=HERE/('solutions' if solution else '')/'0047-saint.ipynb'
    out.parent.mkdir(exist_ok=True);nbf.write(nb,out)
    print(out)

if __name__=='__main__':
    build();build(True)
