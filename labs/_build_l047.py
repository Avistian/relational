"""Build a notebook-first SAINT lesson, with live student tasks and portable figures.

Regenerate figures separately with _figures_l047.py when visuals/results change.
Canonical model chunks are extracted by AST, never maintained as a second model.
"""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path
import nbformat as nbf
from _colab import bootstrap_cells

HERE = Path(__file__).resolve().parent
FIGURES = HERE / 'figures/l047'
MODEL = (HERE / 'relkit/saint.py').read_text()


def extract(source, names, omit_methods=()):
    """Keep original comments/formatting/decorators, omit only named methods."""
    lines = source.splitlines()
    chunks = []
    for node in ast.parse(source).body:
        if getattr(node, 'name', None) not in names:
            continue
        start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])]) - 1
        omitted = set()
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if getattr(child, 'name', None) in omit_methods:
                    omitted.update(range(child.lineno - 1, child.end_lineno))
        chunks.append('\n'.join(lines[i] for i in range(start, node.end_lineno) if i not in omitted).rstrip())
    assert len(chunks) == len(names), names
    return '\n\n\n'.join(chunks)


def build(solution=False):
    cells = []
    def md(s):
        cells.append(nbf.v4.new_markdown_cell(s.strip()))
    def code(s):
        cells.append(nbf.v4.new_code_cell(s.strip()))
    def figure(name, alt, caption):
        data = base64.b64encode((FIGURES / f'{name}.png').read_bytes()).decode()
        cell = nbf.v4.new_markdown_cell(f'![{alt}](attachment:{name}.png)\n\n{caption}')
        cell['attachments'] = {f'{name}.png': {'image/png': data}}
        cells.append(cell)
    def provided(names, description, omit_methods=()):
        code('# PROVIDED — ' + description + '\n' + extract(MODEL, names, omit_methods))
    def todo(student, answer):
        code(answer if solution else student)

    md('''# Lab 047 · SAINT
## Build row attention. Measure what its companions change.

[Lesson 047](../lessons/0047-saint.html) · [Reproduction reference](../reference/saint-reproduction.html) · [Paper: Somepalli et al. (2021)](https://arxiv.org/abs/2106.01342v1)

**From Lesson 046:** you already turned each feature into a token and read a learned `[CLS]` summary. SAINT adds a second attention axis: whole rows can exchange information. Your goal is to implement that path and determine what belongs in its evaluation protocol.

**By EXIT you will have:** an invertible row-packing operator, independently checked attention, a stage whose forward pass you wrote, and a three-dataset experiment using that stage. You will also implement the paper's contrastive pairing term and explain why this alone does not reproduce pretraining.

| Route | What you do | Evidence you produce |
|---|---|---|
| 1 · Understand | Trace feature attention versus row attention | A prediction about changing a companion |
| 2 · Build | Complete Tasks 1–3, inspect real tensors | Values, gradients, and stage-wiring checks |
| 3 · Audit | Change companions; implement Task 4's loss | An intervention and a pairing check |
| 4 · Compare | Train 3 models × 3 datasets × 3 seeds | AUROC, uncertainty, ranks, context report |
| 5 · Reproduce | Attempt Bank with the visible scale-up runner | Saved artifacts and an honest scope verdict |

**How to work:** PROVIDED = read/run; TODO = write the focused blanks; CHECK = run immediately; EXIT = explain your actual evidence. Predict before executing each probe. Keep written answers in the marked markdown cells. The core lab is CPU-sized: the revised solution's comparison took 109 seconds on the authoring machine; reading and implementation take longer, and runtime varies by machine. Bank is a separate, deliberately gated compute step.

All mechanism diagrams are embedded in this file and visible before execution. The lesson offers their interactive versions. The student notebook intentionally has empty code outputs; the clearly labeled author-reference panels later in this notebook show an actual recorded run.''')
    md('''### The reproduction contract

| Evidence | Scope in this notebook | What it cannot establish |
|---|---|---|
| **Architecture** | Full supervised released-code path, implemented from scratch | That every PDF equation matches the release |
| **Mechanism probes** | Synthetic tensors isolate axes, pairing, and companion effects | Accuracy on a population |
| **Local training · Tier A** | Real `credit_g`, `diabetes`, `blood_transfusion`; 3 seeds; fixed split | Reproduction of the paper's Table 2 |
| **Pretraining** | One-way contrastive loss from §4, Eq. 5, as a key-parts exercise | Full augmentation/denoising training or Table 3 results |
| **Paper-results track** | Bank/Table 2 runner, three presets, persistent artifacts | A completed full-scale run merely because code exists |

The three local tables are substitutes. `credit_g` is **not** the paper's credit-card-fraud dataset. Paper tasks such as Bank, Blastchar, Arrhythmia, Arcene, Forest, Shoppers, Income, HTRU2, KDD99, Philippine, QSAR Bio, Shrutime, Spambase, Credit, Volkert, and MNIST are outside this local comparison; Bank has its own required next step.

The implementation target is the [official release pinned at `e288e84`](https://github.com/somepago/saint/tree/e288e84c77a54cfd2ffb55a53678fb7cbbb16630), audited against arXiv v1. The measured reference run is `_verify_l047_results.json`; `_sources_l047.json` records source provenance; `requirements-l047.lock.txt` snapshots the authoring environment. A downscaled experiment stays a different experiment even if it gives an attractive score.''')
    md('''### Recall first · the new axis

Without looking back at Lesson 046, write one sentence for each:
1. What information is initially present in `[CLS]`, and how does it acquire a row summary?
2. In feature self-attention, can changing another row change this row's output in evaluation mode?
3. Predict whether SAINT's answer will depend on the **order** of a fixed batch, its **membership**, both, or neither.

**Your prediction:** _write here before running the intervention._''')
    md(r'''## 1 · Concept recap: a batch becomes part of the input

A **token** is a learned vector for a feature; `[CLS]` is a learned summary slot, never the target label. Let $B$ be rows in the current batch, $T$ be tokens per row **including CLS**, and $d$ be coordinates per token.

Feature attention treats each row as a separate sequence: $[B,T,d]$. Each token can read other tokens from that same row. SAINT's **intersample attention** flattens every row into one vector of width $Td$ and treats all $B$ rows as one sequence: $[1,B,Td]$. The leading 1 means one sequence containing the whole batch; it does not mean there is only one data row.

The distinction is about **who can read whom**. Flattening preserves feature order. Transposing to attend down each column would leave each feature in a separate sequence and implement a different operator. After row attention, we restore $[B,T,d]$ so CLS and the feature tokens retain their slots.''')
    figure('feature-axis', 'Feature attention connects tokens within each separate row, keeping the batch axis independent.',
           '**Read the blue highlight:** one sequence per row. The drawing has B = 3 and T = 3 (CLS, age, job), so the input is `[3,3,d]` and the per-head attention matrix has shape `[3,H,3,3]`. The shaded matrix row marks the keys available to the CLS query; it does not show learned weights. See §3.2 / Algorithm 1.')
    figure('row-axis', 'Intersample attention packs each full row, then connects the rows in a single sequence.',
           '**Read the teal highlight:** each sequence element now contains every coordinate from one row. The same input becomes `[1,3,3*d]`, and the per-head attention matrix becomes `[1,H,3,3]`. Although both illustrated matrices happen to be 3×3, their axes mean different things: feature slots versus whole rows. No target labels are packed.')
    md(r'''### A worked attention calculation

A query $q$ asks what to retrieve; keys $k_j$ determine match scores; values $v_j$ carry the information. For one head,

$$s_{ij} = q_i^	op k_j / sqrt{d_h},\qquad A_{ij}=\frac{e^{s_{ij}}}{\sum_\ell e^{s_{i\ell}}},\qquad o_i=\sum_j A_{ij}v_j.$$

Here $d_h$ is **head width**, not batch size. Each row of $A$ is a probability distribution over candidate keys. For scores `[1,0]` and scalar values `[2,8]`, weights are approximately `[0.731,0.269]` and the output is `0.731×2 + 0.269×8 = 3.614`. If only the first item remains, its weight becomes 1 and its output becomes 2. The query did not change; its available context did.

This will matter at deployment: $f(x_i;\mathcal B)$ can depend on the companion set $\mathcal B$. An evaluation batch size is consequently part of the prediction protocol.''')
    for cell in bootstrap_cells():
        cells.append(nbf.v4.new_markdown_cell(cell['source']) if cell['cell_type'] == 'markdown'
                     else nbf.v4.new_code_cell(cell['source']))
    code('''# PROVIDED — setup; local: open from repo root, labs/, or labs/solutions/
import os, sys, json, copy, hashlib
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '1')
for candidate in (Path.cwd(), Path.cwd().parent, Path.cwd() / 'labs'):
    if (candidate / 'relkit').is_dir():
        LABS = candidate.resolve()
        sys.path.insert(0, str(LABS))
        break
else:
    raise RuntimeError('Open this notebook from the course repo or run the Colab bootstrap.')
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from relkit.saint_experiment import prepare, environment
from relkit.saint_report import score_table, paired_table, plot_scores, plot_ranks, plot_context
torch.set_num_threads(1)
torch.manual_seed(47)
plt.rcParams.update({'font.size': 11, 'figure.dpi': 110})
checks = {}
def check(name, condition, hint):
    checks[name] = bool(condition)
    assert checks[name], f'{name}: {hint}'
    print('PASS ·', name)
display(pd.Series(environment()['versions'], name='Installed version').to_frame())
print('Python:', environment()['python'], '| Device: CPU for the learning lab')''')
    md('''## 2 · Task 1 — pack and restore whole rows

**Goal:** make the two shape transformations in Algorithm 1 invertible while preserving every coordinate's owner.
**Why it matters:** a plausible shape can hide a feature/row permutation and train the wrong model.
**Hint boundary:** keep the row order and the within-row flattening order; use the supplied token count to recover the embedding width. Do not transpose columns into separate sequences.

Before running the check, predict where original coordinate `x[1,2,3]` will live in the packed tensor.''')
    todo('''# TODO — Task 1: two focused shape operations
def pack_rows(tokens):
    b, t, d = tokens.shape
    return ____

def unpack_rows(rows, n_tokens):
    return ____''', '# TODO — Task 1 (teacher solution)\n' + extract(MODEL, {'pack_rows', 'unpack_rows'}))
    code('''# CHECK — values, ownership, non-contiguous input, and gradients
x = torch.arange(60., requires_grad=True).reshape(3, 4, 5)
packed = pack_rows(x)
check('Packing shape', packed.shape == (1, 3, 20), 'The whole batch is one sequence of three rows.')
check('Coordinate ownership', torch.equal(packed[0, 1], x[1].flatten()), 'Keep all coordinates of row 1 together.')
check('Round trip', torch.equal(unpack_rows(packed, 4), x), 'Restore both feature and coordinate order.')
y = x.transpose(1, 2)
check('Non-contiguous input', torch.equal(unpack_rows(pack_rows(y), 5), y), 'The operator must preserve values even for a view.')
grad, = torch.autograd.grad(unpack_rows(packed, 4).square().sum(), x)
check('Packing preserves gradients', torch.equal(grad, 2*x), 'Reshaping must keep the autograd path.')
display(pd.DataFrame(packed.detach()[0].numpy(), index=['row 0', 'row 1', 'row 2']))''')
    md('''### Task 2 — normalize over candidate keys

**Goal:** implement the scaled scores and attention probabilities.
**Why it matters:** normalizing over queries can have a plausible shape but does not give each query a distribution over its companions.
**Hint boundary:** the final Q/K axis is head width. Scores need one axis for queries and one for keys. The test deliberately uses different query and key counts.

After your implementation passes, the provided multi-head module below will call **your function**.''')
    todo('''# TODO — Task 2: scores and probabilities
def attention_weights(q, k):
    scores = ____
    return ____''', '# TODO — Task 2 (teacher solution)\n' + extract(MODEL, {'attention_weights'}))
    code('''# CHECK — independent PyTorch attention kernel, unequal query/key counts
generator = torch.Generator().manual_seed(47)
q = torch.randn(2, 3, 4, 7, generator=generator, dtype=torch.float64, requires_grad=True)
k = torch.randn(2, 3, 6, 7, generator=generator, dtype=torch.float64)
v = torch.randn(2, 3, 6, 5, generator=generator, dtype=torch.float64)
a = attention_weights(q, k)
check('Attention shape', a.shape == (2, 3, 4, 6), 'Four queries each read six keys.')
check('Key probabilities sum to one', torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1))), 'Check the softmax axis.')
expected = F.scaled_dot_product_attention(q, k, v)
check('Independent kernel agreement', torch.allclose(a @ v, expected, atol=1e-12), 'Check the scale and key transpose.')
g1, = torch.autograd.grad((a @ v).square().sum(), q, retain_graph=True)
g2, = torch.autograd.grad(expected.square().sum(), q)
check('Attention gradient agreement', torch.allclose(g1, g2, atol=1e-11), 'A matching output must also train correctly.')''')
    md('''### PROVIDED · projected multi-head attention (§3.2)

`to_qkv` makes three projections and separates heads. Each head computes its own probability matrix; `a @ v` retrieves values; concatenation and `to_out` return to the input width. The **sequence length changes between feature and row attention**, while this module stays the same.

Read the shapes in order: `[batch, sequence, width] → [batch, heads, sequence, head_dim] → attention → [batch, sequence, width]`. In the release, feature heads have width 16 and row heads width 64. These head widths need not equal `d / heads`: the learned projections choose the internal width.''')
    provided({'Attention'}, '§3.2: multi-head attention uses your Task 2')
    md('''## 3 · Assemble a released-code SAINT stage

The diagram opens one stage: feature attention and feed-forward transformation, then row attention and feed-forward transformation. A feed-forward network transforms each sequence element independently; attention is what allows information exchange. The model restores feature slots before reading CLS.

**Predict:** if the row block is removed, which part of the companion effect must disappear?''')
    figure('architecture', 'SAINT architecture with category and numeric embeddings, CLS, feature and row sublayers, then a classification head.',
           '**Follow the tensor shapes, then read the code below.** This is the released supervised path used in this lab. The teal row stage operates at width `T*d`; the prediction head reads only the restored CLS slot.')
    md(r'''### Paper versus executable code · a choice that affects the function

A reproduction must state which source defines the operator. The pinned release and the PDF are not identical:

| Detail | PDF description | Pinned release / this notebook |
|---|---|---|
| Numeric embedding | Linear layer followed by ReLU | Per-feature `1 → 100 → d` MLP, with ReLU between layers |
| Residual/normalization | $x+\mathrm{LN}(F(x))$ | $u=\mathrm{LN}(x)$, then $u+F(u)$ |
| Feed-forward activation | Less explicit prose | GEGLU: split value and gate; multiply value by GELU(gate) |
| Attention dropout | Dropout is part of the configuration | Declared upstream but unused in attention forward; FF dropout is active |

**LayerNorm** normalizes the last dimension for each sequence element, then applies learned scale and bias. For feature attention this dimension is `d`; for row attention it is `T*d`. **GEGLU** expands width `w` to `8w`, splits it into two `4w` halves, gates them, and projects back to `w`.

The skip path below adds the **normalized** input. Replacing it with a familiar conventional residual block changes the model. `_check_l047.py` independently transplanted weights from the pinned official stage and recorded exact forward and input-gradient agreement; the notebook's later check compares your stage with that audited local implementation. Those are two distinct checks.''')
    provided({'GEGLU', 'ReleasedResidual', 'feedforward'}, 'Released gates and normalization; inspect the residual return value')
    md('''### Task 3 — wire the feature stage into the row stage

**Goal:** complete the forward pass of one whole SAINT stage.
**Why it matters:** implementing the attention equation is insufficient if it runs along the wrong axis, omits a sublayer, or fails to restore the token slots.
**Hint boundary:** run the feature sublayers first; preserve the token count before packing; each row sublayer consumes the packed representation. Restore the original representation afterward. The constructor is provided below; its `forward` is deliberately omitted.

`variant='col'` is the depth-matched feature-only control, `'row'` skips feature attention, and `'colrow'` runs both. The stage check exercises all three.''')
    provided({'SaintStage'}, 'Stage parameters only; you supply the actual forward pass next', omit_methods={'forward'})
    todo('''# TODO — Task 3: three missing steps in the real stage
def saint_stage_forward(self, x):
    for layer in self.col:
        x = ____
    if self.row:
        t = x.shape[1]
        x = ____
        for layer in self.row:
            x = layer(x)
        x = ____
    return x

SaintStage.forward = saint_stage_forward''', '''# TODO — Task 3 (teacher solution)
def saint_stage_forward(self, x):
    for layer in self.col:
        x = layer(x)
    if self.row:
        t = x.shape[1]
        x = pack_rows(x)
        for layer in self.row:
            x = layer(x)
        x = unpack_rows(x, t)
    return x

SaintStage.forward = saint_stage_forward''')
    code('''# CHECK — copied weights isolate implementation, rather than training randomness
from relkit.saint import SaintStage as AuditedStage  # checker only; never the trained model
for variant in ('col', 'row', 'colrow'):
    torch.manual_seed(47)
    student = SaintStage(4, 8, ff_dropout=0., variant=variant).double().eval()
    reference = AuditedStage(4, 8, ff_dropout=0., variant=variant).double().eval()
    reference.load_state_dict(student.state_dict())
    x = torch.randn(5, 4, 8, dtype=torch.float64, requires_grad=True)
    got, expected = student(x), reference(x)
    check(f'{variant}: stage output', torch.allclose(got, expected, atol=1e-11), 'Check sublayer order and reshape boundaries.')
    g1, = torch.autograd.grad(got.square().sum(), x, retain_graph=True)
    g2, = torch.autograd.grad(expected.square().sum(), x)
    check(f'{variant}: stage gradient', torch.allclose(g1, g2, atol=1e-10), 'Keep the same differentiable operator.')
print('Official-source parity is recorded separately in _check_l047_results.json.')''')
    md('''### PROVIDED · tokens, missing values, and the prediction head

Each categorical feature owns an embedding table; each numeric feature owns an MLP. Numeric NaNs select a learned **feature-specific missing token**. Categorical code 0 is reserved for missing/unseen values by our training-only encoder. These choices preserve missingness while keeping a fixed token layout.

Read `tokenize` first: **CLS → categorical tokens → numeric tokens**. `encode` applies your stages. `forward` takes position 0 and maps `d → 1000 → 2` logits. Softmax is applied later when probabilities are needed; training cross-entropy consumes logits directly.

The constructor below resolves the `SaintStage` you just completed. The local comparison and the notebook's Bank runner will train this model.''')
    provided({'SAINT'}, '§3 / Fig. 1: full supervised model, using your Task 3 stage')
    md('''### Inspect real data before training

We use the real German-credit table here to inspect **representations**, not to make a predictive claim. The loader fixes split seed 5, forms a stratified 65/15/20 split, fits numeric means/scales and category vocabularies on training rows, and records the exact row IDs. Validation/test features use those fitted transforms; their labels do not enter tokenization.

The PDF's stated 65/15/25 percentages sum to 105%; the release uses 65/15/20. Our local **stratified** split differs from the release's random split assignment. The Bank runner later uses the released assignment, with remaining gaps still declared.''')
    code('''# PROVIDED — inspect the actual fitted preprocessing and intermediate tensors
frame = prepare('credit_g')
meta = frame['meta']
display(pd.DataFrame({'Split': ['train', 'valid', 'test'],
                      'Rows': [len(frame[k]) for k in ('train', 'valid', 'test')]}))
display(pd.DataFrame({'Numeric feature': meta['numeric_columns'],
                      'Training mean': meta['train_numeric_mean'],
                      'Training scale': meta['train_numeric_std']}))
display(pd.DataFrame({'Categorical feature': meta['categorical_columns'],
                      'Training vocabulary size': [len(v) for v in meta['categorical_vocabularies']]}))
idx = frame['train'][:4]
xn = torch.tensor(frame['xn'][idx], dtype=torch.float32)
xc = torch.tensor(frame['xc'][idx], dtype=torch.long)
torch.manual_seed(47)
inspection_model = SAINT(xn.shape[1], frame['cards'], ff_dropout=0.).eval()
with torch.no_grad():
    tokens = inspection_model.tokenize(xn, xc)
    packed = pack_rows(tokens)
    encoded = inspection_model.encode(xn, xc)
    logits = inspection_model(xn, xc)
display(pd.DataFrame({'Tensor': ['numeric inputs', 'category codes', 'tokens', 'packed rows', 'encoded tokens', 'class logits'],
                      'Shape': list(map(lambda a: str(tuple(a.shape)), [xn, xc, tokens, packed, encoded, logits]))}))
check('Real token count', tokens.shape[1] == 1 + xn.shape[1] + xc.shape[1], 'Include CLS and every feature.')
check('Binary logits', logits.shape == (4, 2) and torch.isfinite(logits).all(), 'The head should produce two finite logits per row.')
print('Raw row IDs:', np.array(meta['row_ids'])[idx].tolist(), '| random untrained weights: no accuracy claim')''')
    md('''## 4 · Intervene on the context

**Goal:** separate permutation equivariance from independence of batch membership.
**Why it matters:** a model can treat row order symmetrically while still depending on which rows are present.

The figure uses a small synthetic weighted sum. Changing a companion's key changes its attention weight; the query remains fixed. In the code probe, we change one companion's full token representation and inspect the unchanged query row. We then permute the same batch and align the outputs again. Dropout is disabled and both stages are in evaluation mode, so stochastic dropout cannot explain a difference.''')
    figure('companion', 'Fixed query with self and companion scores, normalized weights, values, and a changed weighted output.',
           '**Synthetic mechanism illustration.** The weights sum to one over available keys. This explains a possible dependency; it is not a trained probability or an accuracy result.')
    code('''# PROVIDED + CHECK — hold the query fixed, manipulate only companions
torch.manual_seed(47)
x = torch.randn(5, 4, 8)
row = SaintStage(4, 8, ff_dropout=0.).eval()
col = SaintStage(4, 8, ff_dropout=0., variant='col').eval()
changed = x.clone()
changed[1] = torch.randn_like(x[1]) * 3  # row 0, our query, is untouched
with torch.no_grad():
    row_delta = (row(x)[0] - row(changed)[0]).abs().max().item()
    col_delta = (col(x)[0] - col(changed)[0]).abs().max().item()
    perm = torch.tensor([3, 1, 4, 0, 2])
    permutation_error = (row(x)[perm] - row(x[perm])).abs().max().item()
check('Companion changes row-attention output', row_delta > 1e-6, 'Trace whether row 0 can read row 1.')
check('Feature-only control is independent', col_delta < 1e-6, 'Feature attention must not mix different rows.')
check('Permutation equivariance', permutation_error < 1e-5, 'Reordering the same rows should reorder outputs.')
display(pd.DataFrame({'Intervention': ['Change companion / feature only', 'Change companion / feature + row', 'Permute same batch / align outputs'],
                      'Max absolute change': [col_delta, row_delta, permutation_error]}))''')
    md('''**Explain the intervention before continuing:** _Was your opening prediction right? Why does the permutation check not imply independence of batch membership? If a deployed query is evaluated alone, what needs to be recorded alongside the weights?_''')
    md('''### Task 4 · What does the contrastive objective actually pair?

The paper also proposes self-supervised pretraining (§4). It makes a corrupted view through **CutMix on raw feature values** (replace selected features with another row's) and **mixup in embedding space** (a convex combination of representations). The clean and augmented paths go through the encoder and projection heads to produce vectors `z` and `z_view`. Contrastive matching and feature reconstruction provide two distinct learning signals.

This notebook trains only the supervised path. The next task isolates the contrastive loss; it does **not** implement the full augmentation sampler, projection/denoising training, or the Table 3 experiment. Read the diagram to understand where the loss inputs would come from.''')
    figure('views', 'CutMix replaces selected raw features, then embedding mixup forms a convex combination for an augmented view.',
           '**Synthetic view construction, §4 / Fig. 1.** Follow the donor feature, then the weighted embedding coordinates. These values illustrate the transformations and are not learned representations from our benchmark.')
    md(r'''**Goal:** compute Eq. 5's one-way contrastive pairing term using no class labels.
**Why it matters:** the positive is the **same row's augmented view**, not an arbitrary example with the same target class.
**Hint boundary:** create the pairwise similarity matrix, scale by positive temperature $\tau$, and identify the corresponding view by its row index.

$$\ell_i=-\log\frac{\exp(z_i^\top z'_i/\tau)}{\sum_j \exp(z_i^\top z'_j/\tau)}.$$

We use the **mean** over rows, a batch-size-independent rescaling of the paper's sum. No vector normalization is implicit in this exercise; the released pretraining code also offers a symmetric, normalized variant. Decreasing temperature sharpens a fixed similarity distribution; it does not magically correct a wrong positive pairing.

**Worked baseline:** with four identical candidate scores, each candidate gets probability 1/4 and loss is $-\log(1/4)=\log 4\approx1.386$. Denoising is a separate objective: reconstruct original features from the corrupted view. Class labels enter neither of these self-supervised targets.''')
    figure('contrast', 'Contrastive similarity matrix with diagonal same-row positives and a separate feature-reconstruction branch.',
           '**Read the diagonal:** one correct augmented view per clean row. Off-diagonal candidates compete in the softmax. The reconstruction branch is conceptually separate and is not trained in this lab.')
    todo('''# TODO — Task 4: paired-view contrastive loss
def info_nce(z, z_view, temperature=.7):
    logits = ____
    return ____''', '# TODO — Task 4 (teacher solution)\n' + extract(MODEL, {'info_nce'}))
    code('''# CHECK — pairing, uniform baseline, and differentiability
z = torch.eye(4, requires_grad=True)
check('Correct pairing beats a shuffled view', info_nce(z, z) < info_nce(z, z.roll(1, 0)), 'The positive view keeps the same row index.')
uniform_loss = float(info_nce(torch.zeros(4, 3), torch.zeros(4, 3)))
check('Uniform loss = log(B)', abs(uniform_loss - np.log(4)) < 1e-6, 'Use a mean cross-entropy over candidates.')
loss = info_nce(z, z)
grad, = torch.autograd.grad(loss, z)
check('Contrastive gradient is finite and nonzero', torch.isfinite(grad).all() and grad.abs().max() > 0, 'Keep the loss as a differentiable tensor.')
print(f'Uniform baseline: {uniform_loss:.4f}; matching identity vectors: {loss.item():.4f}')''')
    code('''# PROVIDED — see how temperature sharpens the SAME pairing
z = torch.eye(4)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), layout='constrained')
for ax, temperature in zip(axes, (.2, .7, 2.)):
    probabilities = (z @ z.T / temperature).softmax(-1).numpy()
    ax.imshow(probabilities, vmin=0, vmax=1, cmap='Blues')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{probabilities[i,j]:.2f}', ha='center', va='center',
                    color='white' if probabilities[i,j] > .55 else '#183049')
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xlabel('Candidate view j'); ax.set_ylabel('Clean row i')
    ax.set_title(f'τ = {temperature} · loss {info_nce(z,z,temperature).item():.3f}')
fig.suptitle('Synthetic identity vectors · same-row positives lie on the diagonal', fontsize=13)
plt.show()''')
    md('''**Interpret, then transfer:** _Why is a class label the wrong target for this matrix? Which blocks and experiment are still missing before you could claim to reproduce SAINT pretraining?_''')
    md('''## 5 · Train the model you built

**Read the evaluator before the trainer.** `predict_saint` makes ascending-index batches within the supplied split, converts logits to positive-class probabilities, and never takes labels. Changing its `batch_size` changes companion membership. We keep validation and test rows in separate prediction calls.

The training loop then shuffles **training rows only**, minimizes supervised cross-entropy, and selects the best checkpoint using validation AUROC. It uses the whole fixed epoch budget rather than early stopping. The model seed must be set **before construction** as well as inside the training loop: reseeding after initialization cannot recover identical starting weights.''')
    provided({'predict_saint'}, 'Within-split evaluation; the companion policy is explicit')
    md('''### PROVIDED · supervised optimization and checkpoint selection

Follow one epoch: permute training indices → form a batch → forward → cross-entropy → backward → optimizer step → evaluate validation split → retain best weights. `best` is a deep copy so later updates cannot silently alter the saved model. Test AUROC is measured only after checkpoint selection.

There is no contrastive pretraining call in this loop. Task 4's function is used by its own loss checks/visualization; Tasks 1–3 are used by the supervised model.''')
    provided({'train_saint'}, 'Visible supervised training loop; no model implementation is imported')
    md('''### The comparison frame · freeze it before looking at test scores

| Choice | Local protocol |
|---|---|
| Data and split | 3 real Tier-A tables; split seed 5; stratified 65/15/20; identical splits across arms |
| Preprocessing | Train-only numeric means/scales and categorical vocabulary; shared prepared inputs |
| Neural models | Feature only vs feature + row; d=8, depth=1, 4 heads, FF dropout .1 |
| Training | Model seeds 0/1/2, 20 epochs, AdamW lr=.001, weight decay=.01, batch 64 |
| Selection | Validation AUROC after each epoch, restore best weights |
| CatBoost | 300 iterations, depth 6, lr=.05, same seeds/splits, best validation-AUC model |
| Test context | Ascending row index within the test split, fixed batches of 64, no labels in forward |
| Context audit | Hold trained weights fixed; compare batch 64 with singleton evaluation |

**What is controlled:** feature-only and combined neural models share depth and the training budget. **What also changes:** adding the row block adds parameters and compute. Inspect their counts below; this is not a parameter-matched causal estimate of attention alone. The feature-only arm is not the paper's six-stage SAINT-s. CatBoost is a budgeted, untuned external baseline, not the best possible tuned tree.

**Uncertainty:** sample SD describes variation over three model seeds on one fixed split. The reported 95% seed intervals are conditional on that split and fragile at n=3; they are not population confidence intervals. Cross-dataset ranks use one seed-mean score per model per dataset, not nine independent datasets.''')
    code('''# PROVIDED — make the added capacity visible on the inspected real table
counts = []
for variant in ('col', 'colrow'):
    torch.manual_seed(0)
    candidate = SAINT(frame['xn'].shape[1], frame['cards'], variant=variant)
    counts.append({'Model': variant, 'Trainable parameters': sum(p.numel() for p in candidate.parameters() if p.requires_grad)})
display(pd.DataFrame(counts))''')
    md('''**Pre-register your expectation:** _Which dataset, if any, do you expect row attention to help? What result would make you keep the feature-only model? Name a context-policy detail you must hold fixed._''')
    code(r'''# PROVIDED — the harness injects YOUR visible model, trainer, and predictor
from _verify_l047 import run
result = run(model_cls=SAINT, train_fn=train_saint, predict_fn=predict_saint)
# Save both metrics/protocol and the executed cell source that produced them.
executed_source = '\n\n# --- next executed cell ---\n\n'.join(get_ipython().history_manager.input_hist_raw[1:])
source_hash = hashlib.sha256(executed_source.encode()).hexdigest()
artifact_dir = Path('l047_student_artifacts')
artifact_dir.mkdir(exist_ok=True)
(artifact_dir / 'executed_cells.py').write_text(executed_source)
result['notebook_source_sha256'] = source_hash
(artifact_dir / 'results.json').write_text(json.dumps(result, indent=2))
print(f'Saved to {artifact_dir.resolve()} · training elapsed {result["wall_s"]:.1f}s')''')
    code('''# CHECK — completeness, protocol boundaries, and the independent-row control
check('Three datasets', len(result['results']['per_dataset']) == 3, 'Run the full local comparison.')
for dataset, arms in result['results']['per_dataset'].items():
    check(f'{dataset}: three seeds per model', all(len(a['scores']) == 3 for a in arms.values()), 'Do not turn seed runs into separate datasets.')
for dataset, protocol in result['protocols'].items():
    tr, va, te = [set(protocol[k]) for k in ('train', 'valid', 'test')]
    check(f'{dataset}: all split boundaries', not (tr & va or tr & te or va & te), 'Training, validation, and test must be disjoint.')
controls = [p['max_abs_probability_change'] for name, p in result['context_probe'].items() if name.endswith('/col')]
check('Trained feature-only context control', max(controls) < 1e-5, 'Independent-row predictions should be unchanged within floating-point tolerance.')
display(score_table(result).round(4))''')
    md('''### Read your result · first within each dataset

The chart shows each measured seed, its mean, and ±1 **sample SD**. The common AUROC axis makes differences visible without pooling datasets. AUROC measures positive/negative ranking quality; it does not establish calibrated probabilities or deployment utility.

The paired table subtracts feature-only AUROC from feature+row AUROC at the same seed. Positive means the combined model did better in that pair. Pairing aligns the experiment's seed labels; the architectures have different parameter sets, so it does not make their initial weights identical. Look for disagreement across seeds before summarizing a mean gap.''')
    code('''# PROVIDED — measured results, not a prewritten winner
fig = plot_scores(result)
fig.savefig(artifact_dir / 'auroc.png', dpi=150, facecolor='white')
plt.show()
display(paired_table(result).pivot(index='Dataset', columns='Seed', values='Row − feature AUROC').round(4))''')
    md('''### Then across datasets · ranks and a critical difference

Rank the **seed-mean** AUROC within each dataset (1 = best), then average ranks. This gives datasets equal weight rather than allowing a large table to dominate. Friedman tests for an overall rank difference; Nemenyi compares pairwise mean-rank gaps with its critical difference (CD).

Use the CD bar on the same rank scale as the points. If a gap is smaller than the CD, this comparison has not distinguished that pair at the chosen threshold. With only three datasets, lack of a detected difference is unsurprising and **does not prove equivalence**. The chart must report the measured result, even when SAINT does not lead.''')
    code('''# PROVIDED — cross-dataset evidence and its limited resolution
fig = plot_ranks(result)
fig.savefig(artifact_dir / 'ranks.png', dpi=150, facecolor='white')
plt.show()
summary = result['results']
print(f'Friedman statistic={summary["friedman"]["statistic"]:.3f}, p={summary["friedman"]["p"]:.4f}; CD={summary["nemenyi_cd"]:.3f}')''')
    md('''### Audit the trained model's context sensitivity

Every dot below belongs to one already-trained model. For the **same test rows and weights**, the harness evaluates batches of 64 and batches of 1 and reports the largest absolute probability change. This is a sensitivity diagnostic, not an estimate of which context is optimal. We do not choose a batch size using these test results.

A large probability shift can coexist with a small AUROC change because AUROC depends on ordering. Inspect both. For an operational evaluation, predeclare batch size, ordering/grouping, companion source, split boundaries, and the rule for a final incomplete batch.''')
    code('''# PROVIDED — fixed-weights context report
fig = plot_context(result)
fig.savefig(artifact_dir / 'context.png', dpi=150, facecolor='white')
plt.show()
probe = result['context_probe']['diabetes/0/colrow']
display(pd.DataFrame({'Context': ['batch 64', 'singleton'],
                      'Diabetes seed 0 test AUROC': [probe['auc_batch64'], probe['auc_batch1']]}))
print(f'Maximum probability shift: {probe["max_abs_probability_change"]:.6f}')''')
    md('''### Author reference run · evidence available before you execute

The next three panels are **embedded snapshots of the committed author run**, not outputs from your kernel. They were generated from `_verify_l047_results.json` using the protocol above. Your report is the output of the preceding cells. Exact agreement depends on versions and environment; a difference calls for a protocol/code comparison, not editing scores.

In this recorded run, the combined model improves the seed-mean AUROC on credit_g, while feature-only leads on diabetes and blood_transfusion. The neural arms tie in mean rank (1.67); CatBoost's mean rank is 2.67. That is a description of this small experiment, not a universal model ranking.''')
    figure('scores', 'Recorded author run: per-dataset AUROC dots for three seeds, means, and sample-standard-deviation bars.',
           '**Measured reference · fixed split, seeds 0/1/2.** Credit_g: feature only 0.7617 ± 0.0173, feature + row 0.7873 ± 0.0153; diabetes: 0.7978 ± 0.0063 vs 0.7911 ± 0.0057; blood transfusion: 0.7576 ± 0.0139 vs 0.7558 ± 0.0084. These are means ± sample SD, not paper results.')
    figure('ranks', 'Recorded author run mean ranks, Friedman p value, and Nemenyi critical-difference bar.',
           '**Measured reference · three datasets.** Friedman p = 0.368; CD = 1.914. The largest mean-rank gap is 1.00, below the CD. This sample does not distinguish the models statistically and does not establish equivalence.')
    figure('context-results', 'Recorded author run: changing evaluation batches from 64 to one changes combined-model probabilities but not feature-only predictions.',
           '**Measured reference · same trained weights.** Diabetes seed 0 has maximum probability change 0.2673 and AUROC 0.7920 → 0.7648. Feature-only predictions stay unchanged within numerical tolerance. This does not authorize selecting a test-time batch size from test scores.')
    md('''## EXIT TICKET · your evidence and your explanation

Run the cell, then answer these prompts in your own words. Automated checks establish implementation behavior; they cannot grade your understanding.

1. **Axes:** explain what one element of `[1,B,T*d]` contains and why a column-wise transpose is a different model.
2. **Intervention:** explain the changed-companion and same-batch-permutation results. State the context policy you would put in a model card.
3. **Comparison:** report one dataset's mean ± SD and the paired-seed gaps. Interpret the ranks, Friedman p, and CD without assuming a winner or equivalence.
4. **Paper audit:** name one PDF/release discrepancy, what you implemented, and one reason the local result cannot verify the paper's Table 2.
5. **Pretraining:** identify the correct positive in Task 4 and what a full Table 3 reproduction still needs.

**Your explanation:** _write here; paste it with EXIT output to your teacher. Ask about any step that remains unclear._''')
    code('''# EXIT — verifiable evidence with no enforced winning model
assert all(checks.values()), 'Fix a failed CHECK before submitting.'
print(f'Implementation/protocol checks: {sum(checks.values())}/{len(checks)} passed')
display(score_table(result).round(4))
print('Mean ranks:', result['results']['mean_ranks'])
print('Friedman:', result['results']['friedman'], '| Nemenyi CD:', result['results']['nemenyi_cd'])
print('Diabetes seed 0 context shift:', probe['max_abs_probability_change'])
print('Executed source SHA256:', source_hash)
print('Artifacts:', artifact_dir.resolve())
print('Scope: local supervised ablation; isolated contrastive loss; full pretraining NOT_RUN.')
print('Full paper-scale Bank: NOT_RUN by this core lab; continue below.')''')
    md('''## 6 · REQUIRED NEXT STEP — attempt the paper's supervised Bank result

The target is **Table 2, Bank, supervised SAINT AUROC 0.9330**, a five-trial mean. It is not the 14-binary-task mean 0.9313. Table 6 reports **SE 0.0009**, not SD. An absolute tolerance of ±0.01 is a declared reproduction target only after the protocols agree. A numerically close result with unresolved protocol gaps remains **INCOMPARABLE**.

| Preset | Data / epochs / seeds | Width / heads / batch | Purpose |
|---|---|---|---|
| `smoke` | 600 Bank rows / 1 / 1 | 4 / 2 / 32 | Confirm data, training, checkpoint, and artifact plumbing |
| `closer` | Full Bank / 20 / 3 | 16 / 4 / 256 | A larger measured attempt |
| `paper` | Full Bank / 100 / 5 | 32 / 8 / 256 | Adopt paper budgets while retaining declared gaps |

The committed **smoke** run reached AUROC 0.742204 and is explicitly INCOMPARABLE. Full `closer` and `paper` results have not been measured here. Their runtime is unmeasured; start with smoke and budget from that timing. Full augmentation/denoising pretraining is still absent.

The source below is split into the **target/configuration ledger** and the **runner**. Read `GAPS` before running. The runner calls the same SAINT class and training functions visible above, saves best weights and test predictions, and prints three separate buckets: verified local findings, paper claim, and scale-up result.''')
    paper_source = (HERE / '_paper_repro_l047.py').read_text()
    # Keep imports/config verbatim, but never overwrite the student's model functions.
    tree = ast.parse(paper_source)
    main_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    prelude = paper_source.splitlines()[:main_node.lineno - 1]
    prelude = [line for line in prelude if not line.startswith('from __future__') and not line.startswith('from relkit.saint import')]
    code('# PROVIDED — Bank target, presets, and explicit protocol gaps\n' + '\n'.join(prelude).strip())
    md('''### Read the runner as an experiment audit

The Bank track uses the released split assignment (65/15/20, split seed 5). Unlike the local learning comparison, its checkpoint choice follows the released binary trainer: **validation accuracy**, checked at zero-based epochs 0, 5, 10, … (reported epochs 1, 6, 11, …). Test AUROC remains the final metric. Do not silently transfer the local validation-AUC protocol to this attempt.

Completed seeds resume only when their recorded identity matches configuration, source files, data/splits, and environment; interrupted seeds restart. Saved best weights support inference, not a full optimizer-state resume. The notebook gate also saves executed cell source and separates output directories by its digest, so editing student code cannot reuse a previous notebook run's completed seeds.

Inspect `protocol.json`, the per-seed histories, prediction row IDs, and the result ledger before comparing the final number with 0.9330.''')
    code('# PROVIDED — full Bank runner; uses the model and loops you built above\n' + extract(paper_source, {'main'}))
    md('''### Deliberate compute · persist the artifacts

Set `RUN_PAPER_REPRO=True` after choosing a budget. Start with `PRESET='smoke'`, then run `closer`. For Colab, attach a GPU and mount Drive before setting `OUTPUT_DIR` to a persistent directory. Free Colab may disconnect during unattended work; use the supplied Modal runner for longer runs:

```bash
modal run --detach modal/l047_paper_repro.py --preset closer
modal volume get relational-artifacts l047/closer ./l047-bank-results
```

Modal runs the canonical from-scratch implementation from the repo; the gated notebook trains your checked inlined implementation. Both use the same architecture/protocol code. The site/notebook must be published before a fresh Colab clone can obtain local changes.

**After the run:** compare the measured mean and uncertainty with the target, enumerate unresolved `GAPS`, and report MATCH/CLOSE/FAIL only if protocol agreement permits it. Otherwise report INCOMPARABLE with the measured number. Until a run executes, its status is NOT_RUN.''')
    code(r'''# PROVIDED — explicit gate; saved source distinguishes notebook implementations
RUN_PAPER_REPRO = False
PRESET = 'closer'  # first deliberate run: change to 'smoke'
OUTPUT_DIR = 'l047_artifacts'  # Colab: use a mounted persistent Drive directory
if RUN_PAPER_REPRO:
    # Re-running identical cells must not create a new run identity or retrain completed seeds.
    notebook_source = '\n\n# --- next executed cell ---\n\n'.join(dict.fromkeys(get_ipython().history_manager.input_hist_raw[1:]))
    notebook_hash = hashlib.sha256(notebook_source.encode()).hexdigest()
    run_dir = Path(OUTPUT_DIR) / f'notebook-{notebook_hash[:16]}'
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'executed_cells.py').write_text(notebook_source)
    paper_result = main(['--preset', PRESET, '--output-dir', str(run_dir)])
else:
    print('NOT_RUN: paper-scale Bank. Start with smoke, then run closer with persistent storage.')''')
    md('''### Revisit after a delay

Tomorrow, without reading this notebook, sketch the two attention axes and state whether shuffling a fixed batch and replacing a batch member are the same intervention. Then name the minimum artifacts needed to regenerate one Bank number. Reopen only the section needed to correct your answer.

**Figure provenance:** `_figures_l047.py` regenerates mechanism diagrams from the lesson's SVG components and author-reference plots from measured JSON; `figures/l047/provenance.json` records hashes. The PNG attachments travel inside this notebook, so understanding it does not depend on running JavaScript or finding an adjacent image folder.''')
    for index, cell in enumerate(cells):
        cell['id'] = f'l047-{index:03d}'
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
        'language_info': {'name': 'python'},
        'lesson': 47, 'canonical_model_sha256': hashlib.sha256(MODEL.encode()).hexdigest(),
        'figure_provenance': json.loads((FIGURES / 'provenance.json').read_text())})
    out = HERE / ('solutions' if solution else '') / '0047-saint.ipynb'
    out.parent.mkdir(exist_ok=True)
    nbf.validate(nb)
    nbf.write(nb, out)
    print(f'{out}: {len(cells)} cells, {sum(bool(c.get("attachments")) for c in cells)} embedded figures')
    if not solution:
        from nbconvert import HTMLExporter
        rendered = copy.deepcopy(nb)
        # The prepared site lives one directory deeper than the downloadable notebook.
        for cell in rendered.cells:
            if cell.cell_type == 'markdown':
                cell.source = cell.source.replace('](../lessons/', '](../../lessons/').replace(
                    '](../reference/', '](../../reference/')
        html, _ = HTMLExporter(template_name='lab').from_notebook_node(rendered)
        html_path = HERE / 'html/0047-saint.html'
        html_path.parent.mkdir(exist_ok=True)
        html_path.write_text(html)
        print(html_path)


if __name__ == '__main__':
    build()
    build(True)
