# @colab-bootstrap — PROVIDED. Makes the lab self-sufficient on Google Colab; a no-op elsewhere.
import os, sys

if "google.colab" in sys.modules:
    if not os.path.isdir("/content/relational"):
        !git clone --depth 1 https://github.com/Avistian/relational.git /content/relational
    %pip install -q -r /content/relational/requirements-labs.txt
    os.chdir("/content/relational/labs")
    print("Colab ready — working dir:", os.getcwd())
else:
    print("Not on Colab — using the local environment as-is.")

# --- next executed cell ---

# PROVIDED — setup; local: open from repo root, labs/, or labs/solutions/
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
print('Python:', environment()['python'], '| Device: CPU for the learning lab')

# --- next executed cell ---

# TODO — Task 1 (teacher solution)
def pack_rows(tokens):
    """Algorithm 1: [B,T,d] → [1,B,T*d], including CLS; no column transpose."""
    b, t, d = tokens.shape
    return tokens.reshape(1, b, t * d)


def unpack_rows(rows, n_tokens):
    """Inverse reshape; preserve feature/embedding order."""
    return rows.reshape(rows.shape[1], n_tokens, rows.shape[-1] // n_tokens)

# --- next executed cell ---

# CHECK — values, ownership, non-contiguous input, and gradients
x = torch.arange(60., requires_grad=True).reshape(3, 4, 5)
packed = pack_rows(x)
check('Packing shape', packed.shape == (1, 3, 20), 'The whole batch is one sequence of three rows.')
check('Coordinate ownership', torch.equal(packed[0, 1], x[1].flatten()), 'Keep all coordinates of row 1 together.')
check('Round trip', torch.equal(unpack_rows(packed, 4), x), 'Restore both feature and coordinate order.')
y = x.transpose(1, 2)
check('Non-contiguous input', torch.equal(unpack_rows(pack_rows(y), 5), y), 'The operator must preserve values even for a view.')
grad, = torch.autograd.grad(unpack_rows(packed, 4).square().sum(), x)
check('Packing preserves gradients', torch.equal(grad, 2*x), 'Reshaping must keep the autograd path.')
display(pd.DataFrame(packed.detach()[0].numpy(), index=['row 0', 'row 1', 'row 2']))

# --- next executed cell ---

# TODO — Task 2 (teacher solution)
def attention_weights(q, k):
    """§3: normalize over KEYS, separately for every query and head."""
    return (q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5).softmax(dim=-1)

# --- next executed cell ---

# CHECK — independent PyTorch attention kernel, unequal query/key counts
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
check('Attention gradient agreement', torch.allclose(g1, g2, atol=1e-11), 'A matching output must also train correctly.')

# --- next executed cell ---

# PROVIDED — §3.2: multi-head attention uses your Task 2
class Attention(nn.Module):
    def __init__(self, width, heads=4, head_dim=16):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.to_qkv = nn.Linear(width, 3 * heads * head_dim, bias=False)
        self.to_out = nn.Linear(heads * head_dim, width)

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = [v.reshape(b, t, self.heads, self.head_dim).transpose(1, 2)
                   for v in self.to_qkv(x).chunk(3, dim=-1)]
        a = attention_weights(q, k)
        return self.to_out((a @ v).transpose(1, 2).reshape(b, t, -1))

# --- next executed cell ---

# PROVIDED — Released gates and normalization; inspect the residual return value
class GEGLU(nn.Module):
    def forward(self, x):
        value, gate = x.chunk(2, dim=-1)
        return value * F.gelu(gate)


class ReleasedResidual(nn.Module):
    """Mirror the executable nesting, not a conventional PreNorm assumption."""
    def __init__(self, width, fn):
        super().__init__()
        self.norm, self.fn = nn.LayerNorm(width), fn

    def forward(self, x):
        u = self.norm(x)
        return u + self.fn(u)


def feedforward(width, dropout):
    return nn.Sequential(nn.Linear(width, 8 * width), GEGLU(),
                         nn.Dropout(dropout), nn.Linear(4 * width, width))

# --- next executed cell ---

# PROVIDED — Stage parameters only; you supply the actual forward pass next
class SaintStage(nn.Module):
    """Fig.1a: feature attention/FF, then entire-row attention/FF."""
    def __init__(self, tokens, d, heads=4, ff_dropout=0.1, variant='colrow'):
        super().__init__()
        if variant not in ('col', 'row', 'colrow'):
            raise ValueError(variant)
        self.variant = variant
        self.col = nn.ModuleList([
            ReleasedResidual(d, Attention(d, heads, 16)),
            ReleasedResidual(d, feedforward(d, ff_dropout)),
        ]) if variant != 'row' else nn.ModuleList()
        width = tokens * d
        self.row = nn.ModuleList([
            ReleasedResidual(width, Attention(width, heads, 64)),
            ReleasedResidual(width, feedforward(width, ff_dropout)),
        ]) if variant != 'col' else nn.ModuleList()

# --- next executed cell ---

# TODO — Task 3 (teacher solution)
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

SaintStage.forward = saint_stage_forward

# --- next executed cell ---

# CHECK — copied weights isolate implementation, rather than training randomness
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
print('Official-source parity is recorded separately in _check_l047_results.json.')

# --- next executed cell ---

# PROVIDED — §3 / Fig. 1: full supervised model, using your Task 3 stage
class SAINT(nn.Module):
    """Released supervised path: CLS, cats, numeric MLPs, blocks, CLS→1000→2.
    Missing numeric inputs are NaN and get feature-specific learned mask tokens;
    categorical code 0 is the missing/unseen token reserved by our train-only encoder.
    """
    def __init__(self, n_num, cards, d=8, depth=1, heads=4,
                 ff_dropout=0.1, variant='colrow'):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, d))
        self.cats = nn.ModuleList([nn.Embedding(c, d) for c in cards])
        self.nums = nn.ModuleList([nn.Sequential(nn.Linear(1, 100), nn.ReLU(),
                                                nn.Linear(100, d)) for _ in range(n_num)])
        self.missing_num = nn.Parameter(torch.randn(n_num, d))
        t = 1 + n_num + len(cards)
        self.stages = nn.ModuleList([SaintStage(t, d, heads, ff_dropout, variant)
                                     for _ in range(depth)])
        self.head = nn.Sequential(nn.Linear(d, 1000), nn.ReLU(), nn.Linear(1000, 2))

    def tokenize(self, x_num, x_cat):
        b = len(x_num)
        parts = [self.cls.expand(b, -1, -1)]
        parts += [emb(x_cat[:, j]).unsqueeze(1) for j, emb in enumerate(self.cats)]
        for j, mlp in enumerate(self.nums):
            value = x_num[:, j:j+1]
            embedded = mlp(torch.nan_to_num(value))
            parts.append(torch.where(torch.isnan(value), self.missing_num[j], embedded).unsqueeze(1))
        return torch.cat(parts, dim=1)

    def encode(self, x_num, x_cat):
        x = self.tokenize(x_num, x_cat)
        for stage in self.stages:
            x = stage(x)
        return x

    def forward(self, x_num, x_cat):
        return self.head(self.encode(x_num, x_cat)[:, 0])

# --- next executed cell ---

# PROVIDED — inspect the actual fitted preprocessing and intermediate tensors
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
print('Raw row IDs:', np.array(meta['row_ids'])[idx].tolist(), '| random untrained weights: no accuracy claim')

# --- next executed cell ---

# PROVIDED + CHECK — hold the query fixed, manipulate only companions
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
                      'Max absolute change': [col_delta, row_delta, permutation_error]}))

# --- next executed cell ---

# TODO — Task 4 (teacher solution)
def info_nce(z, z_view, temperature=0.7):
    """§4 eq.5 contrastive term, mean instead of sum; no implicit normalization.
    The paired original/view is at the same row index. This is one-way eq.5;
    the released pretraining code also offers a symmetric normalized variant.
    """
    logits = z @ z_view.T / temperature
    return F.cross_entropy(logits, torch.arange(len(z), device=z.device))

# --- next executed cell ---

# CHECK — pairing, uniform baseline, and differentiability
z = torch.eye(4, requires_grad=True)
check('Correct pairing beats a shuffled view', info_nce(z, z) < info_nce(z, z.roll(1, 0)), 'The positive view keeps the same row index.')
uniform_loss = float(info_nce(torch.zeros(4, 3), torch.zeros(4, 3)))
check('Uniform loss = log(B)', abs(uniform_loss - np.log(4)) < 1e-6, 'Use a mean cross-entropy over candidates.')
loss = info_nce(z, z)
grad, = torch.autograd.grad(loss, z)
check('Contrastive gradient is finite and nonzero', torch.isfinite(grad).all() and grad.abs().max() > 0, 'Keep the loss as a differentiable tensor.')
print(f'Uniform baseline: {uniform_loss:.4f}; matching identity vectors: {loss.item():.4f}')

# --- next executed cell ---

# PROVIDED — see how temperature sharpens the SAME pairing
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
plt.show()

# --- next executed cell ---

# PROVIDED — Within-split evaluation; the companion policy is explicit
@torch.no_grad()
def predict_saint(model, xn, xc, batch_size=64, device='cpu'):
    """Within-split sequential batches. Their order/membership are model inputs.
    Never mix validation and test rows, and never pass labels into forward.
    """
    model.eval()
    out = []
    for start in range(0, len(xn), batch_size):
        num = torch.as_tensor(xn[start:start+batch_size], dtype=torch.float32, device=device)
        cat = torch.as_tensor(xc[start:start+batch_size], dtype=torch.long, device=device)
        out.append(model(num, cat).softmax(-1)[:, 1].cpu().numpy())
    return np.concatenate(out)

# --- next executed cell ---

# PROVIDED — Visible supervised training loop; no model implementation is imported
def train_saint(model, xn, xc, y, train, valid, *, seed=0, epochs=20,
                batch_size=64, lr=1e-3, device='cpu', checkpoint=None,
                select_metric='auc', validate_every=1):
    """Supervised AdamW, validation-only selection, checkpoint best weights.
    Call torch.manual_seed BEFORE constructing the model as well as here.
    No early stopping: a fixed budget makes the local ablation interpretable.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    num = torch.as_tensor(xn, dtype=torch.float32, device=device)
    cat = torch.as_tensor(xc, dtype=torch.long, device=device)
    labels = torch.as_tensor(y, dtype=torch.long, device=device)
    best, best_score, history = None, -float('inf'), []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train)
        for start in range(0, len(order), batch_size):
            idx = order[start:start+batch_size]
            optimizer.zero_grad()
            loss = F.cross_entropy(model(num[idx], cat[idx]), labels[idx])
            loss.backward()
            optimizer.step()
        if epoch % validate_every == 0:
            p = predict_saint(model, xn[valid], xc[valid], batch_size, device)
            score = (float(np.mean((p >= .5) == y[valid])) if select_metric == 'accuracy'
                     else float(roc_auc_score(y[valid], p)))
            history.append({'epoch': epoch + 1, 'valid_score': score})
            if score > best_score:
                best_score, best = score, copy.deepcopy(model.state_dict())
                if checkpoint:
                    torch.save({'weights': best, 'epoch': epoch + 1, 'seed': seed,
                                'valid_score': score}, checkpoint)
    model.load_state_dict(best)
    return model, history

# --- next executed cell ---

# PROVIDED — make the added capacity visible on the inspected real table
counts = []
for variant in ('col', 'colrow'):
    torch.manual_seed(0)
    candidate = SAINT(frame['xn'].shape[1], frame['cards'], variant=variant)
    counts.append({'Model': variant, 'Trainable parameters': sum(p.numel() for p in candidate.parameters() if p.requires_grad)})
display(pd.DataFrame(counts))

# --- next executed cell ---

# PROVIDED — the harness injects YOUR visible model, trainer, and predictor
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
print(f'Saved to {artifact_dir.resolve()} · training elapsed {result["wall_s"]:.1f}s')