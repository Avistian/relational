"""Build L054 student + solution notebooks: visible TabM code, portable figures."""
import ast, base64, os
from pathlib import Path
from urllib.parse import urlsplit
import nbformat as nbf
from bs4 import BeautifulSoup
from _colab import bootstrap_cells
from _build_l049 import mdtext
ROOT = Path(__file__).resolve().parent; SLUG = '0054-tabm-parameter-efficient-ensembling'
SOURCE = (ROOT / 'relkit/tabm.py').read_text(); EXPERIMENT = (ROOT / 'relkit/tabm_experiment.py').read_text()
LESSON = BeautifulSoup((ROOT.parent / 'lessons' / f'{SLUG}.html').read_text(), 'html.parser')


def extract(source, names):
    tree = ast.parse(source)
    return '\n\n'.join(ast.get_source_segment(source, n) for n in tree.body
                       if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)


def build(solution=False, write=True):
    cells = []
    def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
    def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))
    def figure(name, caption):
        md('![' + caption + '](data:image/png;base64,' +
           base64.b64encode((ROOT / 'figures/l054' / f'{name}.png').read_bytes()).decode() + ')\n\n' + caption)
    def section(name):
        node = BeautifulSoup(str(LESSON.find('section', id=name)), 'html.parser').find('section')
        for el in node.find_all(['figure', 'script']): el.decompose()
        for el in node.find_all('div', id=True): el.decompose()
        for link in node.find_all('a', href=True):
            parts = urlsplit(link['href'])
            if not parts.scheme and parts.path:
                link['href'] = os.path.relpath((ROOT.parent / 'lessons' / parts.path).resolve(), ROOT) + \
                    ('#' + parts.fragment if parts.fragment else '')
        md(mdtext(node))
    def task(name, signature, prompt):
        md(prompt)
        code('# TODO — ' + name + '\n' + (extract(SOURCE, {name}) if solution
             else signature + '\n    raise NotImplementedError("Implement this live function")'))

    md('''# Lab 054 · TabM & parameter-efficient ensembling

[Lesson](../lessons/0054-tabm-parameter-efficient-ensembling.html) · [Reference](../reference/tabm-parameter-efficient-ensembling.html)

**Skill:** implement the numeric TabM-mini (a BatchEnsemble of MLPs) and compare four deployment procedures — a single MLP, a deep ensemble MLP×k, TabM-mini, and tuned XGBoost — while measuring the collective-versus-individual submodel effect. **Scope:** numeric forward path only; no feature embeddings or categorical inputs, and not the paper's 46-dataset benchmark. Four TODO functions feed the visible model and training code. PROVIDED = read/run; CHECK = immediate diagnostic feedback; EXIT = explain your result.

**Reproducibility contract:** real California Housing, House 16H and Higgs Small data from the TabR release, not the TabM benchmark splits. Preserve released boundaries; label-blind row caps 1200/600/600, selection seeds 53/54/55; model seeds 0/1/2. z-score statistics fit on training rows only. TabM-mini: width 64, depth 3, k=32, 64 epochs, batch 256, Adam. **INCOMPARABLE** to the paper benchmark. See [contract](l054-reproduction.md).

**Note on runtime:** the deep ensemble trains k separate MLPs, so a full k=32 / 3-seed run takes ~20 CPU minutes. This notebook's live cell uses a lighter default (1 seed, k=16); the author evidence in the figures used k=32 and three seeds.

**Recall first:** state the BatchEnsemble identity Wᵢ = W ⊙ (sᵢ rᵢᵀ) from memory, then predict whether a narrow (width-64) TabM should beat a plain MLP on 1200 rows.''')
    for c in bootstrap_cells():
        cells.append(nbf.v4.new_markdown_cell(c['source']) if c['cell_type'] == 'markdown'
                     else nbf.v4.new_code_cell(c['source']))
    code('''# PROVIDED — runtime and data setup
import os, sys, copy, math, hashlib, json, time, importlib.metadata
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/relational-matplotlib')
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.stats import rankdata, friedmanchisquare, studentized_range
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier, XGBRegressor
torch.set_num_threads(1)
for candidate in (Path.cwd(), Path.cwd() / 'labs', Path.cwd().parent):
    if (candidate / 'relkit').is_dir():
        LABS = candidate.resolve(); os.chdir(LABS); sys.path.insert(0, str(LABS)); break
else:
    raise RuntimeError('Use the Colab bootstrap or run inside this repository')
from _fetch_l052 import fetch
from relkit.tabr_experiment import seed_interval
from relkit.realmlp_experiment import load_task, error, fit_trees, SEARCH
fetch()
print({p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'scipy', 'scikit-learn', 'xgboost']})
print('Data manifest SHA256', hashlib.sha256((LABS / '_data_l052.json').read_bytes()).hexdigest())''')

    section('opportunity')
    section('architecture')
    figure('architecture', 'TabM-mini forward path: broadcast to k copies, a first ±1 adapter diversifies them, shared blocks, k heads, mean prediction.')
    figure('variants', 'The variant ladder from an explicit deep ensemble down to one shared MLP. We build TabM-mini.')

    section('batchensemble')
    figure('batchensemble', 'A member reuses the shared W with a rank-one ±1 mask; here column 2 flips sign.')
    task('batchensemble_linear', 'def batchensemble_linear(x, weight, R, S, bias):', '''### TODO 1 · The BatchEnsemble layer
**Goal:** return `((x ⊙ R) weight) ⊙ S + bias` for a batch of k members. **Why:** this is the one operation that lets k submodels share `weight` yet compute different functions. `x` is `[B, k, d_in]`, `weight` is `[d_in, d_out]`, `R` is `[k, d_in]` or `None`, `S` is `[k, d_out]` or `None`, `bias` is `[k, d_out]`. Skip R and S when they are `None` (the mini variant). Use `torch.einsum('bki,io->bko', ...)` for the shared matmul so gradients flow.''')
    code('''# CHECK — equivalence to Wᵢ = W ⊙ (s rᵀ), the no-op init, and member diversity
torch.manual_seed(0)
B, k, din, dout = 5, 3, 4, 4
x = torch.randn(B, k, din); W = torch.randn(din, dout)
R = torch.randint(0, 2, (k, din)) * 2. - 1; S = torch.randint(0, 2, (k, dout)) * 2. - 1
Bb = torch.zeros(k, dout)
out = batchensemble_linear(x, W, R, S, Bb)
for i in range(k):                       # explicit per-member matrix route must match
    Wi = W * torch.outer(R[i], S[i])     # weight is [d_in,d_out], so mask is r_i (rows) ⊗ s_i (cols)
    torch.testing.assert_close(out[:, i, :], x[:, i, :] @ Wi)
one = torch.ones(k, din); oneo = torch.ones(k, dout)   # R=S=1 -> Wi = W for every member
noop = batchensemble_linear(x, W, one, oneo, Bb)
for i in range(k):
    torch.testing.assert_close(noop[:, i, :], x[:, i, :] @ W)
assert not torch.allclose(out[:, 0, :], out[:, 1, :])  # different adapters -> different members
print('PASS: adapter route = Wᵢ·x, R=S=1 is a no-op, members differ')''')
    code('# PROVIDED — the layer and packed head that call your op\n' + extract(SOURCE, {'kaiming_', 'sign_pm1', 'BatchEnsembleLinear', 'PackedHead'}))
    task('packed_head', 'def packed_head(h, weight, bias):', '''### TODO 2 · k independent heads
**Goal:** return `[B, k, d_y]` where head i reads submodel i's representation. **Why:** each submodel needs its own output. `h` is `[B, k, d]`, `weight` is `[k, d, d_y]`, `bias` is `[k, d_y]`. Use `torch.einsum('bkd,kdo->bko', ...)` then add the bias.''')
    code('''# CHECK — each head acts only on its own member
torch.manual_seed(1)
h = torch.randn(6, 3, 5); Wh = torch.randn(3, 5, 2); bh = torch.randn(3, 2)
out = packed_head(h, Wh, bh)
assert out.shape == (6, 3, 2)
for i in range(3):
    torch.testing.assert_close(out[:, i, :], h[:, i, :] @ Wh[i] + bh[i])
print('PASS: packed heads are per-member linears')''')

    section('sharing')
    task('member_mean_loss', 'def member_mean_loss(out, y, regression):', '''### TODO 3 · Mean-over-members loss
**Goal:** return the mean over the k members of each member's supervised loss. **Why:** TabM explicitly optimises the average of the members' losses, not the loss of the average. `out` is `[B, k, d_y]`. For regression use squared error on `out[:,:,0]` vs `y`; for classification use cross-entropy (`log_softmax` + gather the true class) averaged over both rows and members.''')
    task('ensemble_predict', 'def ensemble_predict(out, regression):', '''### TODO 4 · Ensemble prediction
**Goal:** return the mean prediction over the k members. **Why:** this is how TabM predicts at inference. For regression return the mean of `out[:,:,0]`; for classification return the mean of the per-member softmax probabilities.''')
    code('''# CHECK — loss is a scalar; prediction averages members and is a valid distribution
torch.manual_seed(2)
out_r = torch.randn(7, 4, 1); y_r = torch.randn(7)
lr = member_mean_loss(out_r, y_r, True)
assert lr.shape == () and torch.isfinite(lr)
torch.testing.assert_close(ensemble_predict(out_r, True), out_r[:, :, 0].mean(1))
out_c = torch.randn(7, 4, 2); y_c = torch.randint(0, 2, (7,))
p = ensemble_predict(out_c, False)
assert p.shape == (7, 2); torch.testing.assert_close(p.sum(1), torch.ones(7))
torch.testing.assert_close(p, out_c.softmax(-1).mean(1))
print('PASS: mean-over-members loss and averaged probabilities')''')
    code('# PROVIDED — the full visible models (MLP baseline and TabM)\n' + extract(SOURCE, {'MLP', 'TabM'}))
    code('''# CHECK — TabM produces k distinct submodel predictions and averages them
torch.manual_seed(0)
model = TabM(4, k=8, width=16, depth=2, regression=False, arch='mini', seed=0)
model.eval()                                   # freeze dropout so the two forward passes match
x = torch.randn(10, 4)
members = model.member_predictions(x)          # [B, k, C]
assert members.shape[1] == 8
spread = members.std(1).mean().item()
assert spread > 0, 'submodels collapsed to one prediction'
torch.testing.assert_close(model.predict(x), members.mean(1))
print(f'PASS: 8 distinct submodels (mean std {spread:.3f}), prediction = their mean')''')

    section('diversity')
    figure('diversity', 'Author-reference measurement: on every task the collective error is below the mean individual submodel.')
    section('kknob')
    figure('kcurve', 'Author-reference k-sweep on California: collective stays below the mean individual and improves with k up to a point.')

    section('comparison')
    code('# PROVIDED — preprocessing, per-arm fitters, and the suite (calls your ops)\n' +
         extract(EXPERIMENT, {'_standardise', '_fit_one_mlp', '_test_pred', 'fit_mlp',
                              'fit_deep_ensemble', 'fit_tabm', 'run_suite'}))
    md('''### Read the selection boundary before training
`fit_tabm` selects its checkpoint on the *ensemble's* validation error and records each submodel's individual test error alongside the collective error. `fit_deep_ensemble` trains k independent MLPs and averages them. `fit_trees` (from L053) selects one of six XGBoost candidates on validation, then scores on test. Predict which arm leads on each task before running. No CHECK requires a particular winner.''')
    code('''# PROVIDED — live local experiment (lighter than the author evidence: 1 seed, k=16)
live_results = run_suite(seeds=(0,), k=16, epochs=48)
rows = []
for name, data in live_results['results'].items():
    for arm, s in data['summary'].items():
        rows.append(dict(dataset=name, method=arm, metric=data['metric'], mean=s['mean'], sd=s['sd']))
    rows.append(dict(dataset=name, method='TabM submodel (mean)', metric=data['metric'],
                     mean=data['diversity']['mean'], sd=data['diversity']['sd']))
display(pd.DataFrame(rows))
display(pd.DataFrame([live_results['ranks']['means']]).T.rename(columns={0: 'mean rank'}))
print('Friedman p', live_results['ranks']['friedman_p'], 'CD', live_results['ranks']['nemenyi_cd'])''')
    code('''# CHECK — the diversity effect: collective ≤ mean individual on every task
for name, data in live_results['results'].items():
    collective = data['summary']['TabM-mini']['mean']
    individual = data['diversity']['mean']
    assert collective <= individual + 1e-9, f'{name}: collective {collective} > individual {individual}'
    print(f'{name}: collective {collective:.4f} ≤ mean individual {individual:.4f}')
print('PASS: averaging weak diverse submodels helps on every task')''')

    section('evidence')
    figure('results', 'Author-reference local measurement (k=32, 3 seeds), separate from your lighter live run above.')
    figure('ranks', 'Author-reference ranks across three tasks. A low-power test is not evidence of equivalence.')
    md('''### EXIT TICKET · complete in your own words
Report one task's four arm errors with seed uncertainty, plus TabM's collective and mean-individual error. Using the variance formula σ²(ρ + (1−ρ)/k), explain why the collective can beat the individuals and why more k stops helping. State the paper verdict and two protocol deviations from the benchmark. Name one change (width, rows, embeddings, temporal splits) that would move this regime toward the paper's.

Paste your output and explanation to the teacher for feedback; passing numerical CHECKs alone does not grade the explanation.''')
    code('''# EXIT — replace the blank explanation before submitting
interpretation = ""  # TODO: write your own conclusion and protocol limits
print('Paper verdict:', live_results['verdict'])
print('Dataset count:', len(live_results['results']))
print('Mean ranks:', live_results['ranks']['means'])
print('Your explanation:', interpretation or 'WRITE YOUR INTERPRETATION BEFORE SUBMITTING')''')
    section('paperclaim')
    md('''### Return tomorrow
Without reopening the formula, reconstruct the BatchEnsemble identity and explain why weight sharing plus joint training beats an explicit deep ensemble. Then state what changes when a benchmark's split stops being random — the L055 question. Ask the teacher about any CHECK or derivation you cannot defend.''')

    nb = nbf.v4.new_notebook(cells=cells, metadata=dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python', version='3')))
    if write:
        path = ROOT / ('solutions' if solution else '') / f'{SLUG}.ipynb'
        path.parent.mkdir(exist_ok=True); nbf.write(nb, path); print(path)
    return nb


if __name__ == '__main__':
    build(False); build(True)
