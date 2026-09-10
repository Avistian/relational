"""Portable L054 figures. Static computation figures plus measured-evidence figures.
Measured figures read _verify_l054_v2_results.json and are skipped if it is absent."""
import os, json
os.environ.setdefault('MPLCONFIGDIR', '/tmp/relational-matplotlib')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures/l054'; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': '#fffdf8',
                     'axes.facecolor': '#fffdf8', 'savefig.facecolor': '#fffdf8'})
BLUE = '#225b78'; ORANGE = '#b65c29'; GREEN = '#377355'; GREY = '#8a8a8a'


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=160, bbox_inches='tight'); plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(8.2, 8.6)); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    def box(x, y, w, h, color='#e7eff0'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.008', facecolor=color, edgecolor='#b8c7cd'))
    def txt(x, y, t, size=11, color='#222', weight='normal', ha='left'):
        ax.text(x, y, t, fontsize=size, color=color, weight=weight, ha=ha, va='center')
    def arrow(x, y, xx, yy):
        ax.annotate('', xy=(xx, yy), xytext=(x, y), arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.6))
    txt(.02, .975, 'TabM (mini): one model, k submodels', 18, weight='bold')
    txt(.02, .945, 'Numeric input x [B, p]. One preprocessing, k shared copies.', 10)
    box(.02, .86, .96, .06)
    txt(.04, .89, 'INPUT + z-score preprocessing   →   broadcast to k copies   [B, k, p]', 11, weight='bold')
    arrow(.5, .855, .5, .83)
    box(.02, .735, .96, .09, '#f3ece0')
    txt(.04, .805, 'FIRST ADAPTER R₁  (the one critical adapter)  [k, p]', 11, weight='bold')
    txt(.04, .77, 'X ⊙ R₁ with R₁ starts at ±1; then learns real scalars: the k copies now differ before W ever mixes features', 10, ORANGE)
    arrow(.5, .73, .5, .705)
    txt(.04, .688, 'N SHARED BLOCKS: Linear (shared W) → ReLU → Dropout   [B, k, d]', 11, weight='bold')
    for i in range(3):
        x = .03 + i * .33; box(x, .585, .28, .075)
        txt(x + .14, .64, 'shared W · ReLU', 10, ha='center')
        txt(x + .14, .607, '[B,k,d] → [B,k,d]', 9, ha='center')
        if i < 2: arrow(x + .288, .622, x + .324, .622)
    txt(.04, .556, 'mini: only R₁ differs between members; every W and bias here is shared', 9, GREY)
    txt(.04, .534, 'full TabM: each block also has R,S,B adapters, initialised R=S=1 (no-op at start)', 9, GREY)
    arrow(.5, .512, .5, .487)
    box(.02, .40, .96, .085)
    txt(.04, .462, 'PACKED HEADS: k independent linear heads   [B, k, d] → [B, k, d_y]', 11, weight='bold')
    txt(.04, .425, 'submodel 1 → ŷ₁ … submodel k → ŷ_k   (k predictions per object)', 10, BLUE)
    arrow(.5, .397, .5, .372)
    box(.02, .285, .96, .085, '#dcece3')
    txt(.04, .347, 'PREDICTION = mean over k submodels', 11, weight='bold')
    txt(.04, .31, 'regression: mean of ŷ ;  classification: mean of softmax probabilities', 10, GREEN)
    box(.02, .12, .96, .13, '#f3ece0')
    txt(.04, .225, 'WHAT IS SHARED vs NOT', 11, weight='bold')
    txt(.04, .19, 'Shared (≈ one MLP): the backbone weights W of every block', 10)
    txt(.04, .162, 'Mini not shared: only R₁ [k,p] and the k heads; no S or member bias', 10)
    txt(.04, .134, 'Only new hyperparameter vs MLP: k. Paper default k = 32, not tuned.', 10, BLUE)
    txt(.02, .085, 'Trained jointly: one optimiser step = k parallel steps; stop on the ensemble’s validation score.', 9)
    txt(.02, .055, 'This lab: numeric only; no feature embeddings (paper’s † variants) or categorical inputs.', 9, GREY)
    txt(.02, .025, 'B = batch, p = features, d = width, d_y = outputs, k = submodels.', 9, GREY)
    save(fig, 'architecture')


def batchensemble():
    W = np.array([[1., 3.], [2., 4.]]); r = np.array([1, -1]); s = np.array([1, 1])
    Wi = (s[:, None] * r[None, :]) * W
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))
    for ax, M, title, hl in [(axes[0], W, 'Column-vector W (transpose of code W)', None),
                             (axes[1], np.outer(s, r), 'sᵢ rᵢᵀ  (outer product of ±1)', None),
                             (axes[2], Wi, 'Wᵢ = W ⊙ (sᵢ rᵢᵀ)', (r * s[0] < 0))]:
        ax.imshow(np.zeros_like(M), cmap='Greys', vmin=0, vmax=1); ax.set_title(title, fontsize=11)
        for (o, i), v in np.ndenumerate(M):
            flip = (s[o] * r[i] < 0)
            ax.text(i, o, f'{v:.2f}', ha='center', va='center', fontsize=13,
                    color=ORANGE if (title.startswith('Wᵢ') and flip) else '#222',
                    weight='bold' if (title.startswith('Wᵢ') and flip) else 'normal')
        ax.set_xticks([0, 1], ['in₁', 'in₂']); ax.set_yticks([0, 1], ['out₁', 'out₂'])
        ax.set_xticks(np.arange(-.5, 2), minor=True); ax.set_yticks(np.arange(-.5, 2), minor=True)
        ax.grid(which='minor', color='#ccc'); ax.tick_params(which='minor', length=0)
    fig.suptitle('A member with r = [+1, −1], s = [+1, +1]: column 2 flips sign; the rest of W is reused',
                 fontsize=11)
    fig.tight_layout(); save(fig, 'batchensemble')


def variants():
    fig, ax = plt.subplots(figsize=(12, 4.8)); ax.axis('off')
    rows = [['MLP×k', 'k full MLPs; own checkpoints', 'k × P(MLP)', 'member-wise validation selection'],
            ['TabM_packed', 'k full MLPs; collective selection', '≈ k × MLP', 'unlike member-wise MLP×k stopping'],
            ['TabM_naive', 'R,S random ±1; member bias', 'W + R,S,bias + heads', 'sharing constrains member functions'],
            ['TabM_mini', 'one first adapter R₁ only', 'backbone + R₁ + heads', 'the minimal effective ensemble'],
            ['TabM (default)', 'all R,S=1 except random first R', 'W + R,S,bias + heads', 'reported strongest base variant']]
    table = ax.table(cellText=rows, colLabels=['Variant', 'What differs per member', 'Size', 'Takeaway'],
                     loc='center', cellLoc='left', colWidths=[.24, .30, .18, .28])
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 2.9)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#d4d1c8')
        if r == 0: cell.set_facecolor('#e7eff0')
        elif r == 4: cell.set_facecolor('#dcece3')
    ax.set_title('From an expensive deep ensemble to one shared MLP (paper §3.3)', pad=16, weight='bold')
    fig.tight_layout(); save(fig, 'variants')


def measured():
    path = ROOT / '_verify_l054_v2_results.json'
    if not path.exists():
        print('no results yet; skipping measured figures'); return
    res = json.loads(path.read_text()); arms = res['ranks']['arms']
    colors = {'MLP': GREY, 'MLP-xk': BLUE, 'TabM-mini': GREEN, 'XGB-tuned': ORANGE}

    # results across datasets
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.6))
    for ax, (name, d) in zip(axes, res['results'].items()):
        for i, a in enumerate(arms):
            stat = d['summary'][a]; v = [r['error'] for r in d['runs'][a]]
            ci = stat.get('ci95')
            yerr = [[stat['mean'] - ci[0]], [ci[1] - stat['mean']]] if ci else None
            ax.errorbar(i, stat['mean'], yerr=yerr, fmt='o', color=colors[a], capsize=4)
            ax.scatter(i + np.linspace(-.09, .09, len(v)), v, color=colors[a], s=16, alpha=.7)
        ax.set_title(name); ax.set_xticks(range(len(arms)), [('MLP\n×32' if a=='MLP-xk' else a.replace('-', '\n')) for a in arms])
        ax.set_ylabel(d['metric'] + ' ↓'); ax.grid(axis='y', alpha=.15)
    fig.suptitle('Corrected v2 local errors · dots = seeds; bars = conditional 95% t intervals', fontsize=13)
    fig.tight_layout(); save(fig, 'results')

    # diversity: individual mean vs collective vs single MLP
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    for ax, (name, d) in zip(axes, res['results'].items()):
        coll = d['summary']['TabM-mini']['mean']; ind = d['diversity']['mean']
        mlp = d['summary']['MLP']['mean']; de = d['summary']['MLP-xk']['mean']
        bars = ['member\nerror', 'mean\npredictor', 'MLP', 'MLP×32']
        vals = [ind, coll, mlp, de]; cols = [GREY, GREEN, '#c0392b', BLUE]
        ax.bar(bars, vals, color=cols)
        for i, v in enumerate(vals): ax.text(i, v, f'{v:.0f}' if name=='house' else f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_title(name); ax.set_ylabel(d['metric'] + ' ↓'); ax.grid(axis='y', alpha=.15)
        ax.set_ylim(0, max(vals) * 1.18)
    fig.suptitle('Corrected v2: first two bars are TabM member-average and collective error', fontsize=13)
    fig.tight_layout(); save(fig, 'diversity')

    # k-sweep
    ks = res.get('k_sweep')
    if ks:
        rows = ks['rows']; k = [r['k'] for r in rows]
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(k, [r['individual_mean'] for r in rows], 'o--', color=GREY, label='mean individual submodel')
        ax.plot(k, [r['collective'] for r in rows], 'o-', color=GREEN, label='collective (mean of k)')
        ax.set_xscale('log', base=2); ax.set_xticks(k, [str(x) for x in k])
        ax.set(xlabel='number of submodels k', ylabel=ks['metric'] + ' ↓',
               title=f"k-sweep on {ks['dataset']} (seed {ks['seed']}, width 64) — measured, one dataset")
        ax.legend(); ax.grid(alpha=.15); fig.tight_layout(); save(fig, 'kcurve')

    # ranks
    fig, ax = plt.subplots(figsize=(8, 3.8)); ranks = res['ranks']; cd = ranks['nemenyi_cd']
    order = sorted(arms, key=lambda a: ranks['means'][a])
    for i, a in enumerate(order):
        r = ranks['means'][a]; ax.scatter(r, i, color=colors[a]); ax.text(r + .04, i, f"{'MLP×32' if a=='MLP-xk' else a}: {r:.3f}", va='center')
    ax.plot([1, 1 + cd], [len(arms) - .3, len(arms) - .3], color='#333')
    ax.text(1 + cd / 2, len(arms) - .12, f'CD = {cd:.3f}', ha='center')
    ax.set(xlim=(.7, len(arms) + .9), ylim=(-.5, len(arms) - .0), yticks=[],
           xlabel='mean rank across 3 datasets (lower is better)',
           title=f"Friedman p = {ranks['friedman_p']:.4f} · exploratory, low power (3 datasets)")
    fig.tight_layout(); save(fig, 'ranks')


if __name__ == '__main__':
    architecture(); batchensemble(); variants(); measured()
    print('figures written to', OUT)
