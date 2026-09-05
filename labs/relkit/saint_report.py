"""Readable L047 experiment reports. No model code or training is hidden here."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LABELS = {'col': 'Feature only', 'colrow': 'Feature + row', 'catboost': 'CatBoost'}
COLORS = {'col': '#2864ad', 'colrow': '#087f78', 'catboost': '#ac5b16'}


def score_table(result):
    records = []
    for dataset, arms in result['results']['per_dataset'].items():
        for model, arm in arms.items():
            records.append({'Dataset': dataset, 'Model': LABELS[model],
                            'Mean AUROC': arm['mean'], 'Seed SD': arm['sample_std'],
                            'Seed CI half-width': arm['seed_ci95_halfwidth']})
    return pd.DataFrame(records)


def paired_table(result):
    records = []
    for dataset, arms in result['results']['per_dataset'].items():
        delta = np.array(arms['colrow']['scores']) - np.array(arms['col']['scores'])
        for seed, value in zip(result['seeds'], delta):
            records.append({'Dataset': dataset, 'Seed': seed, 'Row − feature AUROC': value})
    return pd.DataFrame(records)


def _style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', color='#dce3ea', linewidth=.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=7)


def plot_scores(result, title='Your run · AUROC across three model seeds'):
    """Shared axis, visible seed points, SD bars; no aggregation across datasets."""
    datasets = list(result['results']['per_dataset'])
    fig, axes = plt.subplots(len(datasets), 1, figsize=(8.6, 2.25 * len(datasets)),
                             sharex=True, layout='constrained', squeeze=False)
    for ax, dataset in zip(axes[:, 0], datasets):
        for j, (model, label) in enumerate(LABELS.items()):
            arm = result['results']['per_dataset'][dataset][model]
            ax.scatter(arm['scores'], np.full(len(arm['scores']), j) + np.linspace(-.10, .10, len(arm['scores'])),
                       color=COLORS[model], alpha=.55, s=26, zorder=3)
            ax.errorbar(arm['mean'], j, xerr=arm['sample_std'], color=COLORS[model],
                        fmt='D', capsize=5, markersize=6, linewidth=2, zorder=4)
        ax.set_yticks(range(3), LABELS.values())
        ax.set_ylim(2.55, -.65)
        ax.set_title(dataset.replace('_', ' '), loc='left', fontsize=12, fontweight='bold')
        _style(ax)
    axes[-1, 0].set_xlabel('Test AUROC · dots = seeds; diamond = mean; bar = ±1 sample SD')
    fig.suptitle(title, fontsize=15, fontweight='bold')
    return fig


def plot_ranks(result, title='Your run · how much can three datasets distinguish?'):
    ranks = result['results']['mean_ranks']
    cd = result['results']['nemenyi_cd']
    p = result['results']['friedman']['p']
    fig, ax = plt.subplots(figsize=(8.6, 3.5), layout='constrained')
    for j, model in enumerate(LABELS):
        ax.scatter(ranks[model], j, color=COLORS[model], s=85, zorder=3)
        ax.annotate(f'{ranks[model]:.2f}', (ranks[model], j), xytext=(10, 0),
                    textcoords='offset points', va='center')
    # Same rank scale as the points; CD is a threshold for pairwise rank gaps.
    ax.plot([1, 1 + cd], [-.85, -.85], color='#334155', linewidth=2)
    ax.plot([1, 1], [-.95, -.75], color='#334155')
    ax.plot([1 + cd, 1 + cd], [-.95, -.75], color='#334155')
    ax.text(1 + cd / 2, -1.08, f'Nemenyi critical difference = {cd:.2f} (α = .05)',
            ha='center', va='bottom', fontsize=10)
    ax.set_yticks(range(3), LABELS.values())
    ax.set_xticks([1, 1.5, 2, 2.5, 3])
    ax.set_xlim(.9, 3.25)
    ax.set_ylim(2.6, -1.55)
    ax.set_xlabel('Mean rank across datasets · 1 is best')
    _style(ax)
    verdict = ('No overall rank difference detected; this does not establish equivalence.' if p >= .05
               else 'Overall rank difference detected; compare individual gaps with the CD.')
    fig.suptitle(title + f'\nFriedman p = {p:.3f}. ' + verdict, fontsize=11)
    return fig


def plot_context(result, title='Your run · change companions, keep trained weights fixed'):
    records = []
    for key, probe in result['context_probe'].items():
        dataset, seed, model = key.split('/')
        records.append({'Dataset': dataset, 'Seed': int(seed), 'Model': model, **probe})
    data = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(8.6, 3.6), layout='constrained')
    datasets = list(result['results']['per_dataset'])
    for j, dataset in enumerate(datasets):
        for offset, model in ((-.13, 'col'), (.13, 'colrow')):
            rows = data[(data.Dataset == dataset) & (data.Model == model)]
            ax.scatter(rows.max_abs_probability_change, np.full(len(rows), j + offset),
                       color=COLORS[model], s=42, alpha=.75, label=LABELS[model] if j == 0 else None)
    ax.set_yticks(range(len(datasets)), [d.replace('_', ' ') for d in datasets])
    ax.set_ylim(len(datasets) - .45, -.65)
    ax.set_xlim(left=-.015)
    ax.set_xlabel('Maximum |p(batch 64) − p(singleton)| · each dot is one trained seed')
    _style(ax)
    ax.legend(loc='lower right', frameon=False)
    fig.suptitle(title, fontsize=13, fontweight='bold')
    return fig


def save_reference_figures(result, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, plot in [('scores', plot_scores), ('ranks', plot_ranks), ('context-results', plot_context)]:
        fig = plot(result, title='Author reference run · ' + {
            'scores': 'test AUROC, fixed split, three seeds',
            'ranks': 'mean ranks and critical difference',
            'context-results': 'evaluation batch 64 → singleton'}[name])
        fig.savefig(destination / f'{name}.png', dpi=150, facecolor='white')
        plt.close(fig)
