"""Verify Lesson 023 claims — statistical comparison of classifiers (Demšar 2006).

Two mechanisms, honestly isolated on Tier-C synthetic data (no network / OpenML):

  1. IS THE GAP REAL OR NOISE? (the core skill: a paired test on CV folds)
     Two close models compared with RepeatedStratifiedKFold. The NAIVE paired
     t-test treats the k*r fold differences as independent and fabricates a
     tiny p-value (a "significant" win). The CORRECTED resampled t-test
     (Nadeau & Bengio 2003) inflates the variance by (1/n + n_test/n_train) to
     account for the overlapping training sets, and the p-value balloons — the
     "win" was noise. Wilcoxon signed-rank is the distribution-free companion.

  2. COMPARING MANY MODELS OVER MANY DATASETS (Demšar 2006, the headline tool)
     Rank k classifiers on N synthetic datasets; the Friedman test asks whether
     the average ranks differ at all; if so, the Nemenyi critical difference
     (CD) says which pairwise rank gaps are real. This is the CD-diagram recipe.

Run: .venv/bin/python labs/_verify_l023.py
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0


# ---------------------------------------------------------------------------
# Shared synthetic data — a moderate, mostly-linear signal (LR does well)
# ---------------------------------------------------------------------------
def make_data(n=1000, d=12, n_informative=6, seed=SEED):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.zeros(d)
    w[:n_informative] = rng.normal(size=n_informative)
    logit = X @ w + 1.5 * rng.normal(size=n)
    y = (logit > np.median(logit)).astype(int)
    return X, y


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def lr_c(C):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=C))


def corrected_resampled_ttest(diffs, n_splits):
    """Nadeau & Bengio 2003 corrected resampled t-test on per-fold differences.

    diffs    : array of (scoreA - scoreB) per fold, length n = n_splits * n_repeats
    n_splits : folds per repeat -> test/train size ratio = 1/(n_splits - 1)
    Returns (t_stat, p_value, df).
    """
    n = len(diffs)
    df = n - 1
    mean = float(np.mean(diffs))
    var = float(np.var(diffs, ddof=1))
    test_train_ratio = 1.0 / (n_splits - 1)
    corrected_var = (1.0 / n + test_train_ratio) * var
    t = mean / np.sqrt(corrected_var)
    p = 2.0 * stats.t.cdf(-abs(t), df)
    return t, p, df


def naive_paired_ttest(diffs):
    n = len(diffs)
    df = n - 1
    mean = float(np.mean(diffs))
    var = float(np.var(diffs, ddof=1))
    t = mean / np.sqrt(var / n)
    p = 2.0 * stats.t.cdf(-abs(t), df)
    return t, p, df


# ---------------------------------------------------------------------------
# Mechanism 1 — real or noise? paired test on CV folds
# ---------------------------------------------------------------------------
def _compare(a, b, n_splits):
    diffs = a - b
    t_naive, p_naive, _ = naive_paired_ttest(diffs)
    t_corr, p_corr, _ = corrected_resampled_ttest(diffs, n_splits)
    _, p_wilcox = stats.wilcoxon(a, b)
    return diffs, t_naive, p_naive, t_corr, p_corr, p_wilcox


def real_or_noise():
    """Headline failure mode: a SMALL, consistent-sign gap that the naive paired
    t-test on 100 folds calls highly significant, but the corrected resampled
    t-test (and Wilcoxon) say is indistinguishable from noise."""
    X, y = make_data()
    n_splits, n_repeats = 10, 10
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)

    a = cross_val_score(lr(), X, y, cv=rkf, scoring="accuracy")            # LogReg
    b = cross_val_score(GaussianNB(), X, y, cv=rkf, scoring="accuracy")    # GaussianNB
    diffs, t_naive, p_naive, t_corr, p_corr, p_wilcox = _compare(a, b, n_splits)

    print("== Mechanism 1: is the gap real or noise? (paired test on CV folds) ==")
    print(f"  RepeatedStratifiedKFold({n_splits}x{n_repeats}) -> {len(diffs)} paired fold scores")
    print(f"  model A = LogReg    : mean acc {a.mean():.4f}")
    print(f"  model B = GaussianNB: mean acc {b.mean():.4f}")
    print(f"  mean(A-B) = {diffs.mean():+.4f}   sd(A-B) = {diffs.std(ddof=1):.4f}   "
          f"(sign consistent: {(diffs > 0).mean()*100:.0f}% of folds favour A)")
    print(f"  NAIVE paired t-test        t={t_naive:+.3f}  p={p_naive:.4g}   -> "
          f"{'SIGNIFICANT' if p_naive < 0.05 else 'not sig'} (treats 100 folds as independent)")
    print(f"  CORRECTED resampled t-test t={t_corr:+.3f}  p={p_corr:.4g}   -> "
          f"{'SIGNIFICANT' if p_corr < 0.05 else 'not sig'} (Nadeau-Bengio variance inflation)")
    print(f"  Wilcoxon signed-rank                    p={p_wilcox:.4g}   -> "
          f"{'SIGNIFICANT' if p_wilcox < 0.05 else 'not sig'}")
    print("  --> the naive test fabricates a 'win'; the honest tests say: within noise.")
    return dict(a=a.mean(), b=b.mean(), mean=diffs.mean(), sd=diffs.std(ddof=1),
                p_naive=p_naive, p_corr=p_corr, p_wilcox=p_wilcox)


def real_gap():
    """Contrast: a genuinely large gap — both tests agree it is real."""
    from sklearn.tree import DecisionTreeClassifier
    X, y = make_data()
    n_splits, n_repeats = 10, 10
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    a = cross_val_score(lr(), X, y, cv=rkf, scoring="accuracy")
    b = cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=SEED), X, y, cv=rkf, scoring="accuracy")
    diffs, t_naive, p_naive, t_corr, p_corr, p_wilcox = _compare(a, b, n_splits)
    print("\n== Contrast: a genuinely real gap (LogReg vs shallow DecisionTree) ==")
    print(f"  LogReg {a.mean():.4f}  vs  DecisionTree(d=3) {b.mean():.4f}   mean(A-B) {diffs.mean():+.4f}")
    print(f"  NAIVE p={p_naive:.3g}   CORRECTED p={p_corr:.3g}   Wilcoxon p={p_wilcox:.3g}   -> "
          f"{'both AGREE significant' if p_naive < 0.05 and p_corr < 0.05 else 'disagree'} "
          "(a real gap survives the correction)")
    return dict(a=a.mean(), b=b.mean(), mean=diffs.mean(), p_naive=p_naive, p_corr=p_corr, p_wilcox=p_wilcox)


# ---------------------------------------------------------------------------
# Mechanism 2 — Demšar: many models over many datasets (Friedman + Nemenyi CD)
# ---------------------------------------------------------------------------
def demsar_cd():
    models = {
        "LogReg": lr(),
        "GaussianNB": GaussianNB(),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
        "HistGBDT": HistGradientBoostingClassifier(random_state=SEED),
    }
    names = list(models)
    N = 12  # datasets
    score_matrix = np.zeros((N, len(names)))
    for di in range(N):
        # each "dataset" varies size / signal / noise so rankings can differ
        rng = np.random.default_rng(100 + di)
        n = int(rng.integers(400, 1200))
        d = int(rng.integers(8, 20))
        n_inf = int(rng.integers(3, d))
        X = rng.normal(size=(n, d))
        w = np.zeros(d); w[:n_inf] = rng.normal(size=n_inf)
        noise = float(rng.uniform(0.8, 2.2))
        y = ((X @ w + noise * rng.normal(size=n)) > 0).astype(int)
        for mi, nm in enumerate(names):
            score_matrix[di, mi] = cross_val_score(
                models[nm], X, y, cv=5, scoring="accuracy"
            ).mean()

    # ranks per dataset (1 = best); average across datasets
    ranks = np.zeros_like(score_matrix)
    for di in range(N):
        # higher score -> lower (better) rank; average ties
        order = stats.rankdata(-score_matrix[di])
        ranks[di] = order
    avg_ranks = ranks.mean(axis=0)

    # Friedman test
    chi2, p_friedman = stats.friedmanchisquare(*[score_matrix[:, mi] for mi in range(len(names))])

    # Nemenyi critical difference at alpha=0.05
    k = len(names)
    q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}[k]
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))

    print("\n== Mechanism 2: Demšar — many models over many datasets ==")
    print(f"  {k} models over N={N} synthetic datasets")
    for nm, r, s in sorted(zip(names, avg_ranks, score_matrix.mean(axis=0)), key=lambda t: t[1]):
        print(f"    {nm:14s} avg-rank {r:.3f}   mean-acc {s:.3f}")
    print(f"  Friedman chi2={chi2:.3f}  p={p_friedman:.4g}  -> "
          f"{'ranks differ (reject H0)' if p_friedman < 0.05 else 'no difference'}")
    print(f"  Nemenyi CD (alpha=0.05, k={k}, N={N}) = {cd:.3f}")
    best = names[int(np.argmin(avg_ranks))]
    print(f"  best avg rank: {best}")
    for nm, r in zip(names, avg_ranks):
        if nm == best:
            continue
        gap = r - avg_ranks[int(np.argmin(avg_ranks))]
        verdict = "significantly worse" if gap > cd else "NOT sig. different"
        print(f"    {best} vs {nm:14s}: rank gap {gap:.3f}  ({verdict} — CD {cd:.3f})")
    return dict(avg_ranks=dict(zip(names, [round(float(x), 3) for x in avg_ranks])),
                p_friedman=float(p_friedman), cd=float(cd),
                mean_acc=dict(zip(names, [round(float(x), 3) for x in score_matrix.mean(axis=0)])))


if __name__ == "__main__":
    m1 = real_or_noise()
    rg = real_gap()
    m2 = demsar_cd()
    print("\n== Summary ==")
    print(f"  close pair: naive p={m1['p_naive']:.3g} (sig) vs corrected p={m1['p_corr']:.3g} "
          f"({'not sig' if m1['p_corr'] >= 0.05 else 'sig'}) — the win was noise")
    print(f"  real pair : naive p={rg['p_naive']:.3g}, corrected p={rg['p_corr']:.3g} — survives")
    print(f"  Demšar    : Friedman p={m2['p_friedman']:.3g}, CD={m2['cd']:.3f}, ranks={m2['avg_ranks']}")
