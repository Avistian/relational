"""L056 versioned extension: transparent current-source Bradley-Terry analysis.

Historical leaderboard.py and measured rank evidence remain unchanged.
"""
import numpy as np


def paired_wins(errors):
    """Return D x M x M outcomes; each dataset averages its own split contests."""
    if errors.empty or not np.isfinite(errors.to_numpy()).all():
        raise ValueError('Require a complete finite error matrix')
    blocks=[]
    for _, frame in errors.groupby(level='dataset',sort=True):
        e=frame.to_numpy(dtype=float)
        # Axis 1 is the winning candidate; axis 2 is its opponent.
        outcome=(e[:,:,None]<e[:,None,:])+.5*(e[:,:,None]==e[:,None,:])
        blocks.append(outcome.mean(axis=0))
    return np.stack(blocks)


def fit_elo(dataset_wins):
    """Fit complete balanced contests; centre ratings at 1000, not paper RF anchor.

    Matches e7cc6b0 EloHelper ridge and likelihood, using an explicit pair matrix.
    Input diagonal is .5 for display but contributes no self-contests to the fit.
    """
    from scipy.optimize import minimize
    from scipy.special import expit
    w=np.asarray(dataset_wins,dtype=float)
    if (w.ndim!=3 or w.shape[0]<1 or w.shape[1]<2 or w.shape[1]!=w.shape[2]
        or not np.isfinite(w).all() or np.any((w<0)|(w>1))
        or not np.allclose(w+w.transpose(0,2,1),1)):
        raise ValueError('Expected D x M x M reciprocal win probabilities')
    d,m,_=w.shape
    i,j=np.triu_indices(m,1)
    observed=w.sum(axis=0)[i,j]
    ridge=.5/(1e6*np.log(10)**2)
    def objective(t):
        delta=t[i]-t[j]
        loss=np.sum(d*np.logaddexp(0,delta)-observed*delta)+ridge*(t@t)
        residual=d*expit(delta)-observed
        gradient=2*ridge*t
        np.add.at(gradient,i,residual);np.add.at(gradient,j,-residual)
        return loss,gradient
    result=minimize(objective,np.zeros(m),jac=True,method='L-BFGS-B',
                    options={'maxiter':10000,'ftol':1e-15,'gtol':1e-12})
    if not result.success and np.max(np.abs(result.jac))>1e-6:
        raise RuntimeError('Rating optimizer failed: '+result.message)
    return 1000+400*(result.x-result.x.mean())/np.log(10)


def rating_audit(errors, n_boot=100, seed=56):
    """All outputs depend on the live paired_wins and fit_elo implementations."""
    from scipy.special import expit
    w=paired_wins(errors)
    ratings=fit_elo(w)
    # A dataset mean error supplies one contest per dataset (Appendix A.1 order).
    means=errors.groupby(level='dataset',sort=True).mean()
    mean_ratings=fit_elo(paired_wins(means))
    rng=np.random.default_rng(seed)
    # Anchor contrasts to CatBoost in every draw, keeping paired methods together.
    anchor=list(errors.columns).index('CatBoost')
    boot=np.array([fit_elo(w[rng.integers(len(w),size=len(w))]) for _ in range(n_boot)])
    contrasts=boot-boot[:,anchor,None]
    predicted=expit((ratings[:,None]-ratings[None,:])*np.log(10)/400)
    return dict(methods=list(errors.columns),datasets=len(w),ratings=ratings.tolist(),
        mean_error_ratings=mean_ratings.tolist(),observed_wins=w.mean(0).tolist(),
        fitted_wins=predicted.tolist(),max_matchup_residual=float(np.abs(predicted-w.mean(0)).max()),
        catboost_contrast=(ratings-ratings[anchor]).tolist(),
        catboost_contrast_interval=np.quantile(contrasts,[.025,.975],axis=0).T.tolist(),
        bootstrap_draws=n_boot,bootstrap_seed=seed,
        convention='Each regime separately; field mean 1000; current ridge; no RF anchor',
        paper_table='INCOMPARABLE')
