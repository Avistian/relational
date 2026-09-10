"""L059 v2: Cawley–Talbot equations and a declared finite-grid protocol extension.

Historical benchmark_core.py / _verify_l059_results.json remain unchanged.
No upstream software parity or original figure replication is claimed.
"""
import numpy as np
from scipy.linalg import solve
from scipy.stats import t, binom


def choose_validation(errors):
    errors = np.asarray(errors, float)
    if errors.ndim != 1 or not errors.size or not np.isfinite(errors).all():
        raise ValueError('A nonempty finite one-dimensional loss vector is required')
    return int(np.argmin(errors))


def mixture(n, seed):
    """Paper §3.1: equiprobable components, variance .04 on each coordinate."""
    rng = np.random.default_rng(seed)
    centers = np.array([[.4,.7],[-.3,.7],[-.7,.3],[.3,.3]])
    component = rng.integers(0, 4, n)
    return centers[component] + rng.normal(0, .2, (n, 2)), np.where(component < 2, 1., -1.)


def kernel(x, z, eta):
    """Paper §4 ARD kernel; scalar eta recovers isotropic RBF."""
    return np.exp(-np.sum((x[:,None,:]-z[None,:,:])**2*np.asarray(eta), axis=2))


def krr_fit(x, y, regularization, eta):
    """Paper Eq (3): unpenalized intercept b, constraint sum(alpha)=0."""
    x, y = np.asarray(x,float), np.asarray(y,float)
    eta = np.asarray(eta,float)
    if x.ndim != 2 or len(x)<2 or y.shape!=(len(x),) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite X[n,d], y[n] with n>=2 required')
    if not np.isfinite(regularization) or regularization<=0 or not np.isfinite(eta).all() or (eta<=0).any() or eta.shape not in [(),(x.shape[1],)]:
        raise ValueError('Positive lambda and scalar or d-coordinate eta required')
    n=len(x); c=np.empty((n+1,n+1))
    c[:n,:n]=kernel(x,x,eta)+regularization*np.eye(n)
    c[:n,n]=1.; c[n,:n]=1.; c[n,n]=0.
    inverse=solve(c,np.eye(n+1),assume_a='sym')
    coefficients=inverse @ np.r_[y,0.]
    return dict(x=x,alpha=coefficients[:n],bias=float(coefficients[n]),
                inverse_diagonal=np.diag(inverse)[:n],eta=eta)


def predict_krr(model, x):
    return kernel(np.asarray(x),model['x'],model['eta'])@model['alpha']+model['bias']


def loo_residuals(alpha, inverse_diagonal):
    """Paper §2.1 deleted residual y_i-f^(-i)(x_i), not fitted residual."""
    a,d=np.asarray(alpha,float),np.asarray(inverse_diagonal,float)
    if a.ndim!=1 or not a.size or a.shape!=d.shape or not np.isfinite(a).all() or not np.isfinite(d).all() or (d<=0).any():
        raise ValueError('Aligned finite vectors and positive inverse diagonal required')
    return a/d


def candidate_grid():
    # Fixed before outcomes: 3 regularizers × 3 scales per coordinate = 27.
    return [dict(regularization=l,eta=[e1,e2])
            for l in [.01,.1,1.] for e1 in [.25,2.,16.] for e2 in [.25,2.,16.]]


def select_krr(x, y, candidates):
    """Return chosen model/index and mean PRESS for every declared candidate."""
    if not candidates:raise ValueError('Candidates must be nonempty')
    models=[krr_fit(x,y,**config) for config in candidates]
    losses=np.array([np.mean(loo_residuals(m['alpha'],m['inverse_diagonal'])**2) for m in models])
    selected=choose_validation(losses)
    return models[selected], selected, losses


def nested_predictions(x, y, folds, candidates):
    """Outer OOF predictions: rerun the entire selection on complement rows."""
    x,y,folds=np.asarray(x,float),np.asarray(y,float),np.asarray(folds)
    if folds.shape!=(len(x),) or y.shape!=(len(x),) or not np.issubdtype(folds.dtype,np.integer) or len(np.unique(folds))<2:
        raise ValueError('At least two nonempty integer-labeled outer folds required')
    predictions=np.empty(len(x)); trace=[]
    for fold in np.unique(folds):
        fit=np.flatnonzero(folds!=fold); held=np.flatnonzero(folds==fold)
        model,index,losses=select_krr(x[fit],y[fit],candidates)
        predictions[held]=predict_krr(model,x[held])
        trace.append(dict(fold=int(fold),fit_ids=fit.tolist(),held_ids=held.tolist(),
                          selected=index,selection_mse=float(losses[index])))
    return predictions,trace


def paired_difference(left, right):
    """Mean(left-right), sample SD, Monte Carlo SE and t95 interval across repeats."""
    a,b=np.asarray(left,float),np.asarray(right,float)
    if a.ndim!=1 or a.shape!=b.shape or len(a)<2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Aligned finite one-dimensional repeated observations, n>=2')
    d=a-b;mean=float(d.mean());sd=float(d.std(ddof=1));se=sd/np.sqrt(len(d))
    half=float(t.ppf(.975,len(d)-1)*se)
    return dict(n=len(d),mean=mean,sd=sd,mc_se=float(se),t95=[mean-half,mean+half])


def exact_null_minimum(n, candidates):
    """E[min_j Binomial(n,.5)/n] from discrete survival sums; independent candidates."""
    if not isinstance(n,int) or not isinstance(candidates,int) or n<1 or candidates<1:
        raise ValueError('Positive integer row and candidate counts required')
    return float(np.sum(binom.sf(np.arange(n),n,.5)**candidates)/n)


def null_reanalysis(rows):
    """Historical 200 paired simulation repeats, no new random stream or relabeling."""
    seeds=sorted({r['seed'] for r in rows}); budgets=sorted({r['candidates'] for r in rows})
    table={(r['seed'],r['candidates']):r for r in rows}
    if len(table)!=len(rows) or len(table)!=len(seeds)*len(budgets):raise ValueError('Incomplete or duplicated simulation panel')
    summary=[]
    for budget in budgets:
        val=np.array([table[s,budget]['validation_error'] for s in seeds])
        test=np.array([table[s,budget]['test_error'] for s in seeds])
        base=np.array([table[s,budgets[0]]['test_error']-table[s,budgets[0]]['validation_error'] for s in seeds])
        summary.append(dict(candidates=budget,validation_mean=float(val.mean()),test_mean=float(test.mean()),
            exact_validation=exact_null_minimum(80,budget),optimism=paired_difference(test,val),
            paired_optimism_increase=paired_difference(test-val,base)))
    return summary


def experiment(repetitions=30, n=64, n_test=4096, start_seed=5900):
    """Internal vs external §5.3 extension on §3.1 mixture, same-size fresh targets.

    Inner LOO uses Eq3 including b; no standardization learned outside folds.
    Outer4 folds are label-blind shuffled equal blocks. Test data are generated
    only after selection. Independent D draws, not correlated folds, form units.
    """
    if repetitions<2 or n<8 or n%4 or n_test<1:raise ValueError('Need repeats>=2, n>=8 divisible by4 and test rows>=1')
    configs=candidate_grid(); records=[]
    for rep in range(repetitions):
        seed=start_seed+rep;x,y=mixture(n,seed)
        folds=np.random.default_rng(seed+20000).permutation(np.arange(n)%4)
        full,external_index,full_losses=select_krr(x,y,configs)
        internal,trace=nested_predictions(x,y,folds,configs)
        # Freeze all selected indices before creating fresh evaluation data.
        fresh_x,fresh_y=mixture(n_test,seed+100000)
        full_prediction=predict_krr(full,fresh_x)
        external=np.empty(n); inner_fresh=[];external_fresh=[]
        for item in trace:
            fit=np.asarray(item['fit_ids']);held=np.asarray(item['held_ids'])
            inner_model=krr_fit(x[fit],y[fit],**configs[item['selected']])
            external_model=krr_fit(x[fit],y[fit],**configs[external_index])
            external[held]=predict_krr(external_model,x[held])
            p_inner=predict_krr(inner_model,fresh_x);p_external=predict_krr(external_model,fresh_x)
            inner_fresh.append(float(np.mean((fresh_y-p_inner)**2)))
            external_fresh.append(float(np.mean((fresh_y-p_external)**2)))
            item['internal_fresh_mse']=inner_fresh[-1];item['external_fresh_mse']=external_fresh[-1]
        records.append(dict(seed=seed,x=x.tolist(),y=y.tolist(),folds=folds.tolist(),trace=trace,
            external_selected=external_index,full_selection_losses=full_losses.tolist(),
            selected_loo_mse=float(full_losses[external_index]),full_fresh_mse=float(np.mean((fresh_y-full_prediction)**2)),
            internal_mse=float(np.mean((y-internal)**2)),external_mse=float(np.mean((y-external)**2)),
            internal_fresh_mse=float(np.mean(inner_fresh)),external_fresh_mse=float(np.mean(external_fresh)),
            internal_predictions=internal.tolist(),external_predictions=external.tolist(),
            full_fresh_error=float(np.mean((full_prediction>=0)!=(fresh_y>=0)))))
    summaries={}
    for name,left,right in [('selected_loo_optimism','full_fresh_mse','selected_loo_mse'),
       ('internal_optimism','internal_fresh_mse','internal_mse'),
       ('external_optimism','external_fresh_mse','external_mse'),
       ('protocol_gap','internal_mse','external_mse')]:
        summaries[name]=paired_difference([r[left] for r in records],[r[right] for r in records])
    means={k:float(np.mean([r[k] for r in records])) for k in ['selected_loo_mse','full_fresh_mse','internal_mse','external_mse','internal_fresh_mse','external_fresh_mse','full_fresh_error']}
    return dict(scope='Paper equations/data generator with new finite-grid nested protocol',
       config=dict(repetitions=repetitions,n=n,n_test=n_test,start_seed=start_seed,outer_folds=4,inner='analytic LOO PRESS/n',candidates=configs),
       means=means,summary=summaries,records=records,paper_reproduction='INCOMPARABLE',
       uncertainty='Independent synthetic data repeats; outer folds are averaged inside each repeat')
