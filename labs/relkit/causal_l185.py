"""Original L185 SCM and estimators. Assumptions are authored, not discovered."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEEDS = tuple(range(5))


def intervene(frame, badge=None, action=None):
    """Replace a mechanism, reuse E, and return potential outcomes in input order."""
    if badge not in (None, 0, 1) or action not in (None, 0, 1):
        raise ValueError('Interventions must be binary')
    changed = frame.copy()
    if badge is not None:
        changed['B'] = badge
    if action is not None:
        changed['A'] = action
    # B has no arrow to Y in this declared SCM. Setting B never changes U or A.
    probability = .05 + .85 * changed['U'] + .05 * changed['A']
    return (changed['E'].to_numpy() < probability.to_numpy()).astype(np.int64)


def adjusted_effect(frame):
    """Standardize action risk differences over this population's demand mix."""
    means = frame.groupby(['U', 'A'])['Y'].mean()
    weights = frame['U'].value_counts(normalize=True)
    total = 0.0
    for u, weight in weights.items():
        if (u, 0) not in means or (u, 1) not in means:
            raise ValueError('No treatment overlap within a demand stratum')
        total += weight * (means.loc[(u, 1)] - means.loc[(u, 0)])
    return float(total)


def assemble(tables):
    """Validate keys, join at customer grain, and enforce the availability contract."""
    c, q, a, y = (tables[k].copy() for k in ('companies','customers','actions','outcomes'))
    for table, key in ((c,'company_id'),(q,'customer_id'),(a,'customer_id'),(y,'customer_id')):
        if table[key].isna().any() or table[key].duplicated().any():
            raise ValueError('Null or duplicate primary key')
    ids = set(q.customer_id)
    if set(a.customer_id) != ids or set(y.customer_id) != ids:
        raise ValueError('Missing or extra action/outcome identity')
    if not set(q.company_id).issubset(set(c.company_id)):
        raise ValueError('Missing company foreign key')
    f = q.merge(c,on='company_id',validate='many_to_one').merge(a,on='customer_id',validate='one_to_one').merge(y,on='customer_id',validate='one_to_one')
    if not ((f.available_at <= f.cutoff) & (f.action_at <= f.cutoff) & (f.outcome_at > f.cutoff)).all():
        raise ValueError('Feature/action availability or outcome horizon violated')
    if f.groupby('company_id')['split'].nunique().max() != 1:
        raise ValueError('Company leaks across splits')
    if f[['customer_id','cutoff']].duplicated().any():
        raise ValueError('Duplicate query identity')
    return f.sort_values(['customer_id','cutoff']).reset_index(drop=True)


def generate(seed):
    """Independent streams for company traits, assignment, outcomes, and split."""
    traits, assign, outcome, splitter = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    u = traits.binomial(1,.5,1000)
    b = u ^ traits.binomial(1,.05,1000)
    order = splitter.permutation(1000)
    split = np.empty(1000,dtype=object)
    split[order[:600]], split[order[600:800]], split[order[800:]] = 'train','validation','test'
    company = np.repeat(np.arange(1000),20)
    customer = np.arange(20000)
    action = assign.binomial(1,.1+.8*u[company])
    noise = outcome.random(20000)
    frame = pd.DataFrame({'U':u[company],'B':b[company],'A':action,'E':noise})
    return {
        'companies':pd.DataFrame({'company_id':np.arange(1000),'U':u,'B':b,'available_at':0}),
        'customers':pd.DataFrame({'customer_id':customer,'company_id':company,'cutoff':1,'split':split[company]}),
        'actions':pd.DataFrame({'customer_id':customer,'A':action,'action_at':1}),
        'outcomes':pd.DataFrame({'customer_id':customer,'Y':intervene(frame),'E':noise,'outcome_at':2})}


def predict_frequency(train, query, columns):
    """Training labels only; global training mean if a combination is unseen."""
    if set(columns) - {'U','B','A'}:
        raise ValueError('Illegal predictor column')
    if not (train['split']=='train').all():
        raise ValueError('Fitting is restricted to training rows')
    means = train.groupby(columns)['Y'].mean().rename('score').reset_index()
    return query[columns].merge(means,on=columns,how='left',validate='many_to_one',sort=False)['score'].fillna(train.Y.mean()).to_numpy()


def run_seed(seed):
    tables = generate(seed)
    f = assemble(tables)
    train = f[f.split=='train']
    test = f[f.split=='test'].copy()
    prediction_parts, metrics = [], {'seed':int(seed)}
    for split in ('validation','test'):
        q = f[f.split==split]
        for name, columns in [('badge',['B']),('action',['A']),('demand_action',['U','A'])]:
            score = predict_frequency(train,q,columns)
            part = q[['customer_id','cutoff','Y']].copy()
            part['split'],part['model'],part['score'] = split,name,score
            prediction_parts.append(part)
            metrics[split+'_'+name+'_auroc'] = float(roc_auc_score(q.Y,score))
    potentials = test[['customer_id','cutoff','company_id','U','B','A','Y','E']].copy()
    for name,kwargs in [('badge0',{'badge':0}),('badge1',{'badge':1}),('action0',{'action':0}),('action1',{'action':1})]:
        potentials[name] = intervene(test,**kwargs)
    association = test.groupby('A').Y.mean()
    metrics.update({
        'naive_action_difference':float(association.loc[1]-association.loc[0]),
        'adjusted_action_effect':adjusted_effect(test),
        'paired_action_effect':float((potentials.action1-potentials.action0).mean()),
        'paired_badge_effect':float((potentials.badge1-potentials.badge0).mean()),
        'expected_action_effect':.05,'expected_badge_effect':0.0,
        'observed_purchase_rate':float(test.Y.mean()),
        'badge1_policy_gain':float((potentials.badge1-potentials.Y).mean()),
        'action1_policy_gain':float((potentials.action1-potentials.Y).mean()),
        'expected_action1_policy_gain':float((.05*(1-test.A)).mean()),
        'train_rows':len(train),'validation_rows':int((f.split=='validation').sum()),'test_rows':len(test)})
    return tables,pd.concat(prediction_parts,ignore_index=True),potentials,metrics


def full_experiment():
    """The entire approved experiment, without data-size or seed shortcuts."""
    outputs = [run_seed(seed) for seed in SEEDS]
    rows = [x[3] for x in outputs]
    keys = [k for k,v in rows[0].items() if isinstance(v,float)]
    summary = {k:{'mean':float(np.mean([r[k] for r in rows])),
                  'sd_across_seeds':float(np.std([r[k] for r in rows],ddof=1))} for k in keys}
    report = {'experiment':'L185 relational-shortcut-intervention-v1','status':'COMPLETE_SYNTHETIC_EXPERIMENT',
              'seeds':list(SEEDS),'rows_per_seed':20000,'per_seed':rows,'summary':summary,
              'paper_reproduction':'NOT_ESTABLISHED','real_world_causality':'NOT_ESTABLISHED',
              'learner':'PENDING_WRITTEN_DEFENSE','cloud_usd':0}
    return outputs,report
