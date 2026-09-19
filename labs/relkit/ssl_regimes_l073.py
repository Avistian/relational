"""L073 paired label-budget study. Canonical visible implementation, not a paper reproduction."""
import copy
import hashlib
import platform
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats import t, rankdata, friedmanchisquare, studentized_range
from relkit.contrastive_l072 import SCARF, corrupt, draw_view, scarf_loss


def nested_labels(train, fractions, seed):
    """One label-blind permutation, then increasing prefixes of floor(f*n) rows."""
    if len(set(train)) != len(train) or not fractions or any(f<=0 or f>1 for f in fractions):
        raise ValueError('Unique training IDs and fractions in (0,1] required')
    if any(a>=b for a,b in zip(fractions,fractions[1:])):
        raise ValueError('Fractions must increase')
    order=np.random.default_rng(seed).permutation(train)
    sizes=[int(f*len(train)) for f in fractions]
    if min(sizes)<2:raise ValueError('At least two labels per budget required')
    return [order[:n].tolist() for n in sizes]


def paired_gains(rows, treatment, control):
    """One dataset and fraction at a time. Join by seed, never by list position."""
    maps={a:{} for a in [treatment,control]}
    for r in rows:
        if r['arm'] in maps:
            m=maps[r['arm']]
            if r['seed'] in m:raise ValueError('Duplicate pair key')
            m[r['seed']]=r['accuracy']
    a,b=maps[treatment],maps[control]
    if not a or a.keys()!=b.keys():raise ValueError('Incomplete matched comparison')
    return [a[s]-b[s] for s in sorted(a)]


def crossing_brackets(fractions, gains):
    """Adjacent strict sign reversals of means only; a zero is an observed tie."""
    if len(fractions)!=len(gains) or any(a>=b for a,b in zip(fractions,fractions[1:])):
        raise ValueError('Aligned increasing grid required')
    if not np.isfinite(gains).all():raise ValueError('Finite gains required')
    return [[float(fractions[i]),float(fractions[i+1])] for i in range(len(gains)-1)
            if gains[i]*gains[i+1]<0]


def load_regime(name, seed):
    data={'wine':load_wine,'breast_cancer':load_breast_cancer,'digits':load_digits}[name]()
    raw=data.data.astype('float32');y=data.target
    tr,held=train_test_split(np.arange(len(y)),test_size=.3,stratify=y,random_state=seed)
    va,te=train_test_split(held,test_size=2/3,stratify=y[held],random_state=seed)
    scale=StandardScaler().fit(raw[tr])
    return scale.transform(raw).astype('float32'),y,len(data.target_names),{
        'train':tr.tolist(),'validation':va.tolist(),'test':te.tolist(),
        'feature_mean':scale.mean_.tolist(),'data_sha256':hashlib.sha256(raw.tobytes()+y.tobytes()).hexdigest()}


def initialize(x, seed, epochs=40):
    """Create paired random and pretrained encoders using only training features."""
    torch.manual_seed(seed);random=SCARF(x.shape[1]);net=copy.deepcopy(random)
    opt=torch.optim.Adam(net.parameters(),lr=.001);x=torch.tensor(x);history=[]
    for epoch in range(epochs):
        losses=[]
        for ix in torch.randperm(len(x)).split(64):
            if len(ix)<2:continue
            b=x[ix];loss=scarf_loss(net(b),net(draw_view(b,x,.6)))
            opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return random.eval(),net.eval(),history


def predict_probe(features, y, ids):
    tr,va,te=[ids[k] for k in ['labeled','validation','test']]
    if len(np.unique(y[tr]))<2:raise ValueError('Label-blind budget has one class; report failure, do not resample')
    scale=StandardScaler().fit(features[tr]);f=scale.transform(features)
    trials=[];best=None
    for c in [.1,1.,10.]:
        model=LogisticRegression(C=c,max_iter=2000).fit(f[tr],y[tr])
        score=float(np.mean(model.predict(f[va])==y[va]));trials.append({'C':c,'validation_accuracy':score})
        if best is None or score>best[0]:best=(score,model,c)
    return best[1].predict(f[te]),{'trials':trials,'selected_C':best[2]}


def predict_tuned(net, x, y, ids, classes, seed, epochs=60):
    """Same head initialization, batches and steps in scratch/pretrained arms."""
    torch.manual_seed(seed+1000)
    encoder=copy.deepcopy(net.encoder)
    model=nn.Sequential(encoder,nn.Linear(64,classes))
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    tr=ids['labeled'];xt=torch.tensor(x[tr]);yt=torch.tensor(y[tr],dtype=torch.long)
    initial=copy.deepcopy(encoder.state_dict())
    model.train()
    for epoch in range(epochs):
        for ix in torch.randperm(len(tr)).split(64):
            loss=F.cross_entropy(model(xt[ix]),yt[ix]);opt.zero_grad();loss.backward();opt.step()
    model.eval()
    with torch.no_grad():pred=model(torch.tensor(x[ids['test']])).argmax(1).numpy()
    delta=sum(float((encoder.state_dict()[k]-v).abs().sum()) for k,v in initial.items())
    return pred,{'encoder_delta':delta,'supervised_epochs':epochs}


def summarize_regimes(records, fractions):
    summaries=[];contrasts=[];crossings=[];rank_audits=[]
    datasets=sorted({r['dataset'] for r in records});arms=sorted({r['arm'] for r in records})
    for d in datasets:
        for f in fractions:
            rows=[r for r in records if r['dataset']==d and r['fraction']==f]
            for a in arms:
                values=[r['accuracy'] for r in rows if r['arm']==a]
                summaries.append({'dataset':d,'fraction':f,'arm':a,'mean':float(np.mean(values)),
                                  'sd':float(np.std(values,ddof=1)),'seed_values':values})
            for treatment,control in [('scarf_ft','scratch'),('scarf_frozen','random'),('scarf_ft','raw'),('scarf_ft','tree')]:
                g=np.array(paired_gains(rows,treatment,control));mean=float(g.mean())
                radius=float(t.ppf(.975,len(g)-1)*g.std(ddof=1)/np.sqrt(len(g)))
                contrasts.append({'dataset':d,'fraction':f,'treatment':treatment,'control':control,
                                  'gains':g.tolist(),'mean':mean,'interval':[mean-radius,mean+radius]})
        for treatment,control in [('scarf_ft','scratch'),('scarf_frozen','random')]:
            gains=[next(c['mean'] for c in contrasts if c['dataset']==d and c['fraction']==f and c['treatment']==treatment and c['control']==control) for f in fractions]
            crossings.append({'dataset':d,'comparison':treatment+' - '+control,'brackets':crossing_brackets(fractions,gains),
                              'ties':[f for f,g in zip(fractions,gains) if g==0]})
    for f in fractions:
        scores=np.array([[next(s['mean'] for s in summaries if s['dataset']==d and s['fraction']==f and s['arm']==a) for a in arms] for d in datasets])
        k=len(arms);n=len(datasets);ranks=np.array([rankdata(-row) for row in scores])
        rank_audits.append({'fraction':f,'arms':arms,'mean_ranks':ranks.mean(0).tolist(),
            'friedman_p':float(friedmanchisquare(*scores.T).pvalue) if n>=3 else None,
            'cd':float(studentized_range.ppf(.95,k,np.inf)/np.sqrt(2)*np.sqrt(k*(k+1)/(6*n)))})
    return summaries,contrasts,crossings,rank_audits


def run_regimes(datasets=('wine','breast_cancer','digits'), seeds=(0,1,2), fractions=(.1,.2,.4,.7,1.), pre_epochs=40, fine_epochs=60):
    started=time.time();torch.set_num_threads(1);records=[];splits=[];traces=[]
    for d in datasets:
        for seed in seeds:
            x,y,k,s=load_regime(d,seed);groups=nested_labels(s['train'],fractions,seed+73)
            splits.append({'dataset':d,'seed':seed,**s,'labeled_groups':groups})
            random,ssl,history=initialize(x[s['train']],seed,pre_epochs)
            traces.append({'dataset':d,'seed':seed,'loss':history})
            with torch.no_grad():
                features={'raw':x,'random':random.represent(torch.tensor(x)).numpy(),
                          'scarf_frozen':ssl.represent(torch.tensor(x)).numpy()}
            for f,labels in zip(fractions,groups):
                ids={**s,'labeled':labels}
                for arm in ['raw','random','scarf_frozen','scratch','scarf_ft','tree']:
                    if arm in features:pred,extra=predict_probe(features[arm],y,ids)
                    elif arm=='tree':
                        model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,min_samples_leaf=5,l2_regularization=1.,early_stopping=False,random_state=seed).fit(x[labels],y[labels])
                        pred=model.predict(x[s['test']]);extra={'iterations':100}
                    else:pred,extra=predict_tuned(random if arm=='scratch' else ssl,x,y,ids,k,seed,fine_epochs)
                    records.append({'dataset':d,'seed':seed,'fraction':f,'arm':arm,
                        'accuracy':float(np.mean(pred==y[s['test']])),'prediction':pred.tolist(),'target':y[s['test']].tolist(),
                        'train_labels':len(labels),'validation_labels':len(s['validation']),'total_development_labels':len(labels)+len(s['validation']),**extra})
            print(d,seed,'completed',flush=True)
    summary,contrasts,crossings,ranks=summarize_regimes(records,fractions)
    return {'config':{'datasets':list(datasets),'seeds':list(seeds),'fractions':list(fractions),'pre_epochs':pre_epochs,'fine_epochs':fine_epochs,'batch':64,'lr':.001,'corruption':.6,'tau':1.},
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'sklearn':sklearn_version},
        'elapsed_seconds':time.time()-started,'records':records,'splits':splits,'traces':traces,
        'summary':summary,'contrasts':contrasts,'crossings':crossings,'rank_audits':ranks,
        'interval_scope':'Descriptive paired t95 intervals over three overlapping split/model repetitions; no independent-dataset or simultaneous coverage claim.',
        'verdict':'INCOMPARABLE to original paper benchmarks; fixed local label-budget experiment.'}
