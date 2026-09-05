"""L048 local experiment: 3 real tables × 3 seeds, validation-selected models."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
from relkit.dcnv2 import DCNv2, train_dcn, predict_dcn
from relkit.saint_experiment import prepare, environment, summarize


def run(datasets=('credit_g', 'diabetes', 'blood_transfusion'), seeds=(0, 1, 2), epochs=20,
        model_cls=DCNv2, train_fn=train_dcn, predict_fn=predict_dcn):
    torch.set_num_threads(1)
    begin = time.time()
    per, losses, protocols, diagnostics = {}, {}, {}, {}
    for name in datasets:
        fr = prepare(name, split_seed=5)
        protocols[name] = fr['meta']
        protocols[name]['evaluation_context'] = 'Independent rows; no batch normalization or row attention.'
        xn, xc, y, tr, va, te = [fr[k] for k in ('xn','xc','y','train','valid','test')]
        per[name], losses[name] = [], []
        for seed in seeds:
            aucs, logs = {}, {}
            for arm in ('mlp', 'dense', 'lowrank'):
                torch.manual_seed(seed)
                model = model_cls(xn.shape[1], fr['cards'], layout='mlp' if arm=='mlp' else 'parallel',
                                  kind='dense' if arm=='mlp' else arm,
                                  rank=min(4,max(1,(xn.shape[1]+4*len(fr['cards']))//4)))
                model, history = train_fn(model, xn, xc, y, tr, va, seed=seed, epochs=epochs)
                p = predict_fn(model, xn[te], xc[te])
                aucs[arm] = float(roc_auc_score(y[te], p))
                logs[arm] = float(log_loss(y[te], p, labels=[0,1]))
                single = predict_fn(model, xn[te][:10], xc[te][:10], batch_size=1)
                diagnostics[f'{name}/{seed}/{arm}'] = {
                    'parameters': sum(v.numel() for v in model.parameters()),
                    'best_epoch': min(history, key=lambda r:r['valid_loss'])['epoch'],
                    'max_batch_change': float(np.max(np.abs(single-p[:10]))),
                    'test_probability_sha256': hashlib.sha256(p.tobytes()).hexdigest()}
            features = np.column_stack([np.nan_to_num(xn),xc]).astype(object)
            cats = list(range(xn.shape[1],features.shape[1]))
            for j in cats: features[:,j] = features[:,j].astype(str)
            cb = CatBoostClassifier(iterations=300, depth=6, learning_rate=.05, verbose=False,
                                    random_seed=seed, thread_count=1, allow_writing_files=False,
                                    eval_metric='Logloss')
            cb.fit(features[tr],y[tr],cat_features=cats,eval_set=(features[va],y[va]),use_best_model=True)
            p = cb.predict_proba(features[te])[:,1]
            aucs['catboost'] = float(roc_auc_score(y[te],p))
            logs['catboost'] = float(log_loss(y[te],p,labels=[0,1]))
            per[name].append(aucs); losses[name].append(logs)
            print(name,seed,aucs,flush=True)
    models = ('mlp','dense','lowrank','catboost')
    return {'lesson':48, 'environment':environment(), 'seeds':list(seeds),
            'config':{'epochs':epochs,'embedding_dim':4,'cross_depth':2,'rank':'min(4, max(1, d//4))','hidden':[32,32],
                      'batch_size':64,'lr':.001,'optimizer':'Adam','clip_norm':10,
                      'selection':'validation log loss each epoch', 'split_seed':5,
                      'catboost':{'iterations':300,'depth':6,'learning_rate':.05}},
            'protocols':protocols, 'results':summarize(per,models),
            'logloss_by_seed':losses, 'diagnostics':diagnostics,'wall_s':time.time()-begin,
            'scope':'Substitute tables, untuned fixed budgets; unequal parameter counts. Not Table 6 reproduction.'}


if __name__ == '__main__':
    result=run()
    Path(__file__).with_name('_verify_l048_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['results'],indent=2))
