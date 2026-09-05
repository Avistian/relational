"""L049 closer-to-paper operators. Single-dataset score attempt, not suite replication."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import torch
from _verify_l049 import experiment,ARMS
from relkit.claim_models import ExcelFormer,train_excel,predict_excel,spa_attention,feat_mix
from relkit.claim_data import paper_data
from relkit.saint_experiment import summarize,environment

PRESETS={
    'smoke':dict(epochs=2,d=16,layers=1,heads=2,lr=.001,batch=256,patience=2,dropout=.1,trees=20),
    'closer':dict(epochs=100,d=256,layers=3,heads=32,lr=.0001,batch=256,patience=32,dropout=.3,trees=1000),
    'paper':dict(epochs=1000,d=256,layers=3,heads=32,lr=.0001,batch=256,patience=32,dropout=.3,trees=4096)}


def run(preset='closer',out='data/cache/l049-closer',device=None,model_cls=ExcelFormer,
        train_fn=train_excel,predict_fn=predict_excel,attention_fn=spa_attention,mix_fn=feat_mix):
    """Resume only identical source/config/data/device runs. Keep the student's code live."""
    cfg=PRESETS[preset];folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    data=paper_data('pima');device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    # In notebooks inspect.getsource can fail. Hash bytecode plus source where available.
    identities=[]
    dependencies=model_cls.__init__.__globals__
    for obj in (model_cls,dependencies['GatedTokenizer'],dependencies['SPABlock'],
                train_fn,predict_fn,attention_fn,mix_fn):
        try: identities.append(inspect.getsource(obj))
        except (OSError,TypeError):
            import marshal
            funcs=[obj.__init__,obj.forward] if inspect.isclass(obj) else [obj]
            identities.extend(marshal.dumps(f.__code__).hex() for f in funcs)
    contract={'preset':preset,'config':cfg,'data':data['meta'],'environment':environment(),
              'device':device,'implementation':identities}
    digest=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    contract_path=folder/'contract.json'
    if contract_path.exists():
        assert json.loads(contract_path.read_text())['digest']==digest,'Changed implementation/protocol: choose a fresh output folder.'
    else: contract_path.write_text(json.dumps({'digest':digest,'contract':contract},indent=2))
    rows=[];seeds=(0,) if preset=='smoke' else (0,1,2)
    for seed in seeds:
        path=folder/f'seed-{seed}.json'
        if path.exists(): r=json.loads(path.read_text())
        else:
            r=experiment(data,seeds=(seed,),device=device,model_cls=model_cls,train_fn=train_fn,predict_fn=predict_fn,
                         artifact_dir=folder/'weights',**cfg)
            path.write_text(json.dumps(r,indent=2))
        rows.extend(r['auroc'])
        print('completed seed',seed,flush=True)
    result={'preset':preset,'summary':summarize({'pima':rows},ARMS),'contract_sha256':digest,
            'environment':environment(),'paper_target':{'table':'ExcelFormer v5 Table 14, Pima-Indians-Diabetes','feat_mix_auroc':.8356,'catboost_tuned_auroc':.7528},
            'verdict':'INCOMPARABLE','gaps':['One released split versus published five-run mean.',
            'Chosen batch size/epoch cap; no paper hyperparameter search.',
            'Our per-row Bernoulli Feat-Mix sampler is a documented implementation choice; source augmentation parity not established.',
            'Published upstream preprocessing fit scope is not reconstructed.',
            'Single dataset; the 96/21-table suites and full Trompt benchmark remain NOT_RUN.']}
    (folder/'summary.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preset',choices=PRESETS,default='smoke');p.add_argument('--out',default='data/cache/l049-repro');p.add_argument('--device',default=None)
    a=p.parse_args();print(json.dumps(run(a.preset,a.out,a.device),indent=2))
