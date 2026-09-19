"""CARTE released downstream benchmark replay with full visible upstream source.

The appendix reconstructs sources/carte_paper before running this operator. The full
model, graph transform, ensemble and training loop are therefore readable/editable
in the notebook. Published hyperparameters and reference scores are keyed by
(dataset, budget, split); no test-driven retuning and no compact linear-head substitute.
"""
import ast
import gc
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

ROOT=Path(__file__).resolve().parent
COMMIT='f54690da4cddbedd1e1a9113a312f85783d2c125'  # public example data and score archive
PAPER_COMMIT='e1079b5f470094427935763daab74e33a23e3ed0'
DATASETS=('wina_pl','wine_dot_com_prices','wine_vivino_price')


def fetch_carte_data(name,cache):
    """Pinned full public tables; wina_pl is the benchmark's spelling of wine_pl."""
    filename='wine_pl' if name=='wina_pl' else name
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    paths=[]
    for file in [f'{filename}.parquet',f'config_{filename}.json']:
        path=cache/file
        if not path.exists():
            urllib.request.urlretrieve(f'https://raw.githubusercontent.com/soda-inria/carte/{COMMIT}/data/single_tables/{file}',path)
        paths.append(path)
    return pd.read_parquet(paths[0]),json.loads(paths[1].read_text()),hashlib.sha256(paths[0].read_bytes()).hexdigest()


def load_carte_sources():
    sys.path.insert(0,str(ROOT/'sources'))
    from carte_ai.src.carte_estimator import CARTERegressor
    from carte_ai.src.carte_table_to_graph import Table2GraphTransformer
    from carte_ai.src.evaluate_utils import set_split
    # sklearn 1.8 removed the old mixin marker used by this pinned estimator.
    if not hasattr(CARTERegressor,"_estimator_type"):
        CARTERegressor._estimator_type="regressor"
    return CARTERegressor,Table2GraphTransformer,set_split


def run_carte(fasttext_path,checkpoint,smoke=False,device='cpu',output=None,
              datasets=DATASETS,budgets=(32,64,128,256,512,1024,2048),seeds=tuple(range(1,11)),release='paper-era'):
    torch.set_num_threads(1)
    if release=='paper-era':
        from carte_compat import load_paper_carte
        CARTERegressor,Table2GraphTransformer,set_split=load_paper_carte(fasttext_path,checkpoint)
    else:CARTERegressor,Table2GraphTransformer,set_split=load_carte_sources()
    params=pd.read_csv(ROOT/'carte_bestparams.csv');scores=pd.read_csv(ROOT/'carte_singletable.csv')
    if smoke:datasets=datasets[:1];budgets=(32,);seeds=(1,)
    start=time.time()
    out=dict(target='CARTE released single-table benchmark rows',release=PAPER_COMMIT if release=='paper-era' else COMMIT,data_release=COMMIT,smoke=smoke,records=[],
             config=dict(datasets=list(datasets),budgets=list(budgets),seeds=list(seeds),num_model=1 if smoke else 15,
                         max_epoch=2 if smoke else 500,patience=40,batch=16,validation_fraction=.2),
             verdict='INCOMPARABLE' if smoke else 'RELEASE_REPLICATION_MEASURED',
             remaining=['Downstream replay with released pretrained checkpoint; fresh YAGO pretraining is not rerun.',
                        'The June 2024 source is contemporary with publication; authors did not identify an exact execution commit.',
                        'Public example-table hashes are recorded; historical benchmark input hashes were not published.',
                        'Other benchmark datasets, baseline families and joint source-table experiments are separate targets.'])
    if not Path(fasttext_path).is_file():raise FileNotFoundError('Supply the complete cc.en.300.bin FastText model; the 384-row teaching cache is insufficient')
    if not Path(checkpoint).is_file():raise FileNotFoundError('Supply released kg_pretrained.pt')
    out['checkpoint_sha256']=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    for name in datasets:
        frame,config,digest=fetch_carte_data(name,Path.home()/'.cache/relational-paper/carte')
        from carte_vectors import full_table_vectors
        vectors,vector_identity=full_table_vectors(frame.drop(columns=config['target_name']),fasttext_path,Path.home()/'.cache/relational-paper/carte')
        out.setdefault('vector_identity',{})[name]=vector_identity
        for budget in budgets:
            for seed in seeds:
                key=(params.data_name==name)&(params.num_train==budget)&(params.random_state==seed)
                target_key=(scores.data_name==name)&(scores.num_train==budget)&(scores.random_state==seed)
                if key.sum()!=1 or target_key.sum()!=1:raise ValueError('Missing or ambiguous published configuration/score')
                hp=ast.literal_eval(params.loc[key,'best_param'].iloc[0]);paper=float(scores.loc[target_key,'score'].iloc[0])
                Xtr,Xte,ytr,yte=set_split(frame,config,budget,seed)
                try:
                    transformer=Table2GraphTransformer() if release=='paper-era' else Table2GraphTransformer(fasttext_model_path=str(fasttext_path))
                    transformer.lm_model_=vectors  # exact fixed FastText vectors; numeric fitting still occurs per split
                    graphs=transformer.fit_transform(Xtr,y=ytr)
                    # A smoke test reduces evaluation only, never contaminates paper output.
                    if smoke:Xte=Xte.iloc[:32];yte=yte[:32]
                    test=transformer.transform(Xte)
                    for graph in graphs+test:
                        if not torch.isfinite(graph.x).all() or not torch.isfinite(graph.edge_attr).all():
                            raise FloatingPointError('Released PowerTransformer produced nonfinite graph values; do not silently substitute normalization in a paper run')
                except (ValueError,FloatingPointError) as error:
                    out['records'].append(dict(dataset=name,budget=budget,split_seed=seed,
                        data_sha256=digest,status='FAILED_RELEASE_PREPROCESSING',error=str(error),paper_r2=paper))
                    if output:Path(output).write_text(json.dumps(out,indent=2))
                    del transformer;gc.collect()
                    print(f'CARTE {name}/{budget}/{seed}: FAILED_RELEASE_PREPROCESSING: {error}',flush=True)
                    continue
                # The fixed full-table cache is reused; release split-specific transformer state.
                del transformer;gc.collect()
                checkpoint_kw={} if release=='paper-era' else dict(pretrained_model_path=str(checkpoint))
                model=CARTERegressor(**hp,batch_size=16,num_model=1 if smoke else 15,
                    max_epoch=2 if smoke else 500,n_jobs=1,random_state=0,device=device,
                    **checkpoint_kw)
                model.fit(graphs,ytr);prediction=model.predict(test)
                value=float(r2_score(yte,prediction))
                out['records'].append(dict(dataset=name,budget=budget,split_seed=seed,model_seed=0,
                    params=hp,rows=len(frame),train_indices=Xtr.index.tolist(),test_indices=Xte.index.tolist(),
                    data_sha256=digest,r2=value,paper_r2=paper,gap=value-paper,prediction=prediction.tolist(),
                    target=yte.tolist(),validation_losses=np.asarray(model.valid_loss_).tolist(),
                    comparison='INCOMPARABLE' if smoke else ('WITHIN_0.02_R2' if abs(value-paper)<=.02 else 'OUTSIDE_0.02_R2')))
                out['seconds']=time.time()-start
                if output:Path(output).write_text(json.dumps(out,indent=2))
                print(f'CARTE {name} budget={budget} seed={seed}: R2={value:.5f}, released={paper:.5f}',flush=True)
                del model,graphs,test;gc.collect()
    out['failed_rows']=sum(r.get('status','')=='FAILED_RELEASE_PREPROCESSING' for r in out['records'])
    if out['failed_rows']:out['verdict']='REPLICATION_INCOMPLETE'
    if output:Path(output).write_text(json.dumps(out,indent=2))
    return out

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--fasttext',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--device',default='cpu');p.add_argument('--output',default='carte-paper-results.json')
    p.add_argument('--release',choices=['paper-era','current'],default='paper-era')
    p.add_argument('--datasets',default=','.join(DATASETS));p.add_argument('--budgets',default='32,64,128,256,512,1024,2048');p.add_argument('--seeds',default=','.join(map(str,range(1,11))))
    a=p.parse_args();run_carte(a.fasttext,a.checkpoint,a.smoke,a.device,a.output,tuple(a.datasets.split(',')),tuple(map(int,a.budgets.split(','))),tuple(map(int,a.seeds.split(','))),release=a.release)
