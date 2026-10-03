"""Full released-dimensional HyperFast course inference; immutable per-run artifacts."""
import argparse,hashlib,json,time,sys,gc
from pathlib import Path
import numpy as np
import torch
from relkit.hyper_b07a import class_weights,retrieval_bias
from relkit.serving_b07a import load_hypernetwork,GeneratedPredictor
from relkit.hyperfast_b07a import transform_data_for_main_network,forward_main_network

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def source_prediction_parity(lab,predictor,X):
    """Use unmodified release wrapper on identical fitted state, bypass5GB duplicate load."""
    sys.path.insert(0,str(Path(lab)/'sources/b07a/hyperfast'))
    from hyperfast.hyperfast import HyperFastClassifier
    from hyperfast.utils import TorchPCA as OriginalPCA
    from sklearn.impute import SimpleImputer
    ref=HyperFastClassifier.__new__(HyperFastClassifier)
    ref.device='cpu';ref.batch_size=512;ref.feature_bagging=False;ref.nn_bias_mini_batches=True
    ref._cfg=predictor.cfg;ref._scaler=predictor.scaler;ref._cat_features=[]
    ref._numerical_feature_idxs=np.arange(X.shape[1]);ref._num_imputer=SimpleImputer().fit(X)
    ref.classes_=predictor.classes;ref.n_classes_=len(ref.classes_);ref.n_features_in_=X.shape[1]
    original_pca=OriginalPCA(predictor.cfg.n_dims)
    original_pca.mean_=predictor.pca.mean_;original_pca.components_=predictor.pca.components_
    ref._rfs=[predictor.rf];ref._pcas=[original_pca];ref._main_networks=[predictor.layers];ref._nnbias=[predictor.bias]
    ref._X_preds=[predictor.support];ref._y_preds=[predictor.labels]
    diffs={}
    for retrieval in [False,True]:
        ref.nn_bias=retrieval
        a=ref.predict_proba(X);b=predictor.predict(X,retrieval)
        np.testing.assert_allclose(a,b,rtol=1e-5,atol=2e-6)
        diffs[str(retrieval)]=float(np.max(np.abs(a-b)))
    return diffs

def run_course(lab,output,checkpoint=None,head_fn=class_weights,bias_fn=retrieval_bias):
    lab=Path(lab);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'results.json').exists():raise ValueError('Refusing to overwrite complete evidence')
    cfg=json.loads((lab/'evidence/b07a/course-protocol.json').read_text())
    meta=json.loads((lab/'sources/b07a/checkpoint.json').read_text());checkpoint=Path(checkpoint or lab/'data/b07a/hyperfast.ckpt')
    assert checkpoint.stat().st_size==meta['bytes'] and digest(checkpoint)==meta['sha256'],'Checkpoint authentication failed'
    for name,item in cfg['datasets'].items():assert digest(lab/f'data/b07a/{name}.npz')==item['sha256'],'Data authentication failed'
    for name,sha in cfg['source_pin']['files'].items():assert digest(lab/'sources/b07a/hyperfast'/name)==sha,'Source authentication failed'
    for name,sha in cfg['implementation_hashes'].items():assert digest(lab/name)==sha,'Implementation authentication failed'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.monotonic();model,modelcfg=load_hypernetwork(checkpoint,head_fn);load_seconds=time.monotonic()-start
    records=[];parity=[];refresh=None
    for split in cfg['splits']:
        name,seed=split['dataset'],split['seed'];data=np.load(lab/f'data/b07a/{name}.npz');X,y=data['X'],data['y'];tr,te=split['train'],split['test']
        predictor=GeneratedPredictor(model,modelcfg,bias_fn)
        t=time.monotonic();predictor.fit(X[tr],y[tr],seed,cfg['batch_size']);build=time.monotonic()-t
        identity=predictor.identity();mem=predictor.retained_bytes()
        parity.append(dict(dataset=name,seed=seed,max_abs_difference=source_prediction_parity(lab,predictor,X[te[:32]])))
        with torch.no_grad():
            sx=predictor.support
            _,sh=forward_main_network(transform_data_for_main_network(sx,modelcfg,predictor.rf,predictor.pca),predictor.layers)
            query_hidden=[]
            for j in range(0,len(te),128):
                q=torch.tensor(predictor.scaler.transform(X[te[j:j+128]]),dtype=torch.float32)
                _,qh=forward_main_network(transform_data_for_main_network(q,modelcfg,predictor.rf,predictor.pca),predictor.layers)
                query_hidden.append(qh.numpy())
            feature_path=output/f'{name}-{seed}-features.npz'
            np.savez_compressed(feature_path,support_x=sx.numpy(),support_hidden=sh.numpy(),support_y=predictor.labels.numpy(),query_hidden=np.concatenate(query_hidden))
        for retrieval in [False,True]:
            chunks=[predictor.predict(X[te[j:j+128]],retrieval,True) for j in range(0,len(te),128)]
            probs,raw,logits=[np.concatenate([c[k] for c in chunks]) for k in range(3)]
            arm='retrieval' if retrieval else 'weights_only';path=output/f'{name}-{seed}-{arm}.npz'
            np.savez_compressed(path,row_id=te,target=y[te],probability=probs,raw_logits=raw,logits=logits,output_matrix=predictor.layers[-1][0].numpy(),output_bias=predictor.layers[-1][1].numpy())
            records.append(dict(dataset=name,seed=seed,arm=arm,file=path.name,sha256=digest(path),predictor_sha256=identity,feature_file=feature_path.name,feature_sha256=digest(feature_path),construction_seconds=build,retained_bytes=mem,support_ids=np.array(tr)[predictor.support_indices].tolist(),scaler_mean=predictor.scaler.mean_.tolist(),scaler_scale=predictor.scaler.scale_.tolist(),bias_parameters=predictor.bias.numpy().tolist(),timings=[]))
        pair=records[-2:]
        for size in cfg['timing']['query_rows']:
            q=X[te[:size]]
            for retrieval in [False,True]:predictor.predict(q,retrieval)
            times={False:[],True:[]}
            for rep in range(cfg['timing']['repetitions']):
                for retrieval in ([False,True] if rep%2==0 else [True,False]):
                    t=time.monotonic();predictor.predict(q,retrieval);times[retrieval].append(time.monotonic()-t)
            for rec,retrieval in zip(pair,[False,True]):rec['timings'].append(dict(query_rows=size,seconds=times[retrieval]))
        print(name,seed,'paired prediction COMPLETE; construction',round(build,3),'s',flush=True)
        (output/'partial.json').write_text(json.dumps(dict(records=records,parity=parity),indent=2))
        if name=='banknote' and seed==0:
            changed=GeneratedPredictor(model,modelcfg,bias_fn);updated=tr+[te[0]]
            t=time.monotonic();changed.fit(X[updated],y[updated],seed,cfg['batch_size']);elapsed=time.monotonic()-t
            refresh=dict(dataset=name,seed=seed,added_row_id=te[0],seconds=elapsed,original_sha256=identity,refreshed_sha256=changed.identity(),quality='NOT_EVALUATED after incorporating query label')
            del changed
        del predictor;gc.collect()
    report=dict(name=cfg['name'],protocol_sha256=digest(lab/'evidence/b07a/course-protocol.json'),checkpoint_sha256=meta['sha256'],records=records,source_prediction_parity=parity,refresh=refresh,load_seconds=load_seconds,hypernetwork_parameters=sum(p.numel() for p in model.parameters()),hypernetwork_storage_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),status='COMPLETE')
    (output/'results.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--checkpoint',type=Path);a=p.parse_args();lab=Path(__file__).resolve().parent
    run_course(lab,a.output or lab/'evidence/b07a/runs',a.checkpoint)
