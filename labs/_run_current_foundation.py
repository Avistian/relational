"""L070 version-pinned current-model extension on the EXACT historical panel rows."""
import hashlib,importlib.metadata,json,time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from relkit.foundation_benchmark import random_task,encode_train,error,ROOT
from relkit.benchmark_core import paired_summary


def run(output=None):
    from tabpfn import TabPFNClassifier
    from tabicl import TabICLClassifier
    if importlib.metadata.version('tabpfn')!='8.5.0' or importlib.metadata.version('tabicl')!='2.2.0':
        raise RuntimeError('Use the explicitly pinned current-version environment')
    torch.set_num_threads(1)
    manifest=json.loads((ROOT/'_sources_foundation.json').read_text())['current_checkpoints']
    arms={'TabPFN-2.5-synthetic':'v25','TabPFN-3':'v3','TabICLv2':'tabiclv2'}
    out=Path(output or ROOT/'_verify_l070_current_results.json')
    operator=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result=dict(records=[],datasets={},operator_sha256=operator,checkpoints=manifest,
        versions={p:importlib.metadata.version(p) for p in ['tabpfn','tabicl','torch','numpy','scikit-learn','pandas']},
        scope='Current-checkpoint extension of the fixed five-task L070 panel; one view, identical historical rows',verdict='INCOMPARABLE')
    if out.exists():
        old=json.loads(out.read_text())
        if any(old[k]!=result[k] for k in ['operator_sha256','checkpoints','versions']):raise ValueError('Run identity changed; use a fresh output path')
        result=old
    baseline=json.loads((ROOT/'_verify_l070_results.json').read_text())
    with threadpool_limits(limits=1):
        for name in ['diabetes','blood_transfusion','kc1','phoneme','breast_cancer']:
            raw,y,reg,audit=random_task(name,600,70);x=encode_train(raw);key=name+'/random'
            assert audit==baseline['datasets'][key], 'Dataset/row identity differs from historical panel'
            result['datasets'][key]=audit
            for seed in [0,1,2]:
                for arm,identifier in arms.items():
                    if any(r['dataset']==key and r['seed']==seed and r['arm']==arm for r in result['records']):continue
                    item=manifest[identifier];path=ROOT/'data/cache/foundation'/item['filename']
                    if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Checkpoint checksum changed')
                    begin=time.perf_counter()
                    if arm=='TabICLv2':
                        m=TabICLClassifier(n_estimators=1,device='cpu',random_state=seed,n_jobs=1,
                            model_path=str(path),checkpoint_version=item['filename'],allow_auto_download=False,use_amp=False)
                    else:m=TabPFNClassifier(n_estimators=1,device='cpu',random_state=seed,n_preprocessing_jobs=1,model_path=path)
                    m.fit(x['train'],y['train']);val=m.predict_proba(x['val'])[:,list(m.classes_).index(1)]
                    fit=time.perf_counter()-begin;begin=time.perf_counter();pred=m.predict_proba(x['test'])[:,list(m.classes_).index(1)];predict=time.perf_counter()-begin
                    row=dict(dataset=key,arm=arm,seed=seed,error=error(y['test'],pred,False),metric='log_loss',seconds=fit+predict,
                        fit_selection_seconds=fit,predict_seconds=predict,validation_errors=[error(y['val'],val,False)],selected=0,epoch=None,
                        predictions=pred.tolist(),targets=y['test'].tolist(),operator_sha256=operator,checkpoint_sha256=item['sha256'])
                    result['records'].append(row);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
                    print(name,seed,arm,round(row['error'],5),round(row['seconds'],2),flush=True)
    result['summary']=paired_summary(result['records']);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output');run(p.parse_args().output)
