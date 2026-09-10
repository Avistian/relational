"""Retrain corrected TabM only; preserve aligned archived XGB/TabICL evidence."""
import os,json,hashlib,time,importlib.metadata
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from relkit.cross_experiment_v2 import load_binary,oof_library,evaluate_library,family_ablation,summarize
ROOT=Path(__file__).resolve().parent

def main():
    os.chdir(ROOT);start=time.perf_counter()
    old=json.loads(Path('_verify_l057_results.json').read_text())
    manifest=json.loads(Path('_data_l057.json').read_text());out=Path('data/l057-v2');out.mkdir(exist_ok=True)
    rows=[];ablations=[];entries=[]
    with threadpool_limits(limits=1):
        import torch
        torch.set_num_threads(1)
        for entry in manifest['predictions']:
            name,seed=entry['dataset'],entry['seed'];data=load_binary(name,old['config']['cap'])
            for key in ('dev_ids','test_ids','data_hash','openml_id'):assert data[key]==old['datasets'][name][key],key
            source=Path(entry['path']);assert hashlib.sha256(source.read_bytes()).hexdigest()==entry['sha256']
            archive=np.load(source);np.testing.assert_array_equal(data['y'],archive['y']);np.testing.assert_array_equal(data['yt'],archive['yt'])
            z,zt,audit=oof_library(data,['TabM'],seed,old['config'])
            foldpath=source.with_name(source.stem+'-folds.json')
            assert hashlib.sha256(foldpath.read_bytes()).hexdigest()==entry['folds_sha256']
            assert audit==json.loads(foldpath.read_text())
            oof=archive['oof'].copy();test=archive['test'].copy();oof[:,1]=z[:,0];test[:,1]=zt[:,0]
            for ix in (0,2):
                np.testing.assert_array_equal(oof[:,ix],archive['oof'][:,ix]);np.testing.assert_array_equal(test[:,ix],archive['test'][:,ix])
            result=evaluate_library(oof,test,data['y'],data['yt'],old['families'],old['config']['steps'])
            result.update(dataset=name,seed=seed);rows.append(result)
            ab=family_ablation(oof,test,data['y'],data['yt'],old['families'],old['config']['steps'])
            ablations.extend(dict(dataset=name,seed=seed,**a) for a in ab)
            path=out/source.name
            np.savez_compressed(path,oof=oof,test=test,y=data['y'],yt=data['yt'],
                dev_ids=np.array(data['dev_ids']),test_ids=np.array(data['test_ids']),families=np.array(old['families']),classes=np.array([0,1]))
            fp=path.with_name(path.stem+'-folds.json');fp.write_text(json.dumps(audit))
            entries.append(dict(dataset=name,seed=seed,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                folds_sha256=hashlib.sha256(fp.read_bytes()).hexdigest(),archive_path=str(source),archive_sha256=entry['sha256'],
                unchanged_columns=['XGB','TabICL'],newly_trained_column='TabM',alignment='row IDs, targets, fold arrays and data hash checked exactly'))
            print(name,seed,result['errors'],flush=True)
    hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['relkit/cross_ensemble.py','relkit/cross_experiment_v2.py','relkit/tabm_v2.py','_verify_l057_v2.py']}
    r=dict(preset='lab',config=old['config'],families=old['families'],seeds=old['seeds'],datasets=old['datasets'],rows=rows,summary=summarize(rows),
        ablations=ablations,seconds=time.perf_counter()-start,source_hashes=hashes,
        versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','torch','xgboost']},
        historical_result_sha256=hashlib.sha256(Path('_verify_l057_results.json').read_bytes()).hexdigest(),
        arm_provenance={'TabM':'27 corrected v2 fold fits performed now','XGB':'27 historical fold fits; exact unchanged archived probabilities','TabICL':'27 historical context constructions; exact unchanged archived probabilities; original package/checkpoint in _sources_l057.json'},
        verdict='INCOMPARABLE',scope='Corrected TabM intervention on frozen aligned library; reused outer test is exploratory; no Figure 6 reproduction')
    Path('_verify_l057_v2_results.json').write_text(json.dumps(r,indent=2)+'\n')
    Path('_data_l057_v2.json').write_text(json.dumps(dict(predictions=entries,source_hashes=hashes,arm_provenance=r['arm_provenance']),indent=2)+'\n')
    print(r['summary']);print('seconds',r['seconds'])
if __name__=='__main__':main()
