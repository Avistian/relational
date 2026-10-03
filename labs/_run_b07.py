"""Run all 27 prespecified semantic-ablation fits, retaining every selected head."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from relkit.carte_b07 import load_encoder,normalize_numeric,make_graph,embed
from relkit.semantic_b07 import intervene,fit_probe,encode_table


def run_course(lab,output,intervention=intervene,probe=fit_probe):
    lab=Path(lab);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    config_path=lab/'evidence/b07/course-protocol.json';cfg=json.loads(config_path.read_text())
    for name,digest in {**cfg['portable_inputs'],**cfg['implementation_hashes']}.items():
        if hashlib.sha256((lab/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Input authentication failed: '+name)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    root=lab/'data/b07';meta=json.loads((root/'manifest.json').read_text())
    v=np.load(root/'vectors.npz',allow_pickle=False);vectors=dict(zip(v['names'].tolist(),v['vectors']))
    model=load_encoder(root/'selected_checkpoint.pt',True,0).eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    weights_before={k:v.clone() for k,v in model.state_dict().items()}
    records=[];splits=[];started=time.monotonic()
    for dataset in cfg['datasets']:
        frame=pd.read_parquet(root/(dataset+'.parquet'));target=meta['datasets'][dataset]['target'];y=frame.pop(target).to_numpy(dtype=float)
        if len(frame)!=384:raise ValueError('Original sample size changed')
        for seed in cfg['seeds']:
            ids=np.random.default_rng(seed+74).permutation(384);tr,va,te=ids[:64],ids[64:128],ids[128:]
            normalized,policy=normalize_numeric(frame,tr)
            splits.append(dict(dataset=dataset,seed=seed,train=tr.tolist(),validation=va.tolist(),test=te.tolist(),numeric_policy=policy))
            for arm in cfg['arms']:
                x=intervention(normalized,arm)
                features=encode_table(model,x,vectors,make_graph,embed)
                if not np.isfinite(features).all():raise ValueError('Nonfinite features')
                head=probe(features,y,tr,va,te,alphas=cfg['head']['alphas'])
                run_id=f'{dataset}-{seed}-{arm}'
                np.savez_compressed(output/(run_id+'.npz'),features=features)
                feature_hash=hashlib.sha256((output/(run_id+'.npz')).read_bytes()).hexdigest()
                records.append(dict(dataset=dataset,seed=seed,arm=arm,columns=list(x.columns),test_ids=te.tolist(),source_ids=[meta['datasets'][dataset]['source_rows'][i] for i in te],target=y[te].tolist(),head=head,feature_file=run_id+'.npz',feature_sha256=feature_hash,features_sha256=hashlib.sha256(features.tobytes()).hexdigest()))
            print(dataset,'seed',seed,'three arms complete',flush=True)
    for key,value in model.state_dict().items():
        if not torch.equal(value,weights_before[key]):raise ValueError('Frozen encoder changed')
    report=dict(name=cfg['name'],protocol_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),splits=splits,records=records,encoder_updated=False)
    (output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'timing.json').write_text(json.dumps(dict(seconds=time.monotonic()-started,course_fits=len(records)),indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args();lab=Path(__file__).resolve().parent
    output=a.output or lab/'evidence/b07/runs'
    if (output/'results.json').exists():raise SystemExit('Refusing to overwrite saved evidence; select a fresh --output')
    run_course(lab,output)
