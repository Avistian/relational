"""Fresh complete ten-trial manual-FE search. No test-guided tuning or subsampling."""
import argparse,hashlib,json,os,platform,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from relkit.fe_experiment_l129 import tune_fe
from relkit.manual_fe_l129 import align_predictions
P=Path(__file__).resolve().parent

def run(seed,output,trials=10,rounds=2000):
    start=time.perf_counter();out=Path(output);out.mkdir(parents=True,exist_ok=False)
    data=P/'evidence/l155/fe/matrices.npz';a=np.load(data,allow_pickle=False)
    cats=a['categorical_indices'].tolist()
    matrices={s:(pd.DataFrame(a[s+'_x']),a[s+'_y'] if s!='test' else None,cats) for s in ['train','val','test']}
    model,trace,selected=tune_fe(matrices,seed,num_trials=trials,rounds=rounds,threads=4)
    model.save_model(str(out/'model.txt'))
    arrays={};scores={}
    for split in ['val','test']:
        pred=model.predict(matrices[split][0])
        fk=list(zip(a[split+'_feature_id'],a[split+'_feature_time']));qk=list(zip(a[split+'_query_id'],a[split+'_query_time']))
        aligned=np.array(align_predictions(fk,pred,qk));target=a[split+'_target'];scores[split]=float(np.mean(np.abs(target-aligned)))
        arrays.update({split+'_pred':aligned,split+'_target':target,split+'_entity':a[split+'_query_id'],split+'_time':a[split+'_query_time']})
    np.savez_compressed(out/'predictions.npz',**arrays)
    r=dict(status='COMPLETE',seed=seed,trials=trials,rounds_cap=rounds,selected_trial=selected,best_iteration=model.best_iteration,trace=trace,scores=scores,seconds=time.perf_counter()-start,matrix_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),source_sha256={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [P/'relkit/fe_experiment_l129.py',P/'relkit/manual_fe_l129.py',Path(__file__)]},files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()},platform=platform.platform(),python=sys.version,scope='Released SQL/train/search replay on checksum-pinned v1 data; historical identity NOT_ESTABLISHED',cloud_usd=0)
    (out/'result.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:r[k] for k in ['seed','scores','seconds','selected_trial','best_iteration']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--trials',type=int,default=10);p.add_argument('--rounds',type=int,default=2000);a=p.parse_args();run(a.seed,a.output,a.trials,a.rounds)
