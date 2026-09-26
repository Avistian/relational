"""Require ten complete runs and reconstruct batch metrics independently from predictions."""
import hashlib,importlib.util,json,shutil,statistics,sys
from pathlib import Path
import numpy as np
from _fetch_l103 import fetch
P=Path(__file__).resolve().parent

def metrics(positive,negative):
    scores=np.r_[positive,negative];y=np.r_[np.ones(len(positive)),np.zeros(len(negative))]
    order=np.argsort(-scores,kind='stable');score=scores[order];truth=y[order]
    ends=np.r_[np.flatnonzero(np.diff(score)!=0),len(score)-1]
    tp=np.cumsum(truth)[ends];recall=tp/len(positive);precision=tp/(ends+1)
    ap=np.sum(np.diff(np.r_[0.,recall])*precision)
    neg=np.sort(negative);less=np.searchsorted(neg,positive,side='left');leq=np.searchsorted(neg,positive,side='right')
    auc=np.mean((less+.5*(leq-less))/len(negative))
    accuracy=np.mean((scores>.5)==y)
    return np.array([ap,auc,accuracy])

def collect(root):
    root=Path(root);records=[];identities=[];hashes={};error=0.;source_check=None
    evidence=P/'evidence/l103';evidence.mkdir(parents=True,exist_ok=True)
    audit=json.loads((P/'_audit_l103_results.json').read_text())
    source,_=fetch()
    spec=importlib.util.spec_from_file_location('original_selection',source/'utils.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    for seed in range(10):
        folder=root/f'seed-{seed}'
        result=json.loads((folder/'result.json').read_text());identity=json.loads((folder/'identity.json').read_text())
        assert result['seed']==seed and result['release_quirks'] and result['neighbors']==20 and result['layers']==2
        assert result['train_events']==81028 and result['test_events']==23620 and result['new_test_events']==11714
        assert identity['data']['split_sha256']==audit['data']['split_sha256']
        assert identity['preset']=='paper'
        assert json.loads((folder/'source_parity.json').read_text())['status']=='PASS'
        assert result['epochs_completed']==len(result['trace'])
        assert (result['early_stopped'] and result['epochs_completed']>=4) or result['epochs_completed']==50
        monitor=original.EarlyStopMonitor()
        for epoch,row in enumerate(result['trace']):
            assert row['epoch']==epoch
            stop=monitor.early_stop_check(row['val_ap'])
            assert not stop or epoch==len(result['trace'])-1, 'Training continued after release stop'
        assert stop==result['early_stopped']
        assert result['selected_epoch']==(monitor.best_epoch if stop else epoch), 'Original checkpoint selection mismatch'
        z=np.load(folder/'predictions.npz')
        for lane,key in [('all','test'),('new','new_test')]:
            values=[]
            full_ids=np.r_[z[f'{lane}_e'],np.int64(audit['last_event_by_split'][key])]
            assert hashlib.sha256(full_ids.tobytes()).hexdigest()==audit['data']['split_sha256'][key], 'Prediction event identity mismatch'
            assert len(z[f'{lane}_e'])==result['test_events' if lane=='all' else 'new_test_events']
            for batch in np.unique(z[f'{lane}_batch']):
                mask=z[f'{lane}_batch']==batch
                values.append(metrics(z[f'{lane}_p'][mask],z[f'{lane}_n'][mask]))
            measured=np.mean(values,axis=0);reported=np.array([result[key][k] for k in ['ap','auc','accuracy']])
            assert np.max(np.abs(measured-reported))<1e-12
            error=max(error,float(np.max(np.abs(measured-reported))))
        for name in ['result.json','identity.json','predictions.npz','selected.pt','source_parity.json']:
            hashes[f'seed-{seed}/{name}']=hashlib.file_digest((folder/name).open('rb'),'sha256').hexdigest()
        for name in ['result.json','identity.json','predictions.npz','source_parity.json']:
            dest=evidence/f'seed-{seed}';dest.mkdir(exist_ok=True);shutil.copy2(folder/name,dest/name)
        if (folder/'original-replay.json').exists():
            assert seed==0
            source_check=json.loads((folder/'original-replay.json').read_text())
            assert source_check['status']=='PASS'
            assert source_check['source_checkpoint_sha256']==hashes[f'seed-{seed}/selected.pt']
            for lane,key in [('all','test'),('new','new_test')]:
                assert abs(source_check['populations'][lane]['ap']-result[key]['ap'])<1e-12
            shutil.copy2(folder/'original-replay.json',evidence/'original-replay.json')
        records.append(result);identities.append(identity)
    for key in ['implementation_sha256','python','torch','numpy','pandas','sklearn','device']:
        assert len({x[key] for x in identities})==1, 'Mixed run identity: '+key
    assert identities[0]['implementation_sha256']==hashlib.sha256((P/'relkit/tgat_l103.py').read_bytes()).hexdigest()
    summary={}
    for lane,key,target in [('all','test',95.34),('new','new_test',93.99)]:
        values=[100*x[key]['ap'] for x in records];mean=statistics.mean(values)
        summary[lane]={'mean_ap_percent':mean,'sample_sd_pp':statistics.stdev(values),'paper_ap_percent':target,'absolute_gap_pp':abs(mean-target),'numerical_verdict':'CLOSE' if abs(mean-target)<=.5 else 'FAIL'}
    report={'status':'COMPLETE','summary':summary,'records':records,'identities':identities,'artifact_sha256':hashes,'modal_volume':'l103-tgat-evidence','volume_prefix':'/'+identities[0]['implementation_sha256']+'/paper','modal_app':'https://modal.com/apps/pszar92/main/ap-pt2LyBpEakNbMOfKNV11Cr','metric_reconstruction_max_error':error,'original_source_replay':source_check or 'NOT_CHECKED','historical_identity':'INCOMPARABLE','full_paper':'NOT_ESTABLISHED'}
    (P/'_paper_l103_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(summary)
    return report
if __name__=='__main__':collect(sys.argv[1])
