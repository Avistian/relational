"""Collect all ten GPU runs and independently reconstruct their batch AP/AUC."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
from relkit.tgn_l102 import PAPER_AP,CLOSE_TOLERANCE_PP
P=Path(__file__).resolve().parent

def independent_ap(y,score):
 order=np.argsort(-score,kind='stable');y=np.asarray(y)[order];score=np.asarray(score)[order]
 ends=np.r_[np.flatnonzero(np.diff(score)!=0),len(score)-1]
 tp=np.cumsum(y)[ends];delta=np.diff(np.r_[0,tp])
 return float(np.sum(delta*(tp/(ends+1)))/sum(y))

def independent_auc(positive,negative):
 # Pairwise comparisons, credit half to ties; batches have at most 200 of each.
 delta=positive[:,None]-negative[None,:]
 return float(np.mean((delta>0)+.5*(delta==0)))

def main(root,layout="author"):
 root=Path(root);out=P/'evidence/l102';out.mkdir(parents=True,exist_ok=True)
 selection=json.loads((P/'_run_selection_l102.json').read_text())['seed_to_group'] if layout=='author' else {str(s):('group-0-1-2' if s<3 else 'group-3-4-5' if s<6 else 'group-6-7' if s<8 else 'group-8-9') for s in range(10)}
 paths=[root/selection[str(seed)]/f'seed-{seed}.json' for seed in range(10)]
 assert all(p.exists() for p in paths), 'Need all ten predeclared designated runs before aggregation'
 records=[];identities=[];artifacts={};max_error=0;events_checked=0
 for path in paths:
  r=json.loads(path.read_text());seed=r['seed'];records.append(r)
  identity=json.loads((path.parent/'identity.json').read_text());identities.append(identity)
  assert identity['source']==hashlib.sha256((P/'relkit/tgn_l102.py').read_bytes()).hexdigest()
  par=json.loads((path.parent/'source_parity.json').read_text());assert par['status']=='PASS'
  npz=path.with_name(f'seed-{seed}-predictions.npz');a=np.load(npz)
  for lane,key in [('all','test'),('new','new_test')]:
   pos,neg=a[lane+'_positive'],a[lane+'_negative_score'];assert len(pos)==len(neg)==identity['data']['counts']['test' if lane=='all' else 'new_test']
   aps=[];aucs=[]
   for i in range(0,len(pos),200):
    pp,nn=pos[i:i+200],neg[i:i+200]
    aps.append(independent_ap(np.r_[np.ones(len(pp)),np.zeros(len(nn))],np.r_[pp,nn]));aucs.append(independent_auc(pp,nn))
   err=max(abs(np.mean(aps)-r[key]['ap']),abs(np.mean(aucs)-r[key]['auc']))
   assert err<1e-12,(seed,lane,err);assert np.allclose(aps,r[key]['batch_ap'],rtol=0,atol=1e-12)
   events_checked+=len(pos);max_error=max(max_error,err)
  for file in [path,npz,path.parent/'identity.json',path.parent/'source_parity.json']:
   name=file.name if file in [path,npz] else path.parent.name+'-'+file.name
   shutil.copyfile(file,out/name)
  checkpoint=path.with_suffix('.pt');assert checkpoint.exists()
  artifacts[str(seed)]={'checkpoint_sha256':hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest(),
     'checkpoint_volume_path':'/4a1d877bcbc37106561f8eafc2ea1e05d4b4863c3bf7ff623d2bc7bbf9f5368d/paper/'+path.parent.name+'/'+checkpoint.name,
     'predictions_sha256':hashlib.file_digest(npz.open('rb'),'sha256').hexdigest()}
 records.sort(key=lambda r:r['seed']);assert [r['seed'] for r in records]==list(range(10))
 first=identities[0]
 for a in identities[1:]:assert a==first,'Do not pool different environments/splits'
 report={'scope':'Rossi v3 Table 2 Wikipedia TGN-attn two AP cells, complete selected modern GPU replay',
   'status':'COMPLETE','complete_seeds':list(range(10)),'identity':first,'records':records,'summary':{},
   'historical_identity':'INCOMPARABLE','full_paper_reproduction':'NOT_ESTABLISHED',
   'total_epochs':sum(r['epochs_completed'] for r in records),'artifact_manifest':artifacts,
   'execution_selection':json.loads((P/'_run_selection_l102.json').read_text()) if layout=='author' else {'seed_to_group':selection,'layout':'fresh grouped replay'},
   'modal_apps':['https://modal.com/apps/pszar92/main/'+a for a in json.loads((P/'_run_selection_l102.json').read_text())['modal_apps']] if layout=='author' else [],
   'deviations':['Independent explicit seeds 0-9 instead of one continuous release RNG stream','Modern pinned Python/torch/numpy/sklearn runtime','Set-to-tuple compatibility for Python random.sample; exact split arrays independently checked'],
   'unrun':['Reddit','Twitter','dynamic node classification','baseline models','ablation suite','historical-runtime execution']}
 for lane,key in [('all','test'),('new','new_test')]:
  a=np.array([r[key]['ap']*100 for r in records]);mean=float(a.mean())
  report['summary'][lane]={'mean_ap_percent':mean,'sample_sd_pp':float(a.std(ddof=1)),
    'paper_ap_percent':PAPER_AP[lane],'paper_reported_sd_pp':.1,'gap_pp':mean-PAPER_AP[lane],
    'close_tolerance_pp':CLOSE_TOLERANCE_PP,'numerical_verdict':'CLOSE' if abs(mean-PAPER_AP[lane])<=CLOSE_TOLERANCE_PP else 'OUTSIDE_TOLERANCE'}
 (P/'_paper_l102_results.json').write_text(json.dumps(report,indent=2))
 (P/'_metrics_check_l102_results.json').write_text(json.dumps({'status':'PASS','independent_method':'Threshold-group AP from cumulative counts; pairwise AUC with half-credit ties; equal mean over 200-event batches','runs':10,'real_events_checked':events_checked,'max_metric_error':max_error},indent=2))
 print(json.dumps(report['summary'],indent=2));print('Total epochs:',report['total_epochs'])
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--layout',choices=['author','grouped'],default='author');args=parser.parse_args();main(args.root,args.layout)
