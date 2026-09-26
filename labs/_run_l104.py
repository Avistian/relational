"""Fresh ten-checkpoint released replay and paired temporal access interventions."""
import argparse,hashlib,json,platform,sys,time
from pathlib import Path
import numpy as np
import torch
import pandas,sklearn
from relkit.tgat_l103 import load_wikipedia,TGAT,NeighborFinder
from relkit.leakage_l104 import records_from_archive,score_fixed_questions,replay_release,AuditFinder,paired_ap
P=Path(__file__).resolve().parent
sha=lambda path:hashlib.file_digest(Path(path).open('rb'),'sha256').hexdigest()

def run(seed,checkpoint_root,data_dir,output,device='cpu',pilot=False,max_seconds=2900):
 started=time.monotonic();deadline=started+max_seconds
 m=json.loads((P/'_inputs_l104.json').read_text());src=Path(checkpoint_root)/f'seed-{seed}';out=Path(output);out.mkdir(parents=True,exist_ok=True)
 assert sha(P/'relkit/tgat_l103.py')==m['implementation_sha256']
 for name,expected in m['seeds'][str(seed)].items():assert sha(src/name)==expected,f'Input changed: {name}'
 n,e,d,a=load_wikipedia(data_dir);assert a['split_sha256']==m['split_sha256']
 with np.load(Path(data_dir)/'processed.npz') as z:
  for k,expected in m['processed_arrays'].items():assert hashlib.sha256(z[k].tobytes()).hexdigest()==expected['sha256'],f'Processed data mismatch: {k}'
 identity={'seed':seed,'pilot':pilot,'device':device,'checkpoint_sha256':m['seeds'][str(seed)]['selected.pt'],'input_identity':{**{k:v for k,v in m.items() if k!='seeds'},'seed_files':m['seeds'][str(seed)]},'code':{name:sha(P/name) for name in ['relkit/tgat_l103.py','relkit/leakage_l104.py','_run_l104.py']},'runtime':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'pandas':pandas.__version__,'sklearn':sklearn.__version__,'platform':platform.platform()},'lookahead_seconds':86400,'intervention_rng':104+seed,'neighbors':20,'layers':2}
 if (out/'result.json').exists():
  result=json.loads((out/'result.json').read_text())
  assert result['identity']==identity,'Resume rejected: input/code/runtime changed'
  assert sha(out/'predictions.npz')==result['predictions_sha256'],'Resume rejected: prediction artifact changed'
  return result
 checkpoint=torch.load(src/'selected.pt',map_location=device,weights_only=False)
 model=TGAT(NeighborFinder(d['full'],len(n),release=True),n,e).to(device)
 loaded=model.load_state_dict(checkpoint['weights'],strict=False)
 assert set(loaded.missing_keys)=={'n_feat_th','e_feat_th','edge_raw_embed.weight','node_raw_embed.weight'} and not loaded.unexpected_keys
 archive=np.load(src/'predictions.npz');saved={};replay=None
 if not pilot:
  replay,outputs=replay_release(model,d,checkpoint,archive,deadline)
  for lane,records in outputs.items():
   for key,value in records.items():saved[f'release_{lane}_{key}']=value
  print(json.dumps({'seed':seed,'stage':'release replay PASS','seconds':time.monotonic()-started}),flush=True)
 arms={};audits={}
 for mode in ['strict','inclusive','lookahead']:
  arms[mode]={};audits[mode]={}
  for lane,split in [('all','test'),('new','new_test')]:
   questions=records_from_archive(archive,lane)
   if pilot:questions={k:v[:120] for k,v in questions.items()}
   finder=AuditFinder(d['full'],len(n),mode);model.ngh_finder=finder
   pred=score_fixed_questions(model,d[split],questions,rng_seed=104+seed,deadline=deadline)
   arms[mode][lane]=pred;audits[mode][lane]=finder.audit
   if mode=='strict':assert finder.audit['nonpast_records']==0
   for key,value in pred.items():saved[f'{mode}_{lane}_{key}']=value
  print(json.dumps({'seed':seed,'stage':mode,'seconds':time.monotonic()-started}),flush=True)
 comparisons={lane:{mode:paired_ap(arms['strict'][lane],arms[mode][lane]) for mode in ['inclusive','lookahead']} for lane in ['all','new']}
 np.savez_compressed(out/'predictions.npz',**saved)
 result={'status':'PILOT' if pilot else 'COMPLETE','identity':identity,'release_replay':replay,'comparisons':comparisons,'access_audit':audits,'events':{lane:len(arms['strict'][lane]['e']) for lane in ['all','new']},'elapsed_seconds':time.monotonic()-started,'predictions_sha256':sha(out/'predictions.npz'),'training':'REUSED_L103','full_paper':'NOT_ESTABLISHED','interventions':'COURSE_EXTENSION_NOT_PUBLISHED_EXPERIMENT'}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--checkpoint-root',default=str(P/'results/l103/gpu'));p.add_argument('--data-dir',default=str(P/'l103-cache'));p.add_argument('--output',default=str(P/'results/l104/local'));p.add_argument('--device',default='cpu');p.add_argument('--pilot',action='store_true');p.add_argument('--max-seconds',type=int,default=2900);args=p.parse_args();torch.set_num_threads(1)
 print(json.dumps(run(args.seed,args.checkpoint_root,args.data_dir,Path(args.output)/f'seed-{args.seed}',args.device,args.pilot,args.max_seconds),indent=2))
