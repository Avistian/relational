"""Selected Table23 gate and explicit current-release inference operator.
The released lane needs a caller-supplied authenticated numeric packet; it is
never promoted to historical Table23 parity by matching the rounded score.
"""
import argparse,hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/b08';S=P/'sources/b08'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source_audit():
 for name,h in json.loads((S/'manifest.json').read_text()).items():
  if sha(S/name)!=h:raise ValueError('Source hash mismatch: '+name)
 gate=json.loads((E/'source-gate.json').read_text())
 assert gate['target']=='2509.03505v2 Table23 LimiX-16M Analcatdata BroadwayMult'
 meta=json.loads((S/'checkpoint.json').read_text())
 checkpoint=next(x for x in meta['siblings'] if x['rfilename'].endswith('.ckpt'))
 assert gate['checkpoint']['sha256']==checkpoint['lfs']['sha256'],'checkpoint metadata mismatch'
 assert gate['checkpoint']['revision']==meta['sha'],'checkpoint revision mismatch'
 assert gate['paper_rmse']==.194 and gate['published_mask_rate']==.05,'paper target mismatch'
 return gate

def released(packet,checkpoint,output,device='cpu'):
 """Run full pinned release on a numeric packet, with observed-only inputs.
 Packet JSON pins an NPZ plus shape/identity/type/scaler provenance. NPZ has
 X_support,y_support,X_query,true_X_query,mask and row_ids. X values are already
 scaled using the packet's declared scaler. Only masked cells enter RMSE.
 """
 import numpy as np
 import torch
 gate=source_audit();spec=json.loads(Path(packet).read_text());data=Path(packet).parent/spec['file']
 if sha(data)!=spec['sha256']:raise ValueError('Packet hash mismatch')
 for field in ['dataset_version','split_provenance','mask_provenance','scaler_provenance','feature_types','row_identity_provenance']:
  if not spec.get(field):raise ValueError('Missing '+field)
 if sha(checkpoint)!=gate['checkpoint']['sha256']:raise ValueError('Wrong checkpoint')
 a=np.load(data,allow_pickle=False);mask=a['mask']
 if mask.dtype!=bool or mask.shape!=a['X_query'].shape or not mask.any():raise ValueError('Invalid mask')
 if not np.array_equal(np.isnan(a['X_query']),mask):raise ValueError('Masked cells must be hidden with NaN')
 if not np.array_equal(a['X_query'][~mask],a['true_X_query'][~mask]):raise ValueError('Unmasked input mismatch')
 if len(np.unique(a['row_ids']))!=len(a['row_ids']):raise ValueError('Duplicate query identities')
 if not all(t=='numerical' for t in spec['feature_types']):raise ValueError('This selected RMSE lane requires authenticated numeric columns')
 if Path(output).exists():raise ValueError('Refuse prediction overwrite')
 sys.path.insert(0,str(S/'release'))
 from inference.predictor import LimiXPredictor
 model=LimiXPredictor(device=torch.device(device),model_path=str(checkpoint),inference_config=str(S/'release/config/reg_default_noretrieval_MVI.json'))
 pred=np.asarray(model.predict(a['X_support'],a['y_support'],a['X_query'],task_type='Feature_imputation'))[-len(a['X_query']):]
 if pred.shape!=mask.shape or not np.isfinite(pred).all():raise ValueError('Invalid released output')
 rmse=float(np.sqrt(np.mean((pred[mask].astype('float64')-a['true_X_query'][mask])**2)))
 np.savez_compressed(output,prediction=pred,row_ids=a['row_ids'],mask=mask)
 return dict(status='RELEASED_INFERENCE_ONLY',rmse=rmse,paper_target=.194,historical_parity='NOT_ESTABLISHED',packet_sha256=sha(data),prediction_sha256=sha(output))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--lane',choices=['audit','paper','released'],default='audit');ap.add_argument('--packet');ap.add_argument('--checkpoint');ap.add_argument('--output');ap.add_argument('--device',default='cpu');args=ap.parse_args()
 gate=source_audit()
 if args.lane=='paper':
  print(json.dumps(gate,indent=2));raise SystemExit(2) # No unauthenticated historical dispatch.
 if args.lane=='audit':print(json.dumps(gate,indent=2))
 else:
  if not all([args.packet,args.checkpoint,args.output]):ap.error('released requires --packet --checkpoint --output')
  print(json.dumps(released(args.packet,args.checkpoint,args.output,args.device),indent=2))
