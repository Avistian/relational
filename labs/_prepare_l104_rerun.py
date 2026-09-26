"""Prepare an isolated audit of newly trained checkpoints without overwriting evidence.
Accepts L103's per-seed GPU identities or its common CPU identity.json.
No training or paid compute is launched.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from relkit.tgat_l103 import load_wikipedia
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def prepare(checkpoint_root,data_dir,workspace,seeds):
 src=Path(checkpoint_root).resolve();out=Path(workspace).resolve()
 if out.exists():raise FileExistsError('Use a fresh isolated workspace; existing evidence is not overwritten')
 _,_,_,audit=load_wikipedia(data_dir)
 code_sha=sha(P/'relkit/tgat_l103.py')
 manifest={'schema':1,'named_target':'Independent rerun: Xu et al. 2020 TGAT Wikipedia release evaluation','source_commit':'9293d10d1943c4bd4a186337cf38ba98e4c8bb99','implementation_sha256':code_sha,'raw_sha256':audit['raw_sha256'],'processed_arrays':{},'seeds':{},'split_sha256':audit['split_sha256']}
 with np.load(Path(data_dir)/'processed.npz') as z:
  for k in ['u','v','t','x']:manifest['processed_arrays'][k]={'shape':list(z[k].shape),'dtype':str(z[k].dtype),'sha256':hashlib.sha256(z[k].tobytes()).hexdigest()}
 validated=[]
 for seed in seeds:
  folder=src/f'seed-{seed}';identity_file=folder/'identity.json'
  if not identity_file.exists():identity_file=src/'identity.json'
  identity=json.loads(identity_file.read_text());result=json.loads((folder/'result.json').read_text())
  assert identity['preset']=='paper' and identity['implementation_sha256']==code_sha
  assert identity['data']['split_sha256']==audit['split_sha256'] and identity['data']['raw_sha256']==audit['raw_sha256']
  assert result['seed']==seed and result['layers']==2 and result['neighbors']==20 and result['release_quirks']
  assert result['test_events']==23620 and result['new_test_events']==11714
  sources={name:folder/name for name in ['selected.pt','predictions.npz','result.json']};sources['identity.json']=identity_file
  manifest['seeds'][str(seed)]={name:sha(path) for name,path in sources.items()};validated.append((seed,sources))
 out.mkdir(parents=True);(out/'relkit').mkdir()
 for name in ['_run_l104.py','relkit/tgat_l103.py','relkit/leakage_l104.py']:shutil.copy2(P/name,out/name)
 for seed,sources in validated:
  folder=out/'checkpoints'/f'seed-{seed}';folder.mkdir(parents=True)
  for name,source in sources.items():(folder/name).symlink_to(source.resolve())
 (out/'_inputs_l104.json').write_text(json.dumps(manifest,indent=2)+'\n')
 return {'workspace':str(out),'seeds':seeds,'status':'PREPARED_NOT_EXECUTED','data_dir':str(Path(data_dir).resolve()),'input_links':'Keep original trained checkpoint files; all hashes are verified before use'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint-root',required=True);p.add_argument('--data-dir',required=True);p.add_argument('--workspace',required=True);p.add_argument('--seeds',default='0,1,2,3,4,5,6,7,8,9');a=p.parse_args()
 print(json.dumps(prepare(a.checkpoint_root,a.data_dir,a.workspace,list(map(int,a.seeds.split(',')))),indent=2))
