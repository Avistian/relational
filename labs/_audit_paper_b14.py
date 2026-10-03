"""Independent pairwise AUROC; exact full keys/labels; validation-only selection."""
import hashlib,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;E=P/'evidence/b14'
def auroc(y,p):
 pos=[float(v) for a,v in zip(y,p) if a==1];neg=[float(v) for a,v in zip(y,p) if a==0]
 assert pos and neg and len(pos)+len(neg)==len(y)
 return math.fsum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))
def audit(root,expected):
 r=json.loads((root/'result.json').read_text());assert r['status']=='COMPLETE';trials=r['trials'];full=r['phase']=='full'
 assert len(trials)==(3 if full else 1)
 if full:assert [t['config']['max_depth'] for t in trials]==[2,3,4]
 rows=[]
 for t in trials:
  assert not t['error']
  for split in ['val','test']:
   file=root/(t['config_tag']+'-'+split+'.npz')
   if not file.exists():
    assert split=='test' and t['test_score'] is None;continue
   with np.load(file,allow_pickle=False) as a:
    k=list(zip(a['entity'].tolist(),a['date'].tolist()));y=a['label'];p=a['prediction'].reshape(-1)
    assert len(k)==len(set(k))==len(p);assert np.all(np.isfinite(p))
    exp=expected[split];mapping={(x['entity'],x['date']):x['label'] for x in exp}
    assert set(k)==set(mapping)
    assert all(int(label)==mapping[key] for key,label in zip(k,y))
    score=auroc(y,p);assert abs(score-t[split+'_score'])<1e-12
    rows.append(dict(config=t['config_tag'],depth=t['config']['max_depth'],split=split,rows=len(k),auroc=score,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
 if full:
  selected=max(trials,key=lambda t:t['val_score']);assert selected['config_id']==r['selected']
  default=next(t for t in trials if t['config_tag']=='default')
  assert {x['config'] for x in rows if x['split']=='test'}=={default['config_tag'],selected['config_tag']}
  assert selected['test_score'] is not None
 return dict(status='PASS',phase=r['phase'],evaluations=len(rows),prediction_rows=sum(x['rows'] for x in rows),rows=rows,selected_depth=selected['config']['max_depth'] if full else None,selected_test_auroc=selected['test_score'] if full else None)
if __name__=='__main__':
 import sys
 phase=sys.argv[1] if len(sys.argv)>1 else 'full';expected=json.loads((E/'official-keys.json').read_text());r=audit(E/phase,expected);(E/(phase+'-audit.json')).write_text(json.dumps(r,indent=2)+'\n');print(r)
