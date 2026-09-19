"""Reconstruct saved accuracies; reject missing or inconsistent experiment evidence."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from relkit.gin_l088 import load_mutag,folds_for,select_epoch,summarize_grid
LAB=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--paper-dir',type=Path,default=LAB/'results/l088/paper');a=p.parse_args()
 sources=json.loads((LAB/'_sources_l088.json').read_text())
 for name,digest in sources['files'].items():assert hashlib.sha256((LAB/name).read_bytes()).hexdigest()==digest,name
 graphs=load_mutag(LAB/'data/l088');folds=folds_for(graphs);seen=[];records=[]
 for path in sorted(a.paper_dir.glob('*.json')):
  r=json.loads(path.read_text());records.append(r);npz=path.with_suffix('.npz')
  assert hashlib.sha256(npz.read_bytes()).hexdigest()==r['logits_sha256']
  z=np.load(npz);tr,va=folds[r['fold']]
  assert r['train_ids']==tr.tolist() and r['val_ids']==va.tolist()
  assert set(tr).isdisjoint(va) and len(set(tr)|set(va))==188
  np.testing.assert_array_equal(z['labels'],[graphs[i]['y'] for i in va]);np.testing.assert_array_equal(z['val_ids'],va)
  assert z['logits'].shape==(350,len(va),2) and np.isfinite(z['logits']).all()
  acc=(z['logits'].argmax(-1)==z['labels'][None,:]).mean(1)
  np.testing.assert_array_equal(acc,[c['val_acc'] for c in r['curves']])
  assert [c['lr'] for c in r['curves']]==[.01*.5**(e//50) for e in range(350)]
  seen.append((r['config']['hidden'],r['config']['batch_size'],r['config']['dropout'],r['fold']))
 assert records,'No completed paper-track evidence';assert len(seen)==len(set(seen))
 assert len({r['fingerprint'] for r in records})==1
 report=json.loads((LAB/'_paper_l088_results.json').read_text())
 if len(records)==80:
  fresh=summarize_grid(records);assert fresh['candidates']==report['candidates'] and fresh['selected']==report['selected']
 else:
  assert report['status']!='FULL_GRID_EXECUTED' and report['fold_runs']==len(records)
  for c in report['candidates']:
   rr=sorted([r for r in records if all(r['config'][k]==v for k,v in c['config'].items())],key=lambda r:r['fold'])
   assert [r['fold'] for r in rr]==list(range(10))
   curves=np.array([[e['val_acc'] for e in r['curves']] for r in rr]);epoch=select_epoch(curves)
   assert epoch+1==c['epoch'];np.testing.assert_array_equal(curves[:,epoch],c['fold_values'])
 r={'status':'PASS','source_hashes':len(sources['files']),'fold_records':len(records),'validation_epochs_recomputed':len(records)*350,'all_disjoint_graph_splits':True,'all_predictions_finite':True,'single_fingerprint':True,'full_grid':len(records)==80,'historical_fold_identity':'NOT_CHECKED'}
 (LAB/'_audit_l088_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
if __name__=='__main__':main()
