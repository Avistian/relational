"""Audit lesson/teacher predictions using an independently precomputed source panel.

Build that panel with _reference_l069.py in the pinned original environment.
The base checker still independently reconstructs data, partitions and metrics,
and refits every XGBoost case. Cache records retain exact source row identities.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import _evidence_l069 as audit

def check(evidence,output,reference):
 r=json.loads(Path(evidence).read_text());source=json.loads(Path(reference).read_text());assert source['status']=='PASS' and source['source']=='tabpfn2.0.9'
 assert source['checkpoint_sha256']=='f65a35685aeef42e31b796d9bfa34e68d6fc780bc98e7bff7763802964cf435f'
 for name,metadata in r['datasets'].items():assert metadata['sha256']==source['data_sha256'][name]
 refs={(v['split_id'],v['condition']):v for v in source['records']};assert len(refs)==len(source['records'])==120
 rows=iter(v for v in r['records'] if v['arm']=='v2');splits={v['id']:v for v in r['splits']}
 def cached(model,cx,cy,qx):
  row=next(rows);ref=refs[(row['split_id'],row['condition'])];split=splits[row['split_id']]
  assert ref['context_ids']==split['context_ids'] and ref['query_ids']==split['query_ids']
  classes=np.unique(cy);assert np.array_equal(classes,ref['classes']) and len(qx)==len(ref['probabilities'])
  return np.asarray(ref['probabilities']),classes
 audit.original_model=lambda:(None,source['attention_adapter_checks']);audit.source_predict=cached
 report=audit.check(evidence,output,model_check=True)
 assert next(rows,None) is None
 report.update(reference_sha256=hashlib.sha256(Path(reference).read_bytes()).hexdigest(),reference_worker_sha256=source['worker_sha256'],reference_seconds=source['seconds'],scope='Independent raw CSV, partition, class-map, rank-AUC, grouped-AP, all-row loss, complete-roster and aggregation audit; every v2 probability compared to the independently generated original TabPFN 2.0.9 panel with checked compact attention; every XGBoost case freshly refitted. Original-source prediction cache is independent of lesson code and predictions.')
 Path(output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='checks'}));return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('evidence');p.add_argument('output');p.add_argument('reference');a=p.parse_args();check(a.evidence,a.output,a.reference)
