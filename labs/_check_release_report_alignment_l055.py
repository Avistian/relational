"""Partial release alignment: compare current labels/counts with original report supports.

Equal counts/class totals do not prove original row IDs, feature bytes or target
values match. Read report bytes from pinned Git objects, not mutable worktree files.
"""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from _fetch_l055 import ROOT,REVISION,fetch
from _check_release_l055 import verify_extracted


def check(checkout):
 root=fetch();verify_extracted(root);rows=[]
 for task in ('ecom-offers','homesite-insurance'):
  folder=root/task/task;y=np.load(folder/'y.npy')
  for split in ['default']+[f'{mode}-{i}' for mode in ('random','sliding-window') for i in range(3)]:
   report=Path('paper/exp')/('mlp' if split=='default' else 'temporal-shift-analysis/mlp')/(task if split=='default' else task+'-'+split)/'evaluation/0/report.json'
   raw=subprocess.check_output(['git','-C',str(checkout),'show',REVISION+':'+str(report)])
   original=json.loads(raw)
   for part in ('train','val','test'):
    ids=np.load(folder/'splits'/split/(part+'.npy'))
    current=np.bincount(y[ids].astype(int),minlength=2).tolist()
    counts=[int(original['metrics'][part][str(i)]['support']) for i in range(2)]
    rows.append(dict(task=task,split=split,part=part,current_class_counts=current,reported_class_counts=counts,
                     match=current==counts,source=str(report),source_sha256=hashlib.sha256(raw).hexdigest()))
 return dict(status='PARTIAL_ALIGNMENT',matching_partitions=sum(r['match'] for r in rows),checked_partitions=len(rows),
             scope='Matching counts/class totals support partial alignment, not row/feature/target byte identity. No original arrays are available here to hash.',rows=rows)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkout',type=Path,required=True);a=p.parse_args();r=check(a.checkout)
 (ROOT/'_release_report_alignment_l055_results.json').write_text(json.dumps(r,indent=2)+'\n')
 print({k:v for k,v in r.items() if k!='rows'})
