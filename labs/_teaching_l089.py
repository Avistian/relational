"""PPI partition intervention; equal data passes, not equal optimizer steps."""
import argparse,hashlib,importlib.metadata as md,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'relkit'))
from cluster_gcn_l089 import *
LAB=Path(__file__).resolve().parent
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=LAB/'results/l089/teaching');args=parser.parse_args()
 data=load_ppi(LAB/'data/l089')
 data['a'].data[:]=1  # controlled teaching graph: same binary self-loop convention in all arms
 rows=[]
 for mode,method,p,q in [('full','metis',1,1),('random','random',50,1),('cluster1','metis',50,1),('cluster5','metis',50,5)]:
  for seed in [1,2,3]:
   cfg=dict(preset('smoke'),epochs=10,num_parts=p,q=q,partition_method=method)
   out=args.output/f'{mode}-seed{seed}'
   if out.exists():raise SystemExit('Fresh runs only: existing '+str(out))
   r=train(data,cfg,seed,'cpu',out,validate_every=10)
   r['mode']=mode;r['source_sha256']=hashlib.sha256((LAB/'relkit/cluster_gcn_l089.py').read_bytes()).hexdigest()
   r['versions']={n:md.version(n) for n in ['torch','numpy','scipy','scikit-learn','pymetis']}
   (out/'result.json').write_text(json.dumps(r,indent=2)+'\n');rows.append(r)
 summary=[]
 for mode in ['full','random','cluster1','cluster5']:
  rr=[r for r in rows if r['mode']==mode];values=[r['test']['micro_f1'] for r in rr]
  summary.append({'mode':mode,'test_f1':values,'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1)),
     'max_batch_nodes':max(r['max_batch_nodes'] for r in rr),'hidden_state_proxy_bytes':max(r['hidden_state_proxy_bytes'] for r in rr),
     'mean_train_seconds':float(np.mean([r['training_seconds'] for r in rr])),
     'mean_edge_fraction':float(np.mean([r['mean_retained_edge_fraction'] for r in rr]))})
 (LAB/'_teaching_l089_results.json').write_text(json.dumps({'status':'COMPLETE','lane':'teaching, not Table10','seeds':[1,2,3],'epochs':10,'hidden':32,'layers':2,'rows':summary},indent=2)+'\n')
