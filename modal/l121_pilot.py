"""One bounded optimized-input pilot. Timing is not paper-result evidence."""
from pathlib import Path
import modal
R=Path(__file__).resolve().parents[1];app=modal.App('l121-cvitkovic-cached-pilot')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install('numpy==1.26.4','scikit-learn==1.5.2')
.add_local_file(R/'labs/relkit/cvitkovic_l118.py','/work/relkit/cvitkovic_l118.py')
.add_local_file(R/'labs/relkit/cache_l121.py','/work/relkit/cache_l121.py')
.add_local_dir(R/'labs/data/l118/pilot','/data')
.add_local_file(R/'labs/sources/l118/data/homecreditdefaultrisk/homecreditdefaultrisk.db_info.json','/work/info.json'))
@app.function(image=image,gpu='T4',cpu=2,memory=16384,timeout=1800,retries=0)
def pilot():
 import sys,time,json,torch,numpy as np
 sys.path.insert(0,'/work')
 from relkit.cvitkovic_l118 import PreparedGraphs,collate_graphs,feature_schema,CvitkovicGCN,move_batch
 from relkit.cache_l121 import cache_record,collate_cached
 from torch.nn import functional as F
 torch.set_num_threads(2);torch.manual_seed(1234);start=time.monotonic()
 info=json.load(open('/work/info.json'));ids=json.load(open('/data/ids.json'));ds=PreparedGraphs('/data',ids)
 t=time.monotonic();records=[ds[i] for i in range(len(ds))];raw,labels,identities=collate_graphs(records,info);original=time.monotonic()-t
 t=time.monotonic();cached=[cache_record(r,info) for r in records];preencode=time.monotonic()-t
 collate_times=[]
 for _ in range(5):
  t=time.monotonic();inputs,y,ii=collate_cached(cached);collate_times.append(time.monotonic()-t)
 for key in raw[0]:
  for a,b in zip(raw[0][key],inputs[0][key]):assert torch.equal(a,b)
 for a,b in zip(raw[1:4],inputs[1:4]):assert torch.equal(a,b)
 assert torch.equal(labels,y) and np.array_equal(identities,ii)
 model=CvitkovicGCN(feature_schema(info),info['node_type_to_int']).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0)
 train=[];evaluate=[];transfers=[]
 for _ in range(8):
  torch.cuda.synchronize();t=time.monotonic();inp=move_batch(inputs,'cuda');yy=y.cuda();torch.cuda.synchronize();transfers.append(time.monotonic()-t)
  t=time.monotonic();model.train();optimizer.zero_grad();loss=F.cross_entropy(model(*inp),yy);loss.backward();optimizer.step();torch.cuda.synchronize();train.append(time.monotonic()-t)
  t=time.monotonic();model.eval()
  with torch.no_grad():model(*inp)
  torch.cuda.synchronize();evaluate.append(time.monotonic()-t)
 rate=.00022572;collate=float(np.median(collate_times));transfer=float(np.median(transfers[1:]));tr=float(np.median(train[1:]));ev=float(np.median(evaluate[1:]))
 # Five-fold maximum schedule, including final held-out evaluation (61 batches/fold).
 projected=((collate+transfer+tr)*205+(collate+transfer+ev)*37)*300*5*rate+(collate+transfer+ev)*61*5*rate
 return {'status':'PILOT_ONLY','graphs':len(ids),'nodes':len(inputs[1]),'original_preparation_seconds':original,'one_time_encoding_seconds':preencode,'cached_collation_seconds':collate_times,'transfer_seconds':transfers,'training_step_seconds':train,'evaluation_step_seconds':evaluate,'projected_five_fold_300_epoch_usd':projected,'gpu_compute_only_projection_usd':(tr*205+ev*37)*300*5*rate,'resource_rate_usd_per_second':rate,'worker_seconds':time.monotonic()-start,'tensor_parity':'EXACT','maximum_worker_reservation_usd':1800*rate,'boundary':'one fixed reconstructed real batch; excludes full cache construction and disk IO; no held-out performance measured'}
@app.local_entrypoint()
def main():
 import json,hashlib
 out=R/'labs/_pilot_l121_results.json';budget=R/'labs/_budget_l121.json'
 with out.open('x') as f:json.dump({'status':'RESERVED','maximum_worker_usd':.406296},f)
 budget.write_text(json.dumps({'aggregate_cap_usd':10,'overhead_retry_reserve_usd':2,'pilot_reservation_usd':.406296,'timeout_seconds':1800,'retries':0,'rate_usd_per_second':.00022572,'rates_checked':'2026-09-27','rates_source':'https://modal.com/pricing','five_fold_status':'NOT_LAUNCHED','source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),R/'labs/relkit/cache_l121.py',R/'labs/relkit/cvitkovic_l118.py']}},indent=2))
 result=pilot.remote();result['resource_estimate_usd']=result['worker_seconds']*.00022572;out.write_text(json.dumps(result,indent=2));print(result)
