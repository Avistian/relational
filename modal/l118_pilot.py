"""One bounded full-width real-data timing pilot; maximum worker charge < USD 0.66."""
from pathlib import Path
import modal
R=Path(__file__).resolve().parents[1]
app=modal.App('l118-cvitkovic-pilot')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install('numpy==1.26.4','scikit-learn==1.5.2')
.add_local_file(R/'labs/relkit/cvitkovic_l118.py','/work/relkit/cvitkovic_l118.py')
.add_local_dir(R/'labs/data/l118/pilot','/data')
.add_local_file(R/'labs/sources/l118/data/homecreditdefaultrisk/homecreditdefaultrisk.db_info.json','/work/info.json'))
@app.function(image=image,gpu='T4',cpu=2,memory=16384,timeout=1800,retries=0)
def pilot():
 import sys,time,json,torch,numpy as np
 sys.path.insert(0,'/work')
 from relkit.cvitkovic_l118 import PreparedGraphs,collate_graphs,feature_schema,CvitkovicGCN,move_batch
 from torch.nn import functional as F
 torch.set_num_threads(2);torch.manual_seed(1234)
 start=time.monotonic();info=json.load(open('/work/info.json'));ids=json.load(open('/data/ids.json'))
 ds=PreparedGraphs('/data',ids);t=time.monotonic();inputs,y,_=collate_graphs([ds[i] for i in range(len(ds))],info);prep=time.monotonic()-t
 model=CvitkovicGCN(feature_schema(info),info['node_type_to_int']).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0)
 inp=move_batch(inputs,'cuda');y=y.cuda();times=[];losses=[]
 for step in range(6):
  torch.cuda.synchronize();t=time.monotonic();model.train();optimizer.zero_grad();loss=F.cross_entropy(model(*inp),y);loss.backward();optimizer.step();torch.cuda.synchronize()
  times.append(time.monotonic()-t);losses.append(loss.item())
 model.eval();torch.cuda.synchronize();t=time.monotonic()
 with torch.no_grad():model(*inp)
 torch.cuda.synchronize();evaluation=time.monotonic()-t
 elapsed=time.monotonic()-start
 rate=.000164+2*.0000131+16*.00000222
 # 300 epochs is the release maximum. Early stopping may reduce it, but not budgeted optimistically.
 epoch=(prep+float(np.median(times[1:])))*205+(prep+evaluation)*37
 return {'status':'PILOT_ONLY','graphs':len(ids),'nodes':len(inputs[1]),'parameter_count':sum(p.numel() for p in model.parameters()),'batch_preparation_seconds':prep,'training_step_seconds':times,'evaluation_step_seconds':evaluation,'losses':losses,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'worker_seconds':elapsed,'resource_rate_usd_per_second':rate,'resource_estimate_usd':elapsed*rate,'projected_epoch_seconds':epoch,'projected_five_fold_300_epoch_usd':epoch*5*300*rate,'projected_five_fold_50_epoch_usd':epoch*5*50*rate,'projection_limit':'One fixed real-data batch; excludes full preparation, IO variation, retries and final tests. Repeated training of a batch is a timing pilot, not validation.'}
@app.local_entrypoint()
def main():
 import json
 out=R/'labs/_pilot_l118_results.json'
 if out.exists():raise RuntimeError('Existing pilot evidence; review budget before another paid pilot')
 out.write_text(json.dumps({'status':'RESERVED','maximum_worker_usd':1800*.00022572}))
 result=pilot.remote();out.write_text(json.dumps(result,indent=2));print(result)
