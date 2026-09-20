"""Bounded resource probes: CS load guard, or paper-width update on NN (not CS)."""
import argparse,json,resource,sys,time,traceback
from pathlib import Path
import torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'relkit'))
from hgt_l093 import *
P=Path(__file__).resolve().parent;m=json.loads((P/'_sources_l093.json').read_text());p=argparse.ArgumentParser();p.add_argument('--cs-load',action='store_true');a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(93);start=time.time()
if a.cs_load:
 limit=10*1024**3;resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
 try:
  g=load_oag(P/'data/l093/graph_CS.pk',m['cs_data']['sha256']);result={'status':'LOADED','nodes':{k:len(v) for k,v in g.node_feature.items()}}
 except (MemoryError,RuntimeError) as e:result={'status':'RESOURCE_BLOCKED','error':type(e).__name__+': '+str(e),'virtual_memory_guard_bytes':limit}
 path=P/'_cs_resource_l093_results.json'
else:
 g=load_oag(P/'data/l093/graph_NN.pk',m['nn_data']['sha256']);c,pairs=field_protocol(g);config=protocol_config('paper')
 b,audit=sample_oag(g,pairs['train'],c,93,256,6,128)
 model=HGTModel(b[0].shape[1],256,len(g.get_types()),len(g.get_meta_graph())+1,len(c),8,3)
 opt=torch.optim.AdamW(model.parameters());pred=model(b);loss=F.kl_div(pred,b[-1],reduction='batchmean');loss.backward();opt.step()
 result={'status':'PASS','scope':'One paper-width update on NN; not CS and not a result reproduction','nodes':audit['nodes'],'edges':audit['edges'],'loss':float(loss.detach()),'width':256,'layers':3,'heads':8,'batch_size':256,'sample_width':128,'sample_depth':6}
 path=P/'_capacity_l093_results.json'
result.update(seed=93,seconds=time.time()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);path.write_text(json.dumps(result,indent=2)+'\n');print(result)
