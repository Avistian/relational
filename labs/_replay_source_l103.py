"""Independently replay a saved run's complete test populations with the original model."""
import importlib.util,json,sys,hashlib
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import average_precision_score,roc_auc_score
from _fetch_l103 import fetch
from relkit.tgat_l103 import load_wikipedia
P=Path(__file__).resolve().parent
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def replay(directory,data_directory,device='cpu'):
 out=Path(directory);root,_=fetch();source=module('original_tgat',root/'module.py');graph=module('original_graph',root/'graph.py');utils=module('original_utils',root/'utils.py')
 n,e,d,a=load_wikipedia(data_directory);rows=[[] for _ in range(len(n))]
 for u,v,t,edge in zip(d['full']['u'],d['full']['v'],d['full']['t'],d['full']['e']):rows[u].append((v,edge,t));rows[v].append((u,edge,t))
 finder=graph.NeighborFinder(rows,uniform=True);model=source.TGAN(finder,n,e,num_layers=2,n_head=2,drop_out=.1)
 # Released trainer constructs Adam before moving the model. Verify current-runtime compatibility.
 optimizer=torch.optim.Adam(model.parameters(),lr=.0001);before=[id(p) for group in optimizer.param_groups for p in group['params']]
 model=model.to(device);assert before==[id(p) for p in model.parameters()], 'Release optimizer/device ordering does not preserve parameter identities'
 del optimizer
 checkpoint=torch.load(out/'selected.pt',map_location=device,weights_only=False)
 result=model.load_state_dict(checkpoint['weights'],strict=False)
 assert set(result.missing_keys)=={'n_feat_th','e_feat_th','edge_raw_embed.weight','node_raw_embed.weight'} and not result.unexpected_keys
 model.eval();np.random.set_state(checkpoint['numpy_state']);saved=np.load(out/'predictions.npz');summaries={};max_error=0.
 with torch.no_grad():
  for lane,key,pool in [('all','test','full'),('new','new_test','new_test')]:
   events=d[key];sampler=utils.RandEdgeSampler(d[pool]['u'],d[pool]['v']);offset=0;metrics=[]
   for start in range(0,len(events['u']),30):
    end=min(len(events['u'])-1,start+30)
    if start>=end:continue
    _,negative=sampler.sample(end-start)
    pos,neg=model.contrast(events['u'][start:end],events['v'][start:end],negative,events['t'][start:end],20)
    pos,neg=pos.cpu().numpy(),neg.cpu().numpy();stop=offset+len(pos)
    np.testing.assert_array_equal(negative,saved[f'{lane}_negative'][offset:stop]);np.testing.assert_array_equal(events['e'][start:end],saved[f'{lane}_e'][offset:stop])
    for got,name in [(pos,'p'),(neg,'n')]:
     expected=saved[f'{lane}_{name}'][offset:stop];np.testing.assert_allclose(got,expected,rtol=1e-5,atol=2e-6);max_error=max(max_error,float(np.max(np.abs(got-expected))))
    y=np.r_[np.ones(len(pos)),np.zeros(len(neg))];score=np.r_[pos,neg]
    metrics.append([average_precision_score(y,score),roc_auc_score(y,score),np.mean((score>.5)==y)]);offset=stop
   summaries[lane]={'events':offset,'ap':float(np.mean(np.asarray(metrics)[:,0]))}
 report={'status':'PASS','scope':'Complete seed-0 original-model all/new test prediction replay','device':device,'max_prediction_error':max_error,'optimizer_before_device_move':'PARAMETER_IDENTITIES_PRESERVED','populations':summaries,'source_checkpoint_sha256':hashlib.file_digest((out/'selected.pt').open('rb'),'sha256').hexdigest()}
 (out/'original-replay.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':print(replay(sys.argv[1],sys.argv[2],sys.argv[3] if len(sys.argv)>3 else 'cpu'))
