"""Independent arithmetic, selection, permutation and released-model checks."""
import json,sys,torch,numpy as np
from pathlib import Path
LAB=Path(__file__).resolve().parent
sys.path.insert(0,str(LAB))
from relkit.gin_l088 import neighbor_sum,graph_readout,select_epoch,GIN,load_mutag,collate,summarize_grid

def main():
 torch.set_num_threads(1)
 h=torch.tensor([[1.,2.],[3.,4.],[5.,6.]],requires_grad=True)
 e=torch.tensor([[0,1,1,2],[1,0,2,1]])
 z=neighbor_sum(h,e);torch.testing.assert_close(z,torch.tensor([[3.,4.],[6.,8.],[3.,4.]]))
 z.sum().backward();torch.testing.assert_close(h.grad,torch.tensor([[1.,1.],[2.,2.],[1.,1.]]))
 batch=torch.tensor([0,0,1]);torch.testing.assert_close(graph_readout(h,batch,2),torch.tensor([[4.,6.],[5.,6.]]))
 assert select_epoch([[.9,.8],[.5,.8]])==1 # common epoch, not separate maxima
 assert select_epoch([[.8,.8],[.6,.6]])==0 # earliest tie
 g=load_mutag(LAB/'data/l088');assert len(g)==188
 x,e,b,y=collate(g[:3]);torch.manual_seed(0);model=GIN(7,16,2,0,False).eval()
 p=torch.randperm(len(x));inverse=torch.argsort(p)
 torch.testing.assert_close(model((x,e,b,y)),model((x[p],inverse[e],b[p],y)),rtol=1e-5,atol=1e-5)
 # Original implementation is an independent oracle, loaded from pinned vendored source.
 sys.path.insert(0,str(LAB/'sources/l088/models'))
 from graphcnn import GraphCNN
 from types import SimpleNamespace
 original=GraphCNN(5,2,7,16,2,0,False,'sum','sum',torch.device('cpu')).eval()
 original.load_state_dict(model.state_dict())
 og=[SimpleNamespace(g=range(len(a['x'])),node_features=a['x'],edge_mat=a['edges']) for a in g[:3]]
 expected=original(og);actual=model((x,e,b,y));torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)
 actual.square().sum().backward();expected.square().sum().backward()
 for (n,a),(m,c) in zip(model.named_parameters(),original.named_parameters()):
  assert n==m
  if a.grad is not None:torch.testing.assert_close(a.grad,c.grad,rtol=2e-4,atol=2e-3)
 # Train-mode BN and logit-dropout oracle, including learned epsilon support.
 for learn in [False,True]:
  torch.manual_seed(9);port=GIN(7,16,2,.5,learn).train()
  ref=GraphCNN(5,2,7,16,2,.5,learn,'sum','sum',torch.device('cpu')).train()
  ref.load_state_dict(port.state_dict())
  torch.manual_seed(91);a=port((x,e,b,y))
  torch.manual_seed(91);c=ref(og)
  torch.testing.assert_close(a,c,rtol=1e-4,atol=1e-4)
  for (n,a),(m,c) in zip(port.named_buffers(),ref.named_buffers()):
   assert n==m;torch.testing.assert_close(a,c,rtol=1e-4,atol=1e-4)
 # Full loader oracle: exact categorical features, labels and unordered edges.
 import importlib.util,tempfile,os,shutil
 spec=importlib.util.spec_from_file_location('l088_original_util',LAB/'sources/l088/util.py')
 util=importlib.util.module_from_spec(spec);spec.loader.exec_module(util)
 cwd=os.getcwd()
 with tempfile.TemporaryDirectory() as tmp:
  folder=Path(tmp)/'dataset/MUTAG';folder.mkdir(parents=True)
  shutil.copyfile(LAB/'data/l088/MUTAG.txt',folder/'MUTAG.txt')
  try:
   os.chdir(tmp);released,_=util.load_data('MUTAG',False)
  finally:os.chdir(cwd)
 for a,c in zip(g,released):
  assert a['y']==c.label
  torch.testing.assert_close(a['x'],c.node_features)
  assert set(map(tuple,a['edges'].T.tolist()))==set(map(tuple,c.edge_mat.T.tolist()))
 # A partial grid cannot pass as complete paper selection.
 try:summarize_grid([])
 except AssertionError:pass
 else:raise AssertionError('Incomplete paper grid accepted')
 with tempfile.TemporaryDirectory() as tmp:
  (Path(tmp)/'MUTAG.txt').write_text('corrupted bytes')
  try:load_mutag(tmp)
  except ValueError:pass
  else:raise AssertionError('Corrupted data accepted')
 r={'status':'PASS','arithmetic_and_gradient':'PASS','common_epoch_selection':'PASS','permutation_invariance':'PASS','released_forward_and_parameter_gradients':'PASS','graphs':188,'released_loader_all_graphs':'PASS','train_mode_BN_dropout_epsilon_oracle':'PASS','incomplete_grid_and_corrupt_data_rejected':'PASS'}
 (LAB/'_verify_l088_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
if __name__=='__main__':main()
