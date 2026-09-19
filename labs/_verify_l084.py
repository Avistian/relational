"""Independent dense oracle, gradients, invariances and label-access checks."""
import json
from pathlib import Path
import torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'relkit'))
from gat_l084 import neighbor_softmax, weighted_messages, merge_heads, AttentionHead, GAT, train_cora, load_cora

def verify():
 torch.set_num_threads(1);torch.manual_seed(9)
 edge=torch.tensor([[0,0,1,1,2],[0,1,1,2,2]])
 scores=torch.tensor([0.,1.,2.,-1.,3.],dtype=torch.double,requires_grad=True)
 a=neighbor_softmax(scores,edge[0],3)
 expected=torch.cat([scores[:2].softmax(0),scores[2:4].softmax(0),torch.ones(1,dtype=torch.double)])
 torch.testing.assert_close(a,expected)
 h=torch.randn(3,2,dtype=torch.double,requires_grad=True)
 got=weighted_messages(h,edge,a)
 oracle=torch.stack([a[0]*h[0]+a[1]*h[1],a[2]*h[1]+a[3]*h[2],h[2]])
 torch.testing.assert_close(got,oracle)
 assert torch.autograd.gradcheck(lambda s,x:weighted_messages(x,edge,neighbor_softmax(s,edge[0],3)),(scores,h))
 torch.testing.assert_close(merge_heads([h,h+2],False),h+1)
 assert merge_heads([h,h],True).shape==(3,4)
 head=AttentionHead(4,3).double().eval();x=torch.randn(3,4,dtype=torch.double)
 out,alpha=head(x,edge,return_attention=True)
 z=x@head.w;q=(z@head.a_receiver).flatten()+head.b_receiver;k=(z@head.a_sender).flatten()+head.b_sender
 mask=torch.zeros(3,3,dtype=torch.bool);mask[edge[0],edge[1]]=True
 dense=torch.nn.functional.leaky_relu(q[:,None]+k[None,:],.2).masked_fill(~mask,-torch.inf).softmax(1)
 torch.testing.assert_close(out,dense@z+head.bias)
 torch.testing.assert_close(alpha,dense[edge[0],edge[1]])
 torch.testing.assert_close(head(x,edge[:,torch.tensor([4,2,0,3,1])]),out)
 permutation=torch.tensor([2,0,1]);inverse=permutation.argsort()
 torch.testing.assert_close(head(x[permutation],inverse[edge]),out[permutation])
 data=load_cora(Path(__file__).parent,json.loads((Path(__file__).parent/'_sources_l084.json').read_text()))
 assert data[0].shape==(2708,1433) and [len(t) for t in data[3:]]==[140,500,1000]
 assert data[1].shape==(2,13264)
 r,m=train_cora(data,7,max_epochs=3,return_model=True)
 changed=list(data);changed[2]=data[2].clone();changed[2][data[5]]=(changed[2][data[5]]+1)%7
 r2,m2=train_cora(tuple(changed),7,max_epochs=3,return_model=True)
 for p,q in zip(m.parameters(),m2.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
 assert r['trace']==r2['trace']
 result={'status':'PASS','checks':['segmented softmax','weighted sum dense oracle','double gradient check','head concat/mean','release head dense oracle','edge permutation','node relabeling equivariance','Cora shapes and loops','test-label intervention leaves training/checkpoint unchanged']}
 (Path(__file__).parent/'_verify_l084_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':verify()
