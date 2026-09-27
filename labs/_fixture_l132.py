"""Actual released neural stack: original parity and controlled root-marker ablation."""
import copy,json,sys
from pathlib import Path
import torch,pandas as pd
from torch_frame import stype
from torch_frame.data import Dataset
from torch_geometric.data import HeteroData
P=Path(__file__).resolve().parent
from relkit.identity_l132 import identity_forward,candidate_targets,cycle_witness
sys.path.insert(0,str(P/'sources/l132'))
from model import Model

def neural_fixture():
    torch.set_num_threads(1);torch.manual_seed(132)
    data=HeteroData();stats={}
    for kind in ['conditions','sponsors']:
        table=Dataset(pd.DataFrame({'constant':[1.,1.,1.,1.]}),col_to_stype={'constant':stype.numerical}).materialize()
        data[kind].tf=table.tensor_frame;stats[kind]=table.col_stats
        data[kind].time=torch.tensor([9,9,9,9])
        data[kind].num_nodes=4;data[kind].n_id=torch.tensor([0,1,1,0])
    data['conditions'].batch=torch.tensor([0,1,0,1]);data['conditions'].seed_time=torch.tensor([10,10]);data['conditions'].batch_size=2
    data['sponsors'].batch=torch.tensor([0,0,1,1]);data['sponsors'].n_id=torch.tensor([7,8,7,8])
    edge=torch.tensor([[0,2,3,1],[0,1,2,3]])
    data['conditions','to','sponsors'].edge_index=edge
    data['sponsors','rev','conditions'].edge_index=edge.flip(0)
    original=Model(data,stats,2,16,1,'sum','layer_norm',id_awareness=True)
    reference=copy.deepcopy(original);candidate=copy.deepcopy(original)
    rng=torch.get_rng_state()
    expected=reference.forward_dst_readout(data,'conditions','sponsors')
    torch.set_rng_state(rng)
    actual=identity_forward(candidate,data,'conditions','sponsors')
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    expected.sum().backward();actual.sum().backward();max_grad=0.
    for (name,p),(other,q) in zip(reference.named_parameters(),candidate.named_parameters()):
        assert name==other and (p.grad is None)==(q.grad is None)
        if p.grad is not None:
            assert torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all()
            torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
            max_grad=max(max_grad,float((p.grad-q.grad).abs().max()))
    original.eval()
    with torch.no_grad():
        on=identity_forward(original,data,'conditions','sponsors').flatten()
        off=identity_forward(original,data,'conditions','sponsors',False).flatten()
    assert (on-off).abs().max()>1e-6
    torch.testing.assert_close(off,off[0].expand_as(off),atol=1e-6,rtol=1e-6)
    target=candidate_targets(data['sponsors'].batch,data['sponsors'].n_id,torch.tensor([0,1]),torch.tensor([7,8]),2)
    outcomes={}
    for enabled in [False,True]:
        torch.manual_seed(1320)  # Paired dropout draws for both interventions.
        model=copy.deepcopy(original);optimizer=torch.optim.Adam(model.parameters(),lr=.01);trace=[]
        for epoch in range(80):
            model.train();optimizer.zero_grad();logits=identity_forward(model,data,'conditions','sponsors',enabled).flatten()
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target);loss.backward();optimizer.step();trace.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():prob=torch.sigmoid(identity_forward(model,data,'conditions','sponsors',enabled)).flatten()
        outcomes['marked' if enabled else 'unmarked']=dict(probabilities=prob.tolist(),loss=trace[-1],loss_trace=trace)
    assert outcomes['marked']['loss']<.1 and outcomes['unmarked']['loss']>.68
    witness=cycle_witness();assert [x['root_return'] for x in witness]==[0.,2.]
    return dict(status='PASS',source_output_max_error=float((actual-expected).abs().max().detach()),source_gradient_max_error=max_grad,same_state_on=on.tolist(),same_state_off=off.tolist(),targets=target.tolist(),course_fits=outcomes,cycle_witness=witness,scope='Synthetic four-candidate fixture; not a RelBench score',learner_status='PENDING_WRITTEN_DEFENSE')
if __name__=='__main__':
    r=neural_fixture();(P/'evidence/l132/fixture.json').write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k not in ['course_fits','cycle_witness']});print({k:v['loss'] for k,v in r['course_fits'].items()})
