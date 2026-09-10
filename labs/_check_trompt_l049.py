"""Paper-equation and intervention checks for the complete numeric Trompt path."""
import torch
import relkit.trompt_l049 as tm

def check():
    torch.set_num_threads(1)
    assert hasattr(tm, 'Trompt'), 'Missing complete numeric model'
    torch.manual_seed(49)
    model=tm.Trompt(3,d=8,prompts=4,layers=2).double()
    x=torch.randn(3,3,dtype=torch.double)
    logits,trace=model(x,return_trace=True)
    assert logits.shape==(3,2,2)
    assert torch.equal(trace[0]['weights'][0],trace[0]['weights'][1])
    assert not torch.allclose(trace[1]['weights'][0],trace[1]['weights'][1])
    assert torch.allclose(logits,torch.cat([model(row[None]) for row in x]),atol=1e-10)
    # Independently expand the exact Fig.3 operations using copied parameters.
    c=model.cells[0];previous=x.new_zeros(3,4,8)
    p=c.prompt_norm(c.prompts)[None].expand(3,-1,-1)
    h=c.fusion(torch.cat((p,previous),-1))+p+previous
    m=(h@c.column_norm(c.columns).T).softmax(-1)
    e=c.value_norm(torch.relu(x[:,:,None]*c.value_weight+c.value_bias))
    z=torch.relu(e[:,None]*c.expansion_weight[None,:,None,None])
    expanded=c.group_norm(z)+e[:,None]
    expected=torch.stack([(m[:,i,:,None]*expanded[:,i]).sum(1) for i in range(4)],1)
    actual,details=c(x,previous,return_trace=True)
    assert torch.allclose(actual,expected,atol=1e-12)
    assert torch.allclose(details['expanded'],expanded)
    # Supervision sums cell losses, not loss after averaging predictions.
    y=torch.tensor([0,1,0]);loss=tm.trompt_loss(logits,y)
    independent=sum(torch.nn.functional.cross_entropy(logits[:,i],y) for i in range(2))
    assert torch.allclose(loss,independent)
    assert not torch.allclose(loss,torch.nn.functional.cross_entropy(logits.mean(1),y))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(c.fusion.weight.grad.abs().sum()>0 for c in model.cells)
    assert all(c.expansion_weight.grad.abs().sum()>0 for c in model.cells)
    # A learned expansion must change actual feature vectors and loss gradients.
    assert not torch.allclose(details['expanded'][:,0],details['expanded'][:,1])
    # Student primitives are on the entire repeated prediction path.
    counts={'weights':0,'reduce':0};pw,pr=tm.prompt_weights,tm.prompt_reduce
    def weights(*a):counts['weights']+=1;return pw(*a)
    def reduce(*a):counts['reduce']+=1;return pr(*a)
    try:
        tm.prompt_weights,tm.prompt_reduce=weights,reduce
        model(x)
    finally:tm.prompt_weights,tm.prompt_reduce=pw,pr
    assert counts=={'weights':2,'reduce':2}
    return {'status':'PASS','checks':['zero first state; later row specificity','batch independence','Fig.3 independent expansion','summed cell supervision','all parameter gradients','learned prompt expansion','live student routing'], 'source':'Trompt v2 Fig.2–4, Eq.1–9 and §5.1; local numeric choices explicit'}

if __name__=='__main__':
    import json
    from pathlib import Path
    r=check();Path(__file__).with_name('_check_trompt_l049_results.json').write_text(json.dumps(r,indent=2));print(r)
