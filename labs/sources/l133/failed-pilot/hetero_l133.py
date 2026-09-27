"""Visible relational convolution; dense features and COO edges, sum GraphSAGE only."""
import copy
import torch

def sum_neighbors(source, edge_index, num_destination):
    """Gather source rows, scatter-add into destination rows. Preserve duplicates."""
    if source.ndim != 2 or edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError('Expected dense [N,C] features and long [2,E] edges')
    if num_destination < 0 or (edge_index < 0).any():
        raise ValueError('Negative node index or destination count')
    if edge_index.numel() and (edge_index[0].max() >= len(source) or edge_index[1].max() >= num_destination):
        raise ValueError('Edge index outside its source/destination table')
    out = source.new_zeros((num_destination, source.shape[1]))
    return out.index_add(0, edge_index[1], source[edge_index[0]])

def relation_output(conv, source, destination, edge_index):
    """Pinned sum-SAGE: transformed neighbor sum + relation-specific root transform."""
    if conv.aggr != 'sum' or conv.project or conv.normalize:
        raise ValueError('This visible operator supports unprojected, unnormalized sum-SAGE')
    neighbors = sum_neighbors(source, edge_index, len(destination))
    out = conv.lin_l(neighbors)  # Includes this relation's bias, even for no neighbors.
    if conv.root_weight:
        out = out + conv.lin_r(destination)
    return out

def merge_relations(outputs):
    """Sum relation outputs by destination TYPE, never by source type."""
    grouped = {}
    for (_, _, destination), values in outputs.items():
        grouped.setdefault(destination, []).append(values)
    return {kind: torch.stack(values, dim=0).sum(dim=0) for kind, values in grouped.items()}

def explicit_layer(conv, x_dict, edge_dict):
    """An omitted edge key skips that relation. An empty edge tensor executes it."""
    if conv.aggr != 'sum':
        raise ValueError('Expected outer relation sum')
    terms = {}
    for relation, module in conv.convs.items():
        if relation in edge_dict:
            source, _, destination = relation
            terms[relation] = relation_output(module, x_dict[source], x_dict[destination], edge_dict[relation])
    return merge_relations(terms), terms

def audit_layer(conv, x_dict, edge_dict):
    """Compare independent paths on clones; inputs and original parameters untouched."""
    a=copy.deepcopy(conv);b=copy.deepcopy(conv)
    a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True)
    xa={k:v.detach().clone().requires_grad_(True) for k,v in x_dict.items()}
    xb={k:v.detach().clone().requires_grad_(True) for k,v in x_dict.items()}
    ref=a(xa,edge_dict);out,terms=explicit_layer(b,xb,edge_dict)
    assert ref.keys()==out.keys()
    error=0.
    for k in ref:
        torch.testing.assert_close(ref[k],out[k],rtol=1e-5,atol=1e-5)
        if out[k].numel():error=max(error,float((ref[k]-out[k]).detach().abs().max()))
    sum(v.square().mean() for v in ref.values() if v.numel()).backward()
    sum(v.square().mean() for v in out.values() if v.numel()).backward()
    gradient_error=0.
    for name,p,q in [(n,p,dict(b.named_parameters())[n]) for n,p in a.named_parameters()]+[(k,xa[k],xb[k]) for k in xa]:
        if p.grad is None or q.grad is None:assert p.grad is None and q.grad is None,name
        else:
            assert torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all(),name
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=1e-5)
            if p.grad.numel():gradient_error=max(gradient_error,float((p.grad-q.grad).abs().max()))
    return dict(status='PASS',output_max_error=error,gradient_max_error=gradient_error,
        shapes={k:list(v.shape) for k,v in out.items()},
        relations={'|'.join(k):dict(edges=edge_dict[k].shape[1],first_destination_first_channels=v[:1,:4].detach().cpu().tolist()) for k,v in terms.items()},
        scope='Convolution parameters and input-feature gradients; row encoders outside this check')

def tiny_experiment():
    """Train the visible layer on a tiny schema; course demonstration, not a paper score."""
    from torch_geometric.nn import HeteroConv,SAGEConv
    torch.manual_seed(133);torch.set_num_threads(1)
    x={'orders':torch.tensor([[1.],[2.],[4.],[3.]]),'customers':torch.tensor([[1.],[2.]])}
    e={('orders','belongs','customers'):torch.tensor([[0,1,2,3],[0,0,1,1]]),('customers','self','customers'):torch.tensor([[0,1],[0,1]])}
    conv=HeteroConv({k:SAGEConv((1,1),1,aggr='sum') for k in e},aggr='sum')
    parity=audit_layer(conv,x,e)
    target=torch.tensor([[3.],[7.]])
    opt=torch.optim.Adam(conv.parameters(),lr=.03);losses=[]
    for _ in range(150):
        opt.zero_grad();y,_=explicit_layer(conv,x,e);loss=(y['customers']-target).square().mean();loss.backward();opt.step();losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]*.02
    return dict(status='PASS',parity=parity,initial_mse=losses[0],final_mse=losses[-1],steps=150,scope='Synthetic course fit; not RelBench reproduction')
