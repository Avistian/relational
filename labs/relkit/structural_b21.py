"""B21 finite structural stress experiment. Course mechanism, not paper training."""
import itertools
import numpy as np
import torch


def fixture():
    return dict(parent_time=[0,1,2,12], child_time=[3,4,5,6,7,8],
                session=[0,0,1,2,3,4], original=[0,0,1,2,1,2], cutoff=10,
                parent_x=[[1.,0.],[0.,1.],[1.,1.],[9.,9.]],
                child_x=[[1.,0.],[3.,1.],[2.,0.],[5.,1.],[4.,0.],[6.,1.]],
                target=[2.,4.,6.], query_ids=[0,1,2])


def changed_cells(original, assignment):
    a,b=np.asarray(original),np.asarray(assignment)
    if a.ndim!=1 or a.shape!=b.shape:raise ValueError('FK shape mismatch')
    return int(np.count_nonzero(a!=b))


def validate_assignment(data, assignment, budget):
    a=np.asarray(assignment)
    if not isinstance(budget,(int,np.integer)) or budget<0:raise ValueError('Invalid budget')
    if a.shape!=(len(data['original']),) or not np.issubdtype(a.dtype,np.integer):raise ValueError('Integer FK vector required')
    if np.any(a<0) or np.any(a>=len(data['parent_time'])):raise ValueError('Dangling FK')
    for session in set(data['session']):
        owners=a[np.asarray(data['session'])==session]
        if len(set(owners))!=1:raise ValueError('Session -> account dependency violated')
    if np.any(np.asarray(data['parent_time'])[a]>np.asarray(data['child_time'])):raise ValueError('Parent unavailable at event')
    if np.any(np.asarray(data['child_time'])>data['cutoff']):raise ValueError('Future event')
    if changed_cells(data['original'],a)>budget:raise ValueError('FK cell budget exceeded')
    return True


def direction_score(forward_gradient, reverse_gradient, before, after):
    gf,gr,b,a=map(np.asarray,(forward_gradient,reverse_gradient,before,after))
    if b.ndim!=2 or gf.shape!=b.shape or a.shape!=b.shape or gr.shape!=b.T.shape:raise ValueError('Gradient/adjacency shape mismatch')
    delta=a-b
    return float(np.sum(gf*delta)+np.sum(gr*delta.T))


def adjacency(assignment, n_parent=4):
    a=np.zeros((len(assignment),n_parent),dtype=np.float64)
    a[np.arange(len(assignment)),assignment]=1.
    return a


def legal_states(data, budget):
    groups=sorted(set(data['session']));states=[]
    for owners in itertools.product(range(len(data['parent_time'])),repeat=len(groups)):
        a=[owners[groups.index(g)] for g in data['session']]
        try:validate_assignment(data,a,budget)
        except ValueError:continue
        states.append(a)
    return states


def parameters(seed):
    rng=np.random.default_rng(seed);w={}
    for node in ['parent','child']:w['embed_'+node]=rng.normal(0,.4,(2,4))
    for layer in range(2):
        for node in ['parent','child']:
            for path in ['own','neighbor']:w[f'{layer}_{node}_{path}']=rng.normal(0,.4,(4,4))
            w[f'{layer}_{node}_bias']=np.full(4,.2)
    w['head']=rng.normal(0,.4,4);w['head_bias']=3.
    return w


def forward(data, weights, forward_edges, reverse_edges):
    """A[child,parent] and R[parent,child]. Valid discrete graphs have R=A.T.

    Row-normalized weighted means have zero for empty neighborhoods. The
    continuous mask extension normalizes by summed weight, not candidate count.
    Layers update both types simultaneously. Parameters are fixed float64.
    """
    def t(x):return torch.as_tensor(x,dtype=torch.float64)
    p=t(data['parent_x'])@t(weights['embed_parent'])
    c=t(data['child_x'])@t(weights['embed_child'])
    for layer in range(2):
        incoming_parent=forward_edges.T
        incoming_child=reverse_edges.T
        mp=(incoming_parent@c)/incoming_parent.sum(1,keepdim=True).clamp(min=1e-12)
        mc=(incoming_child@p)/incoming_child.sum(1,keepdim=True).clamp(min=1e-12)
        np_=torch.relu(p@t(weights[f'{layer}_parent_own'])+mp@t(weights[f'{layer}_parent_neighbor'])+t(weights[f'{layer}_parent_bias']))
        nc=torch.relu(c@t(weights[f'{layer}_child_own'])+mc@t(weights[f'{layer}_child_neighbor'])+t(weights[f'{layer}_child_bias']))
        p,c=np_,nc
    return (p@t(weights['head'])+t(weights['head_bias']))[data['query_ids']]


def experiment():
    data=fixture();before=adjacency(data['original']);states=legal_states(data,2)
    report=dict(name='B21-FK-STRESS',status='COMPLETE',data=data,seeds=[0,1,2],budgets=[0,1,2],
                methods=['random','gradient','exhaustive'],states=[],conditions=[],weights={},gradients={},
                claim='Finite random-network stress diagnostic; no training, generalization or paper parity')
    target=torch.tensor(data['target'],dtype=torch.float64)
    for seed in report['seeds']:
        w=parameters(seed);report['weights'][str(seed)]={k:np.asarray(v).tolist() for k,v in w.items()}
        a=torch.tensor(before,requires_grad=True);r=torch.tensor(before.T,requires_grad=True)
        clean_pred=forward(data,w,a,r);loss=((clean_pred-target)**2).mean();loss.backward()
        gf,gr=a.grad.numpy(),r.grad.numpy();report['gradients'][str(seed)]=dict(forward=gf.tolist(),reverse=gr.tolist())
        records=[]
        for state in states:
            af=adjacency(state);validate_graph(data,state,2,af,af.T);pred=forward(data,w,torch.tensor(af),torch.tensor(af.T)).detach().numpy()
            row=dict(seed=seed,assignment=state,cost=changed_cells(data['original'],state),prediction=pred.tolist(),
                     loss=float(np.mean((pred-np.asarray(data['target']))**2)),
                     linear_gain=direction_score(gf,gr,before,af))
            records.append(row);report['states'].append(row)
        for budget in report['budgets']:
            feasible=[x for x in records if x['cost']<=budget]
            exact_cost=[x for x in feasible if x['cost']==budget]
            rng=np.random.default_rng(1000+10*seed+budget)
            random=exact_cost[int(rng.integers(len(exact_cost)))]
            # Strict improvement only; ties prefer fewer edits, then lexicographic assignment.
            gradient=min(feasible,key=lambda x:(-x['linear_gain'],x['cost'],x['assignment']))
            exhaustive=min(feasible,key=lambda x:(-x['loss'],x['cost'],x['assignment']))
            for method,row in [('random',random),('gradient',gradient),('exhaustive',exhaustive)]:
                report['conditions'].append(dict(row,method=method,budget=budget,clean_loss=float(loss.detach()),
                    actual_gain=row['loss']-float(loss.detach()),regret=exhaustive['loss']-row['loss']))
    return report


def validate_graph(data, assignment, budget, forward_edges, reverse_edges):
    validate_assignment(data,assignment,budget)
    expected=adjacency(assignment,len(data['parent_time']))
    if not np.array_equal(forward_edges,expected) or not np.array_equal(reverse_edges,expected.T):
        raise ValueError('Graph directions do not match the database')
    return True
