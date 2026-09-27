"""Measured RDL forward/backward trace, preserving the primary fit's RNG and state."""
import copy
import math
import torch

# %% Learner task 1: measure without severing the live computation

def activation_summary(x):
    if not torch.isfinite(x).all():
        raise ValueError('Nonfinite activation')
    v=x.detach().float().cpu()
    return dict(shape=list(x.shape),mean=float(v.mean()) if v.numel() else None,
                max_abs=float(v.abs().max()) if v.numel() else None,
                zero_fraction=float((v==0).float().mean()) if v.numel() else None,
                first_rows=v[:3,:6].tolist() if v.ndim==2 else v[:6].tolist())

# %% Learner task 2: each node occurrence belongs to a query clock

def relative_days(seed_time, node_time, batch_index):
    if seed_time.ndim!=1 or node_time.ndim!=1 or batch_index.shape!=node_time.shape:
        raise ValueError('One time and one owner per node occurrence required')
    if batch_index.dtype!=torch.long or (batch_index<0).any() or (batch_index>=len(seed_time)).any():
        raise ValueError('Invalid query owner')
    delta=seed_time[batch_index]-node_time
    if not torch.isfinite(delta).all() or (delta<0).any():
        raise ValueError('Future or nonfinite context time')
    return delta/(60*60*24)

# %% Learner task 3: supervise query roots, retaining autograd

def seed_readout(hidden, batch_size):
    if type(batch_size) is not int or hidden.ndim!=2 or not 0<batch_size<=len(hidden):
        raise ValueError('Invalid root count')
    return hidden[:batch_size]

# %% Visible instrumented forward pass

def traced_forward(model, batch, entity_table, detach_encoder=False):
    trace={};seed_time=batch[entity_table].seed_time
    def record(stage,values):
        trace[stage]={k:activation_summary(v) for k,v in values.items()}
    x=model.encoder(batch.tf_dict)
    record('row_encoder',x)
    if detach_encoder:x={k:v.detach() for k,v in x.items()}
    times={};days={}
    for kind,node_time in batch.time_dict.items():
        days[kind]=relative_days(seed_time,node_time,batch.batch_dict[kind])
        times[kind]=model.temporal_encoder.lin_dict[kind](model.temporal_encoder.encoder_dict[kind](days[kind]))
    record('relative_days',days);record('time_encoder',times)
    x={k:v+times[k] if k in times else v for k,v in x.items()}
    for kind,embedding in model.embedding_dict.items():x[kind]=x[kind]+embedding(batch[kind].n_id)
    record('time_added',x)
    for layer,(conv,norms) in enumerate(zip(model.gnn.convs,model.gnn.norms),1):
        x=conv(x,batch.edge_index_dict)
        record(f'layer_{layer}_relation_sum',x)
        x={k:norms[k](v) for k,v in x.items()}
        record(f'layer_{layer}_normalized',x)
        x={k:v.relu() for k,v in x.items()}
        record(f'layer_{layer}_relu',x)
    roots=seed_readout(x[entity_table],int(seed_time.size(0)))
    prediction=model.head(roots)
    record('root_readout',{entity_table:roots});record('prediction',{entity_table:prediction})
    return prediction,trace

# %% Isolated baseline, trace and detach intervention

def gradient_report(model):
    report={}
    groups={'encoder':model.encoder,'time':model.temporal_encoder,'gnn':model.gnn,'head':model.head}
    groups.update({'row:'+k:v for k,v in model.encoder.encoders.items()})
    for name,module in groups.items():
        params=list(module.parameters());grads=[p.grad for p in params if p.grad is not None]
        report[name]=dict(parameters=len(params),with_gradient=len(grads),
            nonfinite=sum(int((~torch.isfinite(g)).sum()) for g in grads),
            finite_l2=math.sqrt(sum(float(g.detach()[torch.isfinite(g)].double().square().sum()) for g in grads)))
    return report

def trace_batch(model,batch,entity_table):
    """Use model copies and restore RNG; never mutate the model being trained."""
    device=next(model.parameters()).device
    devices=[device.index or 0] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all() if devices else None
        def reset_rng():
            torch.set_rng_state(cpu_rng)
            if gpu_rng is not None:torch.cuda.set_rng_state_all(gpu_rng)
        baseline=copy.deepcopy(model);instrumented=copy.deepcopy(model);broken=copy.deepcopy(model)
        for m in [baseline,instrumented,broken]:m.zero_grad(set_to_none=True)
        target=batch[entity_table].y.float().view(-1,1)
        reset_rng();expected=baseline(batch,entity_table);base_loss=(expected-target).abs().mean();base_loss.backward()
        reset_rng();actual,stages=traced_forward(instrumented,batch,entity_table)
        loss=(actual-target).abs().mean();loss.backward()
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(loss,base_loss,rtol=1e-5,atol=1e-6)
        gradient_error=0.;nonfinite_names=[]
        for (name,a),(other,b) in zip(baseline.named_parameters(),instrumented.named_parameters()):
            assert name==other
            if a.grad is None or b.grad is None:assert a.grad is None and b.grad is None,name
            else:
                assert torch.equal(torch.isnan(a.grad),torch.isnan(b.grad)),name
                assert torch.equal(torch.isinf(a.grad),torch.isinf(b.grad)),name
                torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-6,equal_nan=True)
                finite=torch.isfinite(a.grad)
                if not finite.all():nonfinite_names.append(name)
                if finite.any():gradient_error=max(gradient_error,float((a.grad[finite]-b.grad[finite]).abs().max()))
        reset_rng();detached,_=traced_forward(broken,batch,entity_table,detach_encoder=True)
        (detached-target).abs().mean().backward()
        torch.testing.assert_close(detached,expected,rtol=1e-5,atol=1e-6)
        normal_grads=gradient_report(instrumented);broken_grads=gradient_report(broken)
        assert normal_grads['encoder']['finite_l2']>0 and broken_grads['encoder']['with_gradient']==0
        b=int(target.size(0))
        # Store actual feature and query identities, not only shape guesses.
        inputs={k:dict(rows=int(v.num_rows),columns={str(t):names for t,names in v.col_names_dict.items()},
                      node_ids=batch[k].n_id[:3].cpu().tolist(),owners=batch[k].batch[:3].cpu().tolist()) for k,v in batch.tf_dict.items()}
        time_inputs={k:dict(node_seconds=v[:6].cpu().tolist(),owner=batch.batch_dict[k][:6].cpu().tolist(),
                          cutoff_seconds=seed_time.cpu().tolist()) for k,v in batch.time_dict.items() for seed_time in [batch[entity_table].seed_time]}
        return dict(status='PASS',scope='First real training minibatch before any optimizer update; cloned models, training mode',
            batch_size=b,inputs=inputs,time_inputs=time_inputs,
            query_entities=batch[entity_table].n_id[:b].cpu().tolist(),
            query_cutoffs=batch[entity_table].seed_time.cpu().tolist(),
            targets=target.detach().cpu().view(-1).tolist(),predictions=actual.detach().cpu().view(-1).tolist(),
            edges={'|'.join(k):int(v.size(1)) for k,v in batch.edge_index_dict.items()},
            stages=stages,loss=float(loss.detach()),gradients=normal_grads,detached_gradients=broken_grads,
            parity=dict(max_output_error=float((actual-expected).abs().max().detach()),max_gradient_error=gradient_error,nonfinite_gradient_parameters=nonfinite_names,nonfinite_masks='MATCH',
                        detached_output_error=float((detached-expected).abs().max().detach())),
            interpretation='Detach preserves this forward prediction but removes the row encoder gradient path; finite gradients are sensitivity, not causal attribution. Any matched nonfinite gradients are explicitly counted; parity does not establish healthy training.')
