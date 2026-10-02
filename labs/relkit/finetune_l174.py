"""Complete L174 adaptation: visible policy, residual adapter and trainer."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from relkit.multitask_l173 import MaskedCellModel,cell_losses,tensor_packet,predict,score_outputs,task_weights,select_epoch


def adaptation_split(dates):
    """UTC seconds for target identities (not the padded date lookup array)."""
    d=np.asarray(dates).astype('datetime64[s]')
    return np.where(d<np.datetime64('2006-01-01'),-1,np.where(d<np.datetime64('2007-01-01'),0,np.where(d<np.datetime64('2008-01-01'),1,2))).astype(np.int64)


def configure_trainable(model,arm):
    """Optimizer membership and autograd both follow one explicit policy."""
    if arm not in {'freeze','full','adapter','scratch'}:raise ValueError('Unknown adaptation arm')
    for name,parameter in model.named_parameters():
        parameter.requires_grad_(arm in {'full','scratch'} or name.startswith('heads.') or (arm=='adapter' and name.startswith('adapter.')))


class ResidualAdapter(nn.Module):
    """Zero up projection makes x + up(ReLU(down(x))) initially identical to x."""
    def __init__(self):
        super().__init__();self.down=nn.Linear(32,8);self.up=nn.Linear(8,32)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
    def forward(self,x):return x+self.up(F.relu(self.down(x)))


class AdaptationModel(MaskedCellModel):
    """Same L173 encoder and original heads; optional adapter before heads."""
    def __init__(self,tasks,arm):
        super().__init__(tasks)
        self.adapter=ResidualAdapter() if arm=='adapter' else nn.Identity()

    def forward(self,context,task,columns,numeric,categories):
        present=(context!=0).unsqueeze(-1)
        tokens=self.column(columns[context])+self.category(categories[context])+self.numeric(numeric[context,None])
        tokens=tokens*present
        row=tokens[:,:4].sum(1)/present[:,:4].sum(1).clamp(min=1)
        parent=tokens[:,4:].sum(1)/present[:,4:].sum(1).clamp(min=1)
        hidden=self.adapter(self.shared(torch.cat([row,parent,self.column(task+1)],dim=1)))
        output=hidden.new_zeros((len(task),self.max_classes))
        for i,head in enumerate(self.heads):
            selected=task==i
            if selected.any():output[selected,:head.out_features]=head(hidden[selected])
        return output


def state_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def train_adaptation(arrays,tasks,source_state,arm,seed,destination,policy=configure_trainable,split_fn=adaptation_split,select_fn=select_epoch):
    """Ten full passes of 2006; 2007 selects; 2008+ is scored only afterward."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    split=split_fn(arrays['dates'][arrays['cell_ids']])
    train,valid,test=[np.flatnonzero(split==i) for i in range(3)]
    counts=torch.from_numpy(np.bincount(arrays['task'][train],minlength=len(tasks))).float()
    if min(len(train),len(valid),len(test))==0 or (counts<=0).any():raise ValueError('Missing adaptation population')
    model=AdaptationModel(tasks,arm)
    if arm!='scratch':
        missing,unexpected=model.load_state_dict(source_state,strict=False)
        assert not unexpected and set(missing)==({k for k in model.state_dict() if k.startswith('adapter.')} if arm=='adapter' else set())
    policy(model,arm)
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    torch.save(initial,destination/'initial.pt')
    trainable=[name for name,p in model.named_parameters() if p.requires_grad]
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=0)
    data=tensor_packet(arrays);rng=np.random.default_rng(seed);history=[];permutations=[]
    for epoch in range(10):
        model.train();order=rng.permutation(train);permutations.append(hashlib.sha256(order.tobytes()).hexdigest())
        weighted=0.;seen=np.zeros(len(tasks),dtype=np.int64)
        for ix in torch.as_tensor(order).split(1024):
            optimizer.zero_grad(set_to_none=True)
            out=model(data['context'][ix],data['task'][ix],data['columns'],data['numeric'],data['categories'])
            losses=cell_losses(out,data['task'][ix],data['target'][ix],tasks)
            loss=(losses*task_weights(data['task'][ix],counts,'task')).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();optimizer.step();weighted+=float(loss.detach())*len(ix)
            seen+=np.bincount(arrays['task'][ix],minlength=len(tasks))
        validation=score_outputs(predict(model,data,valid),arrays['target'][valid],arrays['task'][valid],tasks)
        history.append(dict(epoch=epoch+1,training_objective=weighted/len(train),training_counts=seen.tolist(),validation=validation))
        torch.save(model.state_dict(),destination/f'epoch-{epoch+1}.pt')
    selected=select_fn([e['validation']['macro_loss'] for e in history])+1
    model.load_state_dict(torch.load(destination/f'epoch-{selected}.pt',weights_only=True))
    output=predict(model,data,test)
    np.savez_compressed(destination/'test.npz',cell_ids=arrays['cell_ids'][test],task=arrays['task'][test],target=arrays['target'][test],output=output)
    changed=[k for k,v in model.state_dict().items() if not torch.equal(v,initial[k])]
    result=dict(arm=arm,seed=seed,source_sha256=state_hash(source_state) if arm!='scratch' else None,initial_sha256=state_hash(initial),initial_backbone_sha256=state_hash({k:v for k,v in initial.items() if not k.startswith(('heads.','adapter.'))}),trainable_names=trainable,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),total_parameters=sum(p.numel() for p in model.parameters()),selected_epoch=selected,selected_sha256=state_hash(model.state_dict()),changed_names=changed,permutation_sha256=permutations,epochs=history,test=score_outputs(output,arrays['target'][test],arrays['task'][test],tasks),seconds=time.monotonic()-start)
    (destination/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'{arm} seed{seed}: epoch{selected}, test macro {result["test"]["macro_loss"]:.6f}',flush=True)
    return result


def load_adaptation_packet(directory):
    """Authenticate every dependency before any output or training is created."""
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:raise ValueError('Input hash mismatch: '+name)
    arrays=dict(np.load(directory/'population.npz',allow_pickle=False))
    tasks=json.loads((directory/'tasks.json').read_text())
    states=[torch.load(directory/f'source-{seed}.pt',weights_only=True) for seed in range(3)]
    return arrays,tasks,states,manifest
