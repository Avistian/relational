"""Visible course model: masked row/FK autocomplete, not RT or KumoRFM."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def erase_target(context, targets):
    """Erase every reference to each example's target, without mutating context."""
    context=np.asarray(context);targets=np.asarray(targets)
    if context.ndim!=2 or targets.shape!=(len(context),) or np.any(targets<=0):
        raise ValueError('One positive target identity per context row required')
    return np.where(context==targets[:,None],0,context)


def task_weights(tasks, counts, arm):
    """Importance weights use the entire training population, not batch counts."""
    if arm not in {'cell','task'} or counts.ndim!=1 or not torch.isfinite(counts).all() or (counts<=0).any():
        raise ValueError('Positive finite training counts and cell/task arm required')
    if arm=='cell':return torch.ones_like(tasks,dtype=torch.float32)
    return counts.sum()/(len(counts)*counts[tasks])


def select_epoch(validation):
    """First minimal finite validation macro loss; never inspect test scores."""
    scores=np.asarray(validation,dtype=float)
    if not len(scores) or not np.isfinite(scores).all():raise ValueError('Finite validation scores required')
    return int(scores.argmin())


def load_packet(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Input hash mismatch: '+name)
    arrays=dict(np.load(directory/'population.npz',allow_pickle=False))
    return arrays,json.loads((directory/'tasks.json').read_text()),manifest


class MaskedCellModel(nn.Module):
    """32-d tokens, separate row/parent pools, shared MLP, task-specific heads."""
    def __init__(self, tasks):
        super().__init__();self.tasks=tasks
        self.column=nn.Embedding(len(tasks)+1,32,padding_idx=0)
        self.category=nn.Embedding(1+sum(t['classes'] for t in tasks),32,padding_idx=0)
        self.numeric=nn.Linear(1,32,bias=False)
        self.shared=nn.Sequential(nn.Linear(96,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.heads=nn.ModuleList(nn.Linear(32,t['classes'] if t['kind']=='category' else 1) for t in tasks)
        self.max_classes=max(t['classes'] for t in tasks)

    def forward(self, context, task, columns, numeric, categories):
        present=(context!=0).unsqueeze(-1)
        tokens=self.column(columns[context])+self.category(categories[context])+self.numeric(numeric[context,None])
        tokens=tokens*present
        row=tokens[:,:4].sum(1)/present[:,:4].sum(1).clamp(min=1)
        parent=tokens[:,4:].sum(1)/present[:,4:].sum(1).clamp(min=1)
        hidden=self.shared(torch.cat([row,parent,self.column(task+1)],dim=1))
        output=hidden.new_zeros((len(task),self.max_classes))
        for i,head in enumerate(self.heads):
            selected=task==i
            if selected.any():output[selected,:head.out_features]=head(hidden[selected])
        return output


def cell_losses(output, task, target, tasks):
    losses=output.new_zeros(len(task))
    for i,spec in enumerate(tasks):
        selected=task==i
        if not selected.any():continue
        if spec['kind']=='number':
            losses[selected]=F.huber_loss(output[selected,0],target[selected],reduction='none',delta=1.)
        else:
            losses[selected]=F.cross_entropy(output[selected,:spec['classes']],target[selected].long(),reduction='none')
    return losses


def tensor_packet(arrays):
    return {k:torch.from_numpy(v.copy()) for k,v in arrays.items() if k in {'context','task','target','split','columns','numeric','categories','cell_ids'}}


def predict(model, data, indices, batch_size=1024):
    model.eval();outputs=[]
    with torch.no_grad():
        for ix in torch.as_tensor(indices,dtype=torch.long).split(batch_size):
            outputs.append(model(data['context'][ix],data['task'][ix],data['columns'],data['numeric'],data['categories']).numpy())
    return np.concatenate(outputs)


def score_outputs(outputs, targets, task_ids, tasks):
    losses=cell_losses(torch.as_tensor(outputs),torch.as_tensor(task_ids),torch.as_tensor(targets),tasks).numpy()
    per_task=[]
    for i,spec in enumerate(tasks):
        mask=task_ids==i
        if not mask.any():raise ValueError('No evaluation targets for '+spec['name'])
        row=dict(task=spec['name'],count=int(mask.sum()),loss=float(losses[mask].astype(float).mean()))
        if spec['kind']=='number':row['mae_raw']=float(np.abs(outputs[mask,0]-targets[mask]).mean()*spec['scale'])
        else:
            row['accuracy']=float((outputs[mask,:spec['classes']].argmax(1)==targets[mask]).mean())
            row['unknown_targets']=int((targets[mask]==0).sum())
        per_task.append(row)
    return dict(macro_loss=float(np.mean([x['loss'] for x in per_task])),cell_loss=float(losses.astype(float).mean()),tasks=per_task)


def train_run(arrays,tasks,arm,seed,destination,weight_fn=task_weights,select_fn=select_epoch):
    """Three full epochs; common validation criterion; test after selection only."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    data=tensor_packet(arrays);model=MaskedCellModel(tasks)
    initial=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    train=np.flatnonzero(arrays['split']==0);valid=np.flatnonzero(arrays['split']==1);test=np.flatnonzero(arrays['split']==2)
    counts=torch.bincount(data['task'][train],minlength=len(tasks)).float()
    rng=np.random.default_rng(seed);history=[];permutations=[]
    for epoch in range(3):
        model.train();order=rng.permutation(train);permutations.append(hashlib.sha256(order.tobytes()).hexdigest())
        task_sums=np.zeros(len(tasks));task_counts=np.zeros(len(tasks),dtype=int);weighted=0.
        for ix in torch.as_tensor(order).split(1024):
            optimizer.zero_grad(set_to_none=True)
            output=model(data['context'][ix],data['task'][ix],data['columns'],data['numeric'],data['categories'])
            losses=cell_losses(output,data['task'][ix],data['target'][ix],tasks)
            weights=weight_fn(data['task'][ix],counts,arm)
            loss=(losses*weights).mean();loss.backward();optimizer.step()
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            weighted+=float((losses.detach()*weights).sum())
            ids=data['task'][ix].numpy()
            task_sums+=np.bincount(ids,weights=losses.detach().numpy(),minlength=len(tasks))
            task_counts+=np.bincount(ids,minlength=len(tasks))
        validation=score_outputs(predict(model,data,valid),arrays['target'][valid],arrays['task'][valid],tasks)
        history.append(dict(epoch=epoch+1,training_objective=weighted/len(train),training_task_losses=(task_sums/task_counts).tolist(),training_task_counts=task_counts.tolist(),validation=validation))
        torch.save(model.state_dict(),destination/f'epoch-{epoch+1}.pt')
        print(f'{arm} seed{seed} epoch{epoch+1}: train={weighted/len(train):.6f}, validation macro={validation["macro_loss"]:.6f}',flush=True)
    chosen=select_fn([x['validation']['macro_loss'] for x in history])
    model.load_state_dict(torch.load(destination/f'epoch-{chosen+1}.pt',weights_only=True))
    predictions=predict(model,data,test)
    np.savez_compressed(destination/'test.npz',cell_ids=arrays['cell_ids'][test],task=arrays['task'][test],target=arrays['target'][test],output=predictions)
    result=dict(arm=arm,seed=seed,initial_sha256=initial,permutation_sha256=permutations,epochs=history,selected_epoch=chosen+1,test=score_outputs(predictions,arrays['target'][test],arrays['task'][test],tasks))
    (destination/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
