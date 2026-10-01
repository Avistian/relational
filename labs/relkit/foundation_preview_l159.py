"""Visible, synthetic objective laboratory. This is NOT a BART implementation.

120 independent four-row tables; three binary reconstruction tasks per table.
Two stages: adapt a tiny row codec, freeze it, then learn graph propagation.
"""
import copy
import hashlib
import json
import math
import time
import torch
from torch import nn


def mask_serializations(tokens, identities, target):
    """Remove every serialization of one semantic identity, leaving targets intact."""
    if tokens.shape != identities.shape or tokens.ndim != 2:
        raise ValueError('Expected equally shaped row-by-token tensors')
    hidden = identities == target
    if target < 0 or not bool(hidden.any()):
        raise ValueError('Unknown semantic target')
    result = tokens.clone()
    result[hidden] = 0  # 0 is the dedicated MASK token, never a target class.
    return result


def masked_cross_entropy(logits, labels, selected):
    """Mean classification loss over explicitly selected targets, not all fields."""
    if logits.shape[:-1] != labels.shape or labels.shape != selected.shape:
        raise ValueError('Target and selection shapes must match logits')
    if selected.dtype != torch.bool or not bool(selected.any()):
        raise ValueError('Select at least one target with a boolean mask')
    scores = logits[selected]
    truth = labels[selected]
    return (torch.logsumexp(scores, dim=-1) - scores.gather(1, truth[:, None]).squeeze(1)).mean()


def freeze_codec(codec):
    """Freeze parameters, not the function: the decoder must differentiate its input."""
    codec.eval()
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


class RowCodec(nn.Module):
    """Embedding mean -> nonlinear row representation -> three binary heads."""
    def __init__(self, width=16):
        super().__init__()
        self.embedding = nn.Embedding(14, width)
        self.encoder = nn.Linear(width, width)
        self.decoder = nn.Linear(width, 6)

    def encode(self, tokens):
        return torch.tanh(self.encoder(self.embedding(tokens).mean(dim=-2)))

    def decode(self, hidden):
        return self.decoder(hidden).reshape(*hidden.shape[:-1], 3, 2)


class TableGraph(nn.Module):
    """One GCN-style layer over a four-row complete graph, including self edges."""
    def __init__(self, width=16):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(width))
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, hidden):
        n = hidden.shape[-2]
        adjacency = torch.ones(n, n, dtype=hidden.dtype, device=hidden.device) / n
        return torch.tanh(adjacency @ hidden @ self.weight + self.bias)


def make_examples(split_seed=159):
    """Split entire synthetic tables; never connect separate train/val/test tables.

    Each level has an independent binary value. Schema tokens repeat across rows;
    a cell's identity is per row even when its value happens to equal another cell.
    Noisy context cues agree with each target 70% of the time. Other observed
    cells share the table's cell class by construction: an intentionally strong
    contextual signal, not a realistic database or a transfer benchmark.
    """
    generator = torch.Generator().manual_seed(split_seed)
    examples = []
    order = torch.randperm(120, generator=generator).tolist()
    assignment = {table: ('train' if i < 84 else 'val' if i < 108 else 'test')
                  for i, table in enumerate(order)}
    for table in range(120):
        labels = torch.randint(0, 2, (3,), generator=generator)
        cues = labels.repeat(4, 1) ^ (torch.rand(4, 3, generator=generator) > .7).long()
        clean = torch.cat((labels.repeat(4, 1) + torch.tensor([2, 4, 6]),
                           cues + torch.tensor([8, 10, 12])), dim=1)
        identities = torch.tensor([[10, 20, 30+r, 100+3*r, 101+3*r, 102+3*r] for r in range(4)])
        for task, target in enumerate([10, 20, 30]):
            tokens = mask_serializations(clean, identities, target)
            selected = torch.zeros(3, dtype=torch.bool); selected[task] = True
            examples.append(dict(key=f'table-{table:03d}/task-{task}', table=table,
                                 split=assignment[table], task=task, tokens=tokens,
                                 labels=labels.clone(), selected=selected))
    return examples


def batch(examples, split):
    rows = [row for row in examples if row['split'] == split]
    return (torch.stack([r['tokens'] for r in rows]),
            torch.stack([r['labels'] for r in rows]),
            torch.stack([r['selected'] for r in rows]), rows)


def accuracy(logits, labels, selected):
    return float((logits.argmax(-1)[selected] == labels[selected]).float().mean())


def fingerprint(state):
    payload = {key: value.detach().cpu().tolist() for key, value in state.items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run_experiment(seeds=(0, 1, 2), epochs=80, max_seconds=120):
    """Complete the declared local lane or raise; never silently reduce its scope."""
    torch.set_num_threads(1)
    start = time.monotonic()
    examples = make_examples()
    train_x, train_y, train_m, _ = batch(examples, 'train')
    val_x, val_y, val_m, _ = batch(examples, 'val')
    test_x, test_y, test_m, test_rows = batch(examples, 'test')
    runs = []
    for seed in seeds:
        torch.manual_seed(seed)
        codec = RowCodec()
        graph = TableGraph()
        histories = {}
        choices = {}
        frozen_before = None
        for stage in ['row', 'graph']:
            model = codec if stage == 'row' else graph
            if stage == 'graph':
                freeze_codec(codec)
                frozen_before = fingerprint(codec.state_dict())
            optimizer = torch.optim.Adam(model.parameters(), lr=.025)
            best_accuracy = -1.0
            best_state = None
            history = []
            for epoch in range(epochs):
                if time.monotonic() - start > max_seconds:
                    raise TimeoutError('INCOMPLETE: local time cap reached; no scope reduction')
                model.train()
                optimizer.zero_grad(set_to_none=True)
                hidden = codec.encode(train_x)
                if stage == 'graph': hidden = graph(hidden)
                logits = codec.decode(hidden[:, 0])
                loss = masked_cross_entropy(logits, train_y, train_m)
                if not bool(torch.isfinite(loss)): raise ValueError('Nonfinite training loss')
                loss.backward()
                if stage == 'graph' and epoch == 0:
                    if graph.weight.grad is None or graph.weight.grad.abs().sum() == 0:
                        raise AssertionError('Frozen decoder did not transmit gradient')
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    hidden = codec.encode(val_x)
                    if stage == 'graph': hidden = graph(hidden)
                    score = accuracy(codec.decode(hidden[:, 0]), val_y, val_m)
                history.append(dict(epoch=epoch, train_loss=float(loss.detach()), val_accuracy=score))
                if score > best_accuracy:  # ties retain the earliest checkpoint
                    best_accuracy = score
                    best_state = copy.deepcopy(model.state_dict())
                    choices[stage] = epoch
            model.load_state_dict(best_state)
            histories[stage] = history
        frozen_after = fingerprint(codec.state_dict())
        if frozen_before != frozen_after: raise AssertionError('Frozen codec changed')
        predictions = []
        # Neither model accesses test until BOTH validation-selected stages are complete.
        with torch.no_grad():
            h = codec.encode(test_x)
            for arm, hidden in [('row', h), ('graph', graph(h))]:
                logits = codec.decode(hidden[:, 0])
                for row, scores in zip(test_rows, logits):
                    task = row['task']
                    predictions.append(dict(key=row['key'], table=row['table'], task=task, arm=arm,
                                            truth=int(row['labels'][task]),
                                            logits=scores[task].tolist(), prediction=int(scores[task].argmax())))
        runs.append(dict(seed=seed, selected_epoch=choices, histories=histories,
                         frozen_before=frozen_before, frozen_after=frozen_after,
                         codec={k:v.tolist() for k,v in codec.state_dict().items()},
                         graph={k:v.tolist() for k,v in graph.state_dict().items()}, predictions=predictions))
    split_tables = {split: sorted({r['table'] for r in examples if r['split']==split})
                    for split in ['train','val','test']}
    return dict(name='L159 synthetic masked-objective mechanism', status='COMPLETE',
                paper_reproduction='NOT_RUN', paper_fidelity='NOT_ESTABLISHED',
                cross_database_transfer='NOT_TESTED', cloud_spend_usd=0,
                config=dict(seeds=list(seeds),split_seed=159,tables=120,rows_per_table=4,
                            tasks=3,epochs=epochs,width=16,optimizer='Adam',lr=.025,
                            selection='maximum validation accuracy; earliest tie',
                            cpu_threads=1,max_seconds=max_seconds),
                splits=split_tables,runs=runs)


def summarize(report):
    lines = ['Synthetic mechanism only; percentages below are not paper scores.',
             '| Task | Row mean ± seed SD | Graph mean ± seed SD |',
             '|---|---:|---:|']
    for task,name in enumerate(['Table name','Column name','Cell value']):
        values = {}
        for arm in ['row','graph']:
            scores=[]
            for run in report['runs']:
                rows=[p for p in run['predictions'] if p['task']==task and p['arm']==arm]
                scores.append(100*sum(p['truth']==p['prediction'] for p in rows)/len(rows))
            mean=sum(scores)/len(scores)
            sd=math.sqrt(sum((x-mean)**2 for x in scores)/(len(scores)-1)) if len(scores)>1 else 0.
            values[arm]=f'{mean:.2f} ± {sd:.2f}%'
        lines.append(f"| {name} | {values['row']} | {values['graph']} |")
    return '\n'.join(lines)
