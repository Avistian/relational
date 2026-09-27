"""Visible OGB setup and train-only constant baseline; no GCN training."""
import hashlib
import json
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
from ogb.nodeproppred import NodePropPredDataset, Evaluator


def audit_splits(splits, num_nodes):
    """Require a complete, nonoverlapping partition of integer node identities."""
    if set(splits) != {'train', 'valid', 'test'}:
        raise ValueError('Expected the three official split names')
    parts = []
    counts = {}
    for name, indices in splits.items():
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.dtype.kind not in 'iu':
            raise ValueError('Split IDs must be a one-dimensional integer array')
        if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= num_nodes):
            raise ValueError('Empty split or node ID outside the graph')
        counts[name] = len(indices)
        parts.append(indices)
    combined = np.concatenate(parts)
    if len(combined) != num_nodes or len(np.unique(combined)) != num_nodes:
        raise ValueError('Missing, repeated or overlapping node IDs')
    return counts


def majority_predictions(labels, train_idx):
    """Fit a constant from training labels only; smallest class wins a tie."""
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.shape[1] != 1:
        raise ValueError('Expected [N,1] labels')
    values, counts = np.unique(labels[train_idx, 0], return_counts=True)
    if not len(values):
        raise ValueError('Training labels are empty')
    chosen = values[np.argmax(counts)]
    return np.full(labels.shape, chosen, dtype=labels.dtype)


def load_official(archive):
    """Authenticate raw bytes; avoid old pickle caches with newer PyTorch."""
    archive = Path(archive)
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve('https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip', archive)
    expected = '49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError('Archive differs from the frozen OGB input')
    with tempfile.TemporaryDirectory(prefix='l111-official-') as tmp:
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp)
        folder = Path(tmp)/'ogbn_arxiv'
        (Path(tmp)/'arxiv').rename(folder)
        (folder/'RELEASE_v1.txt').touch()
        dataset = NodePropPredDataset(name='ogbn-arxiv', root=tmp)
        return dataset[0], dataset.get_idx_split()


def run_contract(archive):
    (graph, labels), splits = load_official(archive)
    counts = audit_splits(splits, graph['num_nodes'])
    predictions = majority_predictions(labels, splits['train'])
    evaluator = Evaluator(name='ogbn-arxiv')
    scores = {}
    for name, idx in splits.items():
        official = evaluator.eval({'y_true': labels[idx], 'y_pred': predictions[idx]})['acc']
        independent = float(np.mean(labels[idx] == predictions[idx]))
        assert official == independent
        scores[name] = official
    years = graph['node_year'].reshape(-1)
    assert years[splits['train']].max() <= 2017
    assert np.all(years[splits['valid']] == 2018)
    assert years[splits['test']].min() >= 2019
    return dict(status='PASS', dataset='ogbn-arxiv', nodes=graph['num_nodes'],
                feature_shape=list(graph['node_feat'].shape), split_counts=counts,
                split_sha256={k:hashlib.sha256(np.asarray(v,dtype='<i8').tobytes()).hexdigest() for k,v in splits.items()},
                label_sha256=hashlib.sha256(np.asarray(labels,dtype='<i8').tobytes()).hexdigest(),
                baseline='Training-majority constant; no tuning or neural training',
                chosen_class=int(predictions[0,0]), accuracy=scores,
                scope='Official-loader/evaluator course baseline, not a published GCN reproduction')

if __name__ == '__main__':
    report = run_contract(os.environ.get('L111_ARCHIVE', 'l111-data/arxiv.zip'))
    Path('l111-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
