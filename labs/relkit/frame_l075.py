"""L075: visible API composition and independently reconstructed numeric tokens.

Tool/API lesson, pinned to pytorch-frame 0.3.0. The five-stype fixture is
synthetic mechanism evidence, not a predictive benchmark or pretrained model.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_frame import stype, NAStrategy
from torch_frame.config import TextEmbedderConfig
from torch_frame.data import Dataset
from torch_frame.nn import (StypeWiseFeatureEncoder, LinearEncoder,
                            EmbeddingEncoder, LinearEmbeddingEncoder, TimestampEncoder)


def fixture():
    train = pd.DataFrame({
        'amount': [10., 20., 30., np.nan],
        'region': ['north', 'north', 'south', 'south'],
        'note': ['red apple', 'green apple', 'red pear', 'green pear'],
        'created': pd.to_datetime(['2024-01-01','2024-01-02','2024-02-01','2024-02-02']),
        'vector': [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,0.]],
    })
    query = pd.DataFrame({
        'amount': [1000., np.nan], 'region': ['east', None],
        'note': ['red apple', 'blue fruit'],
        'created': pd.to_datetime(['2025-01-01', None]),
        'vector': [[1.,0.,1.],[0.,1.,1.]],
    })
    return train, query


def schema():
    return {'amount': stype.numerical, 'region': stype.categorical,
            'note': stype.text_embedded, 'created': stype.timestamp,
            'vector': stype.embedding}


def text_features(texts):
    """Fixed, visible bag-of-words fixture; NOT an LLM or semantic text model.

    'blue fruit' is deliberately out of vocabulary and maps to zero. This
    isolates the text embedding adapter without any network/model download.
    """
    vocabulary = ['red', 'green', 'apple', 'pear']
    return torch.tensor([[s.lower().split().count(w) for w in vocabulary]
                         for s in texts], dtype=torch.float32)


def fit_materializer(train, query, col_to_stype):
    """Fit data-dependent state only on train; apply that converter to query."""
    ds = Dataset(train, col_to_stype=col_to_stype,
                 col_to_text_embedder_cfg=TextEmbedderConfig(text_features))
    ds.materialize()
    held = ds.convert_to_tensor_frame(query)
    return ds, held


def encoder_config():
    """Parent semantic types are encoder keys; text_embedded has parent embedding."""
    return {stype.numerical: LinearEncoder(na_strategy=NAStrategy.MEAN),
            stype.categorical: EmbeddingEncoder(),
            stype.timestamp: TimestampEncoder(),
            stype.embedding: LinearEmbeddingEncoder()}


def numeric_tokens(raw, mean, scale, weight, bias):
    """Reconstruct mean-imputed LinearEncoder; scale includes epsilon.

    raw [B,C], mean/scale [C], weight/bias [C,d] -> tokens [B,C,d].
    Replace missing values with the training mean before differentiable arithmetic.
    """
    filled = torch.where(torch.isnan(raw), mean.unsqueeze(0), raw)
    normalized = (filled - mean) / scale
    tokens = normalized.unsqueeze(-1) * weight.unsqueeze(0) + bias.unsqueeze(0)
    return torch.nan_to_num(tokens, nan=0.)


class RowEncoder(nn.Module):
    """Five typed inputs -> column tokens -> flattened MLP -> row vector.

    This is a small teaching composition, not the paper's FT-Transformer.
    Flattening preserves column positions; the hidden layer mixes columns.
    """
    def __init__(self, dataset, width=8, row_width=6):
        super().__init__()
        self.columns = StypeWiseFeatureEncoder(
            out_channels=width, col_stats=dataset.col_stats,
            col_names_dict=dataset.tensor_frame.col_names_dict,
            stype_encoder_dict=encoder_config())
        self.readout = nn.Sequential(nn.Flatten(start_dim=1),
                                     nn.Linear(dataset.tensor_frame.num_cols * width, 16),
                                     nn.ReLU(), nn.Linear(16, row_width))

    def forward(self, tf):
        tokens, names = self.columns(tf)
        return self.readout(tokens)


def real_table_demo(export=False):
    """Tier A: actual credit_g rows -> typed numeric/categorical row vectors.

    Training-only split, no labels fitted, no performance comparison. The data
    cache is from OpenML 31, shared with earlier labs. Returns portable outputs.
    """
    from relkit.data import load_tier_a
    x, _ = load_tier_a('credit_g')
    order = np.random.default_rng(75).permutation(len(x))
    tr, te = order[:128], order[128:160]
    types = {c: stype.numerical if pd.api.types.is_numeric_dtype(x[c])
             else stype.categorical for c in x.columns}
    ds, held = fit_materializer(x.iloc[tr].copy(), x.iloc[te].copy(), types)
    torch.manual_seed(75)
    model = RowEncoder(ds, width=8, row_width=16).eval()
    with torch.no_grad():
        tokens, names = model.columns(held)
        rows = model(held)
    assert torch.isfinite(rows).all()
    if export:
        torch.save({'row_vectors':rows, 'state_dict':model.state_dict(),
                    'col_stats':ds.col_stats, 'col_names_dict':ds.tensor_frame.col_names_dict,
                    'query_ids':te.tolist()}, 'l075-row-encoder.pt')
        Path('l075-schema.json').write_text(json.dumps({
            'schema':{c:s.value for c,s in types.items()}, 'column_order':names,
            'train_ids':tr.tolist(), 'query_ids':te.tolist(), 'pytorch_frame':'0.3.0',
            'width':8, 'row_width':16, 'weights':'random initialization',
            'numeric_missing':'training mean', 'source':'OpenML 31'}, indent=2))
    return {'dataset':'OpenML 31 credit_g','train_ids':tr.tolist(),'query_ids':te.tolist(),
            'schema':{c:s.value for c,s in types.items()},'column_order':names,
            'token_shape':list(tokens.shape),'row_shape':list(rows.shape),
            'row_vectors':rows.tolist(),'weights':'random initialization, not trained',
            'accuracy':'NOT_MEASURED'}
