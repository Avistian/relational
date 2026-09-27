"""Visible relational entity graph construction; learner tasks use raw keys."""
# %% PROVIDED: dependencies
import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

# %% Task 1: preserve primary-key identity

def key_index(primary_keys):
    lookup = {}
    for position, key in enumerate(primary_keys):
        if pd.isna(key) or key in lookup:
            raise ValueError("Primary keys must be non-null and unique")
        lookup[key] = position
    return lookup

# %% Task 2: resolve a foreign-key column

def relation_edges(primary_keys, foreign_keys):
    lookup = key_index(primary_keys)
    pairs = []
    for source, key in enumerate(foreign_keys):
        if pd.isna(key):
            continue
        if key not in lookup:
            raise ValueError(f"Dangling foreign key: {key!r}")
        pairs.append((source, lookup[key]))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2).T

# %% Task 3: construct typed stores, including isolated rows

def construct_reg(tables, schema):
    if set(tables) != set(schema):
        raise ValueError("Schema must describe every table")
    data = HeteroData()
    features = {}
    for name, frame in tables.items():
        pk = schema[name]['pk']
        if pk is not None:
            key_index(frame[pk].tolist())
        data[name].num_nodes = len(frame)
        excluded = set(schema[name]['fks']) | {pk}
        features[name] = [col for col in frame.columns if col not in excluded]
    for name, frame in tables.items():
        for fk, destination in schema[name]['fks'].items():
            parent_pk = schema[destination]['pk']
            if parent_pk is None:
                raise ValueError("Referenced table needs a primary key")
            edges = relation_edges(tables[destination][parent_pk].tolist(), frame[fk].tolist())
            kind = (name, 'f2p_' + fk, destination)
            data[kind].edge_index = torch.from_numpy(edges)
            reverse = (destination, 'rev_f2p_' + fk, name)
            data[reverse].edge_index = data[kind].edge_index.flip(0).contiguous()
    data.validate(raise_on_error=True)
    return data, features

# %% PROVIDED: a concrete row-to-graph trace

def course_run():
    tables = {
        'person': pd.DataFrame({'id':[90,10,300], 'age':[40.,20.,60.]}),
        'transfer': pd.DataFrame({'id':[7,8,9], 'sender':[10,90,None],
                                  'receiver':[90,10,90], 'amount':[5.,8.,2.]})}
    schema = {'person': {'pk':'id','fks':{}},
              'transfer': {'pk':'id','fks':{'sender':'person','receiver':'person'}}}
    graph, features = construct_reg(tables,schema)
    # A deliberately untrained one-hop sum, solely to reveal the edge direction.
    amounts = torch.tensor(tables['transfer']['amount'].values, dtype=torch.float32)
    received = torch.zeros(graph['person'].num_nodes)
    edge = graph['transfer','f2p_receiver','person'].edge_index
    received.index_add_(0,edge[1],amounts[edge[0]])
    torch.testing.assert_close(received,torch.tensor([7.,8.,0.]))
    return {'status':'PASS','nodes':{k:graph[k].num_nodes for k in graph.node_types},
            'edges':{'|'.join(k):graph[k].edge_index.tolist() for k in graph.edge_types},
            'feature_columns':features,'received_amount_by_person_row':received.tolist(),
            'evidence':'Course construction and untrained sum; not a benchmark score'}
