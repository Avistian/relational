"""
Parameter-count comparison: HeteroSAGE vs HeteroGIN vs HeteroGAT.

All three share the same NodeEncoder, MLPHead, hetero wrapper, 'cat'
across-relation aggregation and config values -- they differ only in the
per-relation message-passing operator (SAGEConv / GINConv / GATConv) and, for
GAT, the extra ('row','self','row') relation that attention needs.

Run:  python examples/compare_model_params.py
"""

import torch
from torch_geometric.data import HeteroData

from autograble.models import (
    HeteroSAGE, SAGEConfig,
    HeteroGIN, GINConfig,
    HeteroGAT, GATConfig, ensure_row_self_loops,
)

# --- a small synthetic graph with the same shape build_hetero_graph produces
K_COLS = 5          # selected columns  -> K value node types, 2K bipartite edges
ROW_FEAT_DIM = 20   # data["row"].x width
N_ROWS = 200
N_VALS = 40

data = HeteroData()
data["row"].x = torch.randn(N_ROWS, ROW_FEAT_DIM)
data["row"].y = torch.randint(0, 2, (N_ROWS,))
for i in range(K_COLS):
    c = f"col{i}"
    data[c].x = torch.zeros(N_VALS, 1)
    ei = torch.stack([torch.randint(0, N_ROWS, (N_ROWS,)),
                      torch.randint(0, N_VALS, (N_ROWS,))])
    data["row", "has", c].edge_index = ei
    data[c, "rev_has", "row"].edge_index = ei.flip(0)

num_value_nodes = {t: data[t].num_nodes for t in data.node_types if t != "row"}


def n_params(m):
    return sum(p.numel() for p in m.parameters())


sage = HeteroSAGE(data.metadata(), ROW_FEAT_DIM, num_value_nodes, SAGEConfig(), out_dim=1)
gin = HeteroGIN(data.metadata(), ROW_FEAT_DIM, num_value_nodes, GINConfig(), out_dim=1)

data_self = ensure_row_self_loops(data.clone())  # adds ('row','self','row')
gat = HeteroGAT(data_self.metadata(), ROW_FEAT_DIM, num_value_nodes, GATConfig(), out_dim=1)

print(f"{'model':<14}{'total params':>14}{'vs SAGE':>12}")
base = n_params(sage)
for name, m in [("HeteroSAGE", sage), ("HeteroGIN", gin), ("HeteroGAT", gat)]:
    p = n_params(m)
    print(f"{name:<14}{p:>14,}{p - base:>+12,}")

print(f"\nGATConfig.attn_heads = {GATConfig().attn_heads}  "
      f"(per-head dim = {GATConfig().hidden_dim // GATConfig().attn_heads}, "
      f"total output width = {GATConfig().hidden_dim} == SAGE hidden_dim)")
