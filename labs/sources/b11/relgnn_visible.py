import math
import torch
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import TransformerConv, SAGEConv
def destination_softmax(scores, destination, num_destinations):
    """Normalize each head over incoming messages for the same destination."""
    index = destination[:, None].expand_as(scores)
    maxima = scores.new_full((num_destinations, scores.shape[1]), -torch.inf)
    maxima.scatter_reduce_(0, index, scores.detach(), reduce='amax', include_self=True)
    weights = (scores - maxima[destination]).exp()
    totals = scores.new_zeros((num_destinations, scores.shape[1]))
    totals.index_add_(0, destination, weights)
    return weights / (totals[destination] + 1e-16)

class RelGNNConv(TransformerConv):
    def __init__(self, attn_type, in_channels, out_channels, heads, aggr,
                 simplified_MP=False, bias=True, **kwargs):
        super().__init__(in_channels, out_channels, heads, bias=bias, **kwargs)
        if aggr != 'sum' or simplified_MP:
            raise ValueError('This audited lesson implements the selected full sum variant')
        self.attn_type = attn_type
        if attn_type == 'dim-fact-dim':
            self.aggr_conv = SAGEConv(in_channels, out_channels, aggr=aggr)
        self.simplified_MP = simplified_MP
        self.final_proj = Linear(heads * out_channels, out_channels, bias=bias)
        self.final_proj.reset_parameters()

    def attend(self, source, destination, edges):
        # [nodes, heads, channels/head-output]; here each head outputs 128 channels.
        h, c = self.heads, self.out_channels
        q = self.lin_query(destination).view(-1, h, c)
        k = self.lin_key(source).view(-1, h, c)
        v = self.lin_value(source).view(-1, h, c)
        src, dst = edges
        scores = (q[dst] * k[src]).sum(-1) / math.sqrt(c)
        alpha = destination_softmax(scores, dst, len(destination))
        messages = v[src] * alpha[:, :, None]
        aggregate = messages.new_zeros((len(destination), h, c))
        aggregate.index_add_(0, dst, messages)
        # Source defaults: concatenate heads, add a destination skip, then project.
        joined = aggregate.reshape(len(destination), h*c) + self.lin_skip(destination)
        return self.final_proj(joined)

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None):
        if edge_attr is not None or return_attention_weights is not None:
            raise ValueError('Selected experiment does not use edge attributes or attention return flags')
        if self.attn_type == 'dim-dim':
            return self.attend(x[0], x[1], edge_index)
        edge_attn, edge_aggr = edge_index
        source, fact, destination = x
        src, dst = edge_aggr
        gathered = source.new_zeros((len(fact), source.shape[1]))
        gathered.index_add_(0, dst, source[src])
        # SAGE sum + transformed fact root; unique to this ordered FK pair.
        intermediate = self.aggr_conv.lin_l(gathered) + self.aggr_conv.lin_r(fact)
        return self.attend(intermediate, destination, edge_attn), intermediate
