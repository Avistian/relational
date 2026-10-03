from relkit.baselines_b11 import explicit_attention
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.nn.dense.linear import Linear

class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average (EMA) for the codebook.
    Adapted from https://github.com/devnkong/GOAT

    Args:
        num_embeddings (int): The number of embeddings in the codebook.
        embedding_dim (int): The dimensionality of each embedding.
        decay (float, optional): The decay rate for the EMA. Defaults to 0.99.

    Attributes:
        _embedding_dim (int): The dimensionality of each embedding.
        _num_embeddings (int): The number of embeddings in the codebook.
        _decay (float): The decay rate for the EMA.
        _embedding (nn.Embedding): The embedding matrix.
        _ema_cluster_size (torch.Tensor): The exponential moving average of the cluster sizes.
        _ema_w (torch.Tensor): The exponential moving average of the embedding updates.
    """

    def __init__(self, num_embeddings, embedding_dim, decay=0.99):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.register_buffer(
            "_embedding", torch.randn(self._num_embeddings, self._embedding_dim)
        )
        self.register_buffer(
            "_embedding_output",
            torch.randn(self._num_embeddings, self._embedding_dim),
        )
        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer(
            "_ema_w", torch.randn(self._num_embeddings, self._embedding_dim)
        )

        self._decay = decay
        self.bn = torch.nn.BatchNorm1d(self._embedding_dim, affine=False)

    def reset_parameters(self):
        nn.init.normal_(self._embedding, mean=0.0, std=1.0)
        nn.init.normal_(self._embedding_output, mean=0.0, std=1.0)
        nn.init.zeros_(self._ema_cluster_size)
        nn.init.normal_(self._ema_w, mean=0.0, std=1.0)

        self.bn.reset_parameters()

    def get_k(self):
        """
        Returns the key tensor of the embedding matrix.
        """
        return self._embedding_output

    def get_v(self):
        """
        Returns the value tensor of the embedding matrix.
        """
        return self._embedding_output[:, : self._embedding_dim]

    def update(self, x):
        inputs_normalized = self.bn(x)
        embedding_normalized = self._embedding

        # Calculate distances
        distances = (
            torch.sum(inputs_normalized**2, dim=1, keepdim=True)
            + torch.sum(embedding_normalized**2, dim=1)
            - 2 * torch.matmul(inputs_normalized, embedding_normalized.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=x.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size.data = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size.data = (
                (self._ema_cluster_size + 1e-5) / (n + self._num_embeddings * 1e-5) * n
            )

            dw = torch.matmul(encodings.t(), inputs_normalized)
            self._ema_w.data = self._ema_w * self._decay + (1 - self._decay) * dw
            self._embedding.data = self._ema_w / self._ema_cluster_size.unsqueeze(1)

            running_std = torch.sqrt(self.bn.running_var + 1e-5).unsqueeze(dim=0)
            running_mean = self.bn.running_mean.unsqueeze(dim=0)
            self._embedding_output.data = self._embedding * running_std + running_mean

        return encoding_indices
class LocalModule(nn.Module):
    def __init__(
        self,
        seq_len,
        input_dim,
        node_only_readout=False,
        n_layers=1,
        num_heads=8,
        hidden_dim=64,
        dropout_rate=0.3,
        attention_dropout_rate=0,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.node_only_readout = node_only_readout
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = 2 * hidden_dim
        self.num_heads = num_heads

        self.n_layers = n_layers

        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        self.att_embeddings_nope = nn.Linear(self.input_dim, self.hidden_dim)

        encoders = [
            EncoderLayer(
                self.hidden_dim,
                self.ffn_dim,
                self.dropout_rate,
                self.attention_dropout_rate,
                self.num_heads,
            )
            for _ in range(self.n_layers)
        ]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

        self.attn_layer = nn.Linear(2 * hidden_dim, 1)
        
    def reset_parameters(self):
        self.att_embeddings_nope.reset_parameters()
        self.attn_layer.reset_parameters()
        self.final_ln.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, batched_data, pretrain_token=False):
        tensor = self.att_embeddings_nope(batched_data)

        # transformer encoder
        for enc_layer in self.layers:
            tensor = enc_layer(tensor)

        output = self.final_ln(tensor)
        
        if pretrain_token:
            return output

        _target = output[:, 0, :].unsqueeze(1).repeat(1, self.seq_len - 1, 1)
        split_tensor = torch.split(output, [1, self.seq_len - 1], dim=1)

        node_tensor = split_tensor[0]
        _neighbor_tensor = split_tensor[1]

        if self.node_only_readout:
            indices = torch.arange(1, self.seq_len, 1)
            neighbor_tensor = _neighbor_tensor[:, indices]
            target = _target[:, indices]
        else:
            target = _target
            neighbor_tensor = _neighbor_tensor

        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))
        layer_atten = F.softmax(layer_atten, dim=1)

        neighbor_tensor = neighbor_tensor * layer_atten
        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)

        output = (node_tensor + neighbor_tensor).squeeze()

        return output

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.bn_in = nn.BatchNorm1d(hidden_size)
        self.bn_out = nn.BatchNorm1d(hidden_size)

        self.ffn_net = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_size, hidden_size),
            nn.Dropout(dropout_rate)
        )
        
    def reset_parameters(self):
        for layer in self.ffn_net:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.bn_in(x)
        x = x.permute(0, 2, 1)
        
        x = self.ffn_net(x)
        
        x = x.permute(0, 2, 1)
        x = self.bn_out(x)
        x = x.permute(0, 2, 1)
        
        return x

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout_rate = attention_dropout_rate

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)

    def reset_parameters(self):
        self.self_attention_norm.reset_parameters()
        
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        self.ffn_norm.reset_parameters()
        self.ffn.reset_parameters()

    def forward(self, x, attn_bias=None):
        # self-attention block with flash attention 
        residual = x
        x_norm = self.self_attention_norm(x)  # [B, L, D]
        
        Q = self.q_proj(x_norm)  # [B, L, D]
        K = self.k_proj(x_norm)
        V = self.v_proj(x_norm)
        B, L, D = Q.shape
        head_dim = D // self.num_heads
        
        # reshape Q, K, V to shape [B, num_heads, L, head_dim].
        Q = Q.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        
        # PyTorch’s fast scaled dot-product attention (flash attention).
        attn_output = explicit_attention(
            Q, K, V,
            attn_mask=attn_bias,
            dropout_p=self.attention_dropout_rate,
            is_causal=False  
        )  # Returns [B, num_heads, L, head_dim]
        
        # reshape back to [B, L, D].
        attn_output = attn_output.transpose(1, 2).reshape(B, L, D)
        
        attn_output = self.out_proj(attn_output)
        attn_output = self.self_attention_dropout(attn_output)
        
        x = residual + attn_output
        
        # Feed-forward block 
        residual = x
        x_norm = self.ffn_norm(x)
        ffn_output = self.ffn(x_norm)
        x = residual + ffn_output
        return x
class RelGTLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        local_num_layers,
        global_dim,
        num_nodes,
        heads=1,
        concat=True,
        ff_dropout=0.0,
        attn_dropout=0.0,
        edge_dim=None,
        conv_type="local",
        num_centroids=None,
        sample_node_len=100,
        **kwargs,
    ):
        super(RelGTLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.local_num_layers = local_num_layers
        self.heads = heads
        self.concat = concat
        self.ff_dropout = ff_dropout
        self.attn_dropout = attn_dropout
        self.edge_dim = edge_dim
        self.conv_type = conv_type
        self.num_centroids = num_centroids
        self._alpha = None
        self.sample_node_len = sample_node_len

        self.local_module = LocalModule(
            seq_len=self.sample_node_len,
            input_dim=in_channels,
            n_layers=local_num_layers,
            num_heads=heads,
            hidden_dim=out_channels,
            dropout_rate=ff_dropout,
            attention_dropout_rate=attn_dropout
        )
        self.layer_norm_local = nn.LayerNorm(out_channels)

        if self.conv_type != "local":
            self.vq = VectorQuantizerEMA(num_centroids, global_dim, decay=0.99)
            c = torch.randint(0, num_centroids, (num_nodes,), dtype=torch.long)
            self.register_buffer("c_idx", c)
            self.attn_fn = F.softmax

            attn_channels = out_channels // heads

            self.lin_proj_g = Linear(in_channels, global_dim)
            self.lin_key_g = Linear(global_dim, heads * attn_channels)
            self.lin_query_g = Linear(global_dim, heads * attn_channels)
            self.lin_value_g = Linear(global_dim, heads * attn_channels)
            self.layer_norm_global = nn.LayerNorm(out_channels)

        self.reset_parameters()
        
    def reset_parameters(self):
        # Reinitialize global attention layers
        if self.conv_type != "local":
            self.lin_proj_g.reset_parameters()
            self.lin_key_g.reset_parameters()
            self.lin_query_g.reset_parameters()
            self.lin_value_g.reset_parameters()
            if hasattr(self, 'vq'):
                self.vq.reset_parameters()
                
        # Reinitialize LocalModule
        if hasattr(self.local_module, 'reset_parameters'):
            self.local_module.reset_parameters()
            
    def forward(self, x_set, x, node_indices):
        if self.conv_type == "local":
            out = self.local_forward(x_set)
            out = self.layer_norm_local(out)

        elif self.conv_type == "global":
            out = self.global_forward(x, node_indices)
            out = self.layer_norm_global(out)

        elif self.conv_type == "full":
            out_local = self.local_forward(x_set)
            out_global = self.global_forward(x, node_indices)
            out_local = self.layer_norm_local(out_local)
            out_global = self.layer_norm_global(out_global)
            out = torch.cat([out_local, out_global], dim=1)

        else:
            raise NotImplementedError

        return out

    def global_forward(self, x, batch_idx):
        d, h = self.out_channels, self.heads
        scale = 1.0 / math.sqrt(d)

        q_x = self.lin_proj_g(x)

        k_buf = self.vq.get_k()
        k_x = k_buf.detach().clone()
        v_buf = self.vq.get_v()
        v_x = v_buf.detach().clone()


        q = self.lin_query_g(q_x)
        k = self.lin_key_g(k_x)
        v = self.lin_value_g(v_x)

        q, k, v = map(lambda t: rearrange(t, "n (h d) -> h n d", h=h), (q, k, v))
        dots = torch.einsum("h i d, h j d -> h i j", q, k) * scale

        c, c_count = self.c_idx.unique(return_counts=True)

        centroid_count = torch.zeros(self.num_centroids, dtype=torch.long).to(x.device)
        centroid_count[c.to(torch.long)] = c_count

        dots = dots + torch.log(centroid_count.view(1, 1, -1))

        attn = self.attn_fn(dots, dim=-1)
        attn = F.dropout(attn, p=self.attn_dropout, training=self.training)

        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h n d -> n (h d)")

        # Update the centroids
        if self.training:
            x_idx = self.vq.update(q_x)
            self.c_idx[batch_idx] = x_idx.squeeze().to(torch.long)

        return out

    def local_forward(self, x_set, pretrain_token=False):
        return self.local_module(x_set, pretrain_token)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads}, "
            f"local_num_layers={self.local_num_layers})"
        )
