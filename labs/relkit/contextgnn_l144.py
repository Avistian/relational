# Adapted from kumo-ai/ContextGNN ca4a9698; MIT, see sources/l144/LICENSE.
from relkit.context_l144 import fuse_scores


# SOURCE: contextgnn/utils/__init__.py
import torch

from enum import Enum


class RHSEmbeddingMode(Enum):
    r"""Specifies how to incorporate shallow RHS representations in link
    prediction tasks.
    """
    # Use trainable look-up embeddings (transductive):
    LOOKUP = 'lookup'
    # Purely rely on shallow RHS input features (inductive):
    FEATURE = 'feature'
    # Fuse look-up embeddings and shallow RHS input features (transductive):
    FUSION = 'fusion'






# SOURCE: contextgnn/nn/encoder.py
from typing import Any, Dict, List, Optional

import torch
import torch_frame
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_frame.nn.models import ResNet
from torch_geometric.nn import PositionalEncoding
from torch_geometric.typing import NodeType

DEFAULT_STYPE_ENCODER_DICT: Dict[torch_frame.stype, Any] = {
    torch_frame.categorical: (torch_frame.nn.EmbeddingEncoder, {}),
    torch_frame.numerical: (torch_frame.nn.LinearEncoder, {}),
    torch_frame.multicategorical: (
        torch_frame.nn.MultiCategoricalEmbeddingEncoder,
        {},
    ),
    torch_frame.embedding: (torch_frame.nn.LinearEmbeddingEncoder, {}),
    torch_frame.timestamp: (torch_frame.nn.TimestampEncoder, {}),
}
SECONDS_IN_A_DAY = 60 * 60 * 24


class HeteroEncoder(torch.nn.Module):
    r"""HeteroStypeWiseEncoder is a simple encoder to encode multi-modal
    data from different node types.

    Args:
        channels (int): The output channels for each node type.
        node_to_col_names_dict (Dict[NodeType, Dict[torch_frame.stype, List[str]]]):  # noqa: E501
            A dictionary mapping from node type to column names dictionary
            compatible to PyTorch Frame.
        node_to_col_stats (Dict[NodeType, Dict[str, Dict[StatType, Any]]]):
            A dictionary mapping from node type to column statistics dictionary
            compatible to PyTorch Frame.
        stype_encoder_cls_kwargs (Dict[torch_frame.stype, Any]):
            A dictionary mapping from :obj:`torch_frame.stype` object into a
            tuple specifying :class:`torch_frame.nn.StypeEncoder` class and its
            keyword arguments :obj:`kwargs`.
        torch_frame_model_cls: Model class for PyTorch Frame. The class object
            takes :class:`TensorFrame` object as input and outputs
            :obj:`channels`-dimensional embeddings. Default to
            :class:`torch_frame.nn.ResNet`.
        torch_frame_model_kwargs (Dict[str, Any]): Keyword arguments for
            :class:`torch_frame_model_cls` class. Default keyword argument is
            set specific for :class:`torch_frame.nn.ResNet`. Expect it to
            be changed for different :class:`torch_frame_model_cls`.
    """
    def __init__(
        self,
        channels: int,
        node_to_col_names_dict: Dict[NodeType, Dict[torch_frame.stype,
                                                    List[str]]],
        node_to_col_stats: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
        stype_encoder_cls_kwargs: Dict[torch_frame.stype, Any],
        torch_frame_model_cls=ResNet,
        torch_frame_model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.encoders = torch.nn.ModuleDict()

        for node_type in node_to_col_names_dict.keys():
            stype_encoder_dict = {
                stype:
                stype_encoder_cls_kwargs[stype][0](
                    **stype_encoder_cls_kwargs[stype][1])
                for stype in node_to_col_names_dict[node_type].keys()
            }

            self.encoders[node_type] = torch_frame_model_cls(
                **torch_frame_model_kwargs,
                out_channels=channels,
                col_stats=node_to_col_stats[node_type],
                col_names_dict=node_to_col_names_dict[node_type],
                stype_encoder_dict=stype_encoder_dict,
            )

    def reset_parameters(self) -> None:
        for encoder in self.encoders.values():
            encoder.reset_parameters()

    def forward(
        self,
        tf_dict: Dict[NodeType, torch_frame.TensorFrame],
    ) -> Dict[NodeType, Tensor]:
        x_dict = {
            node_type: self.encoders[node_type](tf)
            for node_type, tf in tf_dict.items()
        }
        return x_dict


class HeteroTemporalEncoder(torch.nn.Module):
    def __init__(self, node_types: List[NodeType], channels: int) -> None:
        super().__init__()

        self.encoder_dict = torch.nn.ModuleDict({
            node_type:
            PositionalEncoding(channels)
            for node_type in node_types
        })
        self.lin_dict = torch.nn.ModuleDict({
            node_type:
            torch.nn.Linear(channels, channels)
            for node_type in node_types
        })

    def reset_parameters(self) -> None:
        for encoder in self.encoder_dict.values():
            encoder.reset_parameters()
        for lin in self.lin_dict.values():
            lin.reset_parameters()

    def forward(
        self,
        seed_time: Tensor,
        time_dict: Dict[NodeType, Tensor],
        batch_dict: Dict[NodeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        out_dict: Dict[NodeType, Tensor] = {}

        for node_type, time in time_dict.items():
            rel_time = seed_time[batch_dict[node_type]] - time
            rel_time = rel_time / SECONDS_IN_A_DAY

            x = self.encoder_dict[node_type](rel_time)
            x = self.lin_dict[node_type](x)
            out_dict[node_type] = x

        return out_dict


# SOURCE: contextgnn/nn/rhs_embedding.py
from typing import List, Optional

import torch
from torch import Tensor
from torch_frame import TensorFrame
from torch_frame.nn import StypeWiseFeatureEncoder
from torch_frame.nn.models.resnet import FCResidualBlock
from typing_extensions import Self



class RHSEmbedding(torch.nn.Module):
    r"""RHSEmbedding module for GNNs."""
    def __init__(
        self,
        emb_mode: RHSEmbeddingMode,
        embedding_dim: int,
        num_nodes: int,
        col_stats: dict,
        col_names_dict: dict,
        stype_encoder_dict: dict,
        feat: Optional[TensorFrame] = None,
    ):
        super().__init__()
        self.emb_mode = emb_mode
        # Encodes the column features of a table into a shared embedding space.
        self.encoder: Optional[StypeWiseFeatureEncoder] = None
        self.projector: Optional[torch.nn.Sequential] = None
        self._feat = feat
        if self.emb_mode in [
                RHSEmbeddingMode.FEATURE, RHSEmbeddingMode.FUSION
        ]:
            self.encoder = StypeWiseFeatureEncoder(
                out_channels=embedding_dim,
                col_stats=col_stats,
                col_names_dict=col_names_dict,
                stype_encoder_dict=stype_encoder_dict,
            )

            seqs: List[torch.nn.Module] = []
            if feat is None:
                raise ValueError(f"RHSEmbedding mode {self.emb_mode} "
                                 f"requires feat data.")
            seqs += [
                FCResidualBlock(embedding_dim, embedding_dim),
                FCResidualBlock(embedding_dim, embedding_dim),
            ]
            seqs += [torch.nn.LayerNorm(embedding_dim, eps=1e-7)]
            self.projector = torch.nn.Sequential(*seqs)

        self.lookup_embedding: Optional[torch.nn.Embedding] = None
        if self.emb_mode in [RHSEmbeddingMode.LOOKUP, RHSEmbeddingMode.FUSION]:
            self.lookup_embedding = torch.nn.Embedding(num_nodes,
                                                       embedding_dim)
        self._cached_rhs_embedding: Optional[Tensor] = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.lookup_embedding is not None:
            self.lookup_embedding.reset_parameters()
        if self.encoder is not None:
            self.encoder.reset_parameters()
        if self.projector is not None:
            for child in self.projector.children():
                child.reset_parameters()
        self._cached_rhs_embedding = None

    def forward(self, index: Optional[Tensor] = None) -> Tensor:
        if not self.training:
            if self._cached_rhs_embedding is not None:
                return self._cached_rhs_embedding
        outs = []
        if self.lookup_embedding is not None:
            if index is None:
                outs.append(self.lookup_embedding.weight)
            else:
                outs.append(self.lookup_embedding.weight[index, :])
        if self.encoder is not None and self.projector is not None:
            assert self._feat is not None

            if index is None:
                out = self.encoder(self._feat)[0]
            else:
                out = self.encoder(self._feat[index])[0]
            out = self.projector(out)
            # fuse
            out = torch.sum(out, dim=1)
            outs.append(out)
        result = sum(outs)
        assert isinstance(result, Tensor)
        if not self.training:
            self._cached_rhs_embedding = result
        return result

    def to(self, *args, **kwargs) -> Self:
        # Explicitly call `to` on the RHS embedding to move caches to the
        # device.
        if self._feat is not None:
            self._feat = self._feat.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def cpu(self) -> Self:
        if self._feat is not None:
            self._feat = self._feat.cpu()
        return super().cpu()

    def cuda(self, *args, **kwargs) -> Self:
        if self._feat is not None:
            self._feat = self._feat.cuda(*args, **kwargs)
        return super().cuda(*args, **kwargs)


# SOURCE: contextgnn/nn/models/graphsage.py
from typing import Dict, List

import torch
from torch import Tensor
from torch_geometric.nn import HeteroConv, LayerNorm, SAGEConv
from torch_geometric.typing import EdgeType, NodeType


class HeteroGraphSAGE(torch.nn.Module):
    r"""Implementation of GraphSAGE convolutoin."""
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        aggr: str = "mean",
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv(
                        (channels, channels), channels, aggr=aggr)
                    for edge_type in edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

        self.channels = channels

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[NodeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        for i, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict


# SOURCE: contextgnn/nn/models/rhsembeddinggnn.py
from typing import Any, Dict

import torch
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from typing_extensions import Self



class RHSEmbeddingGNN(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        rhs_emb_mode: RHSEmbeddingMode,
        dst_entity_table: str,
        num_nodes: int,
        embedding_dim: int,
    ):
        super().__init__()
        stype_encoder_dict = {
            k: v[0]()
            for k, v in DEFAULT_STYPE_ENCODER_DICT.items()
            if k in data[dst_entity_table]['tf'].col_names_dict.keys()
        }
        self.rhs_embedding = RHSEmbedding(
            emb_mode=rhs_emb_mode,
            embedding_dim=embedding_dim,
            num_nodes=num_nodes,
            col_stats=col_stats_dict[dst_entity_table],
            col_names_dict=data[dst_entity_table]['tf'].col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
            feat=data[dst_entity_table]['tf'],
        )

    def reset_parameters(self):
        self.rhs_embedding.reset_parameters()

    def to(self, *args, **kwargs) -> Self:
        # Explicitly call `to` on the RHS embedding to move caches to the
        # device.
        self.rhs_embedding.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def cpu(self) -> Self:
        self.rhs_embedding.cpu()
        return super().cpu()

    def cuda(self, *args, **kwargs) -> Self:
        self.rhs_embedding.cuda(*args, **kwargs)
        return super().cuda(*args, **kwargs)


# SOURCE: contextgnn/nn/models/shallowrhsgnn.py
from typing import Any, Dict, Optional, Type

import torch
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_frame.nn.models import ResNet
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
from typing_extensions import Self



class ShallowRHSGNN(RHSEmbeddingGNN):
    r"""Implementation of ShallowRHSGNN model."""
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        rhs_emb_mode: RHSEmbeddingMode,
        dst_entity_table: str,
        num_nodes: int,
        num_layers: int,
        channels: int,
        embedding_dim: int,
        aggr: str = 'sum',
        norm: str = 'layer_norm',
        torch_frame_model_cls: Type[torch.nn.Module] = ResNet,
        torch_frame_model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(data, col_stats_dict, rhs_emb_mode, dst_entity_table,
                         num_nodes, embedding_dim)
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
            stype_encoder_cls_kwargs=DEFAULT_STYPE_ENCODER_DICT,
            torch_frame_model_cls=torch_frame_model_cls,
            torch_frame_model_kwargs=torch_frame_model_kwargs,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types
                if "time" in data[node_type]
            ],
            channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        self.head = MLP(
            channels,
            out_channels=1,
            norm=norm,
            num_layers=1,
        )
        self.lhs_projector = torch.nn.Linear(channels, embedding_dim)
        self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        self.id_awareness_emb.reset_parameters()
        self.lhs_projector.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        # Add ID-awareness to the root node
        x_dict[entity_table][:seed_time.size(0
                                             )] += self.id_awareness_emb.weight
        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict,
                                              batch.batch_dict)

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )

        batch_size = seed_time.size(0)
        lhs_emb = self.lhs_projector(x_dict[entity_table][:batch_size])
        rhs_emb = self.rhs_embedding()
        return lhs_emb @ rhs_emb.t()

    def to(self, *args, **kwargs) -> Self:
        return super().to(*args, **kwargs)

    def cpu(self) -> Self:
        return super().cpu()

    def cuda(self, *args, **kwargs) -> Self:
        return super().cuda(*args, **kwargs)


# SOURCE: contextgnn/nn/models/contextgnn.py
from typing import Any, Dict, Optional, Tuple, Type

import torch
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_frame.nn.models import ResNet
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
from torch_geometric.utils.map import map_index
from typing_extensions import Self



class ContextGNN(RHSEmbeddingGNN):
    r"""Implementation of ContextGNN model."""
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        rhs_emb_mode: RHSEmbeddingMode,
        dst_entity_table: str,
        num_nodes: int,
        num_layers: int,
        channels: int,
        embedding_dim: int,
        aggr: str = 'sum',
        norm: str = 'layer_norm',
        torch_frame_model_cls: Type[torch.nn.Module] = ResNet,
        torch_frame_model_kwargs: Optional[Dict[str, Any]] = None,
        rhs_sample_size: Optional[int] = None,
    ) -> None:
        super().__init__(data, col_stats_dict, rhs_emb_mode, dst_entity_table,
                         num_nodes, embedding_dim)

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
            stype_encoder_cls_kwargs=DEFAULT_STYPE_ENCODER_DICT,
            torch_frame_model_cls=torch_frame_model_cls,
            torch_frame_model_kwargs=torch_frame_model_kwargs,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types
                if "time" in data[node_type]
            ],
            channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        self.head = MLP(
            channels,
            out_channels=1,
            norm=norm,
            num_layers=1,
        )
        self.lhs_projector = torch.nn.Linear(channels, embedding_dim)

        self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.lin_offset_idgnn = torch.nn.Linear(embedding_dim, 1)
        self.lin_offset_embgnn = torch.nn.Linear(embedding_dim, 1)
        self.channels = channels
        self.num_rhs_nodes = num_nodes
        self.rhs_sample_size = rhs_sample_size

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        self.id_awareness_emb.reset_parameters()
        self.rhs_embedding.reset_parameters()
        self.lin_offset_embgnn.reset_parameters()
        self.lin_offset_idgnn.reset_parameters()
        self.lhs_projector.reset_parameters()

    def sample_step(self, rhs_idgnn_index, lhs_idgnn_batch, rhs_gnn_embedding,
                    lhs_y_batch, rhs_y_index):
        rnd = torch.rand(self.num_rhs_nodes, device=rhs_idgnn_index.device)
        # Prioritize idgnn logits
        rnd[rhs_idgnn_index] = 3.
        # Ensure we always sample positives
        rhs_y_index = rhs_y_index
        assert rhs_y_index is not None  # always pass in dst index
        rnd[rhs_y_index] = 4.
        rhs_index = rnd.topk(self.rhs_sample_size, sorted=True).indices
        inclusive = rhs_y_index.numel() <= self.rhs_sample_size
        rhs_y_index, mask = map_index(rhs_y_index, rhs_index,
                                      max_index=self.num_rhs_nodes,
                                      inclusive=inclusive)
        lhs_y_batch = lhs_y_batch if inclusive else lhs_y_batch[mask]
        rhs_embedding = self.rhs_embedding(rhs_index)  # num_rhs_nodes, channel
        inclusive = (rhs_y_index.numel() + rhs_idgnn_index.numel()
                     <= self.rhs_sample_size)
        rhs_idgnn_index, mask = map_index(rhs_idgnn_index, rhs_index,
                                          inclusive=inclusive)
        if not inclusive:
            lhs_idgnn_batch = lhs_idgnn_batch[mask]
            rhs_gnn_embedding = rhs_gnn_embedding[mask]
        return (rhs_idgnn_index, rhs_embedding, lhs_idgnn_batch,
                rhs_gnn_embedding, lhs_y_batch, rhs_y_index)

    def construct_logits(self, lhs_embedding_projected, lhs_embedding,
                         rhs_gnn_embedding, rhs_embedding, lhs_idgnn_batch,
                         rhs_idgnn_index):
        embgnn_logits = lhs_embedding_projected @ rhs_embedding.t(
        )  # batch_size, num_rhs_nodes

        # Model the importance of embedding-GNN prediction for each lhs node
        embgnn_offset_logits = self.lin_offset_embgnn(
            lhs_embedding_projected).flatten()
        embgnn_logits += embgnn_offset_logits.view(-1, 1)

        # Calculate idgnn logits
        idgnn_logits = self.head(
            rhs_gnn_embedding).flatten()  # num_sampled_rhs
        # Because we are only doing 2 hop, we are not really sampling info from
        # lhs therefore, we need to incorporate this information using
        # lhs_embedding[lhs_idgnn_batch] * rhs_gnn_embedding
        idgnn_logits += (
            lhs_embedding[lhs_idgnn_batch] *  # num_sampled_rhs, channel
            rhs_gnn_embedding).sum(
                dim=-1).flatten()  # num_sampled_rhs, channel

        # Model the importance of ID-GNN prediction for each lhs node
        idgnn_offset_logits = self.lin_offset_idgnn(
            lhs_embedding_projected).flatten()
        

        return fuse_scores(embgnn_logits, idgnn_logits, lhs_idgnn_batch,
                           rhs_idgnn_index, idgnn_offset_logits)

    def forward_gnn(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ):
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        # Add ID-awareness to the root node
        x_dict[entity_table][:seed_time.size(0
                                             )] += self.id_awareness_emb.weight
        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict,
                                              batch.batch_dict)

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )
        return x_dict

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.forward_gnn(batch, entity_table)

        batch_size = seed_time.size(0)
        lhs_embedding = x_dict[entity_table][:
                                             batch_size]  # batch_size, channel
        lhs_embedding_projected = self.lhs_projector(lhs_embedding)
        rhs_gnn_embedding = x_dict[dst_table]  # num_sampled_rhs, channel
        rhs_idgnn_index = batch.n_id_dict[dst_table]  # num_sampled_rhs
        lhs_idgnn_batch = batch.batch_dict[dst_table]  # batch_size

        rhs_embedding = self.rhs_embedding()  # num_rhs_nodes, channel
        embgnn_logits = self.construct_logits(lhs_embedding_projected,
                                              lhs_embedding, rhs_gnn_embedding,
                                              rhs_embedding, lhs_idgnn_batch,
                                              rhs_idgnn_index)
        return embgnn_logits

    def forward_sample_softmax(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
        src_batch: Optional[Tensor] = None,
        dst_index: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Forward function with RHS sample softmax."""
        seed_time = batch[entity_table].seed_time
        x_dict = self.forward_gnn(batch, entity_table)

        batch_size = seed_time.size(0)
        lhs_embedding = x_dict[entity_table][:
                                             batch_size]  # batch_size, channel
        lhs_embedding_projected = self.lhs_projector(lhs_embedding)
        rhs_gnn_embedding = x_dict[dst_table]  # num_sampled_rhs, channel
        rhs_idgnn_index = batch.n_id_dict[dst_table]  # num_sampled_rhs
        lhs_idgnn_batch = batch.batch_dict[dst_table]  # batch_size

        (rhs_idgnn_index, rhs_embedding, lhs_idgnn_batch, rhs_gnn_embedding,
         lhs_y_batch, rhs_y_index) = self.sample_step(rhs_idgnn_index,
                                                      lhs_idgnn_batch,
                                                      rhs_gnn_embedding,
                                                      src_batch, dst_index)
        embgnn_logits = self.construct_logits(lhs_embedding_projected,
                                              lhs_embedding, rhs_gnn_embedding,
                                              rhs_embedding, lhs_idgnn_batch,
                                              rhs_idgnn_index)
        return embgnn_logits, lhs_y_batch, rhs_y_index

    def to(self, *args, **kwargs) -> Self:
        return super().to(*args, **kwargs)

    def cpu(self) -> Self:
        return super().cpu()

    def cuda(self, *args, **kwargs) -> Self:
        return super().cuda(*args, **kwargs)
