"""
This module contains classes and functions for creating Transformer-based models that operate on
per-feature basis. The main classes are:

- PerFeatureTransformer: A Transformer model that processes each feature separately.
- Ensemble: A collection of models with the same input and output structure.

The module also includes utility functions and classes:

- make_decoder_dict: Creates a dictionary of decoders from a decoder description dictionary.
- LayerStack: A stack of layers with support for passing keyword arguments and recomputing layers.
"""

import random
from typing import Optional, Union, Dict, Tuple, List, Type, Any

import einops
from functools import partial
import torch
import torch.nn as nn
from torch.nn import Module
from torch.utils.checkpoint import checkpoint

from tabpfn import utils
from tabpfn.model import encoders
from tabpfn.model.layer import PerFeatureEncoderLayer
from tabpfn.utils import mean_nested_structures, print_once, SerializableGenerator


def make_decoder_dict(
    decoder_description_dict: Optional[
        Dict[str, Tuple[Optional[Type[nn.Module]], int]]
    ],
    ninp: int,
    nhid: int,
) -> Optional[nn.ModuleDict]:
    """
    Creates a dictionary of decoders from a decoder description dictionary.

    Parameters:
        decoder_description_dict: A dictionary describing the decoders to create.
            Each key is the name of a decoder, and the corresponding value is a tuple
            (decoder_class, output_dimension). If decoder_class is None, a default
            MLP decoder is created.
        ninp: The input dimension of the decoders.
        nhid: The hidden dimension of the decoders.

    Returns:
        An nn.ModuleDict containing the created decoders, or None if no decoders are specified.
    """
    if decoder_description_dict is None or len(decoder_description_dict) == 0:
        return None
    initialized_decoder_dict = {}
    for decoder_key in decoder_description_dict:
        decoder_model, decoder_n_out = decoder_description_dict[decoder_key]
        if decoder_model is None:
            initialized_decoder_dict[decoder_key] = nn.Sequential(
                nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, decoder_n_out)
            )
        else:
            initialized_decoder_dict[decoder_key] = decoder_model(
                ninp, nhid, decoder_n_out
            )
        print(
            "Initialized decoder for",
            decoder_key,
            "with",
            decoder_description_dict[decoder_key],
            " and nout",
            decoder_n_out,
        )
    return nn.ModuleDict(initialized_decoder_dict)


DEFAULT_EMSIZE = 128


class PerFeatureTransformer(Module):
    """
    A Transformer model processes a token per feature and sample.


    This model extends the standard Transformer architecture to operate on a per-feature basis.
    It allows for processing each feature separately while still leveraging the power of self-attention.

    The model consists of an encoder, decoder, and optional components such as a feature positional
    embedding and a separate decoder for each feature.
    """

    def __init__(
        self,
        encoder: nn.Module = encoders.SequentialEncoder(
            encoders.LinearInputEncoderStep(
                1, DEFAULT_EMSIZE, in_keys=["main"], out_keys=["output"]
            ),
        ),
        ninp: int = DEFAULT_EMSIZE,
        nhead: int = 4,
        nhid: int = DEFAULT_EMSIZE * 4,
        nlayers: int = 10,
        y_encoder: nn.Module = encoders.SequentialEncoder(
            encoders.NanHandlingEncoderStep(),
            encoders.LinearInputEncoderStep(
                2,
                DEFAULT_EMSIZE,
                out_keys=["output"],
                in_keys=["main", "nan_indicators"],
            ),
        ),
        decoder_dict: Dict[str, Tuple[Optional[Type[nn.Module]], int]] = {
            "standard": (None, 1)
        },
        init_method: Optional[str] = None,
        activation: str = "gelu",
        recompute_layer: bool = False,
        min_num_layers_layer_dropout: Optional[int] = None,
        repeat_same_layer: bool = False,
        dag_pos_enc_dim: int = 0,
        features_per_group: int = 1,
        feature_positional_embedding: Optional[str] = None,
        zero_init: bool = True,
        use_separate_decoder: bool = False,
        nlayers_decoder: Optional[int] = None,
        use_encoder_compression_layer: bool = False,
        precomputed_kv: Optional[
            List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]
        ] = None,
        cache_trainset_representation: bool = False,
        **layer_kwargs: Any,
    ):
        """
        Parameters:
           encoder: Pass a nn.Module that takes in a batch of sequences of inputs and returns something of the shape (seq_len, batch_size, ninp)
           ninp: Input dimension, also called the embedding dimension
           nhead: Number of attention heads
           nhid: Hidden dimension in the MLP layers
           nlayers: Number of layers, each consisting of a multi-head attention layer and an MLP layer
           y_encoder: A nn.Module that takes in a batch of sequences of outputs and returns something of the shape (seq_len, batch_size, ninp)
           decoder_dict:
           activation: An activation function, e.g. "gelu" or "relu"
           recompute_layer: If True, the transformer layers will be recomputed on each forward pass in training. This is useful to save memory.
           min_num_layers_layer_dropout: if this is set, it enables to drop the last layers randomly during training up to this number.
           repeat_same_layer: If True, the same layer will be used for all layers. This is useful to save memory on weights.
           features_per_group: If > 1, the features will be grouped into groups of this size and the attention is across groups.
           feature_positional_embedding: There is a risk that our models confuse features with each other. This positional embedding is added to the features to help the model distinguish them.
             We recommend setting this to "subspace".
           zero_init: If True, the last sublayer of each attention and MLP layer will be initialized with zeros.
             Thus, the layers will start out as identity functions.
           use_separate_decoder: If True, the decoder will be separate from the encoder.
           nlayers_decoder: If use_separate_decoder is True, this is the number of layers in the decoder. The default is to use 1/3 of the layers for the decoder and 2/3 for the encoder.
           use_encoder_compression_layer: Experimental
           precomputed_kv: Experimental
           layer_kwargs: TODO: document, for now have a look at layer.py:PerFeatureEncoderLayer
        """
        super().__init__()
        self.encoder = encoder
        self.y_encoder = y_encoder
        self.ninp = ninp
        self.nhead = nhead
        self.nhid = nhid
        self.init_method = init_method
        self.features_per_group = features_per_group
        self.cache_trainset_representation = cache_trainset_representation

        layer_creator = lambda: PerFeatureEncoderLayer(
            ninp,
            nhead,
            nhid,
            activation,
            zero_init=zero_init,
            precomputed_kv=precomputed_kv.pop(0)
            if precomputed_kv is not None
            else None,
            **layer_kwargs,
        )
        if repeat_same_layer:
            layer = layer_creator()
            layer_creator = lambda: layer

        nlayers_encoder = nlayers
        if use_separate_decoder and nlayers_decoder is None:
            nlayers_decoder = max((nlayers // 3) * 1, 1)
            nlayers_encoder = max((nlayers // 3) * 2, 1)

        self.transformer_encoder = LayerStack(
            layer_creator,
            nlayers_encoder,
            recompute_each_layer=recompute_layer,
            min_num_layers_layer_dropout=min_num_layers_layer_dropout,
        )

        self.transformer_decoder = None
        if use_separate_decoder:
            self.transformer_decoder = LayerStack(
                layer_creator,
                nlayers_decoder,
            )

        self.global_att_embeddings_for_compression = None
        if use_encoder_compression_layer:
            utils.print_once(
                "ATTENTION: the use_encoder_compression_layer setting is experimental and not tested."
            )
            assert use_separate_decoder

            num_global_att_tokens_for_compression = 512

            self.global_att_embeddings_for_compression = nn.Embedding(
                num_global_att_tokens_for_compression, ninp
            )

            self.encoder_compression_layer = LayerStack(
                layer_creator,
                2,
            )

        self.decoder_dict = make_decoder_dict(decoder_dict, ninp, nhid)

        self.feature_positional_embedding = feature_positional_embedding
        print("using feature positional embedding", self.feature_positional_embedding)
        if feature_positional_embedding == "learned":
            self.feature_positional_embedding_embeddings = nn.Embedding(1_000, ninp)
        elif feature_positional_embedding == "subspace":
            self.feature_positional_embedding_embeddings = nn.Linear(ninp // 4, ninp)

        self.dag_pos_enc_dim = dag_pos_enc_dim
        self.cached_feature_positional_embeddings = None
        self.seed = random.randint(0, 1_000_000)
        self.generator_device = (
            "cpu"  # Device on which the generator was last initialized.
        )
        # If loading from a checkpoint, this might be false, but it will be set to the correct
        # device on the first forward pass.
        self._init_rnd()

    def _init_rnd(self):
        self.generator = SerializableGenerator(device=self.generator_device)
        if self.seed:  # This can be none if set outside of the model.
            self.generator.manual_seed(self.seed)

    def reset_save_peak_mem_factor(self, factor=None):
        """
        Sets the save_peak_mem_factor for all layers.

        This factor controls how much memory is saved during the forward pass in inference mode.
        Setting this factor > 1 will cause the model to save more memory during the forward pass in inference mode.
        A value of 8 is good for a 4x larger width in the fully-connected layers.
        And yields a situation were we need around 2 * num_features * num_items * emsize * 2 bytes of memory for a forward pass (using mixed precision).
        WARNING: It should only be used with post-norm.

        Parameters:
            factor: The save_peak_mem_factor to set. Recommended value is 8.
        """
        for layer in self.transformer_encoder.layers:
            assert hasattr(
                layer, "save_peak_mem_factor"
            ), "Layer does not have save_peak_mem_factor"
            layer.save_peak_mem_factor = factor

    def __setstate__(self, state):
        state.setdefault("features_per_group", 1)
        state.setdefault("feature_positional_embedding", None)
        super().__setstate__(state)

    def forward(self, *args, **kwargs):
        """
        Performs a forward pass through the model.

        This method supports multiple calling conventions:
        - model(train_x, train_y, test_x, **kwargs)
        - model((x,y), **kwargs)
        - model((style,x,y), **kwargs)

        Parameters:
            train_x: The input data for the training set.
            train_y: The target data for the training set.
            test_x: The input data for the test set.
            x: The input data.
            y: The target data.
            style: The style vector.
            single_eval_pos: The position to evaluate at.
            only_return_standard_out: Whether to only return the standard output.
            data_dags: The data DAGs for each example.
            categorical_inds: The indices of categorical features.
            freeze_kv: Whether to freeze the key and value weights.

        Returns:
            The output of the model, which can be a tensor or a dictionary of tensors.
        """
        supported_kwargs = {
            "single_eval_pos",
            "only_return_standard_out",
            "style",
            "data_dags",
            "categorical_inds",
            "freeze_kv",
        }
        if "half_layers" in kwargs:
            assert not kwargs["half_layers"]
            del kwargs["half_layers"]

        if "train_x" in kwargs:
            assert len(args) == 0
            args = [kwargs["train_x"], kwargs["train_y"], kwargs["test_x"]]
            del kwargs["train_x"]
            del kwargs["train_y"]
            del kwargs["test_x"]

        if len(args) == 3:
            supported_kwargs.remove("single_eval_pos")
        spurious_kwargs = set(kwargs.keys()) - supported_kwargs
        assert not spurious_kwargs, spurious_kwargs

        if len(args) == 3:
            # case model(train_x, train_y, test_x, src_mask=None, style=None, only_return_standard_out=True)
            x = args[0]
            if args[2] is not None:
                x = torch.cat((x, args[2]), dim=0)
            return self._forward(x, args[1], single_eval_pos=len(args[0]), **kwargs)
        elif len(args) == 1 and isinstance(args, tuple):
            # case model((x,y), src_mask=None, single_eval_pos=None, only_return_standard_out=True)
            # case model((style,x,y), src_mask=None, single_eval_pos=None, only_return_standard_out=True)
            if len(args[0]) == 3:
                return self._forward(*args[0][1:], style=args[0][0], **kwargs)
            else:
                assert (
                    len(args[0]) == 2
                ), f"Expected tuple of length 2 or 3, got {len(args[0])}"
                return self._forward(*args[0], **kwargs)
        else:
            raise ValueError(
                "Unrecognized input. Please follow the doc string exactly."
            )

    def _forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        single_eval_pos: Optional[int] = None,
        only_return_standard_out: bool = True,
        style: Optional[torch.Tensor] = None,
        data_dags: Optional[List[Any]] = None,
        categorical_inds: Optional[List[List[int]]] = None,
        half_layers: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        The core forward pass of the model.

        Parameters:
            x: The input data. Shape: (seq_len, batch_size, num_features)
            y: The target data. Shape: (seq_len, batch_size)
            single_eval_pos: The position to evaluate at. If None, evaluate at all positions.
            only_return_standard_out: Whether to only return the standard output.
            style: The style vector.
            data_dags: The data DAGs for each example in the batch.
            categorical_inds: The indices of categorical features.
            half_layers: Whether to use half the layers.

        Returns:
            A dictionary of output tensors.
        """
        assert style is None
        if self.cache_trainset_representation:
            if not single_eval_pos:  # none or 0
                assert y is None
        else:
            assert y is not None and single_eval_pos

        single_eval_pos_ = single_eval_pos or 0
        if isinstance(x, dict):
            assert "main" in set(x.keys()), f"Main must be in input keys: {x.keys()}."
        else:
            x = {"main": x}
        seq_len, batch_size, num_features = x["main"].shape

        if y is None:
            # TODO: check dtype.
            y = torch.zeros(
                0, batch_size, device=x["main"].device, dtype=x["main"].dtype
            )

        if isinstance(y, dict):
            assert "main" in set(y.keys()), f"Main must be in input keys: {y.keys()}."
        else:
            y = {"main": y}

        # print(f"missing_to_next: {missing_to_next}", 'num_features', num_features, 'features_per_group', self.features_per_group)
        for k in x:
            num_features_ = x[k].shape[2]

            # pad to multiple of features_per_group
            missing_to_next = (
                self.features_per_group - (num_features_ % self.features_per_group)
            ) % self.features_per_group

            if missing_to_next > 0:
                x[k] = torch.cat(
                    (
                        x[k],
                        torch.zeros(
                            seq_len,
                            batch_size,
                            missing_to_next,
                            device=x[k].device,
                            dtype=x[k].dtype,
                        ),
                    ),
                    dim=-1,
                )
        for k in x:
            # print('x.shape', x.shape)
            x[k] = einops.rearrange(
                x[k], "s b (f n) -> b s f n", n=self.features_per_group
            )  # s b f -> b s #groups #features_per_group

        if categorical_inds is not None:
            new_categorical_inds = []
            for ci in categorical_inds:
                num_subgroups = x["main"].shape[2]
                new_categorical_inds += [
                    [
                        i - subgroup * self.features_per_group
                        for i in ci
                        if (
                            subgroup * self.features_per_group
                            <= i
                            < (subgroup + 1) * self.features_per_group
                        )
                    ]
                    for subgroup in range(num_subgroups)
                ]
            categorical_inds = new_categorical_inds

        for k in y:
            if len(y[k].shape) == 2:
                y[k] = y[k].unsqueeze(-1)  # s b -> s b 1

            y[k] = y[k].transpose(0, 1)  # s b 1 -> b s 1

            if y[k].shape[1] < x["main"].shape[1]:
                assert (
                    y[k].shape[1] == single_eval_pos_
                    or y[k].shape[1] == x["main"].shape[1]
                )
                assert (
                    k != "main" or y[k].shape[1] == single_eval_pos_
                ), "For main y, y must not be given for target time steps (Otherwise the solution is leaked)."
                if y[k].shape[1] == single_eval_pos_:
                    y[k] = torch.cat(
                        (
                            y[k],
                            torch.nan
                            * torch.zeros(
                                y[k].shape[0],
                                x["main"].shape[1] - y[k].shape[1],
                                y[k].shape[2],
                                device=y[k].device,
                                dtype=y[k].dtype,
                            ),
                        ),
                        dim=1,
                    )

            y[k] = y[k].transpose(0, 1)  # b s 1 -> s b 1

        # making sure no label leakage ever happens
        y["main"][single_eval_pos_:] = torch.nan

        embedded_y = self.y_encoder(
            y,
            single_eval_pos=single_eval_pos_,
            cache_trainset_representation=self.cache_trainset_representation,
        ).transpose(0, 1)
        # print('embedded y', embedded_y.shape, embedded_y)
        del y
        assert not torch.isnan(
            embedded_y
        ).any(), f"{torch.isnan(embedded_y).any()=}, make sure to add nan handlers to the ys that are not fully provided (test set missing)"

        extra_encoders_args = {}
        if categorical_inds is not None and isinstance(
            self.encoder, encoders.SequentialEncoder
        ):
            extra_encoders_args["categorical_inds"] = categorical_inds

        for k in x:
            x[k] = einops.rearrange(x[k], "b s f n -> s (b f) n")

        embedded_x = einops.rearrange(
            self.encoder(
                x,
                single_eval_pos=single_eval_pos_,
                cache_trainset_representation=self.cache_trainset_representation,
                **extra_encoders_args,
            ),
            "s (b f) e -> b s f e",
            b=embedded_y.shape[0],
        )  # b s f 1 -> b s f e
        del x

        embedded_x, embedded_y = self.add_embeddings(
            embedded_x,
            embedded_y,
            data_dags,
            num_features,
            seq_len,
            cache_embeddings=self.cache_trainset_representation and single_eval_pos,
            use_cached_embeddings=self.cache_trainset_representation
            and not single_eval_pos,
        )
        del data_dags

        embedded_input = torch.cat(
            (embedded_x, embedded_y.unsqueeze(2)), dim=2
        )  # b s f e + b s 1 e -> b s f+1 e

        assert not torch.isnan(
            embedded_input
        ).any(), f"There should be no NaNs in the encoded x and y. Check that you do not feed NaNs or use a NaN-handling enocder. Your embedded x and y returned the following: {torch.isnan(embedded_x).any()=} and {torch.isnan(embedded_y).any()=}."
        del embedded_y, embedded_x

        # print(f"{embedded_input[:, -1, 0, :10]=}")
        # print(f"{embedded_input[:, -1, -1, :10]=}")

        encoder_out = self.transformer_encoder(
            (
                embedded_input
                if not self.transformer_decoder
                else embedded_input[:, :single_eval_pos_]
            ),
            single_eval_pos=single_eval_pos,
            half_layers=half_layers,
            cache_trainset_representation=self.cache_trainset_representation,
        )  # b s f+1 e -> b s f+1 e

        # If we are using a decoder
        if self.transformer_decoder:
            print_once("Using separate decoder")
            assert not half_layers
            assert encoder_out.shape[1] == single_eval_pos_

            if self.global_att_embeddings_for_compression is not None:
                # TODO: fixed number of compression tokens
                train_encoder_out = self.encoder_compression_layer(
                    self.global_att_embeddings_for_compression,
                    att_src=encoder_out[:, single_eval_pos_],
                    single_eval_pos=single_eval_pos_,
                )

            test_encoder_out = self.transformer_decoder(
                embedded_input[:, single_eval_pos_:],
                single_eval_pos=0,
                att_src=encoder_out,
            )
            encoder_out = torch.cat([encoder_out, test_encoder_out], 1)
            del test_encoder_out

        del embedded_input

        test_encoder_out = encoder_out[:, single_eval_pos_:, -1].transpose(
            0, 1
        )  # out: s b e

        if only_return_standard_out:
            output_decoded = self.decoder_dict["standard"](test_encoder_out)
        else:
            output_decoded = (
                {k: v(test_encoder_out) for k, v in self.decoder_dict.items()}
                if self.decoder_dict is not None
                else {}
            )

            train_encoder_out = encoder_out[:, :single_eval_pos_, -1].transpose(
                0, 1
            )  # out: s b e

            output_decoded["train_embeddings"] = train_encoder_out
            output_decoded["test_embeddings"] = test_encoder_out

        return output_decoded

    def add_embeddings(
        self,
        x,
        y,
        data_dags,
        num_features,
        seq_len,
        cache_embeddings=False,
        use_cached_embeddings=False,
    ):
        if use_cached_embeddings and self.cached_embeddings is not None:
            assert (
                data_dags is None
            ), "Caching embeddings is not supported with data_dags at this point."
            x += self.cached_embeddings[None, None]
            return x, y

        if (
            self.generator_device != self.generator.device
            or self.generator_device != x.device
        ):
            self.generator_device = x.device
            self._init_rnd()

        if self.feature_positional_embedding == "normal_rand_vec":
            embs = torch.randn(
                (x.shape[2], x.shape[3]),
                device=x.device,
                dtype=x.dtype,
                generator=self.generator,
            )
            x += embs[None, None]
        elif self.feature_positional_embedding == "uni_rand_vec":
            embs = (
                torch.rand(
                    (x.shape[2], x.shape[3]),
                    device=x.device,
                    dtype=x.dtype,
                    generator=self.generator,
                )
                * 2
                - 1
            )
            x += embs[None, None]
        elif self.feature_positional_embedding == "learned":
            w = self.feature_positional_embedding_embeddings.weight
            embs = w[
                torch.randint(0, w.shape[0], (x.shape[2],), generator=self.generator)
            ]
            x += embs[None, None]
        elif self.feature_positional_embedding == "subspace":
            embs = torch.randn(
                (x.shape[2], x.shape[3] // 4),
                device=x.device,
                dtype=x.dtype,
                generator=self.generator,
            )
            embs = self.feature_positional_embedding_embeddings(embs)
            x += embs[None, None]
        else:
            assert self.feature_positional_embedding is None

        if cache_embeddings and self.feature_positional_embedding is not None:
            assert (
                data_dags is None
            ), "Caching embeddings is not supported with data_dags at this point."
            self.cached_embeddings = embs
        else:
            self.cached_embeddings = None

        # TODO should this go into encoder?
        # could also be made a bit more concise by moving down to operate on full_x
        if data_dags is not None:
            for b_i, data_dag in enumerate(data_dags):
                g_ = data_dag.copy()  # type: nx.DiGraph
                while utils.add_direct_connections(g_):
                    pass
                subgraph = g_.subgraph(
                    [
                        n
                        for n, info in g_.nodes.items()
                        if (info["is_feature"] or info["is_target"])
                    ]
                )
                k = self.dag_pos_enc_dim
                assert k
                utils.add_pos_emb(subgraph, k=k)
                graph_pos_embs_features = torch.zeros(
                    (num_features, k)
                )  # shape: (num_features, k)
                graph_pos_embs_targets = torch.zeros((1, k))  # shape: (num_targets, k)
                for node, node_info in subgraph.nodes.items():
                    for feature_idx in node_info.get("feature_idxs", []):
                        graph_pos_embs_features[feature_idx] = node_info[
                            "positional_encoding"
                        ]
                    for target_idx in node_info.get("target_idxs", []):
                        graph_pos_embs_targets[target_idx] = node_info[
                            "positional_encoding"
                        ]

                # assert ((graph_pos_embs_features == -1000)[:-1].int() <= (graph_pos_embs_features == -1000)[1:].int()).all(), graph_pos_embs_features.mean(1)

                # print('n',torch.isnan(graph_pos_embs_features).any(), torch.isnan(graph_pos_embs_targets).any())
                # print('o', torch.isnan(x).any(), torch.isnan(y).any())

                graph_pos_embs_targets -= graph_pos_embs_features.mean(0, keepdim=True)
                graph_pos_embs_features -= graph_pos_embs_features.mean(0, keepdim=True)

                graph_pos_embs_features = graph_pos_embs_features[None].expand(
                    seq_len, -1, -1
                )
                x[b_i, :, :, :k] += graph_pos_embs_features.to(y.device, y.dtype)

                graph_pos_embs_targets = (
                    graph_pos_embs_targets[None].expand(seq_len, -1, -1).squeeze(-2)
                )
                y[b_i, :, :k] += graph_pos_embs_targets.to(y.device, y.dtype)
        else:
            assert not hasattr(self, "dag_pos_enc_dim") or not self.dag_pos_enc_dim

        return x, y

    def empty_trainset_representation_cache(self):
        for layer in (self.transformer_decoder or self.transformer_encoder).layers:
            layer.empty_trainset_representation_cache()


class LayerStack(Module):
    """
    Similar to nn.Sequential, but with support for passing keyword arguments to layers and stacks the same layer multiple times.
    """

    def __init__(
        self,
        layer_creator,
        num_layers,
        recompute_each_layer=False,
        min_num_layers_layer_dropout=None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([layer_creator() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.min_num_layers_layer_dropout = (
            min_num_layers_layer_dropout
            if min_num_layers_layer_dropout is not None
            else num_layers
        )
        self.recompute_each_layer = recompute_each_layer

    def forward(self, x, half_layers=False, **kwargs):
        if half_layers:
            assert (
                self.min_num_layers_layer_dropout == self.num_layers
            ), "half_layers only works without layer dropout"
            n_layers = self.num_layers // 2
        else:
            n_layers = torch.randint(
                self.min_num_layers_layer_dropout, self.num_layers + 1, (1,)
            ).item()
        for i, layer in enumerate(self.layers[:n_layers]):
            if self.recompute_each_layer and x.requires_grad:
                x = checkpoint(partial(layer, **kwargs), x, use_reentrant=False)
            else:
                x = layer(x, **kwargs)

        return x
