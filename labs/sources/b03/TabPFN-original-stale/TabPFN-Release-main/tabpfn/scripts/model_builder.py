from __future__ import annotations

import re
from dataclasses import dataclass, asdict
import typing as tp

from torch import nn
import torch
import math

from tabpfn import utils
from tabpfn.utils import UniformSingleEvalPosSampler, WeightedSingleEvalPosSampler
from tabpfn.model import encoders, build_model
from tabpfn.model.multi_head_attention import MultiHeadAttention
from tabpfn.model.bar_distribution import (
    BarDistribution,
)

from tabpfn.model.encoders import (
    InputNormalizationEncoderStep,
    LinearInputEncoderStep,
    VariableNumFeaturesEncoderStep,
    RemoveEmptyFeaturesEncoderStep,
    RemoveDuplicateFeaturesEncoderStep,
    NanHandlingEncoderStep,
    FrequencyFeatureEncoderStep,
)

from tabpfn.model.bar_distribution import (
    FullSupportBarDistribution,
    get_bucket_limits,
)


@dataclass()
class Checkpoint:
    state_dict: tp.Dict[str, tp.Any]
    optimizer_state: tp.Dict[str, tp.Any]
    scaler_state: tp.Optional[tp.Dict[str, tp.Any]]

    config: tp.Dict[str, tp.Any]
    trained_epochs_until_now: int

    def save(self, path: str):
        torch.save(asdict(self), path)

    def prepare_deployment(self):
        return Checkpoint(
            state_dict=self.state_dict,
            optimizer_state=None,
            scaler_state=None,
            config=self.config,
            trained_epochs_until_now=self.trained_epochs_until_now,
        )

    @classmethod
    def load(cls, path: str) -> "Checkpoint":
        print(f"Loading checkpoint from {path}")
        # Inserting 'tabpfn' to sys path because old code had different structure
        #   and pickle load requires same imports
        import sys

        sys.path.insert(0, "tabpfn")

        checkpoint: tp.Dict[str, tp.Any] = torch.load(path, map_location="cpu")
        # this is here for backwards compatibility
        if "trained_epochs_until_now" not in checkpoint:
            checkpoint["trained_epochs_until_now"] = checkpoint["config"][
                "trained_epochs_until_now"
            ]

        return cls(
            state_dict=compatability_fixes(
                checkpoint["state_dict"], checkpoint["config"]
            ),
            optimizer_state=checkpoint["optimizer_state"],
            config={
                **checkpoint["config"],
                "trained_epochs_until_now": checkpoint["trained_epochs_until_now"],
            },
            trained_epochs_until_now=checkpoint["trained_epochs_until_now"],
            scaler_state=checkpoint["scaler_state"]
            if "scaler_state" in checkpoint
            else checkpoint.get("scalar_state"),
        )


def load_model(
    path: str,
    device: str,
    verbose: bool = True,
    overwrite_config_keys: tp.Optional[tp.Dict[str, tp.Any]] = None,
):
    """
    Loads a model from a given path and filename.
    It returns a Transformer model and a config. This is ideal for low-level inference.
    If you want to continue training, it is recommended to go one level deeper and use `Checkpoint.load` directly.

    Args:
        path (str): Path to the model
        device (str): Device to load the model to
        verbose (bool): Whether to print the loaded config
    """
    checkpoint: Checkpoint = Checkpoint.load(path)

    if overwrite_config_keys is not None:
        checkpoint.config = {**checkpoint.config, **overwrite_config_keys}

    model = get_model(
        config=checkpoint.config,
        device=device,
        should_train=False,
        verbose=verbose,
        state_dict=checkpoint.state_dict,
    )

    model[2].to(device)
    model[2].eval()

    return model, checkpoint.config


def preprocess_attention_state_dict(state_dict, config) -> dict:
    if "in_proj_weight" in state_dict:
        utils.print_once(
            "Attention weights in checkpoint are in torch.nn.MultiheadAttention format. Converting to format of the new attention implementation."
        )
        return MultiHeadAttention.convert_torch_nn_multihead_attention_state_dict(
            state_dict, config["nhead"]
        )
    else:
        return state_dict


def preprocess_encoder_layer_state_dict(state_dict, config):
    def handle_attention(attention_name):
        attention_state_dict = utils.get_submodule_from_statedict(
            state_dict, attention_name
        )
        attention_state_dict = preprocess_attention_state_dict(
            attention_state_dict, config
        )
        utils.set_submodule_statedict(state_dict, attention_name, attention_state_dict)

    handle_attention("self_attn_between_features")
    handle_attention("self_attn_between_items")
    if "linear1.weight" in state_dict:
        utils.print_once(
            "Mlp weights in checkpoint are in legacy format. Converting to current format."
        )
        state_dict["mlp.linear1.weight"] = state_dict.pop("linear1.weight")
        state_dict["mlp.linear2.weight"] = state_dict.pop("linear2.weight")
    if "linear3.weight" in state_dict:
        utils.print_once("Converting weights for second mlp to current format.")
        state_dict["second_mlp.linear1.weight"] = state_dict.pop("linear3.weight")
        state_dict["second_mlp.linear2.weight"] = state_dict.pop("linear4.weight")
    return state_dict


def compatability_fixes(state_dict, config):
    """Fixes for compatability with old models"""
    state_dict = {k.replace(".step_module_layer", ""): v for k, v in state_dict.items()}
    module_prefix = "module."
    state_dict = {k.replace(module_prefix, ""): v for k, v in state_dict.items()}
    # for compatibility with the old models, that only have a single decoder
    state_dict = {
        re.sub(r"^decoder\.", "decoder_dict.standard.", k): v
        for k, v in state_dict.items()
    }

    if "seq_len" not in config and "bptt" in config:
        config["seq_len"] = config["bptt"]

    # Backwards compatability, now the attribute is included in save_model.
    # TODO: Can be removed after some time.
    if "trained_epochs_until_now" not in config:
        config["trained_epochs_until_now"] = round(
            config["epochs"] * config.get("done_part_in_training", 1.0)
        )

    if "criterion.bucket_widths" in state_dict:
        del state_dict["criterion.bucket_widths"]

    for layer_index in range(config.get("nlayers_encoder") or config["nlayers"]):
        encoder_layer_prefix = f"transformer_encoder.layers.{layer_index}"
        encoder_layer_state_dict = utils.get_submodule_from_statedict(
            state_dict, encoder_layer_prefix
        )
        encoder_layer_state_dict = preprocess_encoder_layer_state_dict(
            encoder_layer_state_dict, config
        )
        utils.set_submodule_statedict(
            state_dict, encoder_layer_prefix, encoder_layer_state_dict
        )

    if config.get("use_separate_decoder", False):
        for layer_index in range(config["nlayers_decoder"]):
            decoder_layer_prefix = f"transformer_decoder.layers.{layer_index}"
            decoder_layer_state_dict = utils.get_submodule_from_statedict(
                state_dict, decoder_layer_prefix
            )
            decoder_layer_state_dict = preprocess_encoder_layer_state_dict(
                decoder_layer_state_dict, config
            )
            utils.set_submodule_statedict(
                state_dict, decoder_layer_prefix, decoder_layer_state_dict
            )

    if (mqf := config.get("multi_query_factor")) and mqf > 1:
        assert mqf == config["nhead"], "multi_query_factor must be equal to nhead"
        config["multiquery_item_attention_for_test_set"] = True

    return state_dict


def fix_loaded_config_sample(loaded_config_sample, config):
    def copy_to_sample(*k):
        t, s = loaded_config_sample, config
        for k_ in k[:-1]:
            t = t[k_]
            s = s[k_]
        t[k[-1]] = s[k[-1]]

    copy_to_sample("num_classes")
    copy_to_sample(
        "differentiable_hyperparameters", "prior_mlp_activations", "choice_values"
    )


def load_config_sample(path, template_config):
    model_state, optimizer_state, loaded_config_sample = torch.load(
        path, map_location="cpu"
    )
    fix_loaded_config_sample(loaded_config_sample, template_config)
    return loaded_config_sample


def get_default_spec(test_datasets, valid_datasets):
    bptt = 10000
    eval_positions = [
        1000,
        2000,
        3000,
        4000,
        5000,
    ]  # list(2 ** np.array([4, 5, 6, 7, 8, 9, 10, 11, 12]))
    max_features = max(
        [X.shape[1] for (_, X, _, _, _, _) in test_datasets]
        + [X.shape[1] for (_, X, _, _, _, _) in valid_datasets]
    )
    max_splits = 5

    return bptt, eval_positions, max_features, max_splits


def get_y_encoder(config):
    def get_y_encoder_(num_inputs, embedding_size):
        if config.get("canonical_y_encoder", False):
            # encoders.get_Canonical(config["max_num_classes"])
            raise NotImplementedError("Canonical encoder not implemented yet")

        steps = []
        inputs_to_merge = [{"name": "main", "dim": num_inputs}]
        if config.get("nan_handling_y_encoder", True):
            steps += [encoders.NanHandlingEncoderStep()]
            inputs_to_merge += [{"name": "nan_indicators", "dim": num_inputs}]
        if config["max_num_classes"] >= 2:
            steps += [
                encoders.MulticlassClassificationTargetEncoder(),
            ]
        if config.get("is_survival", False):
            steps += [
                encoders.NanHandlingEncoderStep(
                    in_keys=["event"], out_keys=["event", "event_nan_indicators"]
                )
            ]
            inputs_to_merge += [
                {"name": "event", "dim": 1},
                {"name": "event_nan_indicators", "dim": 1},
            ]

        steps += [
            encoders.LinearInputEncoderStep(
                sum([i["dim"] for i in inputs_to_merge]),
                embedding_size,
                in_keys=[i["name"] for i in inputs_to_merge],
                out_keys=["output"],
            )
        ]
        encoder = encoders.SequentialEncoder(
            *steps,
            output_key="output",
        )
        return encoder

    return get_y_encoder_


def get_encoder(config):
    def get_encoder_(num_features, embedding_size):
        inputs_to_merge = {"main": {"dim": num_features}}

        encoder_steps = []
        if config.get("remove_empty_features", True):
            encoder_steps += [RemoveEmptyFeaturesEncoderStep()]

        if config.get("remove_duplicate_features", False):
            encoder_steps += [RemoveDuplicateFeaturesEncoderStep()]

        encoder_steps += [
            NanHandlingEncoderStep(keep_nans=config.get("nan_handling_enabled", False))
        ]

        if config.get("nan_handling_enabled", False):
            inputs_to_merge["nan_indicators"] = {"dim": num_features}

            encoder_steps += [
                VariableNumFeaturesEncoderStep(
                    num_features=num_features,
                    normalize_by_used_features=False,
                    in_keys=["nan_indicators"],
                    out_keys=["nan_indicators"],
                )
            ]

        if config.get("categorical_encoder_per_feature", False):
            # CategoricalInputEncoderPerFeatureWrapper
            raise NotImplementedError("Not Implemented")

        encoder_steps += [
            InputNormalizationEncoderStep(
                normalize_on_train_only=config.get("normalize_on_train_only"),
                normalize_to_ranking=config.get("normalize_to_ranking"),
                normalize_x=config.get("normalize_x"),
                remove_outliers=config.get("remove_outliers"),
            )
        ]

        if config.get("num_frequencies_in_encoding", False):
            num_features = inputs_to_merge["main"]["dim"]
            frequency_encoding = FrequencyFeatureEncoderStep(
                num_features=num_features,
                num_frequencies=config["num_frequencies_in_encoding"],
                freq_power_base=config.get("freq_power_base_in_encoding", 2.0),
                max_wave_length=config.get("max_wave_length_in_encoding", 4.0),
            )
            num_features = frequency_encoding.num_features_out
            inputs_to_merge["main"]["dim"] = num_features
            encoder_steps += [frequency_encoding]

        encoder_steps += [
            VariableNumFeaturesEncoderStep(
                num_features=num_features,
                normalize_by_used_features=config.get("normalize_by_used_features"),
            )
        ]

        separate_cat_encoder = config.get("separate_cat_encoder", False)
        if separate_cat_encoder:
            assert [k for k in inputs_to_merge] == ["main"], inputs_to_merge
            encoder_steps += [
                encoders.CategoricalInputEncoderPerFeatureEncoderStep(
                    num_features=num_features,
                    emsize=embedding_size,
                    base_encoder=encoders.LinearInputEncoder(
                        num_features,
                        embedding_size,
                        bias=config.get("encoder_use_bias", True),
                    ),
                    out_keys=["output"],
                )
            ]
        else:
            if config.get("use_decomposed_number_encoder", False):
                raise NotImplementedError("Not Implemented")
            else:
                encoder_steps += [
                    LinearInputEncoderStep(
                        sum([i["dim"] for i in inputs_to_merge.values()]),
                        embedding_size,
                        bias=config.get("encoder_use_bias", True),
                        in_keys=[name for name in inputs_to_merge],
                        out_keys=["output"],
                    )
                ]

        return encoders.SequentialEncoder(*encoder_steps, output_key="output")

    return get_encoder_


def get_single_eval_pos_gen(config):
    max_len = (
        int(config["seq_len"] - config.get("min_eval_fraction", 0) * config["seq_len"])
        - 1
    )
    if (
        "single_eval_pos_gen_weighted" in config
        and config["single_eval_pos_gen_weighted"] != None
    ):
        single_eval_pos_gen = WeightedSingleEvalPosSampler(
            max_len,
            min_len=config.get("min_eval_pos", 10),
            p=config["single_eval_pos_gen_weighted"],
        )
    else:
        single_eval_pos_gen = UniformSingleEvalPosSampler(
            max_len,
            min_len=config.get("min_eval_pos", 10),
        )
    return single_eval_pos_gen


def preprocess_config(config, task_type=None):
    if "aggregate_k_gradients" not in config or config["aggregate_k_gradients"] is None:
        assert math.log(
            config["batch_size"], 2
        ).is_integer(), "Batch size must be power of 2"
        second_dim_tokens = config.get("num_global_att_tokens ", config["seq_len"])
        memory_factor = (
            config["batch_size"]
            * config["nlayers"]
            * config["emsize"]
            * config["seq_len"]
            * second_dim_tokens
        )
        standard_memory_factor = 16 * 12 * 512 * 1200 * 1200
        config["aggregate_k_gradients"] = math.ceil(
            memory_factor / (standard_memory_factor * 1.1)
        )

        # Make sure that batch size is power of two
        config["batch_size"] = int(
            math.pow(
                2,
                math.floor(
                    math.log(config["batch_size"] / config["aggregate_k_gradients"], 2)
                ),
            )
        )
        config["num_steps"] = math.ceil(
            config["num_steps"] * config["aggregate_k_gradients"]
        )
        # Make sure that batch_size_per_gp_sample is power of two
        assert math.log(config["batch_size_per_gp_sample"], 2) % 1 == 0

    config.setdefault(
        "num_features_sampler_config",
        {
            "min_num_features": 1,
            "max_num_features": config.get("max_num_features", 100),
            "type": "beta",
            "sampler_parameters": {
                "k": config.get("num_features_sampler_config_beta_k", 0.95),
                "b": config.get("num_features_sampler_config_beta_b", 8.0),
            },
        },
    )
    config["max_num_features_in_training"] = utils.NumFeaturesSampler(
        config["num_features_sampler_config"]
    ).config.max_num_features
    config.setdefault("recompute_attn", False)
    # TODO @noah: I don't think eval_positions is used anymore
    config["eval_positions"] = [int(config["seq_len"] * 0.95)]
    if task_type == "regression":
        config = {**config, "max_num_classes": 0, "num_classes": 0}
        if "num_classes" in config["differentiable_hyperparameters"]:
            del config["differentiable_hyperparameters"]["num_classes"]
        print("removed num_classes from differentiable_hyperparameters")
    elif task_type == "multiclass":
        config.setdefault("multiclass_type", "rank")

    return config


def get_model(
    config,
    device,
    should_train=True,
    verbose=False,
    state_dict=None,
    epoch_callback=None,
    step_callback=None,
    continue_model=None,
    optimizer_state=None,
    scaler_state=None,
    config_is_preprocessed=False,
    wandb_track_grads_freq=-1,
):
    # TODO: Integrate new code for pipeline creation by Samuel
    config = {**config}
    verbose_train, verbose_prior = verbose >= 1, verbose >= 2
    config["verbose"] = verbose_prior

    task_type = config.get("task_type", None)

    if not config_is_preprocessed:
        print("Preprocessing config in new way")
        config = preprocess_config(config, task_type=task_type)

    model_proto = None
    extra_kwargs_dict = {}
    use_style = False
    dataloader_kwargs = {}
    dataloader_device = None

    if should_train:
        from . import prior_builder

        (
            model_proto,
            extra_kwargs_dict,
            dataloader_device,
            dataloader_kwargs,
            use_style,
        ) = prior_builder.get_prior(config, device)

    loss = get_loss(
        config=config,
        get_batch_method=model_proto.get_batch if model_proto else None,
        extra_kwargs_dict=extra_kwargs_dict,
        device=device,
        num_features_sampler_config=config["num_features_sampler_config"],
        batch_size=config["batch_size"],
    )

    epochs = 0 if not should_train else config["epochs"]
    extra_train_kwargs = {}  # these are only passed to train
    extra_model_kwargs = (
        {}
    )  # these are passed to train when training and to build_model when not training

    if parallel_sublayers := config.get("parallel_sublayers", False):
        extra_model_kwargs["parallel_sublayers"] = parallel_sublayers

    if recompute_layer := config.get("recompute_layer", False):
        extra_model_kwargs["recompute_layer"] = recompute_layer

    min_num_layers_layer_dropout = config.get("min_num_layers_layer_dropout", None)
    if min_num_layers_layer_dropout is not None:
        extra_model_kwargs[
            "min_num_layers_layer_dropout"
        ] = min_num_layers_layer_dropout

    if repeat_same_layer := config.get("repeat_same_layer", False):
        extra_model_kwargs["repeat_same_layer"] = repeat_same_layer

    if second_mlp := config.get("second_mlp", False):
        extra_model_kwargs["second_mlp"] = second_mlp

    if use_encoder_compression_layer := config.get(
        "use_encoder_compression_layer", False
    ):
        extra_model_kwargs[
            "use_encoder_compression_layer"
        ] = use_encoder_compression_layer

    if dag_pos_enc_dim := config.get("dag_pos_enc_dim", False):
        extra_model_kwargs["dag_pos_enc_dim"] = dag_pos_enc_dim

    if features_per_group := config.get("features_per_group", 1):
        extra_model_kwargs["features_per_group"] = features_per_group

    if feature_positional_embedding := config.get("feature_positional_embedding", None):
        extra_model_kwargs[
            "feature_positional_embedding"
        ] = feature_positional_embedding

    if (zero_init := config.get("zero_init", None)) is not None:
        extra_model_kwargs["zero_init"] = zero_init

    if config.get("dummy_get_batch", False):
        print("Using dummy get_batch")
        print("This is useful for evaluating the code without the prior.")
        from ..priors import test_priors

        get_batch = test_priors.get_batch_deterministic
    else:
        get_batch = model_proto.get_batch if model_proto else None

    if setting := config.get("lr_scheduler", None):
        if setting == "cosine":
            extra_train_kwargs["scheduler"] = utils.get_cosine_schedule_with_warmup
        elif setting == "inverse_sqrt":
            extra_train_kwargs[
                "scheduler"
            ] = utils.get_inverse_sqrt_schedule_with_warmup
        else:
            raise NotImplementedError(f"Unknown lr scheduler {setting}")

    extra_model_kwargs["multiquery_item_attention"] = config.get(
        "multiquery_item_attention", False
    )

    extra_model_kwargs["multiquery_item_attention_for_test_set"] = config.get(
        "multiquery_item_attention_for_test_set", False
    )
    assert (
        config.get("triton_ln", "parameter_free") == "parameter_free"
    ), "only backwards compatible with parameter_free"
    if "triton_ln" in config:
        print("remove your triton_ln config, it is now parameter_free as default.")

    if "attention_init_gain" in config:
        extra_model_kwargs["attention_init_gain"] = config["attention_init_gain"]

    if "two_sets_of_queries" in config:
        extra_model_kwargs["two_sets_of_queries"] = config["two_sets_of_queries"]

    # We use a string-based comparison in addition to a class-based comparison
    # We do this because `reload` changes class hierarchies, making `isinstance` calls wrong when editing the classes
    if ("BarDistribution" in loss.__class__.__name__) or isinstance(
        loss, BarDistribution
    ):
        n_out = loss.num_bars
    elif isinstance(loss, nn.CrossEntropyLoss):
        n_out = config["max_num_classes"]
    else:
        n_out = 1

    model_kwargs = {
        "criterion": loss,
        "encoder_generator": get_encoder(config),
        "num_features": config["max_num_features_in_training"],
        "emsize": config["emsize"],
        "nhid": config["emsize"] * config["nhid_factor"],
        "nlayers": config["nlayers"],
        "nhead": config["nhead"],
        "y_encoder_generator": get_y_encoder(config),
        "decoder_dict": config.get("decoder_dict", {}),
        "load_weights_from_this_state_dict": state_dict,
        "initialize_with_model": continue_model,
        "use_separate_decoder": config.get("use_separate_decoder", False),
        "bias": config.get("bias", True),
        "use_zero_attention": config.get("use_zero_attention", False),
        "layer_norm_with_elementwise_affine": config.get(
            "layer_norm_with_elementwise_affine", False
        ),
        "pre_norm": config.get("pre_norm", False),
        "recompute_attn": config.get("recompute_attn", False),
        "use_flash_attention": config.get("use_flash_attention", False),
        "nlayers_decoder": config.get("nlayers_decoder", None),
        "custom_attention_style_and_activation_and_scale": config.get(
            "custom_attention_style_and_activation_and_scale", None
        ),
        "share_key_and_value_attention_proj": config.get(
            "share_key_and_value_attention_proj", False
        ),
        "scale_softmax_w_dataset_size": config.get(
            "scale_softmax_w_dataset_size", False
        ),
        "n_out": n_out,
    }

    if should_train:
        from ..train import train

        model = train(
            get_batch_method=get_batch,
            continue_model=continue_model,
            num_classes=config["max_num_classes"],
            num_features_sampler_config=config["num_features_sampler_config"],
            batch_size=config["batch_size"],
            epochs=epochs,
            warmup_epochs=config.get("warmup_epochs", None),
            seq_len=config["seq_len"],
            gpu_device=device,
            steps_per_epoch=config["num_steps"],
            single_eval_pos_gen=get_single_eval_pos_gen(config),
            aggregate_k_gradients=config["aggregate_k_gradients"],
            seq_len_curriculum_warmup_factor=config.get(
                "seq_len_curriculum_warmup_factor", 1
            ),
            adaptive_max_seq_len_to_max_full_table_size=config.get(
                "adaptive_max_seq_len_to_max_full_table_size", None
            ),
            fixed_test_set_size=config.get("fixed_test_set_size", None),
            epoch_callback=epoch_callback,
            step_callback=step_callback,
            extra_prior_kwargs_dict=extra_kwargs_dict,
            lr=config["lr"],
            progress_bar=config.get("progress_bar", True),
            dataloader_kwargs=dataloader_kwargs,
            dataloader_device=dataloader_device,
            verbose=verbose_train,
            weight_decay=config.get("weight_decay", 0.0),
            train_mixed_precision=config.get("train_mixed_precision", False),
            optimizer_state=optimizer_state,
            scaler_state=scaler_state,
            trained_epochs_until_now=config.get("trained_epochs_until_now", 0),
            update_parameters=config.get("update_parameters", True),
            wandb_track_grads_freq=wandb_track_grads_freq,
            **model_kwargs,
            **extra_train_kwargs,
            **extra_model_kwargs,
        )
    else:
        model = (
            None,
            None,
            build_model(
                **model_kwargs,
                **extra_model_kwargs,
            ),
            None,
        )

    return model


def get_loss(
    config,
    get_batch_method,
    batch_size,
    extra_kwargs_dict,
    num_features_sampler_config,
    device,
):
    if config["max_num_classes"] == 2:
        loss = Losses.bce
    elif config["max_num_classes"] > 2:
        loss = Losses.ce
    else:
        print("Initializing Bar distribution")

        num_buckets = config.get("num_buckets", 100)

        if config.get("trained_epochs_until_now", 0) < 1:
            from .prior_builder import get_y_sample

            y_sample = get_y_sample(
                get_batch_method,
                batch_size,
                extra_kwargs_dict,
                num_features_sampler_config,
                n=config.get("bar_dist_init_batches", 64),
            )
            borders = get_bucket_limits(
                num_buckets,
                ys=y_sample,
                widen_bucket_limits_factor=config.get(
                    "widen_bucket_limits_factor", 1.0
                ),
                # the idea here is to make our distribution a bit wider s.t. we capture broad distributions well at test time
            )
        else:
            # dummy values, extra bad s.t. one realizes if they are used for training
            borders = torch.arange(num_buckets + 1).float() * 10_000
        borders = borders * config.get("bucket_scaling", 3)
        if config.get("is_survival", False) and config.get("use_censoring_loss", False):
            from tabpfn.model.half_support_bar_distribution import (
                HalfSupportBarDistributionForSurvival,
            )

            loss = HalfSupportBarDistributionForSurvival(
                borders=borders
                # get_bucket_limits(config['num_buckets'], full_range=(-7, 7))
                ,
                ignore_nan_targets=True,
            ).to(device)
        else:
            loss = FullSupportBarDistribution(
                borders=borders
                # get_bucket_limits(config['num_buckets'], full_range=(-7, 7))
                ,
                ignore_nan_targets=True,
            ).to(device)
    return loss


class Losses:
    gaussian = nn.GaussianNLLLoss(full=True, reduction="none")
    mse = nn.MSELoss(reduction="none")
    ce = nn.CrossEntropyLoss(reduction="none")
    bce = nn.BCEWithLogitsLoss(reduction="none")
    get_BarDistribution = BarDistribution
