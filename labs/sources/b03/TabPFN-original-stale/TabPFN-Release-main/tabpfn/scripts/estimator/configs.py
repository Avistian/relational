from __future__ import annotations

import dataclasses
from functools import lru_cache

from dataclasses import dataclass, field, asdict
from typing import Optional, Union, Tuple, Any, Dict, List, Literal
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
import math
import torch

from tabpfn.local_settings import wandb_entity, get_wandb_project


def get_params_from_config(c):
    return (
        {}  # here you add things that you want to use from the config to do inference in transformer_predict
    )


@dataclass(eq=True, frozen=True)
class PreprocessorConfig:
    """
    Configuration for data preprocessors.

    Attributes:
        name (Literal): Name of the preprocessor.
        categorical_name (Literal): Name of the categorical encoding method. Valid options are "none", "numeric",
                                "onehot", "ordinal", "ordinal_shuffled". Default is "none".
        append_original (bool): Whether to append the original features to the transformed features. Default is False.
        subsample_features (float): Fraction of features to subsample. -1 means no subsampling. Default is -1.
        global_transformer_name (str): Name of the global transformer to use. Default is None.
    """

    name: Literal[
        "per_feature",  # a different transformation for each feature
        "power",  # a standard sklearn power transformer
        "safepower",  # a power transformer that prevents some numerical issues
        "power_box",
        "safepower_box",
        "quantile_uni_coarse",  # different quantile transformations with few quantiles up to a lot
        "quantile_norm_coarse",
        "quantile_uni",
        "quantile_norm",
        "quantile_uni_fine",
        "quantile_norm_fine",
        "robust",  # a standard sklearn robust scaler
        "kdi",
        "none",  # no transformation (inside the transformer we anyways do a standardization)
        "kdi_random_alpha",
        "kdi_uni",
        "kdi_random_alpha_uni",
        "adaptive",
        "norm_and_kdi",
        # KDI with alpha collection
        "kdi_alpha_0.3_uni",
        "kdi_alpha_0.5_uni",
        "kdi_alpha_0.8_uni",
        "kdi_alpha_1.0_uni",
        "kdi_alpha_1.2_uni",
        "kdi_alpha_1.5_uni",
        "kdi_alpha_2.0_uni",
        "kdi_alpha_3.0_uni",
        "kdi_alpha_5.0_uni",
        "kdi_alpha_0.3",
        "kdi_alpha_0.5",
        "kdi_alpha_0.8",
        "kdi_alpha_1.0",
        "kdi_alpha_1.2",
        "kdi_alpha_1.5",
        "kdi_alpha_2.0",
        "kdi_alpha_3.0",
        "kdi_alpha_5.0",
    ]
    categorical_name: Literal[
        "none",
        "numeric",
        "onehot",
        "ordinal",
        "ordinal_shuffled",
        "ordinal_very_common_categories_shuffled",
    ] = "none"
    # categorical_name meanings:
    # "none": categorical features are pretty much treated as ordinal, just not resorted
    # "numeric": categorical features are treated as numeric, that means they are also power transformed for example
    # "onehot": categorical features are onehot encoded
    # "ordinal": categorical features are sorted and encoded as integers from 0 to n_categories - 1
    # "ordinal_shuffled": categorical features are encoded as integers from 0 to n_categories - 1 in a random order
    append_original: bool = False
    subsample_features: Optional[float] = -1
    global_transformer_name: Optional[str] = None
    # if True, the transformed features (e.g. power transformed) are appended to the original features

    def __str__(self):
        return (
            f"{self.name}_cat:{self.categorical_name}"
            + ("_and_none" if self.append_original else "")
            + (
                "_subsample_feats_" + str(self.subsample_features)
                if self.subsample_features > 0
                else ""
            )
            + (
                f"_global_transformer_{self.global_transformer_name}"
                if self.global_transformer_name is not None
                else ""
            )
        )

    def can_be_cached(self):
        return not self.subsample_features > 0

    def to_dict(self):
        return {k: str(v) for k, v in asdict(self).items()}


@dataclass
class EnsembleConfiguration:
    """
    Configuration for an ensemble member.

    Attributes:
        class_shift_configuration (torch.Tensor | None): Permutation to apply to classes. Only used for classification.
        feature_shift_configuration (int | None): Random seed for feature shuffling.
        preprocess_transform_configuration (PreprocessorConfig): Preprocessor configuration to use.
        styles_configuration (int | None): Styles configuration to use.
        subsample_samples_configuration (int | None): Indices of samples to use for this ensemble member.
    """

    class_shift_configuration: torch.Tensor | None = None
    feature_shift_configuration: int | None = None
    preprocess_transform_configuration: PreprocessorConfig = PreprocessorConfig("none")
    styles_configuration: int | None = None
    subsample_samples_configuration: int | None = None


@dataclass
class TabPFNConfig:
    """
    Configuration for TabPFN models.

    Check TabPFNBaseEstimator for more information on attributes.
    """

    task_type: str
    model_type: Literal[
        "best",
        "single",
        "ensemble",
        "single_fast",
        "stacking",
        "bagging",
        "rf_pfn",
        "rf_xgb_pfn",
        "weighted_average",
    ]
    paths_config: TabPFNModelPathsConfig
    task_type_config: TabPFNClassificationConfig | TabPFNRegressionConfig | TabPFNSurvivalConfig | None = (
        None
    )
    model_type_config: StackingConfig | TabPFNRFConfig | BaggingConfig | WeightedAverageConfig | None = (
        None
    )

    model_name: str = "tabpfn"  # This name will be tracked on wandb

    preprocess_transforms: Tuple[PreprocessorConfig, ...] = (
        PreprocessorConfig("safepower", categorical_name="numeric"),
        PreprocessorConfig("power", categorical_name="numeric"),
    )
    feature_shift_decoder: Literal[
        "shuffle", "none", "local_shuffle", "rotate", "auto_rotate"
    ] = "shuffle"  # local_shuffle breaks, because no local configs are generated with high feature number
    normalize_with_test: bool = False
    fp16_inference: bool = True
    n_estimators: int = 1
    average_logits: bool = True
    transformer_predict_kwargs: Optional[Dict] = field(default_factory=dict)
    save_peak_memory: Literal["True", "False", "auto"] = "True"
    batch_size_inference: int = None
    add_fingerprint_features: bool = True
    max_poly_features: int = 50
    use_poly_features: bool = False
    softmax_temperature: float = math.log(0.9)
    subsample_samples: float = -1
    remove_outliers: float = -1

    optimize_metric: Optional[str | None] = None
    model_config: Optional[Dict] = field(default_factory=dict)
    model: Optional[torch.nn.Module] = None

    def to_kwargs(self):
        kwargs = dataclasses.asdict(self)
        del kwargs["task_type"]
        del kwargs["model_type"]

        if self.task_type_config is not None:
            kwargs.update(dataclasses.asdict(self.task_type_config))

        if (
            kwargs.get("paths_config", None) is not None
            and kwargs.get("model", None) is not None
        ):
            raise ValueError(
                "Either paths_config or model must be specified, not both."
            )
        elif kwargs.get("paths_config", None) is not None:
            assert (
                len(kwargs["paths_config"]["model_strings"]) == 1
            ), "Only one model can be used as config for our TabPFNBaseEstimator models."
            kwargs["model_path"] = kwargs["paths_config"]["model_strings"][0]
        elif kwargs.get("model", None) is not None:
            kwargs["model_path"] = "tabpfn"
        else:
            raise ValueError(
                f"Either paths_c"
                f"onfig or model must be specified paths_config {kwargs.get('paths_config', None)} model {kwargs.get('model', None)}"
            )

        del kwargs["paths_config"]
        del kwargs["task_type_config"]
        del kwargs["model_type_config"]
        del kwargs["model_name"]

        return kwargs

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            + ", ".join(f"{k}: {repr(v)[:100]}" for k, v in asdict(self).items())
            + ")"
        )


def get_run_from_wandb(wandb_id, task_type="multiclass"):
    import wandb

    wandb.login()  # b561dc0fd10209df25d4df721d5d8826e36a3631
    api = wandb.Api()

    print(task_type, f"{wandb_entity}/{get_wandb_project(task_type)}/{wandb_id}")

    try:
        run = api.run(f"{wandb_entity}/{get_wandb_project(task_type)}/{wandb_id}")
    except wandb.errors.CommError as e:
        if "not found" in str(e) or "not find" in str(e):
            print("wandb run not found", e)
            raise FileNotFoundError
        else:
            raise e
    return run


@lru_cache()
def get_model_string_from_wandb(wandb_id, task_type="multiclass"):
    run = get_run_from_wandb(wandb_id, task_type)
    try:
        return run.summary.model_path
    except KeyError:
        raise FileNotFoundError


@dataclass
class TabPFNModelPathsConfig:
    """
    paths: list of model paths or WANDB_IDs, e.g. ["/path/to/model.pth", "WANDB_ID:1a2b3c4d"]

    The model_strings attribute is automatically generated from the paths attribute, and contains the actual paths to the models on our cluster.
    If a WANDB_ID is given, the model_string is automatically fetched from WANDB.
    If a path is given, the model_string is the path itself.
    """

    paths: list[str]

    model_strings: list[str] = dataclasses.field(init=False)

    task_type: str = "multiclass"

    def __post_init__(self):
        # Initialize Model paths
        self.model_strings = []

        for path in self.paths:
            if path.startswith("WANDB_ID:"):
                self.model_strings.append(
                    get_model_string_from_wandb(path[9:], self.task_type)
                )
            else:
                self.model_strings.append(path)

    def to_dict(self):
        return dataclasses.asdict(self)


### TASK TYPE CONFIGS ###


@dataclass
class TabPFNClassificationConfig:
    multiclass_decoder: Literal[
        "shuffle", "none", "local_shuffle", "rotate"
    ] = "shuffle"


@dataclass
class TabPFNRegressionConfig:
    regression_y_preprocess_transforms: Tuple[str | None, ...] = (
        None,
        "safepower",
    )
    cancel_nan_borders: bool = True
    super_bar_dist_averaging: bool = False


@dataclass
class TabPFNSurvivalConfig:
    pass


### MODEL TYPE CONFIGS ###


@dataclass
class StackingConfig:
    params_stacked: Tuple[Dict[str, Any], ...]
    cv: int
    append_other_model_types: bool

    final_estimator: ClassifierMixin = LogisticRegression()


@dataclass
class WeightedAverageConfig:
    params_stacked: Tuple[Dict[str, Any], ...]
    cv: int
    n_max: int = 3


@dataclass
class TabPFNRFConfig:
    min_samples_split: int = 1000
    min_samples_leaf: int = 5
    max_depth: int = 5
    splitter: Literal["best", "random"] = "best"
    n_estimators: int = 16
    max_features: Literal["sqrt", "auto"] = "sqrt"
    criterion: Literal[
        "gini", "entropy", "log_loss", "squared_error", "friedman_mse", "poisson"
    ] = "gini"
    preprocess_X: bool = False
    preprocess_X_once: bool = False
    adaptive_tree: bool = True
    fit_nodes: bool = True
    adaptive_tree_overwrite_metric: Literal["logloss", "roc"] = None
    adaptive_tree_test_size: float = 0.2
    adaptive_tree_min_train_samples: int = 100
    adaptive_tree_min_valid_samples_fraction_of_train: int = 0.2
    adaptive_tree_max_train_samples: int = 5000
    adaptive_tree_skip_class_missing: bool = True
    max_predict_time: float = -1

    bootstrap: bool = True
    rf_average_logits: bool = False
    dt_average_logits: bool = True


@dataclass
class BaggingConfig:
    n_estimators: int = 32
    max_samples: [float | int] = 2048
    max_features: [float | int] = 1.0
    bootstrap: bool = True
    bootstrap_features: bool = False
