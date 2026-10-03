from __future__ import annotations

import math
from pathlib import Path

from hyperopt import hp

from tabpfn.local_settings import local_model_path


def enumerate_preprocess_transforms():
    from tabpfn.scripts.estimator.configs import PreprocessorConfig

    transforms = []
    for names in [
        ["safepower"],
        ["quantile_uni_coarse"],
        ["quantile_norm_coarse"],
        ["adaptive"],
        ["norm_and_kdi"],
        ["quantile_uni"],
        ["none"],
        ["robust"],
        ["kdi_uni"],
        ["kdi_alpha_0.3"],
        ["kdi_alpha_3.0"],
        ["safepower", "quantile_uni"],
        ["kdi", "quantile_uni"],
        ["none", "power"],
    ]:
        for categorical_name in [
            "numeric",
            "ordinal_very_common_categories_shuffled",
            "onehot",
            "none",
        ]:
            for append_original in [True, False]:
                for subsample_features in [-1, 0.99, 0.95, 0.9]:
                    for global_transformer_name in [None, "svd"]:
                        transforms += [
                            [
                                PreprocessorConfig(
                                    name,
                                    global_transformer_name=global_transformer_name,
                                    subsample_features=subsample_features,
                                    categorical_name=categorical_name,
                                    append_original=append_original,
                                )
                                for name in names
                            ],
                        ]
    return transforms


def get_param_grid_hyperopt(task_type: str) -> dict:
    search_space = {
        # Custom HPs
        "model_type": hp.choice("model_type", ["single", "dt_pfn"]),
        "n_ensemble_repeats": hp.choice("n_ensemble_repeats", [4]),
        # -- inference_config_overwrite HPs
        "average_logits": hp.choice("average_logits", [True, False]),
        "add_fingerprint_features": hp.choice(
            "add_fingerprint_features",
            [True, False],
        ),
        "softmax_temperature": hp.choice(
            "softmax_temperature",
            [
                math.log(0.75),
                math.log(0.8),
                math.log(0.9),
                math.log(0.95),
                math.log(1.0),
            ],
        ),
        "preprocess_transforms": hp.choice(
            "preprocess_transforms",
            enumerate_preprocess_transforms(),
        ),
        "use_poly_features": hp.choice("use_poly_features", [True, False]),
        "remove_outliers": hp.choice("remove_outliers", [-1, 7.0, 9.0, 12.0]),
        "subsample_samples": hp.choice("subsample_samples", [0.99, -1]),
        # Hardcoded config
        "max_poly_features": hp.choice("max_poly_features", [50]),
        "feature_shift_decoder": hp.choice("feature_shift_decoder", ["shuffle"]),
        "fp16_inference": hp.choice("fp16_inference", [True]),
        "batch_size_inference": hp.choice("batch_size_inference", [1]),
        "save_peak_memory": hp.choice("save_peak_memory", ["True"]),
    }

    local_dir = Path(local_model_path).resolve()

    if task_type == "multiclass":
        search_space["multiclass_decoder"] = hp.choice(
            "multiclass_decoder", ["shuffle"]
        )

        model_paths = [
            str(local_dir / "model_hans_classification.ckpt"),
            str(local_dir / "model_hans_classification_od3j1g5m.ckpt"),
            str(local_dir / "model_hans_classification_gn2p4bpt.ckpt"),
            str(local_dir / "model_hans_classification_znskzxi4.ckpt"),
            str(local_dir / "model_hans_classification_llderlii.ckpt"),
            str(local_dir / "model_hans_classification_vutqq28w.ckpt"),
        ]
    elif task_type == "regression":
        model_paths = [
            str(local_dir / "model_hans_regression_09gpqh39.ckpt"),
            str(local_dir / "model_hans_regression.ckpt"),
            str(local_dir / "model_hans_regression_2noar4o2.ckpt"),
            str(local_dir / "model_hans_regression_wyl4o83o.ckpt"),
            str(local_dir / "model_hans_regression_5wof9ojf.ckpt"),
        ]
        search_space["regression_y_preprocess_transforms"] = hp.choice(
            "regression_y_preprocess_transforms",
            [
                (None,),
                (None, "power"),
                ("power",),
                ("safepower",),
                ("adaptive",),
                ("kdi_alpha_0.3",),
                ("kdi_alpha_1.0",),
                ("kdi_alpha_1.5",),
                ("kdi_alpha_0.6",),
                ("kdi_alpha_3.0",),
                ("quantile_uni",),
            ],
        )
    else:
        raise ValueError(f"Unknown task type {task_type} for the search space!")

    search_space["model_path"] = hp.choice("model_path", model_paths)

    return search_space
