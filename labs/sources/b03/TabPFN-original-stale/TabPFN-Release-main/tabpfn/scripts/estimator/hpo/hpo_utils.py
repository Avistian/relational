from __future__ import annotations

import os

from tabpfn.scripts.estimator import TabPFNBaseModel, TabPFNModelPathsConfig


def model_from_tabpfn_config(
    *, param: dict, task_type: str, device: str, seed: int
) -> TabPFNBaseModel:
    """Obtain a TabPFN model from a configuration dictionary.

    Args:
        param: configuration dictionary that contains custom keys and values for inference config overwrite.
            Custom keys:
                - `model_path`: str path to the model.
                - `model_type`: type of the model.
                - `n_ensemble_repeats`: number of ensemble repeats (optionally).
                    If set, it forces all kind of ensemble repeats to be equal to the set value.
            Other keys:
                - Keys for a TabPFN config as usable in `inference_config_overwrite`.
        task_type: task type for the model.
        device: device the model is supposed to use.
        seed: random seed for the TabPFN model.
    """
    from tabpfn.best_models import get_best_tabpfn

    custom_hps = ["model_path", "model_type", "n_ensemble_repeats"]
    inference_config_overwrite = {k: v for k, v in param.items() if k not in custom_hps}

    if "n_ensemble_repeats" in param:
        model_type = param["model_type"]
        if model_type == "single":
            inference_config_overwrite["n_estimators"] = param["n_ensemble_repeats"]
        elif model_type == "dt_pfn":
            inference_config_overwrite["rf_pfn_n_estimators"] = param[
                "n_ensemble_repeats"
            ]
        else:
            raise ValueError(f"Unknown model type {param['model_type']}")

    local_model_overwrite = os.environ.get("LOCAL_MODEL_OVERWRITE", default=None)
    if local_model_overwrite == "LOCAL":
        from pathlib import Path

        from tabpfn.local_settings import local_model_path

        model_postfix = "classification" if task_type == "multiclass" else "regression"
        model_path = str(Path(local_model_path) / f"model_hans_{model_postfix}.ckpt")
    else:
        model_path = param["model_path"]

    return get_best_tabpfn(
        task_type,
        paths_config=TabPFNModelPathsConfig(
            paths=[model_path],
            task_type=task_type,
        ),
        inference_config_overwrite=inference_config_overwrite,
        model_type=param["model_type"],
        seed=seed,
        device=device,
    )
