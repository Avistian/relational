from __future__ import annotations

CLF_DEFAULT = {
    "model_type": "single",
    # Overwrite
    "n_ensemble_repeats": 4,
    "batch_size_inference": 1,
    "save_peak_memory": "True",
    "fp16_inference": True,
}

REG_DEFAULT = {
    "model_type": "single",
    # Overwrite
    "n_ensemble_repeats": 4,
    "batch_size_inference": 1,
    "save_peak_memory": "True",
    "fp16_inference": True,
}


def get_default_for_task_type(task_type: str):
    from copy import deepcopy
    from pathlib import Path

    from tabpfn.local_settings import local_model_path

    if task_type == "multiclass":
        config = deepcopy(CLF_DEFAULT)
        config["model_path"] = str(
            Path(local_model_path) / "model_hans_classification.ckpt"
        )
    elif task_type == "regression":
        config = deepcopy(REG_DEFAULT)
        config["model_path"] = str(
            Path(local_model_path) / "model_hans_regression.ckpt"
        )
    else:
        raise ValueError(f"Unknown task type {task_type} for a default model!")

    return config
