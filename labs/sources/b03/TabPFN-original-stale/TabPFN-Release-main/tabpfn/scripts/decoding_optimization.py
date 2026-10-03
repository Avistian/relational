import typing as tp
import time

from tabpfn.scripts.runner import TabPFNTrainingAgent
from tabpfn.scripts.estimator import get_model_from_wandb
from tabpfn.local_settings import base_path, get_wandb_project
from tabpfn.local_settings import wandb_entity


def evaluate_model(
    task_type: tp.Literal["multiclass", "regression", "binary"],
    device: tp.Literal["cpu", "cuda"],
    split: tp.Literal["debug", "test", "valid"],
    debug: bool = False,
    wandb_id: str = None,
    inference_config_overwrite: dict = None,
    final_evaluation_model_types: list[str] = None,
    tags: list[str] = None,
    n_splits: int = 10,
    wandb_mode: str = "online",
    base_path_local: str = base_path,
):
    """
    Evaluate a model on the benchmark given the provided hyperparameters (in inference_config_overwrite)
    for decoding using decoders in final_evaluation_model_types.
    """
    tags = tags or []
    model, c = get_model_from_wandb(task_type, device=device, wandb_id=wandb_id)

    c["wandb_run_id"] = wandb_id

    config = {
        "task_type": task_type,
        "device": device,
        "wandb_project": get_wandb_project(task_type),  # Log in Wandb to test project
        "wandb_entity": wandb_entity,
        "base_path": base_path_local,  # Save models in tmp
        "model_string": "eval_" + str(time.time()),
        "config_sample": c,
        "mode": wandb_mode,
        "debug": debug,
        "wandb_tags": tags,
        "save_evaluation_results": True,
    }

    agent = TabPFNTrainingAgent(**config)
    agent.full_evaluation(
        model=model,
        split=split,
        n_splits=n_splits,
        model_types=final_evaluation_model_types,
        inference_config_overwrite=inference_config_overwrite,
    )
