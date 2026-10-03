from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from tabpfn.scripts.estimator.hpo.default_models import get_default_for_task_type
from tabpfn.scripts.estimator.hpo.hpo_utils import model_from_tabpfn_config
from tabpfn.scripts.tabular_baselines.utils import get_random_seed
from tabpfn.scripts.tabular_evaluation_utils import DatasetEvaluation
from tabpfn.scripts.tabular_metrics import get_task_type
from tabpfn.utils import print_once

if TYPE_CHECKING:
    from tabpfn.scripts.estimator import TabPFNModelPathsConfig


def _get_task_type(*, metric_used):
    task_type = get_task_type(metric_used)
    assert task_type in [
        "survival",
        "regression",
        "multiclass",
        "quantile_regression",
    ], f"Metric is {task_type}"

    if task_type == "quantile_regression":
        task_type = "regression"

    return task_type


def _warnings(*, max_time):
    if max_time is not None:
        print_once(
            "Maximum time is not enforced on the transformer metric, it is only used too keep the same interface as the other metrics.",
        )


def _get_model(*, input_model, model_type, paths_config, task_type, device, seed):
    from tabpfn.best_models import get_best_tabpfn

    if input_model is not None:
        if input_model == "default":
            return model_from_tabpfn_config(
                param=get_default_for_task_type(task_type),
                task_type=task_type,
                device=device,
                seed=seed,
            )
        return input_model

    return get_best_tabpfn(
        task_type,
        device=device,
        model_type=model_type,
        seed=seed,
        paths_config=paths_config,
    )


def _determine_old_tabpfn(*, input_model) -> bool:
    old_tabpfn = not hasattr(input_model, "set_categorical_features")
    if old_tabpfn:
        warnings.warn(
            "The old TabPFN model is not informed about categorical features, as it does not support `set_categorical_features`.",
            stacklevel=2,
        )
    return old_tabpfn


def _fit(*, input_model, task_type, train_ds, old_tabpfn):
    if not old_tabpfn:
        input_model.set_categorical_features(train_ds.categorical_feats)

    if task_type == "survival":
        input_model.fit_separate_censoring(
            train_ds.x,
            event_times=train_ds.y,
            censoring=train_ds.event_observed,
        )
    elif old_tabpfn:
        input_model.fit(train_ds.x, train_ds.y, overwrite_warning=True)
    else:
        input_model.fit(train_ds.x, train_ds.y)


def _predict(*, input_model, task_type, test_ds, quantiles, full_predict):
    additional_args = {}
    pred_full = {}

    if task_type == "multiclass":
        # TabPFNClassifier
        pred = input_model.predict_proba(test_ds.x)
    elif task_type == "regression":
        # TabPFNRegressor
        if hasattr(input_model, "predict_full") and (
            quantiles is not None or full_predict
        ):
            pred_full = input_model.predict_full(test_ds.x)
            pred = (
                np.stack([pred_full[f"quantile_{q:.2f}"] for q in quantiles], 1)
                if quantiles is not None
                else pred_full[input_model.get_optimization_mode()]
            )
        else:
            pred = input_model.predict(test_ds.x)
    elif task_type == "survival":
        # TabPFNSurvivalRegressor
        pred_full = input_model.predict_full(test_ds.x)
        pred = pred_full[input_model.get_optimization_mode()]
    else:
        raise NotImplementedError(f"Metric is {task_type}")

    if full_predict:
        additional_args["pred_full"] = pred_full
    return pred, additional_args


def transformer_metric(
    train_ds,
    test_ds,
    metric_used,
    max_time=None,
    random_state=None,
    device="cpu",
    classifier=None,  # keep name classifier for backwards compatibility
    model_type: str = "ensemble",  # ["single_light", "single_fast", "rf_pfn"]
    # Allows to hardcode the used model paths
    paths_config: None | TabPFNModelPathsConfig = None,
    full_predict: bool = False,
    **kwargs,  # unused but required for backwards compatibility
):
    task_type = _get_task_type(metric_used=metric_used)

    _warnings(max_time=max_time)
    seed = get_random_seed(random_state, train_ds.y, mod=False)
    model = _get_model(
        input_model=classifier,
        model_type=model_type,
        paths_config=paths_config,
        task_type=task_type,
        device=device,
        seed=seed,
    )
    old_tabpfn = _determine_old_tabpfn(input_model=model)

    _fit(
        input_model=model,
        task_type=task_type,
        train_ds=train_ds,
        old_tabpfn=old_tabpfn,
    )
    pred, additional_args = _predict(
        input_model=model,
        task_type=task_type,
        test_ds=test_ds,
        quantiles=getattr(metric_used, "quantiles", None),
        full_predict=full_predict,
    )

    return DatasetEvaluation(
        y=None,
        pred=pred,
        additional_args=additional_args,
        algorithm_name="TabPFN",
    )
