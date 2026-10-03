from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from tabpfn.scripts.estimator.configs import (
    EnsembleConfiguration,
    PreprocessorConfig,
)
from tabpfn.scripts.estimator.post_hoc_ensembles.abstract_validation_utils import (
    AbstractValidationUtilsClassification,
    AbstractValidationUtilsRegression,
)
from tabpfn.scripts.estimator.preprocessing import (
    ReshapeFeatureDistributionsStep,
)

if TYPE_CHECKING:
    from tabpfn.scripts.estimator.base import TabPFNBaseModel


def evaluate_configurations(
    configurations: list[tuple[str, TabPFNBaseModel | EnsembleConfiguration]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    score_metric: str | None = None,
    classification: bool = False,
    n_folds: int = 5,
    n_repeats: int = 1,
    use_tabpfn_batching: bool = False,
    tabpfn_model: TabPFNBaseModel | None = None,
) -> tuple[list[np.ndarray], list[float]]:
    """Evaluate a list of configurations by calculating the loss of each TabPFNBaseModel."""
    ensUtils = (
        AbstractValidationUtilsClassification
        if classification
        else AbstractValidationUtilsRegression
    )
    ensUtils = ensUtils(
        estimators=configurations,
        n_folds=n_folds,
        n_repeats=n_repeats,
        seed=15423,
        score_metric=score_metric,
        tabpfn_model=tabpfn_model,
        tabpfn_batching=use_tabpfn_batching,
    )
    (
        oof_proba_per_configurations,
        loss_per_configurations,
    ) = ensUtils.get_oof_per_estimator(X_train, y_train, return_loss_per_estimator=True)

    return oof_proba_per_configurations, loss_per_configurations


def _get_oof_for_model(
    input_model: TabPFNBaseModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    classification: bool,
    n_folds: int,
    n_repeats: int,
) -> tuple[np.ndarray, dict]:
    ensUtils = (
        AbstractValidationUtilsClassification
        if classification
        else AbstractValidationUtilsRegression
    )
    tabpfn_model = copy.copy(input_model)

    # Workaround for regression with internal classification
    _extra_processing = False
    if ensUtils == AbstractValidationUtilsRegression:
        # need to load model to figure out the number of bars
        tmp_tabpfn_model = copy.copy(input_model)
        tmp_tabpfn_model.init_model_and_get_model_config()
        n_bars = tmp_tabpfn_model.model_processed_.criterion.num_bars

        class _AbstractValidationUtilsRegression(AbstractValidationUtilsRegression):
            fold_extra_processing: dict = {}

            def _predict_oof(self, base_model, X):
                return base_model.predict_full(X)["logits"]

            def _batch_predict_oof(self, X, configurations):
                return [
                    p["logits"]
                    for p in self.tabpfn_model.custom_batch_predict_full(
                        X=X,
                        configurations=configurations,
                    )
                ]

            def _proba_template(self, X, y):
                return np.full((X.shape[0], n_bars), 0, dtype=float)

            def _extra_processing(self, fold_i, in_tabpfn_model, indices):
                self.fold_extra_processing[fold_i] = (
                    indices,
                    copy.copy(in_tabpfn_model._predict_criterion),
                )

        _extra_processing = True
        ensUtils = _AbstractValidationUtilsRegression

    ensUtils = ensUtils(
        estimators=[("_", tabpfn_model)],
        n_folds=n_folds,
        n_repeats=n_repeats,
        seed=15423,
    )
    y_val_probs = ensUtils.get_oof_per_estimator(
        X_train,
        y_train,
        _extra_processing=_extra_processing,
    )[0]

    _extra_processing_data = (
        ensUtils.fold_extra_processing if _extra_processing else None
    )
    return y_val_probs, _extra_processing_data


def tune_preprocessing(
    input_model: TabPFNBaseModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    classification: bool = False,
    score_metric: str | None = None,
    n_folds: int = 5,
    n_repeats: int = 1,
    selection_size: int = 4,
    return_oof_for_selection: bool = False,
    potential_preprocessors: list[PreprocessorConfig] | None = None,
    use_tabpfn_batching: bool = False,
) -> list[PreprocessorConfig] | tuple[list[PreprocessorConfig], list[np.ndarray]]:
    """Tune the preprocessing steps of a model by evaluating the model with each preprocessing step.

    Args:
        input_model: model to tune
        X_train: training data
        y_train: training labels
        classification: whether the task is classification or not
        n_folds: number of folds for getting OOF
        n_repeats: number of repeats for getting OOF
        selection_size: number of top preprocessing steps to return
        return_oof_for_selection: whether to return the OOF predictions for the selected preprocessing steps
        potential_preprocessors: list of potential preprocessing steps to evaluate. If None, all available preprocessing steps are used.
        use_tabpfn_batching: whether to use the TabPFN batching for evaluation or not.

    Returns: top selected preprocessing steps (and their OOF predictions if return_oof_for_selection is True)
    """
    if potential_preprocessors is None:
        potential_preprocessors: list[PreprocessorConfig] = [
            PreprocessorConfig(m, categorical_name="numeric")
            for m in list(
                ReshapeFeatureDistributionsStep.get_all_preprocessors(10).keys(),
            )
        ]

    ests = []
    if use_tabpfn_batching:
        for pre in potential_preprocessors:
            ests.append(
                (
                    f"tabpfn_{pre.name}",
                    EnsembleConfiguration(preprocess_transform_configuration=pre),
                ),
            )
    else:
        for pre in potential_preprocessors:
            model = copy.copy(input_model)
            model.preprocess_transforms = [pre]
            ests.append((f"tabpfn_{pre.name}", model))

    oof_proba_per_preprocessor, loss_per_preprocessor = evaluate_configurations(
        ests,
        X_train,
        y_train,
        classification=classification,
        n_folds=n_folds,
        n_repeats=n_repeats,
        score_metric=score_metric,
        use_tabpfn_batching=use_tabpfn_batching,
        tabpfn_model=input_model if use_tabpfn_batching else None,
    )
    # print(loss_per_preprocessor)

    sel_list = [
        potential_preprocessors[sel_i]
        for sel_i in np.argsort(loss_per_preprocessor)[:selection_size]
    ]

    if return_oof_for_selection:
        return sel_list, [
            oof_proba_per_preprocessor[sel_i]
            for sel_i in np.argsort(loss_per_preprocessor)[:selection_size]
        ]

    return sel_list


def tune_softmax_temperature(
    input_model: TabPFNBaseModel,
    *,
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    input_OOF_predictions: np.ndarray | None = None,
    input_y_val: np.ndarray | None = None,
    input_extra_processing_data: dict | None = None,
    classification: bool = False,
    n_folds: int = 5,
    n_repeats: int = 1,
    init_val: float = 1,
    max_iter: int = 1000,
    lr: float = 0.01,
    return_log_scaled: bool = True,
    return_tuned_oof: bool = False,
) -> float | tuple[float, np.ndarray]:
    """Tune the softmax temperature scalar term of a model on OOF predictions.

    This code follows AutoGluon's implementation of temperature scaling.
    Does not support a custom metric as we use gradient-based optimization.

    Args:
        input_model: the model to tune
        X_train: training data
        y_train: training labels
        input_OOF_predictions: OOF predictions of the model instead of X_train and y_train!
        input_y_val: test labels if OOF_predictions is not None
        input_extra_processing_data: extra processing data if OOF_predictions is not None
        classification: whether the task is classification or not
        n_folds: number of folds for getting OOF, only used if X_train and y_train are provided.
        n_repeats: number of repeats for getting OOF, only used if X_train and y_train are provided.
        init_val: initial value of the temperature scalar
        max_iter: maximum number of iterations to tune
        lr: learning rate of the tuning algorithm
        return_log_scaled: whether this function returns the raw temperature scalar term or the log of it.

    Returns: the temperature scalar term
    """
    import torch  # lazy import

    if (X_train is not None) and (y_train is not None):
        y_val_probs, _extra_processing_data = _get_oof_for_model(
            input_model=input_model,
            X_train=X_train,
            y_train=y_train,
            classification=classification,
            n_folds=n_folds,
            n_repeats=n_repeats,
        )
        y_val = y_train
    elif (input_OOF_predictions is not None) and (input_y_val is not None):
        y_val_probs = input_OOF_predictions
        y_val = input_y_val
        _extra_processing_data = input_extra_processing_data
    else:
        raise ValueError(
            "Either X_train and y_train xor OOF_predictions and y_test must be provided!",
        )

    temperature_param = torch.nn.Parameter(
        torch.ones(1).fill_(init_val).type(torch.float64),
    )
    y_val_probs = y_val_probs.astype(np.float64)
    if classification:
        y_val = y_val.astype(np.int64)
        y_val_probs = y_val_probs + 1e-15  # Avoid inf values in logits
        y_val_probs = np.log(y_val_probs)
    else:
        # We already have logits for regression
        y_val = y_val.astype(np.float64)
    logits = torch.tensor(y_val_probs)
    y_val_tensor = torch.tensor(y_val)

    nll_criterion = (
        torch.nn.CrossEntropyLoss() if classification else torch.nn.MSELoss()
    )
    optimizer = torch.optim.LBFGS([temperature_param], lr=lr, max_iter=max_iter)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    def _get_new_logits() -> torch.Tensor:
        temp = temperature_param.unsqueeze(1).expand(logits.size(0), logits.size(1))

        # Get new predictions
        new_logits = logits / temp

        if _extra_processing_data is not None:
            # Account for per-fold criterion behavior
            y_pred = torch.zeros((logits.shape[0],), dtype=new_logits.dtype)
            for indices, criterion in _extra_processing_data.values():
                y_pred[indices] = criterion.mean(
                    new_logits[indices].type(criterion.borders.dtype),
                ).type(new_logits.dtype)
        else:
            y_pred = new_logits

        return y_pred

    def temperature_scale_step():
        optimizer.zero_grad()
        y_pred = _get_new_logits()
        loss = nll_criterion(y_pred, y_val_tensor)
        loss.backward()
        scheduler.step()
        # print(temp[0][0], loss)
        return loss

    optimizer.step(temperature_scale_step)

    tuned_oof = (
        np.exp(_get_new_logits().detach().numpy()) - 1e-15
        if classification
        else _get_new_logits().detach().numpy()
    )
    temperature_scale = temperature_param.item()
    if np.isnan(temperature_scale):
        print("Warning: NaN temperature scale value, returning init values!")
        # fallback to default
        temperature_scale = init_val
        tuned_oof = np.exp(y_val_probs) - 1e-15 if classification else y_val_probs

    if return_log_scaled:
        temperature_scale = np.log(temperature_scale)

    if return_tuned_oof:
        return temperature_scale, tuned_oof

    return temperature_scale
