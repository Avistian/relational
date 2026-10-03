from __future__ import annotations

import logging
import time
import warnings

import numpy as np
from scipy.linalg._decomp_update import LinAlgError

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

import contextlib
from functools import partial

import hyperopt
import pandas as pd
from hyperopt import Trials, fmin, rand, space_eval
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from ..tabular_evaluation_utils import DatasetEvaluation
from ..tabular_metrics import QuantileMetric, get_scoring_direction, get_task_type

CV = 5
MULTITHREAD = 8  # Number of threads baselines are able to use at most


def get_random_seed(random_state, y, mod=True):
    """Generates a random seed based on the random_state and the sum
    of the labels. This is to ensure that the seed is different for
    different splits of the same dataset.
    """
    # TODO: @Noah, is the try-except due to possible nans? If so, we should
    # document it.
    try:
        seed = random_state + int(y[:].sum())
    except:
        seed = random_state

    # This is done in order to not change previous results.
    # In some parts of the code the mod operation was applied,
    # while in others it was not.
    # np.random.RandomState would support seeds in [0, 2**32 - 1]
    if mod:
        seed = seed % 10000

    # Only fix after the fact to avoid breaking random seeds that worked previously
    if seed and seed < 0:
        return get_random_seed(random_state, abs(y), mod=mod)

    return seed


def eval_f(params, clf_, x, y, metric_used, use_metric_as_scorer=False, verbose=False):
    scoring = (
        metric_used
        if use_metric_as_scorer
        else make_scorer(
            metric_used,
            needs_proba=get_task_type(metric_used) == "multiclass",
            greater_is_better=get_scoring_direction(metric_used) == 1,
        )  # get_scoring_string(metric_used, usage="sklearn_cv")
    )
    if verbose:
        print(
            f"Starting cross-validation with parameters {params} and scoring {scoring}.",
        )

    try:
        scores = cross_val_score(
            clf_(**params),
            x,
            y,
            n_jobs=1,  # The algorithms use all cores/GPUs available. Large values for n_jobs result in very inefficient fitting.
            cv=CV,
            scoring=scoring,
            error_score="raise",  # to avoid crashing a fold resulting in a better mean performance.
        )
    except Exception as e:
        # Print error to stdout to also be able to find the relationship between
        # tried parameters and the error itself
        print(f"Encountered exception during cross-validation: {e!s}")
        print(f"The error occurred while using these parameters: {params}")
        raise e

    mean_score = np.nanmean(scores)

    if verbose:
        print(
            f"Cross-validation with parameters {params} yielded a score of {mean_score}. Complete score list: {scores}",
        )

    return mean_score * -1


def _cost_fn(params, clf_, x, y, metric_used, use_metric_as_scorer, verbose):
    """Top-level wrapper to support pickling of the cost function for hyperopt."""
    return eval_f(
        params,
        clf_,
        x,
        y,
        metric_used,
        use_metric_as_scorer=use_metric_as_scorer,
        verbose=verbose,
    )


def _hyperopt_random_state(*, rstate):
    return (
        np.random.RandomState(rstate)
        if hyperopt.__version__ in ("0.2.4", "0.2.5")
        else np.random.default_rng(rstate)
    )


def _init_ray() -> None | object:
    """Try to init ray to use for HPO."""
    try:
        import os

        import psutil
        import ray
        import torch
    except ImportError:
        print(
            "Ray or its dependencies is not installed. Not using robust timeout and fitting for hyperopt.",
        )
        return None, None, None

    num_cpus = (
        psutil.cpu_count(logical=True)
        if os.name == "nt"
        else len(os.sched_getaffinity(0))
    )
    num_gpus = torch.cuda.device_count()
    ray_mem_in_b = int(int(os.environ.get("RAY_MEM_IN_GB", default=8)) * (1024.0**3))

    print(f"Found {num_cpus} CPUs and {num_gpus} GPUs.")

    if not ray.is_initialized():
        ray_args = dict(
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            address="local",
            _memory=ray_mem_in_b,
            object_store_memory=ray_mem_in_b,
            include_dashboard=False,
            logging_level=logging.INFO,
            log_to_driver=True,
        )
        if os.environ.get("TABPFN_CLUSTER_SETUP") == "NEMO":
            ray_args["_temp_dir"] = os.environ.get("TMPDIR")
        elif os.environ.get("RAY_INIT_TMPDIR", False):
            from datetime import datetime

            ts = (
                str(time.time())
                + "_"
                + str(
                    np.random.RandomState(int(datetime.now().timestamp())).randint(
                        0,
                        9,
                    ),
                )
            )

            ray_args["_temp_dir"] = f"/tmp_ray/hpo_{ts}"
        ray.init(**ray_args)

    return ray, num_cpus, num_gpus


def _get_eval_func_for_hpo(
    *,
    clf_,
    x,
    y,
    metric_used,
    use_metric_as_scorer,
    verbose,
    start_time,
    max_time,
    method_name,
):
    """Get the eval function for hyperopt.

    If ray is installed, we use a robust function that is able to handle a lot of errors. Otherwise, fallback to the normal eval_f function.
    """
    _ray, num_cpus, num_gpus = _init_ray()

    if _ray is None:
        return (
            lambda params: eval_f(
                params,
                clf_,
                x,
                y,
                metric_used,
                use_metric_as_scorer=use_metric_as_scorer,
                verbose=verbose,
            ),
            None,
        )

    _fn = partial(
        _cost_fn,
        clf_=clf_,
        x=x,
        y=y,
        metric_used=metric_used,
        use_metric_as_scorer=use_metric_as_scorer,
        verbose=verbose,
    )

    # Build function
    def fn(*args, **kwargs):
        rest_time = int(max_time - (time.time() - start_time))

        timeout_reached = False

        if rest_time > 1:
            try:
                remote_fn = _ray.remote(max_calls=1, max_retries=1)(_cost_fn)
                ref = remote_fn.options(num_cpus=num_cpus, num_gpus=num_gpus).remote(
                    *args,
                    clf_=clf_,
                    x=x,
                    y=y,
                    metric_used=metric_used,
                    use_metric_as_scorer=use_metric_as_scorer,
                    verbose=verbose,
                    **kwargs,
                )
                finished, unfinished = _ray.wait(
                    [ref],
                    num_returns=1,
                    # time left with a small overhead of 5s to avoid re-queuing the task before hyperopt times out.
                    timeout=rest_time,
                )
                if finished:
                    res = _ray.get(finished[0])
                else:
                    for f in finished + unfinished:
                        with contextlib.suppress(_ray.exceptions.TaskCancelledError):
                            _ray.cancel(f)
                            time.sleep(5)
                    timeout_reached = True
            except _ray.exceptions.WorkerCrashedError as ex:
                print(f"WORKER CRASHED: {ex}")
                print(
                    "CONFIG FAILED DUE TO UNKNOWN REASON (LIKELY KILLED DUE TO OOM FROM OS).",
                )
                timeout_reached = False
                res = {
                    "status": hyperopt.STATUS_FAIL,
                    "failure": "MemOut",
                }
            except Exception as e:
                # -> fail the trial instead of crashing the whole process if we know the edge case problem

                if method_name == "catboost" and str(e).endswith(
                    "Too few sampling units (subsample=0.8, bootstrap_type=MVS): please increase sampling rate or disable sampling",
                ):
                    # Specific workaround for https://github.com/catboost/catboost/issues/2555
                    warnings.warn(
                        f"Expected CatBoost crash happened with {e!s}. Failing trial instead of HPO.",
                        stacklevel=2,
                    )
                    timeout_reached = False
                    res = {
                        "status": hyperopt.STATUS_FAIL,
                        "failure": "CatBoostBug",
                    }
                elif isinstance(e, ValueError) and str(e).endswith(
                    "Number of classes in y_true not equal to the number of columns in 'y_score'",
                ):
                    # Specific workaround for cross-validation from sklearn not working correctly.
                    warnings.warn(
                        f"Sklearn cross-validation failed with {e!s}. Failing trial instead of HPO.",
                        stacklevel=2,
                    )
                    timeout_reached = False
                    res = {
                        "status": hyperopt.STATUS_FAIL,
                        "failure": "SklearnCVBug",
                    }
                elif (
                    method_name == "linear_quantile"
                    and isinstance(e, TypeError)
                    and str(e).endswith(
                        "params = solution[:n_params] - solution[n_params : 2 * n_params]\nTypeError: 'NoneType' object is not subscriptable",
                    )
                ):
                    # Specific workaround for Linear programming for QuantileRegressor crashing due Numerical difficulties encountered resulting
                    # from the alpha value. Fail the trial instead of HPO.
                    warnings.warn(
                        f"Linear_quantile model failed with {e!s}. Failing trial instead of HPO.",
                        stacklevel=2,
                    )
                    timeout_reached = False
                    res = {
                        "status": hyperopt.STATUS_FAIL,
                        "failure": "LinearProgQuantileRegressorBug",
                    }
                elif method_name == "xgb":
                    from xgboost.core import XGBoostError

                    if isinstance(e, XGBoostError) and (
                        "adaptive.cc:131: Check failed: h_row_set.empty()" in str(e)
                    ):
                        # workaround for edge case quantile regression bug in xgboost
                        warnings.warn(
                            f"XGB model failed with {e!s}. Failing trial instead of HPO.",
                            stacklevel=2,
                        )
                        timeout_reached = False
                        res = {
                            "status": hyperopt.STATUS_FAIL,
                            "failure": "XGBQuantileRegressorBug",
                        }
                    elif (
                        isinstance(e, ValueError)
                        and "mean_pinball_loss" in str(e)
                        and str(e).endswith("ValueError: Input contains NaN.")
                    ):
                        # workaround for edge case where extreme parameters make predictions NaN in xgboost
                        warnings.warn(
                            f"XGB model failed with {e!s}. Failing trial instead of HPO.",
                            stacklevel=2,
                        )
                        timeout_reached = False
                        res = {
                            "status": hyperopt.STATUS_FAIL,
                            "failure": "XGBQuantileRegressorBug",
                        }
                    else:
                        raise e
                else:
                    raise e
        else:
            timeout_reached = True

        if timeout_reached:
            print("TERMINATING DUE TO TIME-OUT.")
            res = {
                "status": hyperopt.STATUS_FAIL,
                "failure": "TimeOut",
            }
            time.sleep(1)  # wait for hyperopt

        return res

    return fn, _ray


def _run_hpo(
    clf_,
    x,
    y,
    max_time: int,
    random_state,
    metric_used,
    use_metric_as_scorer,
    verbose,
    key,
    param_grid,
    default_param: None = None,
):
    # Init args
    failed = False
    trials = Trials()
    rstate = get_random_seed(random_state, y)
    rstate_gen = _hyperopt_random_state(rstate=rstate)
    default_param = default_param if default_param is not None else {}
    best = default_param

    start_time = time.time()

    print("Start HPO Search.")
    print(f"Total time for hpo: {int(max_time-(time.time()-start_time))}")

    fn, _ray = _get_eval_func_for_hpo(
        clf_=clf_,
        x=x,
        y=y,
        metric_used=metric_used,
        use_metric_as_scorer=use_metric_as_scorer,
        verbose=verbose,
        start_time=start_time,
        max_time=max_time,
        method_name=key,
    )

    print("Fit Default")
    default = fn(default_param)
    if isinstance(default, dict):
        print(
            f"Default config ran into {default['failure']}. Failed to train anything!",
        )
        if isinstance(metric_used, QuantileMetric) and (
            default["failure"] == "TimeOut"
        ):
            print(
                "Default config failed due to timeout. Using default params as fallback and skip tuning.",
            )
            best = default_param
        else:
            failed = True
    elif np.isnan(default):
        # In case no CV fold was found that satisfies the split criteria e.g.
        # the classes in train and validation are equal, default returns
        # a nan result. This would lead to fmin() failing with the error
        # hyperopt.exceptions.AllTrialsFailed
        # Reproducibility: analcatdata_marketing fails consistently on
        # split_number 3 as in the overall train portion only 1 sample
        # belongs to class 0, so no split exists that will satisfy our
        # conditions.
        # In that case, we just skip the hpo and use the default params.
        print(
            f"HPO on method {key} failed as no CV split satisfying the conditions was found. Using default params.",
        )
    else:
        try:
            print("Start Hyperopt")

            hpo_best = fmin(
                fn=fn,
                space=param_grid,
                algo=rand.suggest,
                rstate=rstate_gen,
                trials=trials,
                timeout=max(int(max_time - (time.time() - start_time)), 1),
                verbose=True,
                show_progressbar=False,  # The seed is deterministic but varies for each dataset and each split of it
                max_evals=10000,
            )
            valid_trial_losses = [
                t["result"]["loss"]
                for t in trials.trials
                if t["result"]["status"] == hyperopt.STATUS_OK
            ]
            print(valid_trial_losses)
            if valid_trial_losses:
                best_score = np.min(valid_trial_losses)

                # Only use the parameters in case they are better than the default.
                if best_score < default:
                    best = space_eval(param_grid, hpo_best)

            if verbose:
                print("<=========================== HPO ===========================>")
                print(
                    f"Number of hpo configurations evaluated: {len(valid_trial_losses)}",
                )
                print(f"Best parameters: {best}")
                print("<===========================================================>")
        except hyperopt.exceptions.AllTrialsFailed:
            print("HPO failed. Using default params.")

    if _ray is not None:
        _ray.shutdown()

    return best, failed


def _fallback_predictions(metric_used, x, y, test_x, additional_args, method_name):
    print("Failed to train model, returning default predictions!")
    if get_task_type(metric_used) == "survival":
        pred = np.ones(test_x.shape[0])
        return DatasetEvaluation(
            y=None,
            pred=pred,
            additional_args=additional_args,
            algorithm_name=method_name,
        )

    if get_task_type(metric_used) == "multiclass":
        from sklearn.dummy import DummyClassifier

        pred = DummyClassifier(strategy="most_frequent").fit(x, y).predict_proba(test_x)
        return DatasetEvaluation(
            y=None,
            pred=pred,
            additional_args=additional_args,
            algorithm_name=method_name,
        )
    if get_task_type(metric_used) == "regression":
        # regression
        from sklearn.dummy import DummyRegressor

        pred = DummyRegressor(strategy="mean").fit(x, y).predict(test_x)
        return DatasetEvaluation(
            y=None,
            pred=pred,
            additional_args=additional_args,
            algorithm_name=method_name,
        )

    if isinstance(metric_used, QuantileMetric):
        raise NotImplementedError(
            "Fallback predictions for quantile regression is not implemented.",
        )

    raise ValueError(f"Unknown task type: {get_task_type(metric_used)}")


def eval_complete_f(
    x,
    y,
    test_x,
    key,
    param_grid,
    clf_,
    metric_used,
    max_time,
    no_tune: dict | None,  # if None, do HPO. If dict, use dict as default parameters
    random_state,
    use_metric_as_scorer=False,
    verbose: bool = True,
    method_name=None,
    default_param: None | dict = None,
) -> DatasetEvaluation:
    if no_tune is None:
        best, failed = _run_hpo(
            clf_=clf_,
            x=x,
            y=y,
            max_time=max_time,
            random_state=random_state,
            metric_used=metric_used,
            use_metric_as_scorer=use_metric_as_scorer,
            verbose=verbose,
            key=key,
            param_grid=param_grid,
            default_param=default_param,
        )
    else:
        best, failed = no_tune.copy(), False

    start = time.time()
    if not failed:
        clf = clf_(**best)
        try:
            clf.fit(x, y)
        except (
            LinAlgError
        ) as e:  # This can happen for linear models if the data is too ill-conditioned
            print(f"Encountered LinAlgError during fit: {e!s}")
            print(f"The error occurred while using these parameters: {best}")
            failed = True
        except ValueError as e:
            print(f"Encountered ValueError during fit: {e!s}")
            print(f"The error occurred while using these parameters: {best}")
            failed = True
        except AssertionError:
            print("Encountered AssertionError during fit.")
            print(f"The error occurred while using these parameters: {best}")
            failed = True

    fit_time = time.time() - start
    additional_args = {"best_config": best, "failed": failed, "fit_time": fit_time}

    if failed:
        return _fallback_predictions(
            metric_used=metric_used,
            x=x,
            y=y,
            test_x=test_x,
            additional_args=additional_args,
            method_name=method_name,
        )

    start = time.time()
    pred = (
        clf.predict_proba(test_x)
        if get_task_type(metric_used) == "multiclass"
        else clf.predict(test_x)
    )

    inference_time = time.time() - start
    additional_args["inference_time"] = inference_time

    if isinstance(metric_used, QuantileMetric):
        additional_args["quantiles"] = tuple(metric_used.quantiles)

    return DatasetEvaluation(
        y=None,
        pred=pred,
        additional_args=additional_args,
        algorithm_name=method_name,
    )


def make_pd_from_np(
    x: np.ndarray,
    cat_features: list[str],
    is_train: bool = True,
    remove_high_cardinality: bool = False,
):
    data = pd.DataFrame(x)
    # we go in reverse order to keep the indexes in `cat_features` correct
    for c in reversed(cat_features):
        if (
            is_train
            and remove_high_cardinality
            and (len(np.unique(data.iloc[:, c])) > 20)
        ):
            cat_features.remove(c)
            continue
        data[c] = data[c].astype("category")
    return data, cat_features


def preprocess_and_impute(
    x,
    y,
    test_x,
    impute,
    one_hot,
    standardize,
    attribute_names,
    cat_features=None,
    is_classification=True,
    return_pandas=False,
    onehot_drop=None,
):
    import warnings

    # ignore all warnings
    if cat_features is None:
        cat_features = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if is_classification:
            y = y.long()

        x, y, test_x = (
            x.cpu().numpy(),
            y.cpu().numpy(),
            test_x.cpu().numpy(),
        )

        if impute:
            imp_mean = SimpleImputer(
                missing_values=np.nan,
                strategy="mean",
                keep_empty_features=True,
            )
            imp_mean.fit(x)
            x, test_x = imp_mean.transform(x), imp_mean.transform(test_x)

        if one_hot:
            x, cat_features = make_pd_from_np(
                x,
                cat_features,
                is_train=True,
                remove_high_cardinality=True,
            )
            test_x, _ = make_pd_from_np(test_x, cat_features, is_train=False)

            # Create a list of transformers. For columns with indices in cat_features, use OneHotEncoder. For others, just passthrough.
            # The way the ColumnTransformer is built, the relative order of the columns is kept.
            transformers = []
            for idx, col in enumerate(x.columns):
                if idx in cat_features:
                    transformers.append(
                        (
                            f"onehot_{col}",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                drop=onehot_drop,
                            ),
                            [col],
                        ),
                    )
                else:
                    transformers.append((f"passthrough_{col}", "passthrough", [col]))

            transformer = ColumnTransformer(transformers)
            transformer.fit(x)
            x, test_x = transformer.transform(x), transformer.transform(test_x)

            # Get the feature names of the expanded features
            attribute_names = transformer.get_feature_names_out(attribute_names)

            # Update the categorical column indices to be correct
            cat_features = [
                i
                for i, name in enumerate(attribute_names)
                if name.startswith("onehot_")
            ]

            # Restore the original names besides having _{i} suffixes for expanded one hot columns
            attribute_names = [name.split("__", 1)[1] for name in attribute_names]

        if standardize:
            scaler = MinMaxScaler()
            scaler.fit(x)
            x, test_x = scaler.transform(x), scaler.transform(test_x)

    if return_pandas:
        # Nans in categorical features must be encoded as separate class
        x, cat_features = make_pd_from_np(
            x,
            cat_features,
            is_train=True,
            remove_high_cardinality=False,
        )
        test_x, _ = make_pd_from_np(
            test_x,
            cat_features,
            is_train=False,
            remove_high_cardinality=False,
        )

    return x, y, test_x, attribute_names, cat_features
