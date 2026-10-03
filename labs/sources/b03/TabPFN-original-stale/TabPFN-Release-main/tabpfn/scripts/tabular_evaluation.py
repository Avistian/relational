from __future__ import annotations

import argparse
import copy
import datetime
import os
import time
import warnings
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import tqdm.auto
from torch import cuda

from tabpfn import local_settings, utils
from tabpfn.datasets import (
    get_benchmark_for_task,
    load_openml_dataset,
    remove_duplicated_datasets,
)
from tabpfn.scripts import tabular_metrics
from tabpfn.scripts.estimator import tabpfn_model_type_getters
from tabpfn.scripts.tabular_baselines import get_clf_dict
from tabpfn.scripts.tabular_evaluation_utils import (
    DatasetEvaluation,
    DatasetEvaluationCollection,
)
from tabpfn.scripts.tabular_metrics import (
    calculate_score,
    calculate_score_per_method,
    check_metric_fits_task_type,
    get_main_eval_metric,
    get_standard_eval_metrics,
    is_imbalanced,
)
from tabpfn.utils import default_task_settings, np_load_if_exists, print_once

N_SPLITS = 5

if TYPE_CHECKING:
    from tabpfn.datasets import TabularDataset

    from .tabular_metrics import MetricDefinition

try:
    from clusters import monkey_patches

    monkey_patches.verify_patch()
except ModuleNotFoundError:
    # that means we are not in our dev environment, thus we can ignore this
    pass


def evaluate_simple(
    model: Callable[..., DatasetEvaluation],
    test_datasets: list[TabularDataset],
    task_type: Literal["multiclass", "regression", "survival"],
    bptt: int = 5000,
    eval_positions=None,  # TODO: Remove this
    device: str = "cpu",
    max_time: int = 300,
    overwrite: bool = True,
    **evaluate_kwargs: Any,
) -> tuple[dict[str, float], dict[str, DatasetEvaluation | float]]:
    """A simple wrapper around `evaluate`.

    :param model: Model function
    :param test_datasets: List of datasets
    :param task_type: Task type of the evaluations
    :param bptt: Position of which to evaluate into
    :param eval_positions: TODO: Remove this
    :param device: Device to run the evaluation on
    :param max_time: Maximum time to run the evaluation
    :param overwrite: Whether to overwrite existing evaluations
    :param evaluate_kwargs: Additional arguments to pass to `evaluate`
    """
    evaluation_results = evaluate(
        datasets=test_datasets,
        model=model,
        method="unknown",
        device=device,
        metric_used=get_main_eval_metric(task_type),
        overwrite=overwrite,
        save=False,
        return_tensor=True,
        verbose=False,
        bptt=bptt,
        max_time=max_time,
        base_path="",
        path_interfix=f"tabular_{task_type}",
        **evaluate_kwargs,
    )

    global_results: dict[str, DatasetEvaluation | float] = {**evaluation_results}

    metrics = get_standard_eval_metrics(task_type)
    for metric in metrics:
        calculate_score_per_method(
            metric["func"],
            metric["name"],
            global_results,
            test_datasets,
            aggregator=metric["aggregator"],
        )
    metrics_dict = {
        f"{metric['name']}": float(
            global_results[f"{metric['aggregator']}_{metric['name']}"],  # type: ignore
        )
        for metric in metrics
    }

    # TODO: This is rather redundant, all entries in `metrics_dict` is in
    # `global_results`. It would rather make sense to just have metrics stored in
    # the DatasetEvaluation
    return metrics_dict, global_results


def evaluate_and_score(
    valid_datasets: list[TabularDataset],
    valid_metrics: dict | None = None,
    metric_with_model: Callable[..., DatasetEvaluation] | None = None,
    metric_used: Callable | None = None,
    split_name: str = "test",
    log_per_dataset_metrics: bool = False,
    manual_dataset_groups: dict[str, list[str]] | None = None,
    num_splits: int = 5,
    evaluate_subsets: Literal[
        True,
        "all",
        "property_based_subsets",
        False,
    ] = "property_based_subsets",
    max_time: int = 300,
    random_state: int = 0,
    save: bool = False,
    overwrite: bool = False,
    fetch_only: bool = False,
    base_path: str = local_settings.base_path,
    path_interfix: str | None = None,
    method_name: str = "test",
    rename_gpu_runs: bool = False,  # If True, we rename the gpu result files to not overwrite the cpu results
    allow_remap_time: bool = True,  # If True, we remap the maxtime based on the name of the method.
    splits: list[int] | None = None,
):
    """This function evaluates a model on multiple datasets and calculates aggregate results over multiple splits. These results
    are then aggregated into an overall score for each metric across all datasets. Manual groups can also be defined to report
    the aggregated scores for them separately. The function can also log metrics per dataset and store the results. It returns the
    log message as well as all calculated scores.

    Example for the usage of manual dataset groups:
    ```python
    # Assume we have the following datasets represented by their names here
    valid_datasets = ['ds1', 'ds2', 'ds3']

    # Define manual dataset groups
    manual_dataset_groups = {
        'group1': ['ds1', 'ds2'],
        'group2': ['ds2', 'ds3']
    }
    ```

    :param valid_datasets: List of datasets to evaluate on
    :param valid_metrics: List of metrics to evaluate on
    :param metric_with_model: Model function, i.e. partial(transfromer_metric, model=model)
    :param metric_used: Metric function to use for evaluation
    :param split_name: Name of the split to evaluate on, this is only used for logging
    """
    assert metric_with_model is not None, "metric_with_model must be defined"

    if valid_metrics is None:
        valid_metrics = get_standard_eval_metrics(valid_datasets[0].task_type)

    if path_interfix is None:
        path_interfix = valid_datasets[0].task_type

    if metric_used is None:
        metric_used = get_main_eval_metric(valid_datasets[0].task_type)
    # Get the tasks for the datasets
    if evaluate_subsets is True or evaluate_subsets == "all":
        tasks = get_benchmarks_and_groups(valid_datasets)
    elif evaluate_subsets == "property_based_subsets":
        tasks = get_task_groups_for_datasets(valid_datasets)
    elif evaluate_subsets is False:
        tasks = {}
    else:
        raise ValueError(f"{evaluate_subsets} not allowed as evaluate_subsets keyword.")

    # Build a mapping from dataset name to its index in the list of datasets
    ds_name_to_index_mapping = {
        ds.get_dataset_identifier(): i for i, ds in enumerate(valid_datasets)
    }

    # Create a new dictionary that stores the defined dataset objects for each group
    manual_dataset_groups = (
        {} if manual_dataset_groups is None else manual_dataset_groups
    )
    dataset_groups = {group: [] for group in manual_dataset_groups}

    # Map the dataset names to their respective dataset objects
    for group in manual_dataset_groups:
        for ds_name in manual_dataset_groups[group]:
            # Raise an error if the manually defined dataset name is not in the list of provided datasets
            assert (
                ds_name in ds_name_to_index_mapping
            ), f"Dataset {ds_name} defined in the group {group} could not be found in the dataset list."

            # Get the index of this dataset in the list of datasets
            index = ds_name_to_index_mapping[ds_name]

            # Add the dataset object to the respective group
            dataset_groups[group] += [valid_datasets[index]]

    assert tasks.keys().isdisjoint(
        dataset_groups.keys(),
    ), "The benchmark task and manually defined dataset group names should not overlap."

    # Define the subgroups of datasets to be aggregated separately
    subgroups = {**tasks, **dataset_groups}

    split_results = {}

    splits = splits if splits is not None else list(range(1, num_splits + 1))
    # Iterate over the splits
    for split in (pbar := tqdm.tqdm(splits)):
        pbar.set_description(f"Running split {split - 1}/{num_splits}")
        import os

        import psutil

        print(
            psutil.Process(os.getpid()).memory_info().rss / 1024**2,
            "MiB",
            "split",
            split,
        )
        # Evaluate the split
        split_result = evaluate(
            datasets=valid_datasets,
            model=metric_with_model,
            method=method_name,
            metric_used=metric_used,
            overwrite=overwrite,
            save=save,
            return_tensor=False,
            verbose=False,
            split_number=split,
            max_time=max_time,
            random_state=random_state,
            fetch_only=fetch_only,
            base_path=base_path,
            path_interfix=path_interfix,
            rename_gpu_runs=rename_gpu_runs,
            allow_remap_time=allow_remap_time,
        )

        split_results[split] = split_result

    global_results = {}
    for ds in valid_datasets:
        # Note: Some of the split_results can be None in case no split was found.
        global_results[ds.get_dataset_identifier()] = DatasetEvaluationCollection(
            ds.get_dataset_identifier(),
            {k: split_results[k][ds.get_dataset_identifier()] for k in split_results},
        )

    # Aggregate the metrics of each dataset over the number of splits
    # into a total metric, group metrics as well as task metrics.
    for metric in (pbar := tqdm.tqdm(valid_metrics)):
        pbar.set_description(f"Calculating {metric['aggregator']}_{metric['name']}")
        calculate_score_per_method(
            metric["func"],
            metric["name"],
            global_results,
            valid_datasets,
            aggregator=metric["aggregator"],
            subgroups=subgroups,
        )

    log_msg = {
        f"{split_name}/{num_splits}_splits/{metric['aggregator']}_{metric['name']}": float(
            global_results[f"{metric['aggregator']}_{metric['name']}"],
        )
        for metric in valid_metrics
    }

    print(str(log_msg))

    for task in tasks:
        if len(tasks[task]) == 0:
            continue

        log_msg = {
            **log_msg,
            **{
                f"{split_name}/{num_splits}_splits/per_task/benchmark_{task}/{metric['aggregator']}_{metric['name']}": float(
                    global_results[f"{task}_{metric['aggregator']}_{metric['name']}"],
                )
                for metric in valid_metrics
            },
        }

    for group in dataset_groups:
        log_msg = {
            **log_msg,
            **{
                f"{split_name}/{num_splits}_splits/per_group/benchmark_{group}/{metric['aggregator']}_{metric['name']}": float(
                    global_results[f"{group}_{metric['aggregator']}_{metric['name']}"],
                )
                for metric in valid_metrics
            },
        }

    if log_per_dataset_metrics:
        for ds in valid_datasets:
            log_msg = {
                **log_msg,
                **{
                    f"{split_name}/{num_splits}_splits/per_dataset/{ds.name}_{ds.get_dataset_identifier()}/{metric['aggregator']}_{metric['name']}": float(
                        global_results[ds.get_dataset_identifier()].metrics[
                            f"{metric['aggregator']}_{metric['name']}"
                        ],
                    )
                    for metric in valid_metrics
                },
            }

    return log_msg, global_results


def evaluate(
    datasets: list[TabularDataset],
    verbose: bool = False,
    **eval_kwargs: Any,
) -> dict[str, DatasetEvaluation]:
    """Evaluates a list of datasets for a model function.

    :param datasets: List of datasets
    :param eval_kwargs: Keyword arguments for `evaluate_position`
    :return: Dictionary of dataset names and their evaluation results
    """
    overall_result = {}
    it = tqdm.tqdm if verbose else lambda x: x

    for _i, ds in enumerate(it(datasets)):
        if eval_kwargs.get("device", "cpu") != "cpu":
            cuda.empty_cache()
        # uncomment, if you want to debug why your tests are failing..
        # print("evaluating", ds.name, "...")
        assert type(ds) is not list, ValueError("Datasets must be Dataset objects")
        try:
            result = evaluate_position(dataset=ds, verbose=verbose, **eval_kwargs)
        except Exception as e:
            print(f"Error evaluating {ds.name} with id {ds.get_dataset_identifier()}")
            raise e
        # In case the split fails, set the dataset result to None.
        if result is None:
            print(
                f"{ds.name} could not be evaluated on split {eval_kwargs.get('split_number', -1)}. Skipping.",
            )

            overall_result[ds.get_dataset_identifier()] = None
        else:
            overall_result[ds.get_dataset_identifier()] = result

    return overall_result


"""
===============================
INTERNAL HELPER FUNCTIONS
===============================
"""


def evaluate_position(
    dataset: TabularDataset,
    model: Callable[..., DatasetEvaluation],
    method: str,
    metric_used: Callable,
    base_path: str = ".",
    path_interfix: str = "",
    fetch_only: bool = False,
    overwrite_splits: bool = False,
    overwrite: bool = True,
    save: bool = True,
    max_time: int = 300,
    split_number: int = 1,
    random_state: int = 0,
    raise_if_result_not_found: bool = False,
    task_type: str | None = None,
    allow_remap_time: bool = True,
    allow_old_path: bool = True,
    test_suite: None | str = None,
    dry_run: bool = False,
    **kwargs,
) -> DatasetEvaluation | None:
    """Evaluates a dataset with a 'bptt' number of training samples.

    :param dataset: Dataset to evaluate on
    :param model: A function taking in (train_ds=, test_ds=, metric_used=, max_time=)
    :param method: Name of the method, "transformer" or other...
    :param allow_remap_time: If True, the time will be remapped based on the method name if {`transformer`,`default`,`tabpfn`}.
    :param allow_old_path: If True, the old path structure is allowed to be used but prints a warning.
    :param test_suite: Name of the test suite for this evaluation. If not None, the name might be used to truncate the training set,
        e.g., see below for `gbdt_friendly_medium`.
    :param dry_run: If True, the evaluation is not saved to disk. Useful for testing and debugging without overwriting results.
    """
    # Generates a string that identifies the settings of the evaluation result
    max_time_mapped = get_mapped_time(
        method,
        max_time,
        allow_remap_time=allow_remap_time,
    )
    time_string = "_time_" + str(max_time_mapped) if max_time else ""
    metric_used_string = "_" + tabular_metrics.get_scoring_string(metric_used, usage="")
    method = method + time_string + metric_used_string

    assert check_metric_fits_task_type(
        metric_used,
        dataset.task_type,
    ), f"Metric {metric_used} does not fit task type {dataset.task_type}"

    bptt = len(dataset.x)

    num_splits_or_official_split = (
        f"{N_SPLITS}" if dataset.splits is None or overwrite_splits else "official"
    )

    prefix_for_device = (
        ""
        if (
            (kwargs.get("device", "cpu") == "cpu")
            and (not kwargs.get("rename_gpu_runs", False))
        )
        else "gpu_"
    )
    old_type_path = os.path.join(
        base_path,
        f"results/tabular/{path_interfix}/{prefix_for_device}results_{method}_{dataset.get_dataset_identifier()}_{split_number}_{num_splits_or_official_split}.npy",
    )
    if allow_old_path and os.path.exists(old_type_path):
        warnings.warn(
            f"You are using a base_path with the old type of results. Please switch to the new base_path as in local_settings. Your currently used base_path is {base_path}",
        )
        path = old_type_path
    else:
        method_path = os.path.join(
            base_path,
            f"results/tabular/{path_interfix}",
            prefix_for_device + method,
        )
        os.makedirs(method_path, exist_ok=True)
        path = os.path.join(
            method_path,
            f"dataset_{dataset.get_dataset_identifier()}_split_{split_number}_{num_splits_or_official_split}.npy",
        )

    ## Try loading results from disk
    if ((not overwrite) or fetch_only) and (not dry_run):
        result = np_load_if_exists(path)
        if result is not None:
            # print(f"Loaded saved result for {path}")
            result.update({"algorithm_name": method})
            return DatasetEvaluation(
                **{
                    **result,
                    "task_type": task_type
                    or tabular_metrics.utils.get_task_type(metric_used),
                },
            )  # adding the task_type here is only a hot fix for quantile_regression and should be removed once all quantile regression results are saved correctly with task_type='quantile_regression'
        elif fetch_only:
            if raise_if_result_not_found:
                raise ValueError(
                    f"Could not load saved result for {path} but in fetch only mode",
                )
            print(f"Could not load saved result for {path} but in fetch only mode")
            return DatasetEvaluation(y=None, pred=None)

    ## Generate data splits
    train_ds, test_ds = dataset.generate_valid_split(
        n_splits=N_SPLITS,
        splits=None if overwrite_splits else dataset.splits,
        split_number=split_number,
    )

    if (
        (test_suite is not None)
        and (test_suite in ["gbdt_friendly_medium"])
        and (train_ds.x.shape[0] > 10000)
    ):
        # For this benchmark we do not want to throw out large datasets, instead we trim down each training set to be at most 10_000 item large
        from sklearn.model_selection import StratifiedShuffleSplit

        _, sel_index = next(
            StratifiedShuffleSplit(
                n_splits=1,
                test_size=10000,
                random_state=48579,
            ).split(train_ds.x, train_ds.y),
        )
        train_ds = train_ds[sel_index]

    if train_ds is None:
        print(
            f"No dataset could be generated {dataset.name=} {dataset.get_dataset_identifier()=} {bptt=}",
        )
        return None

    print(
        f"============= Dataset {dataset.name} has {train_ds.x.shape[0]} samples and {train_ds.x.shape[1]} features.",
    )
    print(f"Save Path is: {path}")
    device = kwargs.get("device", "cpu")
    utils.print_once(
        f"Running model with: device={device}, max_time={max_time}, metric={metric_used_string}, rng={random_state}.",
    )

    start_time = time.time()
    ds_result = model(
        train_ds=copy.deepcopy(train_ds),
        test_ds=copy.deepcopy(test_ds),
        metric_used=metric_used,
        max_time=max_time,
        random_state=random_state,
        device=device,
    )
    ds_result.timestamp = datetime.datetime.now().isoformat(" ", "seconds")
    ds_result.time = time.time() - start_time

    ds_result.task_type = task_type or dataset.task_type
    ds_result.name = dataset.name
    ds_result.identifier = dataset.get_dataset_identifier()
    ds_result.y = test_ds.y

    if dataset.task_type == "survival":
        ds_result.additional_args = {
            **ds_result.additional_args,
            **{"censoring": test_ds.event_observed},
        }

    if save and (not dry_run):
        with open(path, "wb") as f:
            np.save(f, ds_result.to_dict())
            print(f"Saved results to {path}")

    return ds_result


def get_mapped_time(
    method: str,
    max_time: int,
    *,
    allow_remap_time: bool = True,
) -> int:
    """Remap time string.

    Maps the time string to -1 for methods which do not accept a time parameter. Thus we reload and save all results
    to the same file and don't redo calculations for multiple time budgets if the methods ignore these budgets anyways.

    :param method:
    :param max_time:
    :param allow_remap_time: If True, the time will be remapped based on the method name if {`transformer`,`default`,`tabpfn`}
        in the name or it the metod it is TabPFN's tpye getters, it is mapped to -1.
    """
    if max_time == -1:
        return -1

    remap = (
        "transformer" in method
        or "default" in method
        or "tabpfn" in method
        or method in tabpfn_model_type_getters
    )
    remap = remap and allow_remap_time

    max_time_mapped = -1 if remap else max_time
    if remap:
        print_once(f"Remapped time based on name to: {max_time_mapped}")
    return max_time_mapped


def verify_results(
    global_results,
    test_datasets,
    max_times,
    methods,
    splits,
    scoring_str,
):
    """Verifies that the results are calculated on the same data, i.e. splits, by comparing the checksums of the labels.

    :param global_results:
    :param test_datasets:
    :param max_times:
    :param methods:
    :param splits:
    :param scoring_str:
    :return:
    """
    y_checksums = {}
    not_same = []
    for split_number in splits:
        y_checksums[split_number] = {}
        for ds in test_datasets:
            y_checksums[split_number][ds.get_dataset_identifier()] = {}
            for method in methods:
                for max_time in max_times:
                    res = global_results[scoring_str][method][max_time][split_number][
                        ds.get_dataset_identifier()
                    ]
                    if res.y is None:
                        continue
                    y_checksum = res.y.sum().item()
                    if res.additional_args and "censoring" in res.additional_args:
                        y_checksum += res.additional_args["censoring"].sum().item()
                    if (
                        y_checksum
                        not in y_checksums[split_number][ds.get_dataset_identifier()]
                    ):
                        y_checksums[split_number][ds.get_dataset_identifier()][
                            y_checksum
                        ] = []

                    y_checksums[split_number][ds.get_dataset_identifier()][
                        y_checksum
                    ] += [f"{method}_{max_time}"]

            if len(y_checksums[split_number][ds.get_dataset_identifier()].keys()) > 1:
                not_same.append(
                    f"Split {split_number} for dataset {ds.name} has different labels for methods {y_checksums[split_number][ds.get_dataset_identifier()]}",
                )

    return not_same


def load_evaluations(
    test_datasets: list[TabularDataset],
    task_type: Literal["multiclass", "regression", "quantile_regression", "survival"],
    metric_used: Callable | None = None,
    max_times: list[int] = (300,),
    methods: Iterable[str] | dict[str, Callable] = ("autogluon",),
    base_path: str | None = local_settings.base_path,
    eval_metrics: list[MetricDefinition] | None = None,
    splits: Iterable[int] = range(1, 6),
    ignore_checks: bool = False,
) -> dict:
    if metric_used is None:
        metric_used = get_main_eval_metric(task_type)
    if eval_metrics is None:
        eval_metrics = get_standard_eval_metrics(task_type)
    scoring_str = tabular_metrics.get_scoring_string(metric_used)
    global_results = {scoring_str: {}}

    for method in tqdm.tqdm(methods):
        global_results[scoring_str][method] = {}
        for max_time in max_times:
            global_results[scoring_str][method][max_time] = {}
            for split_number in splits:
                global_results[scoring_str][method][max_time][split_number] = evaluate(
                    method=method,
                    datasets=test_datasets,
                    model=methods,
                    fetch_only=True,
                    save=False,
                    task_type=task_type,
                    verbose=False,
                    max_time=max_time,
                    base_path=base_path,
                    path_interfix=task_type,
                    metric_used=metric_used,
                    test_datasets=test_datasets,
                    split_number=split_number,
                    raise_if_result_not_found=not ignore_checks,
                )

    print("Done loading, verifying integrity...")
    not_same = verify_results(
        global_results,
        test_datasets,
        max_times,
        methods,
        splits,
        scoring_str,
    )
    if len(not_same) > 0:
        msg = f"Done verifying, some (N={len(not_same)}) result files calculated on false data. {not_same}"
        if ignore_checks:
            print(msg)
        else:
            raise ValueError(msg)
    else:
        print("Done verifying, all result files calculated on correct data.")

    for metric in tqdm.tqdm(eval_metrics):
        start = time.time()
        calculate_score(
            metric=metric["func"],
            name=metric["name"],
            global_results=global_results,
            ds=test_datasets,
            aggregator=metric["aggregator"],
            limit_to="",
        )
        print(f"{metric} done, time taken {time.time() - start}")

    return global_results


from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class BenchmarkGroup:
    groups: dict[str, Callable]
    name: str
    # Setup function that can use all datasets to generate statistics (e.g. quantiles for feature numbers)
    setup_func: Callable = field(default_factory=lambda: lambda datasets: {})
    task_types: list = field(
        default_factory=lambda: [
            "regression",
            "multiclass",
            "survival",
            "quantile_regression",
        ],
    )


samples_fine_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def in_percentile(fi, quantiles, v):
    if fi == 0:
        return v <= quantiles[fi]
    return quantiles[fi - 1] < v <= quantiles[fi]


task_groupers = [
    BenchmarkGroup(
        name="Nans",
        groups={
            "has_nans": lambda ds, _: ds.x.isnan().sum() > 0,
            "no_nans": lambda ds, _: ds.x.isnan().sum() == 0,
        },
    ),
    BenchmarkGroup(
        name="BinaryVsMulticlass",
        groups={
            "binary_classification": lambda ds, _: len(ds.y.unique()) <= 2,
            "nonbinary_classification": lambda ds, _: len(ds.y.unique()) > 2,
        },
        task_types=["multiclass"],
    ),
    BenchmarkGroup(
        name="Balanced",
        groups={
            "balanced": lambda ds, _: not is_imbalanced(ds.y),
            "imbalanced": lambda ds, _: is_imbalanced(ds.y),
        },
        task_types=["multiclass"],
    ),
    BenchmarkGroup(
        name="Features",
        groups={
            "small_features": (
                lambda ds, params: ds.x.shape[1] <= params["quantile_33"]
            ),
            "medium_features": lambda ds, params: (
                params["quantile_33"] < ds.x.shape[1] < params["quantile_66"]
            ),
            "large_features": (
                lambda ds, params: ds.x.shape[1] >= params["quantile_66"]
            ),
        },
        setup_func=lambda datasets: {
            "quantile_33": np.quantile([ds_.x.shape[1] for ds_ in datasets], 0.33),
            "quantile_66": np.quantile([ds_.x.shape[1] for ds_ in datasets], 0.66),
        },
    ),
    BenchmarkGroup(
        name="Samples",
        groups={
            "small_sample_size": (
                lambda ds, params: ds.x.shape[0] <= params["quantile_33"]
            ),
            "medium_sample_size": lambda ds, params: (
                params["quantile_33"] < ds.x.shape[0] < params["quantile_66"]
            ),
            "large_sample_size": (
                lambda ds, params: ds.x.shape[0] >= params["quantile_66"]
            ),
        },
        setup_func=lambda datasets: {
            "quantile_33": np.quantile([ds_.x.shape[0] for ds_ in datasets], 0.33),
            "quantile_66": np.quantile([ds_.x.shape[0] for ds_ in datasets], 0.66),
        },
    ),
    BenchmarkGroup(
        name="Samples_fine",
        groups={
            f"Samples in {round(samples_fine_fractions[fi] * 100)}th Percentile": (
                lambda ds, params, fi=fi: in_percentile(fi, params, ds.x.shape[0])
            )
            for fi in range(len(samples_fine_fractions))
        },
        setup_func=lambda datasets: {
            fi: np.quantile(
                [ds_.x.shape[0] for ds_ in datasets],
                samples_fine_fractions[fi],
            )
            for fi in range(len(samples_fine_fractions))
        },
    ),
    BenchmarkGroup(
        name="Features_fine",
        groups={
            f"Feats in {round(samples_fine_fractions[fi] * 100)}th Percentile": (
                lambda ds, params, fi=fi: in_percentile(fi, params, ds.x.shape[1])
            )
            for fi in range(len(samples_fine_fractions))
        },
        setup_func=lambda datasets: {
            fi: np.quantile(
                [ds_.x.shape[1] for ds_ in datasets],
                samples_fine_fractions[fi],
            )
            for fi in range(len(samples_fine_fractions))
        },
    ),
    BenchmarkGroup(
        name="Samples_to_features",
        groups={
            "small_features_per_sample": (
                lambda ds, params: ds.x.shape[1] / ds.x.shape[0]
                <= params["quantile_33"]
            ),
            "medium_features_per_sample": lambda ds, params: (
                params["quantile_33"]
                > (ds.x.shape[1] / ds.x.shape[0])
                > params["quantile_66"]
            ),
            "large_features_per_sample": (
                lambda ds, params: ds.x.shape[1] / ds.x.shape[0]
                >= params["quantile_66"]
            ),
        },
        setup_func=lambda datasets: {
            "quantile_33": np.quantile(
                [ds_.x.shape[1] / ds_.x.shape[0] for ds_ in datasets],
                0.33,
            ),
            "quantile_66": np.quantile(
                [ds_.x.shape[1] / ds_.x.shape[0] for ds_ in datasets],
                0.66,
            ),
        },
    ),
    BenchmarkGroup(
        name="Categoricals",
        groups={
            "purely_numerical": lambda ds, _: len(ds.categorical_feats) == 0,
            "purely_categorical": lambda ds, _: len(ds.categorical_feats)
            == ds.x.shape[1],
            "has_numericals": lambda ds, _: ds.x.shape[1] - len(ds.categorical_feats)
            > 0,
            "has_categoricals": lambda ds, _: len(ds.categorical_feats) > 0,
            "no_categorical_no_nan": lambda ds, _: (
                len(ds.categorical_feats) == 0 and ds.x.isnan().sum() == 0
            ),
        },
    ),
    BenchmarkGroup(
        name="Categorical Shades",
        groups={
            "only_has_numericals": lambda ds, _: len(ds.categorical_feats) == 0,
            "has_categoricals_and_numericals": lambda ds, _: 0
            < len(ds.categorical_feats)
            < ds.x.shape[1],
            "only_has_categoricals": lambda ds, _: len(ds.categorical_feats)
            == ds.x.shape[1],
        },
    ),
    BenchmarkGroup(
        name="Feature Type",
        groups={
            "no_categorical_feat_type": lambda ds, _: len(ds.categorical_feats) == 0,
            "has_categoricals_feat_type": lambda ds, _: len(ds.categorical_feats) > 0,
        },
    ),
    BenchmarkGroup(
        name="Censoring",
        setup_func=lambda datasets: {
            "quantile_33": np.quantile(
                [ds_.event_observed.mean() for ds_ in datasets],
                0.33,
            ),
            "quantile_66": np.quantile(
                [ds_.event_observed.mean() for ds_ in datasets],
                0.66,
            ),
        },
        groups={
            "low_censoring": (
                lambda ds, params: ds.event_observed.mean() <= params["quantile_33"]
            ),
            "medium_censoring": lambda ds, params: (
                params["quantile_33"] < ds.event_observed.mean() < params["quantile_66"]
            ),
            "high_censoring": lambda ds, params: (
                ds.event_observed.mean() >= params["quantile_66"]
            ),
        },
        task_types=["survival"],
    ),
    BenchmarkGroup(
        name="Global Censoring",
        setup_func=lambda datasets: {},
        groups={
            "Yes_global": lambda ds, params: ds.has_global_censoring(),
            "No_global": lambda ds, params: not ds.has_global_censoring(),
        },
        task_types=["survival"],
    ),
    BenchmarkGroup(
        name="Y-outlier",
        setup_func=lambda datasets: {},
        groups={
            "Has > 50 Std": (lambda ds, params: (ds.y > ds.y.std() * 50).any()),
            "Has > 10 Std": (lambda ds, params: (ds.y > ds.y.std() * 10).any()),
            "None > 10 Std": lambda ds, params: (
                lambda ds, params: not (ds.y > ds.y.std() * 10).any()
            ),
        },
        task_types=["survival", "regression"],
    ),
]

"""
Disabled, might use high memory
BenchmarkGroup(
        name="Duplicated Samples",
        setup_func=lambda datasets: {
            "quantile_33": np.quantile(
                [ds_.get_duplicated_samples()[1] for ds_ in datasets], 0.33
            ),
            "quantile_66": np.quantile(
                [ds_.get_duplicated_samples()[1] for ds_ in datasets], 0.66
            ),
        },
        groups={
            "low_duplicates_all": (
                lambda ds, params: ds.get_duplicated_samples()[1]
                <= params["quantile_33"]
            ),
            "medium_duplicates_all": lambda ds, params: (
                params["quantile_33"]
                < ds.get_duplicated_samples()[1]
                < params["quantile_66"]
            ),
            "high_duplicates_all": lambda ds, params: (
                ds.get_duplicated_samples()[1] >= params["quantile_66"]
            ),
        },
    ),
    BenchmarkGroup(
        name="Duplicated Features But Different Ys",
        setup_func=lambda datasets: {
            "quantile_33": np.quantile(
                [
                    ds_.get_duplicated_samples()[1]
                    - ds_.get_duplicated_samples(features_only=True)[1]
                    for ds_ in datasets
                ],
                0.33,
            ),
            "quantile_66": np.quantile(
                [
                    ds_.get_duplicated_samples()[1]
                    - ds_.get_duplicated_samples(features_only=True)[1]
                    for ds_ in datasets
                ],
                0.66,
            ),
        },
        groups={
            "low_duplicates_feats_not_ys": (
                lambda ds, params: ds.get_duplicated_samples()[1]
                - ds.get_duplicated_samples(features_only=True)[1]
                <= params["quantile_33"]
            ),
            "medium_duplicates_feats_not_ys": lambda ds, params: (
                params["quantile_33"]
                < ds.get_duplicated_samples()[1]
                - ds.get_duplicated_samples(features_only=True)[1]
                < params["quantile_66"]
            ),
            "high_duplicates_feats_not_ys": lambda ds, params: (
                ds.get_duplicated_samples()[1]
                - ds.get_duplicated_samples(features_only=True)[1]
                >= params["quantile_66"]
            ),
        },
    ),
"""


def get_task_groups_for_datasets(
    valid_datasets: list[TabularDataset],
) -> dict[str, list[TabularDataset]]:
    """Given a list of valid datasets, return a dictionary of task groups and the datasets that belong to each group.
    These groups are defined in the task_groupers list above, e.g. "contains nans".
    """
    results = {}

    for task_grouper in task_groupers:
        if valid_datasets[0].task_type not in task_grouper.task_types:
            continue
        params = task_grouper.setup_func(valid_datasets)
        assert (
            len(set(results.keys()).intersection(set(task_grouper.groups))) == 0
        ), f"A benchmark sub group {task_grouper.groups} already exists, change the group names to avoid duplicates"
        results.update(
            {
                group: [
                    ds
                    for ds in valid_datasets
                    if task_grouper.groups[group](ds, params)
                ]
                for group in task_grouper.groups
            },
        )

    return results


def get_benchmarks_and_groups(valid_datasets: list[TabularDataset]) -> dict[str, Any]:
    """Given a list of valid datasets, return a dictionary that contains the datasets for each of the following types of keys:
    - "{benchmark_name}": all datasets that belong to the that benchmark
    - "{benchmark_name}_{group_name}": all datasets that belong to that task group inside the benchmark
    - "all_{group_name}": all datasets that belong to that task group across all benchmarks.

    Task groups are things like "contains nans" and benchmarks are things like "automl".
    """
    benchmark_names = {ds.benchmark_name for ds in valid_datasets}
    results = {}
    for benchmark_name in benchmark_names:
        benchmark_datasets = [
            ds for ds in valid_datasets if ds.benchmark_name == benchmark_name
        ]
        benchmark_dict = get_task_groups_for_datasets(benchmark_datasets)
        benchmark_dict = {
            f"{benchmark_name}_{k}": benchmark_dict[k] for k in benchmark_dict
        }
        results.update(benchmark_dict)
        results[f"{benchmark_name}"] = benchmark_datasets

    # Add one unified benchmark
    r = get_task_groups_for_datasets(remove_duplicated_datasets(valid_datasets))
    r = {f"all_{k}": r[k] for k in r}  # prepend "all" to results for all datasets
    results.update(r)

    return results


def get_task_super_group_to_sub_groups(task_type: str) -> dict[str, list[str]]:
    """Given a task type, return a dictionary that maps the task group to the subgroups that belong to that group.
    E.g. "contains nans" -> ["contains nans", "does not contain nans"].
    """
    return {
        super_group.name: list(super_group.groups.keys())
        for super_group in task_groupers
        if task_type in super_group.task_types
    }


def none_or_int(value):
    if value == "None":
        return None
    return int(value)


if __name__ == "__main__":
    max_samples, max_features, max_times, max_classes = default_task_settings()

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="xgb")
    # Optional Arg's for `--loss_function barnll`
    parser.add_argument("--overwrite", type=str, default="True")
    parser.add_argument("--base_path", type=str, default=local_settings.base_path)
    parser.add_argument("--bptt", type=none_or_int, default=max_samples)
    parser.add_argument("--max_features", type=none_or_int, default=max_features)
    parser.add_argument("--max_classes", type=none_or_int, default=max_classes)
    parser.add_argument("--split_number", type=int, default=1)
    parser.add_argument(
        "--task_type",
        type=str,
        choices=["regression", "multiclass", "survival", "quantile_regression"],
        default="regression",
    )
    parser.add_argument(
        "--test_suite",
        type=str,
        default="test",
    )
    parser.add_argument("--max_time", type=int, default=max_times[0])
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--no_of_blocks", type=int, default=-1)
    parser.add_argument("--block_n", type=int, default=0)
    parser.add_argument("--dataset_n", type=int, default=-1)
    parser.add_argument("--rename_gpu_runs", type=str, default="False")
    parser.add_argument("--allow_remap_time", type=str, default="True")
    parser.add_argument("--allow_old_path", type=str, default="True")
    parser.add_argument("--dataset_overwrite", type=str, default="None")
    parser.add_argument("--dry_run", type=str, default="False")

    args = parser.parse_args()
    if args.overwrite == "True":
        args.overwrite = True
    elif args.overwrite == "False":
        args.overwrite = False
    else:
        args.overwrite = bool(args.overwrite)

    if args.dry_run == "True":
        args.dry_run = True
    elif args.dry_run == "False":
        args.dry_run = False
    else:
        raise ValueError(f"Invalid dry_run argument: {args.dry_run}")

    print(f"Renaming Results files for GPU: {args.rename_gpu_runs}")
    if args.rename_gpu_runs == "True":
        args.rename_gpu_runs = True
    elif args.rename_gpu_runs == "False":
        args.rename_gpu_runs = False
    else:
        args.rename_gpu_runs = bool(args.rename_gpu_runs)

    print(f"Remapping results files time string: {args.rename_gpu_runs}")
    if args.allow_remap_time == "True":
        args.allow_remap_time = True
    elif args.allow_remap_time == "False":
        args.allow_remap_time = False
    else:
        args.allow_remap_time = bool(args.allow_remap_time)

    print(f"Allow old path overwrite: {args.allow_old_path}")
    if args.allow_old_path == "True":
        args.allow_old_path = True
    elif args.allow_old_path == "False":
        args.allow_old_path = False
    else:
        args.allow_old_path = bool(args.allow_old_path)

    print(f"Running with args: {args}")
    test_datasets, _ = get_benchmark_for_task(
        task_type=args.task_type,
        split=args.test_suite,
        max_samples=args.bptt,
        max_features=args.max_features,
        max_classes=args.max_classes,
        return_as_lists=False,
        allowed_datasets=[args.dataset_overwrite]
        if args.dataset_overwrite != "None"
        else None,
        load_dummy_data=True,
    )
    del _

    if args.no_of_blocks > 1:
        no_of_ds = len(test_datasets)
        block_size = int(no_of_ds / args.no_of_blocks)
        start = args.block_n * block_size
        end = (args.block_n + 1) * block_size
        if args.block_n == args.no_of_blocks - 1:
            end = no_of_ds + 1
        test_datasets = test_datasets[start:end]

    if args.dataset_n != -1:
        test_datasets = [test_datasets[args.dataset_n]]
    print(f"Running on datasets: {[ds.name for ds in test_datasets]}")

    # Only load one dataset at a time (into RAM) such that we do not run out of memory when processing large datasets.
    for _dataset in test_datasets:
        print(f"Load and evaluate for dataset {_dataset.name}.")

        r = evaluate(
            datasets=[
                load_openml_dataset(
                    name=_dataset.name,
                    did=_dataset.extra_info["openml_did"],
                    tid=_dataset.extra_info["openml_tid"],
                    **_dataset.extra_info["shared_dataset_kwargs"],
                ),
            ],
            model=get_clf_dict(args.task_type)[args.method],
            method=args.method,
            device=args.device,
            max_time=args.max_time,
            split_number=args.split_number,
            metric_used=get_main_eval_metric(args.task_type),
            overwrite=args.overwrite,
            save=True,
            return_tensor=False,
            verbose=False,
            bptt=args.bptt,
            base_path=args.base_path,
            path_interfix=args.task_type,
            rename_gpu_runs=args.rename_gpu_runs,
            allow_remap_time=args.allow_remap_time,
            allow_old_path=args.allow_old_path,
            test_suite=args.test_suite,
            dry_run=args.dry_run,
        )

        print(r)
        del r
