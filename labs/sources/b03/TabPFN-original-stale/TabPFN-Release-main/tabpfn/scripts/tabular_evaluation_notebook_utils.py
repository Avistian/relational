from __future__ import annotations

import torch
import os
from typing import List, Dict, Any, Union, Tuple, Optional, Literal, Sequence
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm.auto
import numpy as np

from .tabular_metrics import (
    get_scoring_string,
    get_scoring_direction,
    metric_renamer,
)
from tabpfn.utils import mean_confidence_interval
from .tabular_evaluation import evaluate
from tabpfn import local_settings
from .tabular_metrics import get_main_eval_metric, get_metric_name
from . import tabular_metrics
from .notebook_utils import to_str_table_vis


def submit_method_evaluation(
    method,
    methods,
    datasets_dict,
    task_type,
    max_time,
    metric_used,
    split_number,
    max_samples,
    base_path,
    test_suite,
    device="cpu",
    submit=True,
    queue=True,
    ex=None,
    overwrite=True,
    no_of_blocks=1,
    block_number=0,
    max_features=100,
    max_classes=10,
    verbose=False,
    dataset_n: int = -1,  # -1 means all, else take n out of list of dataset from current suite
):
    job = None
    if submit:
        cmd = (
            f"python -m scripts.tabular_evaluation"
            f" --task_type {task_type}"
            f" --max_time {max_time}"
            f" --split_number {split_number}"
            f" --method {method}"
            f" --overwrite {overwrite}"
            f" --test_suite {test_suite}"
            f" --no_of_blocks {no_of_blocks}"
            f" --block_n {block_number}"
            f" --bptt {max_samples}"
            f" --max_features {max_features}"
            f" --max_classes {max_classes}"
            f" --device {device}"
            f" --dataset_n {dataset_n}"
        )
        if verbose:
            print(cmd)
        if queue:
            if os.environ["TABPFN_CLUSTER_SETUP"] == "CHARITE":
                os.system(
                    f"bsub -o job.out export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/lib;{cmd}"
                )
            elif os.environ["TABPFN_CLUSTER_SETUP"] == "FREIBURG":

                def exec_():
                    os.system(cmd)
                    return None

                job = ex.submit(exec_)
        else:
            os.system(f"{cmd}")
    else:
        kwargs = {
            "model": methods[method],
            "method": method,
            "task_type": task_type,
            "datasets": datasets_dict[f"{test_suite}_{task_type}"],
            "max_time": max_time,
            "metric_used": metric_used,
            "split_number": split_number,
            "save": True,
            "path_interfix": task_type,
            "bptt": max_samples,
            "overwrite": overwrite,
            "base_path": base_path,
            "device": device,
        }
        if queue:
            job = ex.submit(evaluate, kwargs)
        else:
            job = evaluate(**kwargs)

    return job


def rename_table_vis(table):
    return table.T.rename(
        {
            "blood-transfusion-service-center": "blood-transfus..",
            "jungle_chess_2pcs_raw_endgame_complete": "jungle\_chess..",
            "bank-marketing": "bank-market..",
        }
    ).T


def get_suffix_table_vis(i, k, test_datasets):
    suffix = ""
    suffix = suffix + "s" if test_datasets[i][5]["samples_capped"] == True else suffix
    suffix = suffix + "f" if test_datasets[i][5]["feats_capped"] == True else suffix
    suffix = suffix + "c" if test_datasets[i][5]["classes_capped"] == True else suffix
    suffix = "" if len(suffix) == 0 else f" [{suffix}]"

    return k + suffix


def get_metrics_for_table_vis(task_type):
    if task_type == "regression":
        visualize_metrics = [
            "normalized_rmse",
            "spearman",
            "r2",
            "normalized_mae",
            "normalized_mse",
        ]
    elif task_type == "quantile_regression":
        visualize_metrics = [
            get_metric_name(tabular_metrics.MeanNormalizedIntervalScoreMetric()),
            get_metric_name(tabular_metrics.NormalizedPinballLossMetric()),
            get_metric_name(tabular_metrics.NormalizedSharpnessMetric()),
            get_metric_name(tabular_metrics.NormalizedQuantileMAEMetric()),
            get_metric_name(tabular_metrics.QuantileCalibrationMetric()),
        ]
    elif task_type == "multiclass":
        visualize_metrics = ["roc", "acc", "f1", "ce", "ece"]
    elif task_type == "survival":
        visualize_metrics = [
            tabular_metrics.survival_c_index_metric,
            tabular_metrics.survival_censored_accuracy,
            tabular_metrics.survival_mse_uncensored,
            tabular_metrics.survival_spearman_uncensored,
        ]
        visualize_metrics = [get_metric_name(m) for m in visualize_metrics]
    return visualize_metrics


"""
Old code from get_table_vis:

# Wins vs Us, currently disabled

    # def wins_vs_idx(matrix, idx):
    #    wins_auc = np.array([[(matrix.values[:, j] < matrix.values[:, i]).sum() if i != j else 0 for i,method in enumerate(methods)] for j in [idx]])
    #    ties_auc = np.array([[(matrix.values[:, j] == matrix.values[:, i]).sum() if i != j else 0 for i,method in enumerate(methods)] for j in [idx]])
    #    losses_auc = np.array([[(matrix.values[:, j] > matrix.values[:, i]).sum() if i != j else 0 for i,method in enumerate(methods)] for j in [idx]])

    #    return wins_auc, ties_auc, losses_auc

    # transformer_idx = np.where(rmse_matrix.columns == 'transformer')[0][0]

    # wins_ce_vs_us, ties_ce_vs_us, losses_ce_vs_us = wins_vs_idx(-cross_entropy_matrix, transformer_idx)

    # import scipy
    # wilcoxon_ps = np.array([[scipy.stats.wilcoxon(roc_matrix.T[:, j],roc_matrix.T[:, i]).pvalue if i != j else 0 for i,method in enumerate(methods)] for j in [0,1]])

"""


def get_table_vis(
    global_results: Dict[str, Dict[str, Dict[float, Dict[int, Any]]]],
    methods: Sequence[str],
    test_datasets: List[Any],
    eval_metrics: List[Dict[str, str]],
    main_metric_used_for_optimization: str,
    max_times: List[float],
    splits: Union[List[int], range] = range(0, 10),
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Aggregates and summarizes evaluation metrics for methods.

    Parameters:
        global_results (dict): Nested dictionary containing evaluation results.
        methods (list): List method names to evaluate, should be part of global_results.
        test_datasets (list): List of dataset objects for testing.
        eval_metrics (list): List of metric names to evaluate
        main_metric_used_for_optimization (str): Primary evaluation metric (e.g., "accuracy").
        max_times (list): List of time budgets for evaluation.
        splits (list or range): Range of data splits for cross-validation. Default is range(0, 10).

    Returns:
        tuple:
          - df (pd.DataFrame): A DataFrame containing aggregated evaluation results.
            Columns include 'method', 'max_time', 'split_number', 'scoring_str', 'nansum_count',
            and additional metrics as specified in eval_metrics.
          - df_by_ds (pd.DataFrame): A DataFrame with dataset-specific evaluation results.
            It has the same columns as 'df' but is extended with results for each dataset.
          - results_matrix (dict): A nested dictionary containing summary statistics ('mean', 'rank', 'win', 'std')
            for each evaluation metric and time budget. Each entry is a pivoted DataFrame indexed by 'ds'
            and columns multi-indexed by 'method' and 'split_number'.
    """
    assert (
        type(test_datasets[0]) is not list
    ), "Please pass dataset objects, not the old list format"

    # Initialize empty lists for dataframe entries
    df = []
    df_by_ds = []

    # Convert metric_used into a formatted string
    metric_string = get_scoring_string(main_metric_used_for_optimization)

    # Main loops to traverse through methods, time budgets, and data splits
    for method in methods:
        for max_time in max_times:
            for split_number in splits:
                # Filter results according to current loop variables
                filtered_results = global_results[metric_string]
                filtered_results = filtered_results[method]
                try:
                    filtered_results = filtered_results[max_time]
                except KeyError:
                    raise KeyError(
                        f"max_time {max_time} not in {filtered_results.keys()}"
                    )
                try:
                    filtered_results = filtered_results[split_number]
                except KeyError:
                    raise KeyError(
                        f"split_number {split_number} not in {filtered_results.keys()}"
                    )

                # Prepare a dictionary entry for summary stats
                dataframe_entry = {
                    "method": method,
                    "max_time": max_time,
                    "split_number": split_number,
                    "scoring_str": metric_string,
                    "nansum_count": filtered_results["nansum_count"],
                }

                # Extend dataframe_entry with other evaluation metrics
                for metric in eval_metrics:
                    metric_name, metric_agg = metric["name"], metric["aggregator"]
                    dataframe_entry[f"mean_{metric_name}"] = filtered_results[
                        f"{metric_agg}_{metric_name}"
                    ]

                # Append to df list
                df.append(dataframe_entry)

                # Generate dataset-specific results and append to df_by_ds list
                for ds in test_datasets:
                    ds_results = {}
                    for metric in eval_metrics:
                        metric_name, metric_agg = metric["name"], metric["aggregator"]
                        ds_results[metric_name] = filtered_results[
                            ds.get_dataset_identifier()
                        ].metrics[f"{metric_agg}_{metric_name}"]
                    df_by_ds.append({**dataframe_entry, **ds_results, "ds": ds.name})

    # Convert lists to DataFrames
    df = pd.DataFrame.from_dict(df)
    df_by_ds = pd.DataFrame.from_dict(df_by_ds)

    results_matrix = get_results_matrix(
        df_by_ds[(df_by_ds.scoring_str == metric_string)],
        eval_metrics=[metric["name"] for metric in eval_metrics],
        max_times=max_times,
    )

    # Return the three compiled DataFrames
    return df, df_by_ds, results_matrix


def get_results_matrix(df_by_ds, eval_metrics=("time",), max_times=None):
    """
    Generate a results matrix for a given DataFrame, that contains summary statistics for each evaluation metric like
    mean, normalized mean, rank, win, and std.
    :param df_by_ds: DataFrame containing evaluation results, all sharing the same scoring_str.
    Each row should represent one evaluation of the method (method column) on one split (split_number column) of one dataset (ds column).
    Metrics are additional columns, e.g. `balanced_acc`, `ece`, `roc`, `rmse`, `time`, etc.
    :param eval_metrics: List of evaluation metrics to include in the results matrix, have to be columns in df_by_ds.
    :param max_times: List of time budgets to include in the results matrix, have to be value for some `max_time` column in df_by_ds.
    If None, the values of `max_time` column in df_by_ds are used.
    :return: A dictionary containing summary statistics ('mean', 'rank', 'win', 'std', 'normalized') for each evaluation metric and time budget.
    """
    assert len(df_by_ds.scoring_str.unique()) == 1, "Only one scoring_str allowed"
    # Initialize an empty results_matrix for storing mean, rank, win, and std statistics
    results_matrix = {
        "mean": {},
        "rank": {},
        "win": {},
        "std": {},
        "normalized": {},
        "rank_in_pct": {},
    }

    # Loop through all evaluation metrics
    for metric_name in eval_metrics:
        # Initialize dictionaries within results_matrix
        (
            results_matrix["mean"][metric_name],
            results_matrix["rank"][metric_name],
            results_matrix["win"][metric_name],
            results_matrix["std"][metric_name],
            results_matrix["normalized"][metric_name],
            results_matrix["rank_in_pct"][metric_name],
        ) = ({}, {}, {}, {}, {}, {})

        # Loop through each time budget
        for max_time in max_times or df_by_ds.max_time.unique():
            # Filter dataframe to only include rows with the same time budget and scoring string
            df_by_ds_filtered = df_by_ds[(df_by_ds.max_time == max_time)]

            # Determine if rank should be in ascending order
            ascending = get_scoring_direction(metric_name) == -1

            # Initialize an empty dictionary to hold results for current metric and time
            results = {}

            # Compute mean, rank, and std using groupby and aggregation
            results["rank"] = (
                df_by_ds_filtered.groupby(["ds", "method", "split_number"])[metric_name]
                .mean()
                .groupby(["split_number", "ds"])
                .rank(method="average", ascending=ascending)
            )
            results["rank_in_pct"] = (
                df_by_ds_filtered.groupby(["ds", "method", "split_number"])[metric_name]
                .mean()
                .groupby(["split_number", "ds"])
                .rank(method="average", ascending=ascending, pct=True)
            )
            results["mean"] = df_by_ds_filtered.groupby(
                ["ds", "method", "split_number"]
            )[metric_name].mean()
            results["std"] = df_by_ds_filtered.groupby(
                ["ds", "method", "split_number"]
            )[metric_name].std()

            # Compute win statistics
            results["win"] = (
                df_by_ds_filtered.groupby(["ds", "method", "split_number"])[metric_name]
                .mean()
                .groupby(["split_number", "ds"])
                .rank(method="min", ascending=ascending)
                == 1.0
            )

            num_winners = (
                results["win"].groupby(["split_number", "ds"]).transform("sum")
            )

            results["win"] /= num_winners

            # Compute normalized scores:
            #   The worst scores among all methods is set to 0, the best to 1, and the rest are linearly interpolated
            grouped = df_by_ds_filtered.groupby(["ds", "method", "split_number"])[
                metric_name
            ].mean()
            min_scores = grouped.groupby(["ds", "split_number"]).agg("min")
            max_scores = grouped.groupby(["ds", "split_number"]).agg("max")
            aggs = df_by_ds_filtered.groupby(["ds", "method", "split_number"])[
                metric_name
            ].mean()

            aggs = aggs.to_frame() - min_scores.to_frame()
            aggs = aggs / (max_scores.to_frame() - min_scores.to_frame())
            aggs = aggs.reorder_levels(["ds", "method", "split_number"]).sort_index()

            results["normalized"] = aggs

            # Populate results_matrix with computed statistics
            for aggregator in results_matrix.keys():
                results_matrix[aggregator][metric_name][max_time] = pd.pivot(
                    results[aggregator].reset_index(),
                    index="ds",
                    columns=["method", "split_number"],
                    values=metric_name,
                )
    return results_matrix


def make_metric_over_time_plot(
    df, methods, max_times, metric_key, grouping=False, data_type="mean"
):
    # TODO: Make ranks and wins possible

    df_ = df.groupby(["max_time", "method"]).mean_time.mean()
    df = df.set_index(["max_time", "method"])
    df.loc[df_.index, "step_time"] = df_
    df = df.reset_index()

    max_times_renamer = {
        0.5: "0.5s",
        1: "1s",
        5: "5s",
        15: "15s",
        30: "30s",
        60: "1min",
        300: "5min",
        900: "15min",
        3600: "1h",
        14400: "4h",
    }

    f, ax = plt.subplots(figsize=(7, 7))
    sns.set_palette("tab10")
    df.max_time_log = np.log(df.max_time)
    df.step_time_log = np.log(df.step_time)

    for m in methods:
        x = df.step_time_log if grouping else df.max_time_log
        sns.lineplot(
            x,
            f"{data_type}_{metric_key}",
            data=df[df.method == m],
            linestyle="--",
            marker="o",
            ci="sd",
            ax=ax,
        )

    ax.set_xticks(np.log(max_times))
    ax.set_xticklabels([max_times_renamer[t] for t in max_times])

    ax.set_ylabel(metric_renamer[metric_key])
    if grouping:
        ax.set_xlabel("Time (s, requested, not actual)")
    else:
        ax.set_xlabel("Time taken")

    return ax


def make_results_table_table_vis(results_matrix, visualize_metrics, max_time):
    table = (
        results_matrix["mean"][visualize_metrics[0]][max_time]
        .groupby(axis=1, level=0)
        .mean()
        .copy()
    )
    table_by_df_rows = table.shape[0]

    # TODO: Reenable
    # table.index = [get_suffix_table_vis(i, k, test_datasets_for_results) for i, k in enumerate(table.index[0:table.shape[0]])]

    for metric in visualize_metrics:
        table.loc[f"Mean Normalized {metric}"] = (
            results_matrix["normalized"][metric][max_time]
            .groupby(axis=1, level=0)
            .mean()
            .mean()
            .values
        )

        # table.loc[f"Mean Normalized {metric} Std"] = (
        #    results_matrix["normalized"][metric][max_time]
        #    .groupby(axis=1, level=0)
        #    .std()
        #    .mean()
        #    .values
        # )

        conf_interval = (
            results_matrix["normalized"][metric][max_time]
            .groupby(axis=1, level=0)
            .apply(lambda x: mean_confidence_interval(x.values))
        )
        table.loc[f"Mean Normalized {metric} ConfInt"] = conf_interval.values

    for metric in visualize_metrics:
        table.loc[f"Mean {metric}"] = (
            results_matrix["mean"][metric][max_time]
            .groupby(axis=1, level=0)
            .mean()
            .mean()
            .values
        )
        table.loc[f"Mean {metric} ConfInt"] = (
            results_matrix["mean"][metric][max_time]
            .groupby(axis=1, level=0)
            .std()
            .mean()
            .values
        )

    for metric in visualize_metrics:
        table.loc[f"Ranks {metric}"] = (
            results_matrix["rank"][metric][max_time]
            .groupby(axis=1, level=0)
            .mean()
            .mean()
            .values
        )

    for metric in visualize_metrics:
        table.loc[f"Wins {metric}"] = (
            results_matrix["win"][metric][max_time]
            .groupby(axis=1, level=0)
            .mean()
            .sum()
            .values
        )

    for metric in visualize_metrics:
        continue
        # This code is diabled for now, as we do not want to show in results tables
        # This code is for 1-vs-1 win comparison
        direction = get_scoring_direction(metric)
        table.loc[f"VS {metric}"] = (
            (
                (
                    results_matrix["mean"][metric][max_time]
                    - results_matrix["mean"][metric][max_time].loc[
                        :, incumbant_model_name
                    ]
                )
                * direction
                > 0
            )
            .groupby(axis=1, level=0)
            .mean()
            .mean()
            .values
        )

    # table.loc['Win/T/L AUC vs Us'] = ["{:d}/{:d}/{:d}".format(w, t, l) for w,t,l in zip(wins_roc_vs_us[-1, :], ties_roc_vs_us[-1, :], losses_roc_vs_us[-1, :])]

    table.loc["Mean time (s)"] = (
        results_matrix["mean"]["time"][max_time].groupby(axis=1, level=0).mean().mean()
    )
    conf_interval = (
        results_matrix["mean"]["time"][max_time]
        .groupby(axis=1, level=0)
        .apply(lambda x: mean_confidence_interval(x.values))
    )
    table.loc["Mean time (s) ConfInt"] = conf_interval.values

    table.loc["Non Nan Count"] = (
        results_matrix["mean"]["count"][max_time].groupby(axis=1, level=0).sum().sum()
    )

    return table, table.iloc[table_by_df_rows:]


def style_table(table, eval_metrics, remove_means=True):
    table = rename_table_vis(table)

    for normalized_str in [" Normalized", ""]:
        for metric_name in eval_metrics + ["time (s)"]:
            if normalized_str != "" and metric_name == "time (s)":
                continue
            table.loc[[f"Mean{normalized_str} {metric_name} ConfInt"]] = (
                table.loc[[f"Mean{normalized_str} {metric_name} ConfInt"]]
                .apply(
                    lambda data: to_str_table_vis(data, format_string="%.2g"), axis=1
                )
                .values
            )
            table.loc[[f"Mean{normalized_str} {metric_name}"]] = (
                table.loc[[f"Mean{normalized_str} {metric_name}"]]
                .apply(
                    lambda data: to_str_table_vis(data, format_string="%.4g"), axis=1
                )
                .values
            )
            table.loc[f"Mean{normalized_str} {metric_name}"] = (
                table.loc[f"Mean{normalized_str} {metric_name}"].values
                + "$\pm$"
                + table.loc[f"Mean{normalized_str} {metric_name} ConfInt"].values
            )
            table = table.drop([f"Mean{normalized_str} {metric_name} ConfInt"])

    if remove_means:
        for metric_name in eval_metrics:
            table = table.drop([f"Mean {metric_name}"])

    # print(table.to_latex(escape=False))
    # table_small = table.iloc[-len(keys_min+keys_max)-1-3:]
    # table_small
    # print(table_small.to_latex(escape=False))

    # table = table.copy()
    #
    # table.iloc[:-5] = table.iloc[:-5].apply(lambda data : bold_extreme_values(data),axis=1)
    # table.iloc[-5:-5] = table.iloc[-5:-5].apply(lambda data : bold_extreme_values(data, max_=False),axis=1)
    #
    # table
    # #print(table.to_latex(escape=False))

    # rename(table[-7:]).round(decimals=3).style.highlight_min(axis = 1, props= 'font-weight: bold;').format(precision=3)

    keys_min = [
        c
        for c in table.index
        if (" ce" in c or " ece" in c or "Ranks" in c)
        and not "Wins" in c
        and not "VS" in c
    ]
    keys_max = [c for c in table.index if c not in keys_min]
    keys_min = [c for c in keys_min if "time" not in c]

    keys_rounded = [c for c in table.index if "Rank" in c or "Wins" in c]
    keys_not_rounded = [c for c in table.index if c not in keys_rounded]

    return (
        table.style.format(precision=1, subset=pd.IndexSlice[keys_rounded, :])
        .format(precision=4, subset=pd.IndexSlice[keys_not_rounded, :])
        .highlight_min(
            axis=1, props="font-weight: bold;", subset=pd.IndexSlice[keys_min, :]
        )
        .highlight_max(
            axis=1, props="font-weight: bold;", subset=pd.IndexSlice[keys_max, :]
        )
    )


def plot_scores_grid(
    ax, results, dss, x1: str, x2: str, method_vis: str, clf: str = "linear"
):
    """
    Plots a smooth contour plot of the results on a grid of the two featurizers x1 and x2.
    E.g. how does the rank change with an increasing or decreasing number of features and classes?

    plot_scores_grid(results_matrix['rank'][visualize_metrics[0]][max_time],
                 test_datasets_for_results,
                 'n_features', 'n_classes',
                 'lgb_default')

    :param results:
    :param dss:
    :param x1:
    :param x2:
    :param method_vis:
    :return:
    """

    def extend_df_by_ds(df_by_ds, test_datasets_for_results):
        for i in range(len(df_by_ds)):
            ds = [
                ds for ds in test_datasets_for_results if ds.name == df_by_ds.loc[i].ds
            ][0]
            df_by_ds.loc[i, "n_features"] = ds.x.shape[1]
            df_by_ds.loc[i, "n_classes"] = len(np.unique(ds.y))
            df_by_ds.loc[i, "n_samples"] = ds.x.shape[0]
        return df_by_ds

    # dss = test_datasets_for_results
    # results = results_matrix['rank'][visualize_metrics[0]][max_time]

    # method_vis = 'lgb_default'
    # method_vis = 'knn_default'
    # method_vis = 'logistic_default'

    # x1, x2 = 'n_features', 'n_classes'
    # x1, x2 = 'n_features', 'n_samples'

    df_by_ds_ = (
        results.groupby(level=0)
        .mean()
        .reset_index()
        .melt(id_vars="ds", var_name="method", value_name="score")
    )
    df_by_ds_ = df_by_ds_[df_by_ds_.method == method_vis].copy()
    if len(df_by_ds_) == 0:
        raise ValueError(f"Method {method_vis} not found in results matrix")

    df_by_ds_ = extend_df_by_ds(df_by_ds_.reset_index(drop=True), dss)
    df_by_ds_ = df_by_ds_[~df_by_ds_["score"].isna()]

    X_fit, y_fit = df_by_ds_[[x1, x2]], df_by_ds_["score"]
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.kernel_ridge import KernelRidge

    if clf == "linear":
        clf = LinearRegression()
    elif clf == "ridge":
        clf = Ridge()
    elif clf == "kernel_ridge":
        clf = KernelRidge()
    elif clf == "gaussian_process":
        kernel = DotProduct() + WhiteKernel()
        kernel = 1 * RBF(length_scale=1.0)
        noise_std = 0.75
        clf = GaussianProcessRegressor(
            kernel=kernel, random_state=0, alpha=noise_std**2
        )
    elif clf == "gradient_boosting":
        clf = GradientBoostingRegressor(n_estimators=1000)
    elif clf == "svm":
        from sklearn.svm import SVR
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        clf = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    else:
        raise ValueError(f"Unknown classifier {clf}")

    clf.fit(X_fit.values.astype(float), y_fit.values.astype(float))

    x = np.linspace(0, df_by_ds_[x1].max(), 100)
    y = np.linspace(0, df_by_ds_[x2].max(), 100)

    X, Y = np.meshgrid(x, y)
    Z = np.stack([X, Y], -1).reshape([-1, 2])

    Z = clf.predict(Z).reshape([X.shape[0], -1])

    contour = ax.contourf(X, Y, Z, 50, cmap="RdGy")
    plt.colorbar(contour, ax=ax)

    sns.scatterplot(data=df_by_ds_, x=x1, y=x2, hue="score", ax=ax, palette="RdGy")

    all_methods = results.columns.get_level_values(0).unique()

    plt.title(f"Wins of {method_vis} (Vs: {all_methods.values})")


def check_calibration(datasets, r):
    """
    Visualizes Regression calibration.
    """
    if type(datasets[0]) == list:
        raise NotImplementedError(
            "Please load datasets objects (not the old string format)"
        )

    # Is this ECE?
    def calculate_ratio_within_quantile(r, name, q):
        buckets = torch.tensor(r[name].additional_args["pred_full"]["buckets"]).cpu()
        # per_sample_upper_q = #torch.quantile(buckets, 0.9, axis=1)
        # per_sample_lower_q = #torch.quantile(buckets, 0.1, axis=1)
        buckets_sort = buckets.cpu().sort(descending=True)
        per_sample_lower_q = torch.gather(
            buckets_sort[0],
            1,
            torch.abs(buckets_sort[0].cumsum(1) - q).argmin(1).long().unsqueeze(1),
        )
        per_sample_bucket = (
            r[name]
            .additional_args["pred_full"]["criterion"]
            .cpu()
            .map_to_bucket_idx(r[name].y.cpu())
        )
        in_quantiles = buckets > per_sample_lower_q
        return (
            torch.gather(in_quantiles, 1, per_sample_bucket.unsqueeze(-1))
            .float()
            .mean(),
            in_quantiles.float().mean(),
        )

    names = [datasets[i].get_dataset_identifier() for i in range(len(datasets))]

    quantiles = [
        {
            "quantile": q,
            "within_quantile": float(calculate_ratio_within_quantile(r, name, q)[0]),
        }
        for q in np.arange(0.05, 1, 0.05)
        for name in names
    ]
    quantiles = pd.DataFrame(quantiles)
    ax = sns.lineplot(data=quantiles, x="quantile", y="within_quantile")
    plt.xlabel("Requested quantile")
    plt.ylabel("Actual quantile")

    # Create the x and y ticks
    x_ticks = np.arange(0, 1.1, 0.1)
    y_ticks = np.arange(0, 1.1, 0.1)

    # Set the x and y ticks
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Add gridlines
    ax.grid(True)


def wandb_get_runs(
    metric: str | List[str],
    tunable_params: List[str] | None,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    filters: dict = None,
    epoch_min: int = None,
    hpo_type: Literal["inference", "training"] = "training",
    task_type: Literal["regression", "multiclass", "survival"] = "multiclass",
) -> tuple:
    """
    This function can be used to retrieve wandb runs used in a hyperparameter search.
    It will go through the runs and look at their summaries to find the metric(s) (`metric`).
    If it can find the metric(s) it will add these to a summary table which includes
    the `tunable_paramas` as other columns, as well as some more columns, e.g. name.

    :param metric: The metric(s) to retrieve from the wandb runs. (str)
    :param tunable_params: The tunable parameters to retrieve from the wandb runs (List[str]). Can be None, then all are retrieved.
    :param wandb_entity: The entity to retrieve the runs from. (str) If None, the default entity is used.
    :param wandb_project: The project to retrieve the runs from. (str) If None, the default project is used.
    :param filters: Filters to apply to the runs. The filters are passed to wandb.Api.runs, they can e.g. be {"tags": "inference_optimization_0"}.
    :param epoch_min: The minimum epoch to include in the summary table. (int) If None, all epochs are included.
    :param hpo_type: The type of hyperparameter optimization. (str) Can be "inference" or "training". This has to be set correctly otherwise this might fail.
    :param task_type: The type of task. (str) Can be "regression", "multiclass" or "survival". This has to be set correctly otherwise this might fail.
    """

    import wandb

    api = wandb.Api(timeout=60)

    if wandb_project is None:
        wandb_project = local_settings.get_wandb_project(task_type)

    if wandb_entity is None:
        wandb_entity = local_settings.wandb_entity

    if filters is None:
        filters = {}

    runs = api.runs(
        f"{wandb_entity}/{wandb_project}",
        filters=filters,  # e.g. {"tags": "inference_optimization_0"}
    )

    runs_df, _ = _process_runs(
        runs,
        metric,
        tunable_params,
        epoch_min=epoch_min,
        hpo_type=hpo_type,
        task_type=task_type,
    )

    return runs_df


def compute_pareto_front_on_dataframe(df, columns):
    """
    Computes the n-dimensional Pareto front on a dataframe.
    :param df: DataFrame to compute the Pareto front on.
    :param columns: List of column names to compute the Pareto front on. Columns with a leading '-' are considered to be "larger is better".
    :return: DataFrame filtered to the Pareto front.
    """
    pareto_front = []

    # Adjust for columns where larger values are better by negating the data,
    # so that the problem is always "smaller is better".
    adjusted_df = df.copy()
    for i in range(len(columns)):
        if columns[i][0] == "-":
            columns[i] = columns[i][1:]
            adjusted_df[columns[i]] = -adjusted_df[columns[i]]

    def check_dominance(row):
        comparisons = []
        for column in columns:
            comparisons.append(adjusted_df[column] <= row[column])
        combined_comparison = comparisons[0]
        for comparison in comparisons[1:]:
            combined_comparison &= comparison
        return combined_comparison

    for index, row in adjusted_df.iterrows():
        if not (check_dominance(row) & (adjusted_df.index != index)).any():
            pareto_front.append(row)

    return df.loc[pd.DataFrame(pareto_front).index].sort_values(columns)


def compute_mixed_scores(df_by_ds: pd.DataFrame, metric_names: Sequence[str]):
    """
    Compute scores that are consisting of an average over normalized ranks (0-1) and normalized means (0-1).
    They can also be across multiple `metric_names`

    This function is typically used inside the `add_mixed_scores_to_runs_df` function below inside the `InferenceTunining.ipynb` notebook.

    :param df_by_ds: DataFrame generated by the evaluation_results, it is also saved to disk by default, from where
    you can load it.
    :param metric_names: Tuple of metric names to compute the scores for.
    """
    scores = {}

    for method_set_name, methods in [
        (
            "all",
            [
                "knn_default",
                "xgb_default",
                "catboost_default",
                "lgb_default",
                "knn",
                "xgb",
                "catboost",
                "lightgbm",
                "autogluon",
                "logistic",
                "logistic_default",
                "gpu_tabpfn_1",
                "single",
            ],
        ),
    ]:
        for method in methods:
            assert (
                method in df_by_ds.method.unique()
            ), f"{method} not in {df_by_ds.method.unique()}"
        df_by_ds = df_by_ds[df_by_ds.method.isin(methods)]
        results_matrix = get_results_matrix(df_by_ds, metric_names)
        rank_normalized_scores = []
        for metric_name in metric_names:
            rank_score = results_matrix["rank_in_pct"][metric_name][3600].groupby(
                level=0, axis=1
            ).mean().mean(0)["single"] / len(methods)
            normalized_score = (
                results_matrix["normalized"][metric_name][3600]
                .groupby(level=0, axis=1)
                .mean()
                .mean(0)["single"]
            )
            if tabular_metrics.get_scoring_direction(metric_name) != -1:
                normalized_dec_score = 1 - normalized_score
            else:
                normalized_dec_score = normalized_score

            rank_normalized_scores += [(rank_score, normalized_dec_score)]

            mean_score = (
                results_matrix["mean"][metric_name][3600]
                .groupby(level=0, axis=1)
                .mean()
                .mean(0)["single"]
            )

            scores = {
                **scores,
                f"rank_normalized_dec_mix_on_{method_set_name}_{metric_name}": (
                    rank_score + normalized_dec_score
                )
                / 2,
                f"rank_on{method_set_name}_{metric_name}": rank_score,
                f"normalized_dec_on{method_set_name}_{metric_name}": normalized_dec_score,
                f"mean_{metric_name}": mean_score,
                f"normalized_on_{method_set_name}_{metric_name}": normalized_score,
            }
        scores = {
            **scores,
            f"rank_normalized_dec_mix_on_{method_set_name}_{','.join(metric_names)}": np.mean(
                np.array(rank_normalized_scores)
            ),
        }
    return scores


def add_mixed_scores_to_runs_df(runs_df, metric_names=["roc", "acc"]):
    """
    Adds mixed scores to the runs_df.
    Computes a score that corresponds to a uniformly weighted average of the rank and the normalized score across the different metrics (`metric_names`) passed.

    This requires the evaluation_results_valid_hard_df_by_ds_pickle_path to be set in the runs_df, ensure this by
    passing `runs_df = wandb_get_runs(...,tunable_params=None)`.

    :param runs_df: DataFrame with the runs, computed by wandb_get_runs.
    :param metric_names: List of metric names to compute the mixed scores on.
    """
    scores_and_results = pd.DataFrame(
        [
            compute_mixed_scores(pd.read_pickle(eval_path), metric_names=metric_names)
            for eval_path in tqdm.tqdm(
                runs_df.evaluation_results_valid_hard_df_by_ds_pickle_path.to_list()
            )
        ],
        index=runs_df.index,
    )

    for col in scores_and_results.columns:
        runs_df[col] = scores_and_results[col]


def wandb_get_sweep_runs(
    wandb_entity: str,
    wandb_project: str,
    sweep_id: str,
    metric: str = None,
    epoch_min: int = None,
    load_per_epoch_metrics: bool = False,
    hpo_type: Literal["inference", "training"] = "training",
    verbose: bool = False,
    **kwargs,
) -> tuple:
    """
    Retrieve runs and relevant data for a given Weights & Biases sweep.

    Parameters:
        wandb_entity (str): The W&B entity (usually user or organization).
        wandb_project (str): The W&B project name.
        sweep_id (str): The ID of the sweep to fetch.
        metric (str, optional): Specific metric name to extract. Defaults to None.

    Returns:
        tuple: A DataFrame containing sweep runs, and a list of sweep tunable parameters.
    """

    import wandb

    # Input checks
    if not all(isinstance(arg, str) for arg in [wandb_entity, wandb_project, sweep_id]):
        raise TypeError("All arguments except 'metric' must be of type str.")

    try:
        api = wandb.Api()
        sweep = api.sweep(f"{wandb_entity}/{wandb_project}/{sweep_id}")
    except Exception as e:
        raise Exception(f"Failed to initialize W&B API or obtain sweep: {e}")

    # Extract data
    runs = sweep.runs
    sweep_tunable_params = list(sweep.config["parameters"].keys())

    if metric is None:
        metric_name = sweep.config["metric"]["name"]
    else:
        metric_name = metric

    runs_df, history_df = _process_runs(
        runs,
        metric_name,
        sweep_tunable_params,
        epoch_min=epoch_min,
        load_per_epoch_metrics=load_per_epoch_metrics,
        hpo_type=hpo_type,
        verbose=verbose,
        **kwargs,
    )

    return runs_df, history_df, sweep_tunable_params, metric_name


def _smoothen_metric_history(smoothing_window, epoch_max, metric_history_dict):
    # smoothen metrics
    if smoothing_window > 0:
        for epoch in range(0, epoch_max):
            if epoch < smoothing_window:
                continue
            metric_history_dict[f"metric_history_{epoch}"] = dict(
                zip(
                    metric_history_dict[f"metric_history_{epoch}"].keys(),
                    np.mean(
                        [
                            list(
                                metric_history_dict[
                                    f"metric_history_{epoch - i}"
                                ].values()
                            )
                            for i in range(0, smoothing_window)
                        ],
                        axis=0,
                    ).tolist(),
                )
            )
    return metric_history_dict


def _load_per_epoch_metrics_helper(
    row, run, metric_name: str | list[str], epoch_max, history_df, smoothing_window=0
):
    if not isinstance(metric_name, list):
        metric_name = [metric_name]
    keys = [m for m in metric_name] + ["_runtime"]
    metric_history = run.history(keys=keys)
    row["metric_history"] = metric_history
    metric_history_dict = {}
    last_time = -1

    for epoch in range(0, epoch_max):
        try:
            metric_history_dict[f"metric_history_{epoch}"] = dict(
                zip(
                    metric_history.columns.values[:-1],
                    metric_history.iloc[epoch].values[:-1].tolist(),
                )
            )
            metric_history_dict[f"time_{epoch}"] = metric_history.iloc[epoch].values[-1]

            row_ = row.copy()
            row_["metric"] = metric_history_dict[f"metric_history_{epoch}"]
            row_["epoch"] = epoch
            row_["time"] = metric_history_dict[f"time_{epoch}"]
            last_time = row_["time"]
            history_df.append(row_)
        except Exception as e:
            metric_history_dict[f"metric_history_{epoch}"] = dict(
                zip(metric_name, [np.nan for _ in range(0, len(metric_name))])
            )
            metric_history_dict[f"time_{epoch}"] = np.nan

    metric_history_dict = _smoothen_metric_history(
        smoothing_window, epoch_max, metric_history_dict
    )

    row["time"] = last_time
    row.update(metric_history_dict)

    return row


def _process_runs_for_global_time(runs_df, history_df, epoch_max):
    try:
        runs_df["time_deviation_from_std"] = (
            runs_df["time"] - runs_df["time"].mean()
        ) / runs_df["time"].std()

        sel = (runs_df["time_deviation_from_std"] > -2) | (
            runs_df["time"] > 60 * 60 * 4
        )
        runs_df = runs_df[sel]
        print(f"Removed runs with runtime too short: {(~sel).sum()}")

        history_df["time_c"] = pd.cut(
            history_df["time"], bins=history_df["epoch"].max() // 2, labels=False
        )

        # Select a common time for all runs
        selected_time = runs_df["time"].mean() - runs_df["time"].std() * 0.5
        selected_time = runs_df["time"].quantile(0.25)
        selected_time_available = runs_df["time"] >= selected_time
        print(
            f"Common time selected: {selected_time}, available for {selected_time_available.sum()}/{len(selected_time_available)} rows (Mean ({runs_df['time'].mean()}) - Std ({runs_df['time'].std()}) * 0.5), Max: {runs_df['time'].max()}"
        )

        def get_metric_for_run_at_time(x, selected_time):
            for i in range(0, epoch_max):
                if not any([v == np.nan for v in x[f"metric_history_{i}"].values()]):
                    if x[f"time_{i}"] > selected_time:
                        return x[f"metric_history_{i}"]
            return {m: np.nan for m in x[f"metric_history_{i}"].keys()}

        runs_df.loc[:, "metric_at_common_time"] = runs_df.apply(
            lambda x: get_metric_for_run_at_time(x, selected_time),
            axis=1,
        )
    except Exception as e:
        print(f"Failed to compute time deviation from std: {e}")

    return runs_df, history_df


def _process_runs(
    runs,
    metric_name: str | list[str],
    tunable_params: List[str] | None,
    epoch_min: int = None,
    load_per_epoch_metrics=False,
    epoch_max: int = 500,
    hpo_type: Literal["inference", "training"] = "training",
    smoothing_window: int = 0,
    task_type: str = "multiclass",
    verbose: bool = False,
):
    if not isinstance(metric_name, list):
        metric_name = [metric_name]

    runs_df, history_df = [], []

    for run in tqdm.auto.tqdm(runs):
        row = {}
        row["summary"] = run.summary._json_dict
        row["config"] = {k: v for k, v in run.config.items() if not k.startswith("_")}
        row["name"] = run.name

        # Check if metric is present
        row["metric"] = {m: row["summary"].get(m, None) for m in metric_name}
        if any([metric_value is None for metric_value in row["metric"].values()]):
            if verbose:
                print(
                    f"Skipping run {run.name} because metric is missing, metric: {row['metric']}, available metrics: {row['summary']}"
                )
            continue

        # Check if all tunable parameters are present
        try:
            config = {**row["config"], **row["summary"]}
            if tunable_params is not None:
                row.update({p: config[p] for p in tunable_params if p in config})
                tunable_params_not_in_config = [
                    p for p in tunable_params if p not in config
                ]
            else:
                row.update(
                    {p: config[p] for p in config if p.startswith("inference_config_")}
                )
                tunable_params_not_in_config = []
            if tunable_params_not_in_config:
                assert (
                    "continue_model_wandb_id" in config
                ), f"Run {run.name} is missing some tunable parameters {tunable_params_not_in_config}. Cannot load stuff from its config."
                from .estimator.configs import (
                    get_run_from_wandb,
                )

                try:
                    run = get_run_from_wandb(
                        config["continue_model_wandb_id"], task_type=task_type
                    )
                except FileNotFoundError:
                    print(f"WANDB ID {config['continue_model_wandb_id']} not found.")
                row.update({p: run.config[p] for p in tunable_params_not_in_config})

        except KeyError as e:
            print(f"Run {run.name} is missing some tunable parameters {e}.")
            # print traceback
            import traceback

            traceback.print_exc()

            print(row["config"])

        # Adding hpo type specific metrics
        if hpo_type == "inference":
            time_col = (
                "_mean_".join(metric_name[0].split("_mean_")[:-1]) + "_mean_time_(s)"
            )
            if mean_time := row["summary"].get(time_col, None):
                row["inference_time"] = mean_time

        if epoch_min is not None and hpo_type == "training":
            if row["summary"]["epoch"] < epoch_min:
                continue

            if load_per_epoch_metrics:
                row = _load_per_epoch_metrics_helper(
                    row,
                    run,
                    metric_name,
                    epoch_max,
                    history_df,
                    smoothing_window=smoothing_window,
                )
        runs_df.append(row)

    print(
        f"Loaded {len(runs)} runs, len tunable_params = {len(tunable_params) if tunable_params is not None else None}"
    )
    print(f"Removed runs with missing metrics: {len(runs) - len(runs_df)}")

    if len(runs_df) == 0:
        raise ValueError(
            f"No runs with metrics found, available in summary of first run: {runs[0].summary}"
        )

    runs_df = pd.DataFrame(runs_df)
    history_df = pd.DataFrame(history_df)

    if hpo_type == "training":
        runs_df, history_df = _process_runs_for_global_time(
            runs_df, history_df, epoch_max
        )
        for metric_name_ in metric_name:
            history_df.loc[:, metric_name_] = history_df.loc[:, "metric"].apply(
                lambda x: x[metric_name_]
            )

    for metric_name_ in metric_name:
        runs_df.loc[:, metric_name_] = runs_df.loc[:, "metric"].apply(
            lambda x: x[metric_name_]
        )

    return runs_df, history_df


def visualize_hpo_results(
    runs_df: pd.DataFrame,
    sweep_tunable_params: list,
    metric_name: str | list[str],
    epochs_to_plot: list = None,
    y: str = "metric",
    dependence_parameter: str | None = None,
    hue: str | None = None,
) -> tuple:
    """
    Visualize hyperparameter optimization (HPO) results.

    Parameters:
        runs_df (pd.DataFrame): DataFrame containing sweep runs.
            Columns should include sweep_tunable_params and 'metric'. Rows should be runs.

        sweep_tunable_params (list): List of sweep tunable parameters.
        hue (str, optional): Column name to use for hue. Defaults to None. Example: 'inference_time'.

    Returns:
        tuple: A matplotlib figure and axes.
    """
    assert (hue in runs_df.columns) or (
        hue is None
    ), f"Column {hue=} not in runs_df, which has columns {runs_df.columns}"

    # Input checks
    if not isinstance(runs_df, pd.DataFrame):
        raise TypeError("'runs_df' must be a pandas DataFrame.")

    if not isinstance(sweep_tunable_params, list):
        raise TypeError("'sweep_tunable_params' must be a list.")

    assert dependence_parameter in [None, "epochs"] + sweep_tunable_params

    if dependence_parameter == "epochs":
        assert (
            epochs_to_plot is not None
        ), "epochs_to_plot must be provided if dependence_parameter is epochs"

    if dependence_parameter is None:
        dependence_items = [None]
    else:
        dependence_items = (
            epochs_to_plot
            if dependence_parameter == "epochs"
            else pd.unique(runs_df[dependence_parameter])
        )

    if not isinstance(metric_name, list):
        metric_name = [metric_name]

    # Visualization code
    n_cols = len(dependence_items)
    n_rows = len(sweep_tunable_params)
    fig, axs = plt.subplots(
        ncols=n_cols, nrows=n_rows, figsize=(16, n_rows * 10), sharey=True
    )

    for i, metric_name_ in enumerate(metric_name):
        _scatter_plot_helper(
            runs_df,
            y,
            hue,
            metric_name_,
            sweep_tunable_params,
            dependence_items,
            n_rows,
            n_cols,
            axs,
            dependence_parameter,
            annotate_stats=i == 0,
        )

    plt.tight_layout()
    plt.subplots_adjust(hspace=3.0)

    return fig, axs


def _scatter_plot_helper(
    runs_df,
    y,
    hue,
    metric_name,
    sweep_tunable_params,
    dependence_items,
    n_rows,
    n_cols,
    axs,
    dependence_parameter,
    annotate_stats=True,
):
    runs_df = runs_df.copy(deep=True)

    runs_df.loc[:, y] = runs_df.loc[:, y].apply(lambda x: x[metric_name])

    # replace nans by "nan"
    runs_df = runs_df.fillna(-100)
    runs_df[y] = runs_df[y]

    def make_str(x):
        return x.astype(str) if not pd.api.types.is_numeric_dtype(x.dtype) else x

    runs_df = runs_df.apply(make_str, axis=0)

    for i, param_key in tqdm.tqdm(list(enumerate(sweep_tunable_params))):
        for j, dependence_item in enumerate(dependence_items):
            if n_rows == 1 and n_cols == 1:
                ax = axs
            elif n_rows == 1:
                ax = axs[j]
            elif n_cols == 1:
                ax = axs[i]
            else:
                ax = axs[i, j]

            if dependence_parameter is None:
                runs_df_ = runs_df.copy(deep=True)
            elif dependence_parameter == "epochs":
                runs_df_ = runs_df.copy(deep=True)
                runs_df_.loc[:, y] = runs_df_.loc[
                    :, f"metric_history_{dependence_item}"
                ]
            else:
                runs_df_ = runs_df[
                    runs_df.loc[:, dependence_parameter].astype(str)
                    == str(dependence_item)
                ]

                print(
                    f"Filtering by {dependence_parameter} = {dependence_item}, {len(runs_df_)}"
                )
            if len(runs_df_) == 0:
                continue

            runs_df_ = runs_df_[runs_df_[y] != -100]

            if len(pd.unique(runs_df[param_key])) < len(runs_df) / 2:
                # Categorical parameter
                sns.swarmplot(
                    data=runs_df_,
                    y=y,
                    x=param_key,
                    ax=ax,
                    hue=hue,  # , label=metric_name
                )
                if len(runs_df_) > 1:
                    sns.boxplot(
                        data=runs_df_,
                        y=y,
                        x=param_key,
                        showmeans=True,
                        meanline=True,
                        medianprops={"visible": True},
                        whiskerprops={"visible": False, "color": "red"},
                        zorder=10,
                        showfliers=False,
                        showbox=False,
                        showcaps=True,
                        ax=ax,
                    )

                if len(np.unique(runs_df_[param_key])) < 5 and annotate_stats:
                    # Adds statistical significance annotations
                    add_stat_annotation(runs_df_, ax, param_key, y)

            else:
                # Numerical parameter
                sns.scatterplot(data=runs_df_, y=y, x=param_key, ax=ax, hue=hue)

            ax.set(xlabel="", ylabel="")
            ax.tick_params(axis="x", labelrotation=90)
            ax.set_title(param_key[-50:])
            if dependence_parameter:
                ax.set_title(
                    f"{param_key},\n {dependence_parameter} = {dependence_item}"
                )
            else:
                ax.set_title(f"{param_key}")

            if hue is not None:
                if j + 1 == n_cols:
                    ax.legend(
                        title="Inference Time",
                        bbox_to_anchor=(1.05, 1),
                    )
                else:
                    if legend := ax.get_legend():
                        legend.set_visible(False)


def add_stat_annotation(runs_df, ax, param_key, y):
    """
    Annotates a plot with statistical significance between groups of data.

    This function performs a t-test between each pair of groups specified by `param_key`
    and annotates a given matplotlib axis `ax` with the p-values from the tests.

    Parameters:
    -----------
    runs_df : pandas.DataFrame
        The DataFrame containing the data to be plotted and tested.
        It should have a column named 'metric' and another column named according to `param_key`.

    ax : matplotlib.axes.Axes
        The axes object where the annotations will be added.

    param_key : str
        The column name in `runs_df` that contains the parameter keys to group by
        for the statistical test.

    Returns: None

    Notes:
    ------
    The function assumes that the column named 'metric' contains the numerical data
    to be tested and that the data are normally distributed for the purpose of the t-test.
    """

    import scipy.stats as stats

    # Group data by param_key
    groups = runs_df.groupby(param_key)

    # Perform statistical test (e.g., t-test)
    group_keys = groups.groups.keys()
    p_values = {}
    for key1pos, key1 in enumerate(group_keys):
        for key2pos, key2 in enumerate(group_keys):
            if key1 >= key2:
                continue
            if key2pos - 1 != key1pos:  # Only plots adjacent values
                continue

            stat, p_value = stats.ttest_ind(
                groups.get_group(key1)[y],
                groups.get_group(key2)[y],
            )
            p_values[(key1, key2)] = (p_value, (key1pos, key2pos))

    offset = 0.03
    max_value = runs_df[y].min() + offset + 0 * (runs_df[y].max() - runs_df[y].min())

    # Annotate plot with statistical significance
    for (key1, key2), (p_value, (key1pos, key2pos)) in p_values.items():
        ax.annotate(
            f"p={p_value:.3f}",
            xy=(
                (key1pos + key2pos) / 2,
                max_value,
            ),  # Positioning at the midpoint of key1 and key2
            ha="center",  # Horizontal alignment
            va="bottom",  # Vertical alignment
            arrowprops=dict(arrowstyle="-[, widthB=2.5, lengthB=0.2"),
        )


def create_wandb_sweep(
    task_type: str,
    sweep_space_name: str,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    metric_name: Optional[str] = None,
    sweep_method: str = "random",
) -> Tuple[str, Union[dict, Any]]:
    """
    Creates a sweep configuration for hyperparameter tuning using Weights and Biases (wandb).

    This function sets up a sweep for hyperparameter tuning using the wandb library.
    It allows the user to specify the type of task, sweep space, and various other optional settings.

    Parameters:
    -----------
    task_type : str
        The type of machine learning task. This influences the choice of evaluation metric.

    sweep_space_name : str
        The name of the sweep configuration space, which should be present in `hp_tuning_sweep_configuration.sweep_spaces`.

    wandb_project : Optional[str], default=None
        The name of the Weights and Biases project. If not provided, it defaults to a locally defined setting.

    wandb_entity : Optional[str], default=None
        The entity (user or team) under which the Weights and Biases project resides. Defaults to a locally defined setting.

    metric_name : Optional[str], default=None
        The name of the evaluation metric to use. If not provided, it is inferred from `task_type`.

    sweep_method : str, default="random"
        The hyperparameter search method. Supports "random", "grid", and "bayesian" methods.

    Returns:
    --------
    Tuple[str, Union[dict, Any]]
        A tuple containing the sweep ID and the sweep configuration dictionary.

    """
    import wandb
    from . import hp_tuning_sweep_configuration

    if wandb_entity is None:
        wandb_entity = local_settings.wandb_entity
    if wandb_project is None:
        wandb_project = local_settings.get_wandb_project(task_type)
    if metric_name is None:
        metric_name = get_metric_name(get_main_eval_metric(task_type))

    sweep_space, sweep_space_name = (
        hp_tuning_sweep_configuration.sweep_spaces.get(sweep_space_name),
        sweep_space_name,
    )

    sweep_config = load_sweep_configuration(
        tunable_params=sweep_space,
        tuning_metric=f"valid/5_splits/mean_{metric_name}",
        method=sweep_method,
        name=f"{sweep_space_name}_{task_type}",
    )
    sweep_id = wandb.sweep(
        sweep=sweep_config, project=wandb_project, entity=wandb_entity
    )
    return sweep_id, sweep_config


def load_sweep_configuration(
    name, tunable_params, tuning_metric, goal="maximize", method="random", hyperband=-1
):
    """Loads a sweep configuration for wandb tuning.

    config method: grid, bayes, random

    Sample usage:
    >>> from scripts.tabular_evaluation_notebook_utils import load_sweep_configuration
    >>> from scripts import hp_tuning_sweep_configuration
    >>> sweep_config = load_sweep_configuration(
    >>>                     tunable_params=hp_tuning_sweep_configuration.sweep_configuration_mlp_prior_weight_init
    >>>                     , tuning_metric='valid/spearman'
    >>>                     , name = 'mlp_prior_weight_init')
    """
    hyperband_config = {"early_terminate": {"type": "hyperband", "min_iter": 25}}

    sweep_config = {
        "method": method,
        "name": name,
        "metric": {"name": tuning_metric, "goal": goal},
        "parameters": tunable_params,
    }

    if hyperband > 0:
        sweep_config.update(hyperband_config)

    return sweep_config
