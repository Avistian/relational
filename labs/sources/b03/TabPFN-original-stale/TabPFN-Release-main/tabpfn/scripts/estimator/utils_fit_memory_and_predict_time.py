"""
Workflow to fit the functions for memory and predict time estimation of TabPFN.
Will load a model, run it with memory tracking and time tracking and fit a linear model to the data.
"""

import matplotlib.pyplot as plt

import seaborn as sns
import random
from functools import partial

from ..tabular_evaluation import evaluate_simple
from tabpfn.datasets import *
from tabpfn.best_models import get_best_tabpfn

from .base import (
    LOG_MEMORY_USAGE_PATH,
    LOG_COMPUTE_TIME_PATH,
)
from ..tabular_baselines import transformer_metric


def run_workflow():
    model = get_best_tabpfn(
        task_type,
        model_type="single_fast",
        device=device,
    )

    transformer_metric_with_model = partial(transformer_metric, classifier=model)

    df_exp = []
    debug_datasets = [
        ds for ds in datasets_dict[f"test_{task_type}"] if ds.x.shape[0] < 6000
    ]
    for N_ensembles in range(1, 30, 3):
        for n in range(0, 10):
            ds = copy.deepcopy(debug_datasets[0])
            n_samples, n_features = (
                random.randint(1, 70) ** 2,
                random.randint(1, 70) ** 2,
            )
            ds.x = torch.randn(n_samples, n_features)
            ds.y = torch.randn(n_samples) > 0
            ds.splits = None
            model.use_poly_features = random.random() > 0.5
            model.save_peak_memory = "True" if random.random() > 0.5 else "False"
            model.n_estimators = N_ensembles

            print(
                n_samples,
                n_features,
                model.n_estimators,
                model.use_poly_features,
                model.save_peak_memory,
            )

            split = 1
            try:
                metrics_tabpfn, r_tabpfn = evaluate_simple(
                    transformer_metric_with_model,
                    [ds],
                    task_type=task_type,
                    split_number=split,
                )
                metrics_tabpfn
                df_exp += [
                    {
                        "time": metrics_tabpfn["time"],
                        "ce": metrics_tabpfn["ce"],
                        "ensembles": N_ensembles,
                        "ds": ds.name,
                        "features": ds.x.shape[1],
                        "samples": ds.x.shape[0],
                    }
                ]
            except Exception as e:
                print("failed", e)
                pass

    mode = "compute"
    # mode = "memory"

    if mode == "compute":
        path = LOG_COMPUTE_TIME_PATH
        y = "transformer_time"
        map_f = model._log_compute_time_add_estimated_time
    else:
        path = LOG_MEMORY_USAGE_PATH
        y = "peak_mem"
        map_f = model._log_memory_usage_add_estimated_memory

    time_data = pd.read_csv(path)
    time_data = map_f(time_data)

    time_data = time_data[time_data[y] > 0]
    # time_data = time_data[time_data["n_estimators"] == 1]
    time_data = time_data[time_data["save_peak_mem_factor"] == True]
    time_data["feats_square"] = time_data["num_features"] * time_data["num_features"]
    time_data["samples_square"] = time_data["num_samples"] * time_data["num_samples"]
    time_data["samples_plus_features"] = (
        time_data["num_samples"] + time_data["num_features"]
    )
    time_data["samples_plus_features_square"] = time_data["samples_plus_features"] ** 2
    time_data["samples_times_features"] = (
        time_data["num_samples"] * time_data["num_features"]
    )
    time_data["samples_times_features_square"] = (
        time_data["samples_times_features"] ** 2
    )
    time_data["max_samples_and_features"] = time_data[
        ["num_samples", "num_features"]
    ].max(axis=1)
    time_data["max_samples_and_features_square"] = (
        time_data["max_samples_and_features"] ** 2
    )

    if False:
        time_data["transformer_time_norm"] = (
            time_data["transformer_time"] / time_data["transformer_time"].max()
        )
        time_data["estimated_time_norm"] = (
            time_data["estimated_time"] / time_data["estimated_time"].max()
        )
        time_data["time_diff"] = (
            time_data["transformer_time_norm"] / time_data["estimated_time_norm"]
        )

    from sklearn import datasets, linear_model

    X_keys = [
        "num_features",
        "num_samples",
        "samples_square",
        "feats_square",
        "samples_plus_features_square",
        "samples_plus_features",
        "max_samples_and_features",
        "max_samples_and_features_square",
        "samples_times_features",
        "samples_times_features_square",
    ]
    if mode == "compute":
        X_keys += ["n_estimators", "n_batches"]
    else:
        # For mem
        time_data["samples_times_features_times_batch"] = (
            time_data["samples_times_features"] * time_data["batch_size"]
        )
        X_keys += ["samples_times_features_times_batch"]

    X = time_data[X_keys]
    X_max = X.max()
    X = X / X_max
    alpha = 10000000.0
    clf = linear_model.Lasso(positive=True, alpha=alpha).fit(X, time_data[y])
    time_data["linreg"] = clf.predict(X)

    print("Before", len(time_data))
    time_data = time_data[(time_data[y] - time_data["linreg"]) < 0.9]
    print("After", len(time_data))

    X = time_data[X_keys]
    X_max = X.max()
    X = X / X_max

    clf = linear_model.Lasso(positive=True, alpha=alpha).fit(X, time_data[y])
    time_data["linreg"] = clf.predict(X)

    fig1, ax1 = plt.subplots()
    sns.scatterplot(time_data, x=y, y="estimated", hue="use_poly_features")
    plt.grid()

    fig1, ax1 = plt.subplots()
    sns.scatterplot(time_data, x=y, y="linreg")
    plt.grid()
