from __future__ import annotations

import warnings
from functools import partial

import numpy as np

try:
    from tabpfn.scripts.tabular_baselines.methods import (
        autogluon_metric,
        autosklearn2_metric,
        catboost_metric,
        dummy_metric,
        gp_metric,
        hyperfast_bench_metric,
        hyperfast_opt_bench_metric,
        knn_metric,
        lightgbm_metric,
        logistic_metric,
        mlp_metric,
        only_autogluon_metric,
        random_forest_metric,
        ridge_metric,
        svm_metric,
        tabpfn1_metric,
        tabpfn_bench_ens_metric,
        tabpfn_tuned_metric,
        xgb_metric,
    )
    from .methods.tabular_baselines_quantile_regression import (
        linear_quantile_metric,
        random_forest_quantile_metric,
    )
except ImportError as e:
    warnings.warn(
        f"Due to error {e}, we could not import all baseline methods and are aborting their imports.",
        stacklevel=2,
    )

from .methods import transformer_metric

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# We should keep it that way, since % autoreload in jupyter notebook destroys the from A import B structure
# This destroys calls to isinstance(var , B). But isinstance(var, A.B) still works.
# https://github.com/ipython/ipython/issues/12399

param_grid, param_grid_hyperopt = {}, {}


def _get_additional_sklearn_models(task_type):
    rf_args = dict(
        n_estimators=300,
        max_leaf_nodes=15000,
        n_jobs=-1,
        random_state=np.random.RandomState(42),
        bootstrap=True,
    )
    if task_type == "multiclass":
        from sklearn.ensemble import RandomForestClassifier

        return [("rf", RandomForestClassifier(**rf_args))]
    if task_type == "regression":
        from sklearn.ensemble import RandomForestRegressor

        return [("rf", RandomForestRegressor(**rf_args))]

    raise NotImplementedError(f"Unknown task type {task_type}")


def get_clf_dict(task_type):
    if task_type in ("multiclass", "regression"):
        clf_dict = {
            # ---- Part of the Benchmark
            # -- Default Baselines
            "svm_default": partial(svm_metric, no_tune={}),
            "knn_default": partial(knn_metric, no_tune={}),
            "xgb_default": partial(xgb_metric, no_tune={}),
            "catboost_default": partial(catboost_metric, no_tune={}),
            "lgb_default": partial(lightgbm_metric, no_tune={}),
            "random_forest_default": partial(random_forest_metric, no_tune={}),
            # -- Tuned Baselines
            "knn": knn_metric,
            "svm": svm_metric,
            "xgb": xgb_metric,
            "catboost": catboost_metric,
            "random_forest": random_forest_metric,
            "lightgbm": lightgbm_metric,
            # -- Deep Learning Baselines
            "hyperfast_opt_default": partial(hyperfast_opt_bench_metric, no_tune={}),
            "hyperfast": hyperfast_bench_metric,
            "hyperfast_opt": hyperfast_opt_bench_metric,
            # -- AutoML
            "autogluon_1_bq": partial(
                only_autogluon_metric,
                preset="best_quality",
            ),
            # -- TabPFN Versions
            "tabpfn_bench_default": partial(
                transformer_metric,
                classifier="default",
            ),
            "tabpfn_bench_tuned": partial(tabpfn_tuned_metric),
            # -- TabPFN Ensemble Versions
            "tabpfn_bench_post_hoc_ens_random_portfolio": partial(
                tabpfn_bench_ens_metric,
                extra_kwargs={"tabpfn_base_model_source": "random_portfolio"},
            ),
            "autogluon": autogluon_metric,
            "autogluon_1_hq": partial(
                only_autogluon_metric,
                preset="high_quality",
            ),
            # -- Not used right now
            "transformer": transformer_metric,
            "sklearn_mlp_default": partial(mlp_metric, no_tune={}),
            "gp_default": partial(gp_metric, no_tune={}),
            "sklearn_mlp": mlp_metric,
            "gp": gp_metric,
            "autosklearn2": autosklearn2_metric,
        }
        if task_type == "multiclass":
            clf_dict.update(
                {
                    "logistic": logistic_metric,
                    "logistic_default": partial(logistic_metric, no_tune={}),
                    "dummy_most_frequent_metric_default": partial(
                        dummy_metric,
                        strategy="most_frequent",
                        no_tune={},
                    ),
                    "tabpfn_1": tabpfn1_metric,
                    "gpu_tabpfn_1": partial(tabpfn1_metric, device="cuda"),
                },
            )
        elif task_type == "regression":
            clf_dict.update(
                {
                    "ridge": ridge_metric,
                    "ridge_default": partial(ridge_metric, no_tune={}),
                    "dummy_mean_metric_default": partial(
                        dummy_metric,
                        strategy="mean",
                        no_tune={},
                    ),
                },
            )
    elif task_type == "quantile_regression":
        return {
            "quantile_random_forest_default": partial(
                random_forest_quantile_metric,
                no_tune={},
            ),
            "quantile_random_forest": random_forest_quantile_metric,
            "linear_quantile_default": partial(linear_quantile_metric, no_tune={}),
            "linear_quantile": linear_quantile_metric,
            "xgb": xgb_metric,
            "xgb_default": partial(xgb_metric, no_tune={}),
            "transformer": transformer_metric,
            "autogluon_1_bq": partial(
                only_autogluon_metric,
                preset="best_quality",
            ),
            "autogluon": autogluon_metric,
        }
    elif task_type == "survival":
        from .methods.tabular_baselines_survival import (
            lifelines_coxph_metric,
            survival_auton_metric,
            survival_autoprognosis_metric,
            survival_coxnet_metric,
            survival_coxph_metric,
            survival_coxph_pycox_metric,
            survival_grad_boost_metric,
            survival_ipc_ridge_metric,
            survival_random_forest_metric,
            survival_svm_metric,
            survival_tree_metric,
        )
        from .methods.xgb_survival import survival_xgb_metric

        clf_dict = {
            # "kaplan_meier_default": partial(survival_kaplan_meier_metric, no_tune={}),
            # "kaplan_meier": survival_kaplan_meier_metric,
            ##
            "coxph_default": partial(survival_coxph_metric, no_tune={}),
            "coxph": survival_coxph_metric,
            "lifelines_coxph_metric_default": partial(
                lifelines_coxph_metric,
                no_tune={},
            ),
            "lifelines_coxph_metric": lifelines_coxph_metric,
            "pycox_coxph_metric_default": partial(
                survival_coxph_pycox_metric,
                no_tune={},
            ),
            "pycox_coxph_metric": survival_coxph_pycox_metric,
            ##
            "coxnet_default": partial(survival_coxnet_metric, no_tune={}),
            "coxnet": survival_coxnet_metric,
            ##
            "survival_random_forest_default": partial(
                survival_random_forest_metric,
                no_tune={},
            ),
            "survival_random_forest": survival_random_forest_metric,
            ##
            "survival_ipc_ridge_default": partial(
                survival_ipc_ridge_metric,
                no_tune={},
            ),
            "survival_ipc_ridge": survival_ipc_ridge_metric,
            ##
            "survival_tree_default": partial(survival_tree_metric, no_tune={}),
            "survival_tree": survival_tree_metric,
            ##
            "survival_svm_metric_default": partial(survival_svm_metric, no_tune={}),
            "survival_svm_metric": survival_svm_metric,
            ##
            "survival_grad_boost_default": partial(
                survival_grad_boost_metric,
                no_tune={},
            ),
            "survival_grad_boost": survival_grad_boost_metric,
            ##
            "survival_xgboost_default": partial(survival_xgb_metric, no_tune={}),
            "survival_xgboost": survival_xgb_metric,
            ##
            "survival_auton_dsm_default": partial(survival_auton_metric, no_tune={}),
            "survival_autoprognosis_metric": partial(
                survival_autoprognosis_metric,
                no_tune={},
            ),
        }
    else:
        raise NotImplementedError(f"Unknown task type {task_type}")
    return clf_dict


clf_relabeler = {
    "linear_default": "Linear (default)",
    "transformer": "Tabular PFN",
    "autogluon": "Autogluon",
    "autosklearn2": "Autosklearn2",
    "ridge": "Ridge",
    "gp": "GP (RBF)",
    "bayes": "BNN",
    "tabnet": "Tabnet",
    "logistic": "Log. Regr.",
    "knn": "KNN",
    "catboost": "CatBoost",
    "xgb": "XGB",
    "catboost_default": "CatBoost (default)",
    "xgb_default": "XGB (default)",
    "knn_default": "KNN (default)",
    "lgb_default": "LightGBM (default)",
    "lightgbm": "LightGBM",
    "logistic_default": "Log. Regr. (default)",
    "ridge_default": "Ridge (default)",
    "dummy_most_frequent_metric_default": "Dummy (most frequent)",
    "tabpfn": "TabPFN (default)",
    "tabpfn(single)": "TabPFN (default)",
    "single_fast": "TabPFN (No Ensemble)",
    "single": "TabPFN (Ensembling)",
    "rf_pfn": "TabPFN (RF)",
    "stacking": "TabPFN (Stacking)",
    "bagging": "TabPFN (Bagging)",
    "weighted_average": "TabPFN (Weighted Average)",
    "mlp_default": "MLP (default)",
    "mlp": "MLP",
    "sklearn_mlp_default": "MLP (default)",
    "task_type_agnostic_linear_default": "Linear (default)",
    "sklearn_mlp": "MLP (sklearn)",
    "svm_default": "SVM (default)",
    "svm": "SVM",
    "tabpfn_1": "TabPFN (V1)",
    "autogluon_1_bq": "Autogluon(V1,BQ)",
}


def clf_relabeler_with_time(clf, time):
    def time_mapper(time):
        if time == 3600:
            return "1h "
        else:
            return ""

    if (
        "default" in clf
        or "rf_pfn" in clf
        or "single" in clf
        or "stacking" in clf
        or "tabpfn" in clf
    ):
        return clf_relabeler.get(clf, clf)
    return clf_relabeler.get(clf, clf) + f" ({time_mapper(time)}tuned)"
