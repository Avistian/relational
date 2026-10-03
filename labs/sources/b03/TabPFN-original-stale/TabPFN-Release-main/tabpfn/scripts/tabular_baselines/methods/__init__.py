from .tabpfn import transformer_metric

try:
    from .autogluon import autogluon_metric
    from .autogluon_only_v1 import only_autogluon_metric
    from .autosklearn import autosklearn_metric, autosklearn2_metric
    from .dummy import dummy_metric
    from .logistic import logistic_metric
    from .ridge import ridge_metric
    from .tabpfn_1 import tabpfn1_metric
    from .gaussian_process import gp_metric
    from .lightgbm import lightgbm_metric
    from .random_forest import random_forest_metric
    from .svm import svm_metric
    from .knn import knn_metric
    from .tabnet import tabnet_metric
    from .sklearn_mlp import mlp_metric
    from .xgb import xgb_metric
    from .catboost import catboost_metric
    from .autogluon_v1 import custom_autogluon_metric
    from .hyperfast_default import hyperfast_bench_metric
    from .hyperfast_opt import hyperfast_opt_bench_metric
    from .tabpfn_bench_ens import tabpfn_bench_ens_metric
    from .tabpfn_tuned import tabpfn_tuned_metric
except ImportError:
    pass
