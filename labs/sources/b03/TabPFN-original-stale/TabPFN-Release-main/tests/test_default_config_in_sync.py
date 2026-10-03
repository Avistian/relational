import pytest

from tabpfn.scripts.estimator import base
from tabpfn import best_models

from tabpfn.utils import compare_nested_dicts
from tabpfn.local_settings import overwrite_cluster_settings


@overwrite_cluster_settings("UNITTEST")
@pytest.mark.parametrize("task_type", ["regression", "multiclass"])
def test_default_config_in_sync(task_type):
    """
    The default settings when calling TabPFNClassifier/TabPFNRegressor are always supposed
    to get the same model as our "single" config.
    This test checks if the default settings are in sync with the single config.
    """
    estimator_based_on_config = best_models.get_best_tabpfn(
        task_type, model_type="single"
    )
    estimator_based_on_config.init_model_and_get_model_config()
    if task_type == "regression":
        TabPFNEstimator = base.TabPFNRegressor
    else:
        TabPFNEstimator = base.TabPFNClassifier
    estimator = TabPFNEstimator()
    estimator.init_model_and_get_model_config()

    def compare_estimators():
        # as np.random.default_rng(0) != np.random.default_rng(0), we need to exclude the random number generator
        # as everything is part of the defintion of predict_function_for_shap including rnd, we need to exclude it as well
        drop_keys = ["_rnd", "predict_function_for_shap"]

        estimator_dict = {
            k: v for k, v in estimator.__dict__.items() if k not in drop_keys
        }
        estimator_based_on_config_dict = {
            k: v
            for k, v in estimator_based_on_config.__dict__.items()
            if k not in drop_keys
        }
        assert estimator_dict == estimator_based_on_config_dict, compare_nested_dicts(
            estimator_dict, estimator_based_on_config_dict
        )
