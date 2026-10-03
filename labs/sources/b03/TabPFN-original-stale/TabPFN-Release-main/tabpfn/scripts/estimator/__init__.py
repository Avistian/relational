from . import preprocessing
from .base import (
    TabPFNClassifier,
    TabPFNRegressor,
    TabPFNBaseModel,
    PreprocessorConfig,
    get_single_tabpfn,
    ClassificationOptimizationMetricType,
    RegressionOptimizationMetricType,
)
from .configs import (
    TabPFNConfig,
    TabPFNModelPathsConfig,
    BaggingConfig,
    StackingConfig,
    TabPFNRFConfig,
    WeightedAverageConfig,
    TabPFNClassificationConfig,
    TabPFNRegressionConfig,
    EnsembleConfiguration,
    TabPFNSurvivalConfig,
)
from .meta_models import (
    get_stacking_ensemble,
    get_bagging_ensemble,
    get_tabpfn_outer_ensemble,
    get_tabpfn_rf,
    get_weighted_average_ensemble,
)
from .unsupervised import TabPFNUnsupervisedModel
from .many_class_classifier import ManyClassClassifier
from .classifier_as_regressor import ClassifierAsRegressor

### MODEL CREATION ###


def get_model_from_wandb(task_type, wandb_id, device="cpu"):
    """
    TODO: this is getting old, with the runner being able to load wandb models itself!?
    :param task_type:
    :param wandb_id:
    :param device:
    :return:
    """
    clf = get_tabpfn(
        TabPFNConfig(
            task_type=task_type,
            paths_config=TabPFNModelPathsConfig(
                paths=[f"WANDB_ID:{wandb_id}"], task_type=task_type
            ),
            model_type="single",
        ),
        device=device,
    )
    clf.init_model_and_get_model_config()

    model = clf.model_processed_
    c = clf.c_processed_

    return model, c


def get_tabpfn(config: TabPFNConfig, **kwargs):
    """Get a combined TabPFN Model, e.g. stacking, bagging, ensemble, rf_pfn."""
    return tabpfn_model_type_getters[config.model_type](config, **kwargs)


tabpfn_model_type_getters = {
    "single": get_single_tabpfn,
    "single_fast": get_single_tabpfn,
    "outer_ensemble": get_tabpfn_outer_ensemble,
    "stacking": get_stacking_ensemble,
    "bagging": get_bagging_ensemble,
    "rf_pfn": get_tabpfn_rf,
    "weighted_average": get_weighted_average_ensemble,
}
