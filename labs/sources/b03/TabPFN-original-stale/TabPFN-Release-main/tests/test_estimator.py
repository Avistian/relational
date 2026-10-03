import pytest
import torch
import numpy as np
import time
from tabpfn.scripts.estimator import (
    TabPFNClassifier,
    TabPFNRegressor,
    PreprocessorConfig,
    EnsembleConfiguration,
    ManyClassClassifier,
)
from tabpfn.best_models import get_best_tabpfn


def test_cat_inds():
    test_transformer = torch.nn.Linear(5, 5)
    tabpfn = TabPFNClassifier(
        model=test_transformer,
        model_config={"differentiable_hps_as_style": False},
        feature_shift_decoder="none",
        add_fingerprint_features=False,
    )
    tabpfn._init_rnd()
    tabpfn.is_classification_ = True
    tabpfn.num_classes_ = 2
    tabpfn.set_categorical_features(categorical_features=(0, 3, 4))

    X = torch.tensor(
        [
            [0.0, 1, 2, 3, 4],
            [1.0, 2, 2, 4, 4],
        ]
    )

    feature_transformer = tabpfn.get_feature_preprocessor(
        PreprocessorConfig("none"),
    )
    new_x, categorical_features = feature_transformer.fit_transform(
        X, tabpfn.categorical_features
    )
    assert categorical_features == [0, 2]

    feature_transformer = tabpfn.get_feature_preprocessor(
        PreprocessorConfig("power", categorical_name="numeric"),
    )
    new_x, categorical_features = feature_transformer.fit_transform(
        X, tabpfn.categorical_features
    )
    assert categorical_features == []

    feature_transformer = tabpfn.get_feature_preprocessor(
        PreprocessorConfig("power", categorical_name="numeric", append_original=True),
    )
    new_x, categorical_features = feature_transformer.fit_transform(
        X, tabpfn.categorical_features
    )
    assert new_x.shape[1] == 6, (X, new_x)  # 6 since 3 features are non-trivial
    assert categorical_features == [2, 6], f"Got {categorical_features}"

    feature_transformer = tabpfn.get_feature_preprocessor(
        PreprocessorConfig("none", categorical_name="onehot"),
    )
    new_x, categorical_features = feature_transformer.fit_transform(
        X, tabpfn.categorical_features
    )
    assert categorical_features == [0, 1, 2, 3]

    X = X[:, None]

    # TODO: With feature shifts the indices are not correct
    class_shift = torch.tensor([0, 1])
    # Shuffled: feature_and_class_shift = (torch.tensor([0, 1]), torch.tensor([4,3,1,2,0]))

    tabpfn.infer_categorical_features(X[:, 0, :])

    (
        inputs,
        labels,
        categorical_inds,
        _,
        _,
        _,
        _,
    ) = tabpfn.build_transformer_input_for_each_configuration(
        ensemble_configurations=[
            EnsembleConfiguration(
                class_shift_configuration=class_shift,
                preprocess_transform_configuration=PreprocessorConfig(
                    "none", categorical_name="onehot"
                ),
                styles_configuration=None,
            ),
            EnsembleConfiguration(
                class_shift_configuration=class_shift,
                preprocess_transform_configuration=PreprocessorConfig("none"),
                styles_configuration=None,
            ),
            EnsembleConfiguration(
                class_shift_configuration=class_shift,
                preprocess_transform_configuration=PreprocessorConfig(
                    "power", categorical_name="numeric"
                ),
                styles_configuration=None,
            ),
        ],
        eval_xs=X,
        eval_ys=torch.tensor([0, 1])[None],
        eval_position=2,
    )

    assert categorical_inds == [[0, 1, 2, 3], [0, 2], []]

    ## Split inputs to model into chunks that can be calculated batchwise for faster inference
    ## We split based on the dimension of the features, so that we can feed in a batch
    (
        _,
        _,
        _,
        categorical_inds_,
        _,
    ) = tabpfn.build_transformer_input_in_batches(
        inputs, labels, categorical_inds, batch_size_inference=2
    )

    assert categorical_inds_ == ([[0, 1, 2, 3]], [[0, 2], []])

    (
        inputs,
        labels,
        implied_permutation,
        categorical_inds_,
        _,
    ) = tabpfn.build_transformer_input_in_batches(
        inputs, labels, categorical_inds, batch_size_inference=1
    )

    assert categorical_inds_ == ([[0, 1, 2, 3]], [[0, 2]], [[]])


def test_is_sklearn_compatible():
    pytest.skip("This test is very slow (80s), it seems hard to speed up though.")
    # WARNING: This test never errors
    # This test is very slow (80s), it seems hard to speed up though. A smaller model that still
    # has okay performance could be a way to speed it up.
    from sklearn.utils.estimator_checks import check_estimator
    from tabpfn.best_models import get_best_tabpfn

    clf = get_best_tabpfn(model_type="single_fast", task_type="multiclass")
    clf.sklearn_compatible_precision = True
    clf.fp16_inference = False
    for estimator, check in check_estimator(clf, generate_only=True):
        passed = False
        tries = 0
        while not passed and tries < 5:  # Some checks fail non deterministically
            try:
                check(estimator)
                passed = True
            except Exception as e:
                print(e)
                tries += 1


def test_splitting_of_fit_and_predict_preprocessing():
    tabpfn: TabPFNClassifier = get_best_tabpfn(  # type: ignore
        "multiclass",
        model_type="single",
        device="cpu",
        debug=True,
        inference_config_overwrite={},
    )

    torch.manual_seed(0)
    eval_xs = torch.randn(30, 1, 5)
    eval_ys = (eval_xs @ torch.randn(5) > 0).to(int)
    test_xs = torch.randn(10, 1, 5)

    tabpfn.num_classes_ = 2
    tabpfn._categorical_features = []

    ## all in one

    tabpfn.init_model_and_get_model_config()
    ensemble_configurations = tabpfn.get_ensemble_configurations(
        eval_xs,
        eval_ys,
        eval_position=len(eval_xs),
    )
    print(ensemble_configurations)

    (
        inputs,
        labels,
        categorical_inds,
        adapted_bar_dists,
        logit_cancel_masks,
        descending_borders,
        additional_ys,
    ) = tabpfn.build_transformer_input_for_each_configuration(
        ensemble_configurations=ensemble_configurations,
        eval_xs=torch.cat((eval_xs, test_xs)),
        eval_ys=eval_ys,
        eval_position=len(eval_xs),
        bar_dist=None,
        additional_ys=None,
        cache_trainset_transforms=False,
    )
    print(inputs)
    print([i.shape for i in inputs])

    ## split
    tabpfn.init_model_and_get_model_config()
    new_ensemble_configurations = tabpfn.get_ensemble_configurations(
        eval_xs,
        eval_ys,
        eval_position=len(eval_xs),
    )
    assert (
        new_ensemble_configurations[0].feature_shift_configuration
        == ensemble_configurations[0].feature_shift_configuration
    )

    (
        inputs_train,
        labels_train,
        categorical_inds_train,
        adapted_bar_dists_train,
        logit_cancel_masks_train,
        descending_borders_train,
        additional_ys_train,
    ) = tabpfn.build_transformer_input_for_each_configuration(
        ensemble_configurations=ensemble_configurations,
        eval_xs=eval_xs,
        eval_ys=eval_ys,
        eval_position=len(eval_xs),
        bar_dist=None,
        additional_ys=None,
        cache_trainset_transforms=True,
    )

    (
        inputs_test,
        labels_test,
        categorical_inds_test,
        adapted_bar_dists_test,
        logit_cancel_masks_test,
        descending_borders_test,
        additional_ys_test,
    ) = tabpfn.build_transformer_input_for_each_configuration(
        ensemble_configurations=ensemble_configurations,
        eval_xs=test_xs,
        eval_ys=torch.tensor([]),
        eval_position=0,
        bar_dist=None,
        additional_ys=None,
        cache_trainset_transforms=True,
    )

    for input, input_train, input_test in zip(inputs, inputs_train, inputs_test):
        concatenated = torch.cat((input_train, input_test))
        max_discrepancy = (input - concatenated).abs().max().item()
        if max_discrepancy > 1e-5:
            print("max discrepancy", max_discrepancy)
            print(input - concatenated)
        assert input == pytest.approx(concatenated), f"{input} != {concatenated}"

    for label, label_train, label_test in zip(labels, labels_train, labels_test):
        concatenated = torch.cat((label_train, label_test))
        assert label == pytest.approx(concatenated)

    for categorical_ind, categorical_ind_train, categorical_ind_test in zip(
        categorical_inds, categorical_inds_train, categorical_inds_test
    ):
        assert categorical_ind == categorical_ind_train == categorical_ind_test


@pytest.mark.parametrize("task_type", ["multiclass", "regression"])
def test_splitting_of_fit_and_predict_works(task_type):
    tabpfn: TabPFNClassifier | TabPFNRegressor = get_best_tabpfn(  # type: ignore
        task_type,
        model_type="single",
        device="cpu",
        debug=True,
        inference_config_overwrite={},
        seed=0,
    )

    tabpfn.fit_at_predict_time = True

    torch.manual_seed(0)
    np.random.seed(0)
    x = np.random.randn(100, 5)
    y = x @ np.random.randn(5)
    if task_type == "multiclass":
        y = (y > 0).astype(int)
    test_x = np.random.randn(10, 5)

    tabpfn.fit(x, y)

    predict_func = tabpfn.predict_proba if task_type == "multiclass" else tabpfn.predict

    lazy_preds = predict_func(test_x)

    tabpfn.fit_at_predict_time = False

    t = time.time()
    tabpfn.fit(x, y)
    fit_time = time.time() - t

    t = time.time()
    eager_preds = predict_func(test_x)
    predict_time = time.time() - t

    assert fit_time > predict_time

    if task_type == "multiclass":
        # split comps into unsure and sure ones
        sure_inds = lazy_preds.max(1) > 0.9

        assert lazy_preds[~sure_inds] == pytest.approx(
            eager_preds[~sure_inds], abs=1e-2
        )

        assert lazy_preds[sure_inds] == pytest.approx(eager_preds[sure_inds], abs=1e-2)
    else:
        assert lazy_preds == pytest.approx(eager_preds, abs=1e-2)


def test_many_class():
    tabpfn: TabPFNClassifier = get_best_tabpfn(  # type: ignore
        "multiclass",
        model_type="single",
        device="cpu",
        debug=True,
        inference_config_overwrite={"n_estimators": 2},
    )
    n_classes = 20
    clf = ManyClassClassifier(tabpfn)

    x = np.random.randn(500, 5)
    y = np.random.randint(low=0, high=n_classes, size=500)

    clf.fit(x, y)
    preds = clf.predict_proba(x)

    assert (
        preds.shape[1] == n_classes
    ), f"Got {preds.shape[1]} classes, expected {n_classes}"
