from functools import partial

import numpy as np

from tabpfn.scripts.estimator import preprocessing


def test_interface():
    """
    We want to enforece that preprocessing steps do all "training" in fit and all "transforming" in transform,
    that means we do not want any interdependence between examples in transform.
    """
    defaults = [
        cls
        for cls in preprocessing.__dict__.values()
        if (
            isinstance(cls, type)
            and issubclass(cls, preprocessing.FeaturePreprocessingTransformerStep)
            and cls is not preprocessing.FeaturePreprocessingTransformerStep
        )
    ]
    extras = [
        partial(
            preprocessing.ReshapeFeatureDistributionsStep,
            "none",
            append_to_original=True,
            global_transformer_name="svd",
            apply_to_categorical=False,
        )
    ]
    for cls in defaults + extras:
        print(f"Testing {cls}")
        x = np.random.rand(20, 4)
        x2 = np.random.rand(20, 4)
        cat_inds = [1, 3]
        x[:, cat_inds] = np.round(np.linspace(0, 3, len(x))[:, None])[
            np.random.permutation(len(x))
        ].astype(float)
        x2[:, cat_inds] = np.round(np.linspace(0, 3, len(x))[:, None])[
            np.random.permutation(len(x))
        ].astype(float)
        obj = cls().fit(x, cat_inds)

        # there should not be an interdependence between examples in transform
        # we check for this heuristically
        print("Testing interdependence for", cls, "...")

        def check_equality(x, x2):
            assert (x == x2).all(), (x, x2, x - x2)

        # calling transform again should not change the result
        check_equality(obj.transform(x2).X, obj.transform(x2).X)

        if type(obj) != preprocessing.AddFingerprintFeaturesStep:
            # AddFingerprintFeaturesStep is not invariant to sample shuffling
            #   it assign each datapoint a unique fingerprint, this must be random or dependent on position.
            check_equality(obj.transform(x2).X, obj.transform(x2[::-1]).X[::-1])
            check_equality(obj.transform(x2).X[:4], obj.transform(x2[:4]).X)
        assert (
            obj.transform(x2).categorical_features
            == obj.transform(x2[:4]).categorical_features
        )
