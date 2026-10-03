"""
Copyright 2023

Author: Lukas Schweizer <schweizer.lukas@web.de>
"""
# TODO: Clean this, don't do data preprocessing for TabPFN inputs,
#  make sure that categoricals are passed correctly, Make number of leaves depend on size of data

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin
from .utils import preprocess_data
from sklearn.tree import BaseDecisionTree


class DecisionTreeTabPFNClassifier(BaseDecisionTree, ClassifierMixin):
    """
    Class that implements a DT-TabPFN model based on sklearn package
    """

    task_type = "multiclass"

    def __init__(
        self,
        tabpfn=None,
        min_samples_split=1000,
        max_features=None,
        random_state=None,
        unique_tab_fit=False,
        criterion="gini",
        max_depth=None,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
        categorical_features=None,
        verbose=False,
        monotonic_cst=None,
        **kwargs,
    ):
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            splitter="best",
            monotonic_cst=monotonic_cst,
        )
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.tabpfn = tabpfn
        if self.tabpfn:
            self.tabpfn.seed = None  # Make sure that TabPFN is not seeded
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.unique_tab_fit = unique_tab_fit
        self.n_classes_ = 0
        self.classes_ = []
        self.decision_tree = None
        self.leaf_nodes = []
        self.leaf_train_data = {}
        self.categorical_features = categorical_features
        self.verbose = verbose

    def set_categorical_features(self, categorical_features):
        """
        Sets categorical features
        :param categorical_features: Categorical features
        :return: None
        """
        self.categorical_features = categorical_features

    def _fit(
        self,
        X,
        y,
        sample_weight=None,
        check_input=True,
        missing_values_in_feature_mask=None,
    ):
        """
        Method that fits the DecisionTree-TabPFN model
        :param X: Feature training data
        :param y: Label training data
        :param sample_weight: NOT IMPLEMENTED
        :param check_input:
        :return: None
        """
        if check_input:
            pass
            # X = check_array(X)
        # Handle X and y
        X = pd.DataFrame(X).reset_index().drop("index", axis=1)
        X_preprocessed = preprocess_data(X, nan_values=True, normalization=False)
        self.classes_ = np.unique(y)
        y = pd.DataFrame(y).reset_index().drop("index", axis=1)
        self.n_classes_ = int(y.max().iloc[0] + 1)
        y_train = pd.DataFrame()
        y_train["class"] = y
        # Create DecisionTree and get leaf nodes of each datapoint
        self.decision_tree = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
        )
        self.decision_tree.fit(X_preprocessed, y_train, sample_weight=sample_weight)
        if sample_weight is None:
            sample_weight = np.ones((X_preprocessed.shape[0],))
        # Create bootstraps from sample_weights
        sample_weight = sample_weight.astype(int)
        bootstrap_X = np.repeat(X, sample_weight, axis=0)
        bootstrap_X_pre = np.repeat(X_preprocessed, sample_weight, axis=0)
        bootstrap_y = np.repeat(y_train, sample_weight, axis=0)
        if self.unique_tab_fit:
            bootstrap_X, idx = np.unique(bootstrap_X, axis=0, return_index=True)
            bootstrap_X_pre = np.unique(bootstrap_X_pre, axis=0)
            bootstrap_y = np.take(bootstrap_y, idx, axis=0)
        X_train_leaf_nodes = self.decision_tree.apply(bootstrap_X_pre)
        self.leaf_nodes = sorted(np.unique(X_train_leaf_nodes))
        self.leaf_train_data = {}
        if self.verbose:
            print(f"Leaf Nodes: {len(self.leaf_nodes)}")
        # Store train data point for each leaf
        for leaf_id in self.leaf_nodes:
            indices = np.argwhere(X_train_leaf_nodes == leaf_id).ravel()
            X_train_samples = np.take(bootstrap_X, indices, axis=0)
            y_train_samples = np.array(np.take(bootstrap_y, indices, axis=0)).ravel()

            if self.verbose:
                print(f"Leaf: {leaf_id} Shape: {X_train_samples.shape}")

            self.leaf_train_data[leaf_id] = (X_train_samples, y_train_samples)

    def predict(self, X, check_input=True):
        """
        Predicts X_test
        :param X: Data that should be evaluated
        :param check_input:
        :return: Labels of the predictions
        """
        return np.array(
            list(map(np.argmax, self.predict_proba(X, check_input=check_input)))
        )

    def predict_proba(self, X, check_input=True):
        """
        Predicts X
        :param X: Data that should be evaluated
        :param check_input:
        :return: Probabilities of each class
        """
        if check_input:
            pass
            # X = check_array(X)
        # Handle X
        X = pd.DataFrame(X).reset_index().drop("index", axis=1)
        X_preprocessed = preprocess_data(X, nan_values=True, normalization=False)
        y_eval_prob = np.zeros((X.shape[0], self.n_classes_))
        # Get leaf nodes of each datapoint in X
        X_test_leaf_nodes = self.decision_tree.apply(X_preprocessed)
        tabpfn_model = self.tabpfn
        tabpfn_model.set_categorical_features(self.categorical_features)
        # Make prediction via TabPFN on each leaf node
        for leaf_id in sorted(np.unique(X_test_leaf_nodes)):
            indices = np.argwhere(X_test_leaf_nodes == leaf_id).ravel()
            # if no test datapoint comes to a leaf skip it
            if len(indices) == 0:
                continue
            # Get training data of leaf
            X_train_samples, y_train_samples = self.leaf_train_data[leaf_id]
            classes = np.unique(list(map(int, y_train_samples)))
            # Make actual prediction
            if len(classes) > 1:
                tabpfn_model.seed = leaf_id
                tabpfn_model._init_rnd()
                tabpfn_model.fit(X_train_samples, y_train_samples)
                # TODO: Leafs use correct categorical indices
                y_eval_prob[
                    indices[:, None], np.array(classes)
                ] = tabpfn_model.predict_proba(X.iloc[indices])
            else:
                y_eval_prob[indices[:, None], np.array(classes)] = 1
        return y_eval_prob
