from sklearn.base import BaseEstimator
from typing import Optional
import torch
import copy
import numpy as np

from .base import TabPFNRegressor, TabPFNClassifier


class TabPFNUnsupervisedModel(BaseEstimator):
    """TabPFN unsupervised model for imputation, outlier detection, and synthetic data generation.

    This model combines a TabPFNClassifier for categorical features and a TabPFNRegressor for
    numerical features to perform various unsupervised learning tasks on tabular data.

    Parameters:
        tabpfn_clf : TabPFNClassifier, optional
            TabPFNClassifier instance for handling categorical features. If not provided, the model
            assumes that there are no categorical features in the data.

        tabpfn_reg : TabPFNRegressor, optional
            TabPFNRegressor instance for handling numerical features. If not provided, the model
            assumes that there are no numerical features in the data.

    Attributes:
        categorical_features : list
            List of indices of categorical features in the input data.

    Examples:
        >>> tabpfn_clf = TabPFNClassifier()
        >>> tabpfn_reg = TabPFNRegressor()
        >>> model = TabPFNUnsupervisedModel(tabpfn_clf, tabpfn_reg)
        >>>
        >>> X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        >>> model.fit(X)
        >>>
        >>> X_imputed = model.impute(X)
        >>> X_outliers = model.outliers(X)
        >>> X_synthetic = model.generate_synthetic_data(n_samples=100)
    """

    def _more_tags(self):
        return {"allow_nan": True}

    def __init__(
        self,
        tabpfn_clf: Optional[TabPFNClassifier] = None,
        tabpfn_reg: Optional[TabPFNRegressor] = None,
    ) -> None:
        """Initialize the TabPFNUnsupervisedModel.

        Parameters:
            tabpfn_clf : TabPFNClassifier, optional
                TabPFNClassifier instance for handling categorical features. If not provided, the model
                assumes that there are no categorical features in the data.

            tabpfn_reg : TabPFNRegressor, optional
                TabPFNRegressor instance for handling numerical features. If not provided, the model
                assumes that there are no numerical features in the data.

        Raises:
            AssertionError
                If both tabpfn_clf and tabpfn_reg are None.
        """
        assert (
            tabpfn_clf is not None or tabpfn_reg is not None
        ), "You cannot set both `tabpfn_clf` and `tabpfn_reg` to None. You can set one to None, if your table exclusively consists of categoricals/numericals."

        self.tabpfn_clf = tabpfn_clf
        self.tabpfn_reg = tabpfn_reg
        self.estimators = [self.tabpfn_clf, self.tabpfn_reg]

        self.categorical_features = []

    def set_categorical_features(self, categorical_features):
        self.categorical_features = categorical_features
        for estimator in self.estimators:
            estimator.set_categorical_features(categorical_features)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fit the model to the input data.

        Parameters:
            X : array-like of shape (n_samples, n_features)
                Input data to fit the model.

            y : array-like of shape (n_samples,), optional
                Target values.

        Returns:
            self : TabPFNUnsupervisedModel
                Fitted model.
        """
        self.X_ = copy.deepcopy(X)
        self.y = copy.deepcopy(y)
        self.tabpfn_clf.infer_categorical_features(X)
        self.categorical_features = self.tabpfn_clf.categorical_features

    def impute_for_gen(self, X: torch.tensor, t: float = 0.000000001) -> torch.tensor:
        """
        Impute missing values (np.nan) in X by sampling all cells independently from the trained models

        :param X: Input data of the shape (num_examples, num_features) with missing values encoded as np.nan
        :param t: Temperature for sampling from the imputation distribution, lower values are more deterministic
        :return: Imputed data, with missing values replaced
        """
        train_X = self.X_
        imputed_X = copy.deepcopy(X)
        from tqdm import tqdm

        categorical_features = self.tabpfn_clf.infer_categorical_features(self.X_)

        for column_idx in tqdm(range(0, X.shape[1])):
            if column_idx > 0:
                # If not the first feature, use all previous features
                mask = torch.zeros_like(train_X).bool()
                mask[:, :column_idx] = True
                X_, y_ = (
                    train_X[mask].reshape(train_X.shape[0], -1),
                    train_X[:, column_idx],
                )

                mask = torch.zeros_like(X).bool()
                mask[:, :column_idx] = True
                X_pred, y_pred = (
                    imputed_X[mask].reshape(imputed_X.shape[0], -1),
                    imputed_X[:, column_idx],
                )
            else:
                # If the first feature, use a zero feature as input
                # Because of preprocessing, we can't use a zero feature, so we use a random feature
                X_, y_ = torch.rand_like(train_X[:, 0:1]), train_X[:, 0]
                X_pred, y_pred = torch.rand_like(imputed_X[:, 0:1]), imputed_X[:, 0]

            if torch.isnan(y_pred).sum() == 0:
                continue

            cat_column = column_idx in categorical_features
            model = self.tabpfn_clf if cat_column else self.tabpfn_reg

            X_where_y_is_nan = X_pred[torch.isnan(y_pred)]

            # TODO: If the model can't do semisupervised, need to remove missing ys
            model.fit(X_, y_)

            if not cat_column:
                pred = model.predict_full(X_where_y_is_nan)
                pred = pred["criterion"].sample(torch.tensor(pred["logits"]), t=t)
            else:
                pred = model.predict_proba(X_where_y_is_nan)
                # sample from the predicted distribution
                pred = (
                    torch.distributions.Categorical(probs=torch.tensor(pred))
                    .sample()
                    .float()
                )
            imputed_X[torch.isnan(y_pred), column_idx] = pred
        return imputed_X

    def impute(self, X: torch.tensor, t: float = 0.000000001) -> torch.tensor:
        """Impute missing values in the input data.

        Parameters:
            X : torch.Tensor of shape (n_samples, n_features)
                Input data with missing values encoded as np.nan.

            t : float, default=0.000000001
                Temperature for sampling from the imputation distribution. Lower values result in
                more deterministic imputations.

        Returns:
            torch.Tensor of shape (n_samples, n_features)
                Imputed data with missing values replaced.
        """
        train_X = self.X_
        imputed_X = copy.deepcopy(X)
        from tqdm import tqdm

        categorical_features = self.tabpfn_clf.infer_categorical_features(self.X_)

        for column_idx in tqdm(range(0, X.shape[1])):
            mask = torch.zeros_like(train_X).bool()
            mask[:, column_idx] = True
            X_, y_ = train_X[~mask].reshape(train_X.shape[0], -1), train_X[mask]

            mask = torch.zeros_like(X).bool()
            mask[:, column_idx] = True
            X_pred, y_pred = (
                imputed_X[~mask].reshape(imputed_X.shape[0], -1),
                imputed_X[mask],
            )

            if torch.isnan(y_pred).sum() == 0:
                continue

            cat_column = column_idx in categorical_features
            model = self.tabpfn_clf if cat_column else self.tabpfn_reg

            X_where_y_is_nan = X_pred[torch.isnan(y_pred)]

            # TODO: If the model can't do semisupervised, need to remove missing ys
            model.fit(X_, y_)

            if not cat_column:
                pred = model.predict_full(X_where_y_is_nan)
                pred = pred["criterion"].sample(torch.tensor(pred["logits"]), t=t)
            else:
                pred = model.predict_proba(X_where_y_is_nan)
                # sample from the predicted distribution
                pred = (
                    torch.distributions.Categorical(probs=torch.tensor(pred))
                    .sample()
                    .float()
                )
            imputed_X[torch.isnan(y_pred), column_idx] = pred
        return imputed_X

    def get_embeddings(self, X: torch.tensor) -> torch.tensor:
        """
        Get the transformer embeddings for the test data X.

        :param X:
        :return:
        """
        model = self.tabpfn_reg
        model.fit(
            self.X_,
            self.y
            if self.y is not None
            else (torch.zeros_like(self.X_[:, 0])),  # Must contain more than one class
        )  # Fit the data for random labels
        embs = model.get_embeddings(X, additional_y=None)
        return embs.reshape(X.shape[0], -1)

    def get_embeddings_per_column(self, X: torch.tensor) -> torch.tensor:
        """
        Alternative implementation for get_embeddings, where we get the embeddings for each column as a label
         separately and concatenate the results. This alternative way needs more passes but might be more accurate
        """
        embs = []
        for column_idx in range(0, X.shape[1]):
            mask = torch.zeros_like(self.X_).bool()
            mask[:, column_idx] = True
            X_train, y_train = (
                self.X_[~(mask)].reshape(self.X_.shape[0], -1),
                self.X_[mask],
            )

            X_pred, y_pred = X[~(mask)].reshape(X.shape[0], -1), X[mask]

            model = (
                self.tabpfn_clf
                if column_idx in self.categorical_features
                else self.tabpfn_reg
            )
            model.fit(X_train, y_train)
            embs += [model.get_embeddings(X_pred, additional_y=None)]

        return torch.cat(embs, 1).reshape(embs[0].shape[0], -1)

    def init_model_and_get_model_config(self):
        for estimator in self.estimators:
            estimator.init_model_and_get_model_config()

    def outliers(self, X: torch.tensor) -> torch.tensor:
        """
        Preferred implementation for outliers, where we calculate the sample probability for each sample in X by
        multiplying the probabilities of each feature according to chain rule of probability. The first feature is
        estimated by using a zero feature as input.

        :param X: Samples to calculate the sample probability for, shape (n_samples, n_features)
        :return: Sample probability for each sample in X, shape (n_samples,)
        """
        self.init_model_and_get_model_config()

        p = torch.ones_like(X[:, 0])  # Start with a probability of 1
        for column_idx in range(0, X.shape[1]):
            mask = torch.zeros_like(X).bool()
            if column_idx > 0:
                # If not the first feature, use all previous features
                mask[:, :column_idx] = True
                X_train, y_train = X[mask], X[:, column_idx]
                X_train = X_train.reshape(X.shape[0], -1)
            else:
                # If the first feature, use a zero feature as input
                # Because of preprocessing, we can't use a zero feature, so we use a random feature
                X_train, y_train = torch.rand_like(X[:, 0:1]), X[:, 0]
            X_train, y_train = X_train.numpy(), y_train.numpy()

            model = (
                self.tabpfn_clf
                if column_idx in self.categorical_features
                and len(np.unique(y_train)) < self.tabpfn_clf.max_num_classes_
                else self.tabpfn_reg
            )

            model.fit(X_train, y_train)

            pred = model.predict_y_proba(X_train, y_train)

            p = p * pred

        return p

    def outliers_leave_one_out(self, X: torch.tensor) -> torch.tensor:
        """
        Alternative implementation for outliers, where we calculate the sample probability for each sample in X by
        leaving one feature out at a time and multiplying the probabilities.

        :param X: Samples to calculate the sample probability for, shape (n_samples, n_features)
        :return: Sample probability for each sample in X, shape (n_samples,)
        """
        p = torch.ones_like(X[:, 0])
        for column_idx in range(0, X.shape[1]):
            mask = torch.zeros_like(X).bool()
            mask[:, column_idx] = True
            X_train, y_train = X[~(mask)], X[mask]
            X_train = X_train.reshape(X.shape[0], -1)

            model = (
                self.tabpfn_clf
                if column_idx in self.categorical_features
                else self.tabpfn_reg
            )

            model.fit(X_train, y_train)

            pred = model.predict_y_proba(X_train, y_train)

            p = p * pred

        return p

    def generate_synthetic_data(self, n_samples=100, t=1.0):
        """
        Generate synthetic data using the trained models. Uses imputation method to generate synthetic data, passed with
        a matrix of nans. Samples are generated feature by feature in one pass, so samples are not dependent on each
        other per feature.

        Parameters:
            n_samples : int, default=100
                Number of synthetic samples to generate.

            t : float, default=1.0
                Temperature for sampling from the imputation distribution. Lower values result in
                more deterministic samples.

        Returns:
            torch.Tensor of shape (n_samples, n_features)
                Generated synthetic data.

        Raises:
            AssertionError
                If the model is not fitted.
        """
        # TODO: Test what happens if we generate one feature at a time, with train data only for that featur
        #  and previous ones, like outliers
        assert hasattr(
            self, "X_"
        ), "You need to fit the model before generating synthetic data"

        X = torch.zeros(n_samples, self.X_.shape[1]) * np.nan
        return self.impute_for_gen(X, t=t)
