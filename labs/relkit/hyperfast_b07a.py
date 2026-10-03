"""Full HyperFast source, adapted for the live output-head exercise.

AI-sandbox/HyperFast commit d1f1c3b0c45572dc09173733b194c0e7384cd696.
CC BY-NC4.0; original LICENSE and unchanged source in labs/sources/b07a.
Includes optional downstream optimizer for inspection; B07a does not call it.
Only generated output-head pooling is delegated to a learner-owned function.
"""
import os
import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)


def nn_bias_logits(
    test_logits, test_samples, train_samples, train_labels, bias_param, n_classes, mini_batches
):
    with torch.no_grad():
        nn = NN(train_samples, train_labels)
        preds = nn.predict(test_samples, mini_batches=mini_batches)
        preds_onehot = F.one_hot(preds, n_classes)
    test_logits[preds_onehot.bool()] += bias_param
    return test_logits


def forward_main_network(x, main_network):
    for n, layer in enumerate(main_network):
        if n % 2 == 0:
            residual_connection = x
        matrix, bias = layer
        x = torch.mm(x, matrix) + bias
        if n % 2 == 1 and n != len(main_network) - 1:
            x = x + residual_connection

        if n != len(main_network) - 1:
            x = F.relu(x)
            if n == len(main_network) - 2:
                intermediate_activations = x
    return x, intermediate_activations


def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = torch.argmax(torch.abs(u), axis=0)
        signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, None]
    else:
        # rows of v, columns of u
        max_abs_rows = torch.argmax(torch.abs(v), axis=1)
        signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, None]

    return u, v


class TorchPCA:
    def __init__(self, n_components=None, fit="full"):
        self.n_components = n_components
        self.fit = fit

    def _fit(self, X):
        if self.n_components is None:
            n_components = min(X.shape)
        else:
            n_components = self.n_components

        n_samples, n_features = X.shape
        if n_components > min(X.shape):
            raise ValueError(
                f"n_components should be <= min(n_samples: {n_samples}, n_features: {n_features})"
            )

        self.mean_ = torch.mean(X, axis=0)
        X -= self.mean_

        if self.fit == "full":
            U, S, Vt = torch.linalg.svd(X, full_matrices=False)
            # flip eigenvectors' sign to enforce deterministic output
            U, Vt = svd_flip(U, Vt)
        elif self.fit == "lowrank":
            U, S, Vt = torch.pca_lowrank(X)

        self.components_ = Vt[:n_components]
        self.n_components_ = n_components

        return U, S, Vt

    def fit(self, X):
        self._fit(X)
        return self

    def transform(self, X):
        assert self.mean_ is not None
        X -= self.mean_
        return torch.matmul(X, self.components_.T)

    def fit_transform(self, X):
        U, S, Vt = self._fit(X)
        U = U[:, : self.n_components_]
        U *= S[: self.n_components_]
        return U


class MainNetworkTrainable(nn.Module):
    def __init__(self, cfg, n_classes, rf, pca, main_network, nn_bias):
        super(MainNetworkTrainable, self).__init__()
        self.cfg = cfg
        self.n_classes = n_classes
        self.pca = pca
        self.clip_data_value = cfg.clip_data_value
        self.main_network = main_network
        self.nn_bias = nn_bias
        self.use_nn_bias = cfg.nn_bias
        self.device = cfg.device

        self.rf = rf
        self.rf[0].weight.requires_grad = True
        self.pca_mean = (
            nn.Parameter(pca.mean_)
            if cfg.torch_pca
            else nn.Parameter(torch.from_numpy(pca.mean_))
        )
        self.input_features, self.output_features = pca.components_.shape
        self.pca_components = nn.Linear(
            self.input_features, self.output_features, bias=False
        )
        self.pca_components.weight = (
            nn.Parameter(pca.components_)
            if cfg.torch_pca
            else nn.Parameter(torch.from_numpy(pca.components_))
        )

        self.layers = nn.ModuleList()
        for matrix, bias in main_network:
            linear_layer = nn.Linear(matrix.shape[0], matrix.shape[1])
            linear_layer.weight = nn.Parameter(matrix.T)
            linear_layer.bias = nn.Parameter(bias)
            self.layers.append(linear_layer)

    def forward(self, X, y=None):
        intermediate_activations = [X]
        X = self.rf(X)
        X = X - self.pca_mean
        X = self.pca_components(X)
        X = torch.clamp(X, -self.clip_data_value, self.clip_data_value)

        for n, layer in enumerate(self.layers):
            if n % 2 == 0:
                residual_connection = X

            X = layer(X)
            if n % 2 == 1 and n < len(self.layers) - 1:
                X = X + residual_connection

            if n < len(self.layers) - 1:
                X = F.relu(X)
                if n == len(self.layers) - 2:
                    intermediate_activations.append(X)

        if self.use_nn_bias:
            assert y is not None
            for ii, bias in enumerate(self.nn_bias):
                nn = NN(intermediate_activations[ii], y)
                preds = nn.predict_from_training_with_LOO()
                preds_onehot = F.one_hot(preds, self.n_classes)
                X[preds_onehot.bool()] += bias
        return X

    def get_main_network_parts(self):
        rf_reconstructed = self.rf
        rf_reconstructed[0].weight.requires_grad = False

        pca_reconstructed = self.pca
        pca_reconstructed.mean_ = (
            self.pca_mean.detach()
            if self.cfg.torch_pca
            else self.pca_mean.cpu().detach().numpy()
        )
        pca_reconstructed.components_ = (
            self.pca_components.weight.detach()
            if self.cfg.torch_pca
            else self.pca_components.weight.cpu().detach().numpy()
        )

        main_network_reconstructed = []

        for layer in self.layers:
            weight_matrix = layer.weight.data.T
            bias_vector = layer.bias.data
            main_network_reconstructed.append((weight_matrix, bias_vector))

        return (
            rf_reconstructed,
            pca_reconstructed,
            main_network_reconstructed,
            self.nn_bias,
        )


def fine_tune_main_network(
    cfg,
    X,
    y,
    n_classes,
    rf,
    pca,
    main_network_layers,
    nn_bias,
    device,
    optimize_steps,
    batch_size,
):
    main_model = MainNetworkTrainable(
        cfg, n_classes, rf, pca, main_network_layers, nn_bias
    ).to(device)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(main_model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10
    )

    for step in range(optimize_steps):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = main_model(inputs, targets)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            # print(f"Step: [{step+1}/{optimize_steps}], Loss: {loss.item()}")

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss.item())
            else:
                scheduler.step()
    return main_model.get_main_network_parts()


def get_main_weights(x, hn, weight_gen=None):
    emb = hn(x)
    if weight_gen is not None:
        global_emb = torch.mean(emb, dim=0)
        w = weight_gen(global_emb)
    else:
        w = emb
    return w


def forward_linear_layer(x, w, hs):
    w = w.view(-1, hs)
    m = w[:-1, :]
    b = w[-1, :]
    x = torch.mm(x, m) + b
    return x, (m, b)


def transform_data_for_main_network(X, cfg, rf, pca):
    with torch.no_grad():
        X = rf(X)
    if cfg.torch_pca:
        X = pca.transform(X)
    else:
        X = torch.from_numpy(pca.transform(X.cpu().numpy())).to(cfg.device)
    X = torch.clamp(X, -cfg.clip_data_value, cfg.clip_data_value)
    return X


"""
For the following code:
Author: Josue N Rivera (github.com/JosueCom)
Date: 7/3/2021
Description: Snippet of various clustering implementations only using Pyth
Full project repository: https://github.com/JosueCom/Lign (A graph deep learning framework that works alongside Pyth)
"""


def distance_matrix(x, y=None, p=2):
    y = x if y is None else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = (
        torch.linalg.vector_norm(x - y, p, 2)
        if torch.__version__ >= "1.7.0"
        else torch.pow(x - y, p).sum(2) ** (1 / p)
    )

    return dist


class NN:
    def __init__(self, X=None, Y=None, p=2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def predict(self, x, mini_batches=True):
        if self.train_pts is None or self.train_label is None:
            name = self.__class__.__name__
            raise RuntimeError(
                f"{name} wasn't trained. Need to execute {name}.train() first"
            )

        if not mini_batches:
            with torch.no_grad():
                dist = distance_matrix(x, self.train_pts, self.p)
                labels = torch.argmin(dist, dim=1)
        else:
            with torch.no_grad():
                batch_size = 128
                num_batches = math.ceil(x.shape[0] / batch_size)
                labels = []
                for ii in range(num_batches):
                    x_ = x[batch_size * ii:batch_size * (ii + 1), :]
                    dist = distance_matrix(x_, self.train_pts, self.p)
                    labels_ = torch.argmin(dist, dim=1)
                    labels.append(labels_)
                labels = torch.cat(labels)

        return self.train_label[labels]

    def predict_from_training_with_LOO(self, mini_batches=True):
        if self.train_pts is None or self.train_label is None:
            name = self.__class__.__name__
            raise RuntimeError(
                f"{name} wasn't trained. Need to execute {name}.train() first"
            )

        if not mini_batches:
            dist = distance_matrix(self.train_pts, self.train_pts, self.p)
            dist.fill_diagonal_(float("inf"))
            labels = torch.argmin(dist, dim=1)
        else:
            batch_size = 128
            num_batches = math.ceil(self.train_pts.shape[0] / batch_size)
            labels = []
            for ii in range(num_batches):
                x_ = self.train_pts[batch_size * ii:batch_size * (ii + 1), :]
                dist = distance_matrix(x_, self.train_pts, self.p)
                dist.fill_diagonal_(float("inf"))
                labels_ = torch.argmin(dist, dim=1)
                labels.append(labels_)
            labels = torch.cat(labels)

        return self.train_label[labels]


import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from relkit.hyper_b07a import class_weights


class HyperFast(nn.Module):
    def __init__(self, cfg, head_fn=class_weights):
        super().__init__()
        self.head_fn = head_fn
        self.n_dims = cfg.n_dims
        self.max_categories = cfg.max_categories
        self.rf_size = cfg.rf_size
        self.torch_pca = cfg.torch_pca
        self.clip_data_value = cfg.clip_data_value
        self.hn_n_layers = cfg.hn_n_layers
        self.hn_hidden_size = cfg.hn_hidden_size
        self.main_n_layers = cfg.main_n_layers

        middle_layers = []
        for n in range(self.hn_n_layers - 2):
            middle_layers.append(nn.Linear(self.hn_hidden_size, self.hn_hidden_size))
            middle_layers.append(nn.ReLU())
        self.num_input_features_hn = self.n_dims + self.max_categories

        self.hypernetworks = nn.ModuleList()
        self.hn_emb_to_weights = nn.ModuleList()

        for n in range(self.main_n_layers - 1):
            if n > 0:
                self.num_input_features_hn = self.n_dims * 2 + self.max_categories
            num_input_features_hn = self.num_input_features_hn + self.n_dims * 2

            hn_layers = []
            hn_layers.append(nn.Linear(num_input_features_hn, self.hn_hidden_size))
            hn_layers.append(nn.ReLU())
            hn_layers = hn_layers + middle_layers

            self.hypernetworks.append(nn.Sequential(*hn_layers))
            self.output_size_hn = (self.n_dims + 1) * self.n_dims
            self.hn_emb_to_weights.append(
                nn.Linear(self.hn_hidden_size, self.output_size_hn)
            )

        hn_layers = []
        last_hn_output_size = self.n_dims + 1
        self.num_input_features_hn += self.n_dims * 2

        hn_layers.append(nn.Linear(self.num_input_features_hn, self.hn_hidden_size))
        hn_layers.append(nn.ReLU())
        hn_layers = hn_layers + middle_layers
        hn_layers.append(nn.Linear(self.hn_hidden_size, last_hn_output_size))
        self.hypernetworks.append(nn.Sequential(*hn_layers))
        self.nn_bias = nn.Parameter(torch.ones(2))

    def forward(self, X, y, n_classes):
        X = X.flatten(start_dim=1)
        rf_linear = nn.Linear(X.shape[1], self.rf_size, bias=False)
        nn.init.kaiming_normal_(rf_linear.weight, mode="fan_out", nonlinearity="relu")
        rf_linear.weight.requires_grad = False
        rf = nn.Sequential(rf_linear, nn.ReLU()).to(X.device)
        with torch.no_grad():
            X = rf(X)
        if self.torch_pca:
            self.pca = TorchPCA(n_components=self.n_dims)
        else:
            self.pca = PCA(n_components=self.n_dims)
        if self.torch_pca:
            X = self.pca.fit_transform(X)
        else:
            X = torch.from_numpy(self.pca.fit_transform(X.cpu().numpy())).to(X.device)
        X = torch.clamp(X, -self.clip_data_value, self.clip_data_value)

        out = X
        pca_global_mean = torch.mean(out, axis=0)
        pca_perclass_mean = []
        for lab in range(n_classes):
            if torch.sum((y == lab)) > 0:
                class_mean = torch.mean(out[y == lab], dim=0, keepdim=True)
            else:
                class_mean = torch.mean(out, dim=0, keepdim=True)
            pca_perclass_mean.append(class_mean)
        pca_perclass_mean = torch.cat(pca_perclass_mean)

        pca_concat = []
        for ii, lab in enumerate(y):
            if pca_perclass_mean.ndim == 1:
                pca_perclass_mean = pca_perclass_mean.unsqueeze(0)
            if out.ndim == 1:
                out = out.unsqueeze(0)

            lab_index = lab.item() if torch.is_tensor(lab) else lab
            lab_index = min(lab_index, pca_perclass_mean.size(0) - 1)

            row = torch.cat((out[ii], pca_global_mean, pca_perclass_mean[lab_index]))
            pca_concat.append(row)
        pca_output = torch.vstack(pca_concat)
        y_onehot = F.one_hot(y, self.max_categories)

        main_network = []
        for n in range(self.main_n_layers - 1):
            if n > 0:
                data = torch.cat((out, pca_output, y_onehot), dim=1)
            else:
                data = torch.cat((pca_output, y_onehot), dim=1)
            if n % 2 == 0:
                residual_connection = out

            weights = get_main_weights(
                data, self.hypernetworks[n], self.hn_emb_to_weights[n]
            )
            out, main_linear_layer = forward_linear_layer(out, weights, self.n_dims)
            if n % 2 == 0:
                out = F.relu(out)
            else:
                out = out + residual_connection
                out = F.relu(out)
            main_network.append(main_linear_layer)
        data = torch.cat((out, pca_output, y_onehot), dim=1)
        weights_per_sample = get_main_weights(data, self.hypernetworks[-1])

        weights = self.head_fn(weights_per_sample, out, y, n_classes)
        out, last_linear_layer = forward_linear_layer(out, weights, n_classes)
        main_network.append(last_linear_layer)

        return rf, self.pca, main_network, self.nn_bias
