import torch
import torch.nn.functional as F
from torchmetrics.regression import (
    MeanSquaredError,
    MeanSquaredLogError,
    MeanAbsoluteError,
)
from torchmetrics.classification import AUROC, BinaryAUROC


def compute_metric(outputs, labels, metric):
    if metric == "rmse":
        score = - MeanSquaredError(squared=False)(outputs.flatten(), labels).item()
    elif metric == "mae":
        score = - MeanAbsoluteError()(outputs.flatten(), labels).item()
    elif metric == "hr@1":
        score = torch.mean((outputs.argmax(dim=1) == labels).to(torch.float)).item()
    elif metric == "retrieval_auroc":
        score = AUROC(task="multiclass", num_classes=outputs.shape[1])(
            outputs, labels
        ).item()
        """
        if outputs.shape[1] == 2: #, "only binary auroc"
            metric = "binary AUC"
            outputs = outputs[:, 1] - outputs[:, 0]
            score = BinaryAUROC()(outputs, labels).item()
        else:
            metric = "multi AUC"
            score = AUROC(task="multiclass", num_classes=outputs.shape[1])(outputs, labels).item()
        """
    elif metric == "retrieval_logloss":
        score = - F.cross_entropy(outputs, labels)  # nn.()(outputs, labels).item()
    else:
        raise NotImplementedError(metric)
    return score