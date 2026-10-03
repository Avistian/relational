import warnings
from datetime import datetime
import torch
import sys
import numpy as np
import copy
import platform
import os
import psutil

from . import tabular_metrics
from tabpfn import utils


class DatasetEvaluation:
    def __init__(
        self,
        y: torch.tensor,
        pred: torch.tensor,
        algorithm_name=None,
        additional_args=None,
        name=None,
        time=None,
        task_type=None,
        **kwargs,
    ):
        self.y = y
        if y is not None:
            self.y = y if not torch.is_tensor(y) else y.detach().cpu().numpy()
        self.pred = pred if not torch.is_tensor(pred) else pred.detach().cpu().numpy()
        self.datetime = datetime.now()
        self.algorithm_name = algorithm_name
        self.additional_args = utils.move_to_device(additional_args, "cpu")
        self.time = time  # time in seconds, if splits are added together this is the sum of the splits
        self.environment = sys.argv[0]
        self.metrics = {}
        self.name = name
        try:
            cpu_freq = psutil.cpu_freq().current
        except NotImplementedError:
            # We need this, as on HELIX, cpu_freq is not available.
            cpu_freq = None
        self.system_info = {
            "python_version": sys.version,
            "system": platform.platform(),
            "node": platform.node(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "cpu_freq": cpu_freq,
            "processors": platform.processor(),
            "cpu_count": str(os.cpu_count()),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        self.task_type = task_type

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __getitem__(self, indices):
        assert (
            len(self.metrics) == 0
        ), "Slicing of DatasetEvaluation is currently only allowed before metrics were calculated."

        ds = copy.deepcopy(self)
        ds.y = ds.y[indices]
        ds.pred = ds.pred[indices]

        return ds

    def to_dict(self):
        return self.__dict__

    def calculate_metric(self, metric, name, aggregator):
        if (self.additional_args is not None) and self.additional_args.get(
            "failed", False
        ):
            self.metrics[f"{aggregator}_{name}"] = np.nan
            return

        if self.y is None:
            self.metrics[f"{aggregator}_{name}"] = np.nan
            return

        if getattr(metric, "__name__", "none") == "time_metric":
            self.metrics[f"{aggregator}_{name}"] = self.time
            return

        if (
            len(self.pred.shape) > 1
            and (
                not hasattr(metric, "quantiles")
                and self.task_type != "quantile_regression"
            )
            and len(np.unique(self.y)) != self.pred.shape[1]
        ):
            if len(np.unique(self.y)) > self.pred.shape[1]:
                warnings.warn(
                    "Particularly bad, we have less classes in train than test. Setting the score to nan."
                )
                self.metrics[f"{aggregator}_{name}"] = np.nan
                return
            else:
                # print warning filters
                utils.print_once(
                    "We have less classes in test than train. This is not a problem for the metric.."
                )

        pred = self.pred
        if self.additional_args is not None and (
            pred_full := self.additional_args.get("pred_full")
        ):
            if isinstance(metric, tabular_metrics.QuantileMetric):
                self.additional_args["quantiles"] = metric.quantiles
                pred = torch.stack(
                    [
                        torch.tensor(pred_full[f"quantile_{q:.2f}"])
                        for q in metric.quantiles
                    ],
                    dim=1,
                )

        if (
            isinstance(metric, tabular_metrics.QuantileMetric)
            and "quantiles" in self.additional_args
            and tuple(metric.quantiles) != tuple(self.additional_args["quantiles"])
        ):
            raise ValueError(
                f"Quantiles of metric {metric.quantiles} do not match quantiles of dataset {self.additional_args['quantiles']}"
            )

        try:
            if "survival" in getattr(metric, "__name__", "none"):
                self.metrics[f"{aggregator}_{name}"] = float(
                    metric(self.y, pred, self.additional_args["censoring"])
                )
            else:
                self.metrics[f"{aggregator}_{name}"] = float(metric(self.y, pred))
        except Exception as err:
            print(
                "ALGO",
                self.algorithm_name,
                "DS NAME",
                self.name,
                "AGG",
                aggregator,
                "METRIC",
                name,
                "ERR",
                err,
                "TYPE",
                type(err),
            )
            print("pred", self.pred)
            raise err
            # warnings.warn(f"Error calculating metric with {err}, {type(err)} on {name}")
            # self.metrics[f"{aggregator}_{name}"] = np.nan


class DatasetEvaluationCollection:
    def __init__(self, name, evaluations: dict[DatasetEvaluation]):
        self.name = name
        self.evaluations = evaluations
        self.metrics = {}

    def calculate_metric(self, metric, name, aggregator):
        # For time and count the metrics are summed. Time lists then the total time spent with
        # evaluating all splits. Count lists the total number of splits evaluated.
        aggregator_f = tabular_metrics.get_aggregator_f(aggregator)

        metric_values = []

        # Aggregate the metric over all evaluations
        failed_once = False
        for i, evaluation in enumerate(self.evaluations.values()):
            evaluation.calculate_metric(metric, name, aggregator)

            if evaluation is None:
                raise ValueError(
                    f"Invalid split evaluation {i} for dataset {self.name}."
                )

            failed = (
                False
                if (evaluation.additional_args is None)
                else evaluation.additional_args.get("failed", False)
            )
            if failed:
                failed_once = True
                # at least one fold failed, hence any aggregator will fails as well.
                continue

            metric_values.append(evaluation.metrics[f"{aggregator}_{name}"])

            if np.isnan(metric_values[-1]):
                error_msg = (
                    f"Invalid metric value {metric_values[-1]} for split {i} of dataset {self.name}. In '{aggregator}_{name}'."
                    "See prints above for info on the error."
                )
                # TODO uncomment this again, when we have fixed the splits with the official ones from openml
                # raise ValueError(error_msg)
                print(error_msg)

        if failed_once:
            self.metrics[f"{aggregator}_{name}"] = np.nan
            return

        aggr_metric = aggregator_f(metric_values)

        # Store the calculated mean metric
        self.metrics[f"{aggregator}_{name}"] = aggr_metric
