from __future__ import annotations

if __name__ == "__main__":
    # Add parent directory to path if running this file directly
    # We do this, as we can't run with `python -m` if we use `torchrun`
    import sys

    sys.path.insert(0, ".")

import secrets
import string
import re
import os
from typing import Literal, Tuple
import time
import argparse
import numpy as np
from functools import partial, wraps
import torch
import torch.distributed
import tqdm.auto
import torch.distributed
import tqdm
import pickle
import typing as tp

import datetime
import pytz
import signal
import sys
import collections
from enum import Enum

from tabpfn import utils
from tabpfn import datasets

from . import tabular_metrics
from .model_builder import get_model, preprocess_config, Checkpoint
from tabpfn.datasets import get_benchmark_for_task
from tabpfn.local_settings import base_path, get_wandb_project, wandb_entity

from tabpfn.utils import init_dist, default_task_settings
from tabpfn.best_models import get_best_tabpfn
from tabpfn.scripts.estimator import configs
from .model_configs import (
    load_config,
    remap_config_for_hp_run,
    set_from_dict,
)
from .benchmark.evaluate_model import (
    FullTabularResultsTablePlot,
)
from .tabular_metrics import (
    get_standard_eval_metrics,
    get_main_eval_metric,
    get_standard_eval_baselines,
)
from .tabular_evaluation import evaluate_and_score
from .tabular_evaluation_utils import (
    DatasetEvaluationCollection,
)
from .tabular_baselines import transformer_metric
from tabpfn.train import EarlyStoppingException

from .benchmark import get_evaluation_config_task_type
from .benchmark.evaluate_model import (
    EvaluationResults,
    evaluate_and_plot,
    get_plotter_collection,
)
from .benchmark.evaluation_storers import (
    EvaluationStorer,
    WandbStorer,
)
from .benchmark import tabular_results_plots

# Tags that should be added to all runs, even of baseline models
# You can use this e.g. to mark the revision of the eval code that is used
DEFAULT_TAGS = ["PullRequest47", "NewValidSets"]
NUM_SPLITS = 5
NUM_SPLITS_FINAL = 5


def generate_wandb_id(length: int = 8) -> str:
    """Generate a random base-36 string of `length` digits."""
    # There are ~2.8T base-36 8-digit strings. If we generate 210k ids,
    # we'll have a ~1% chance of collision.
    # taken from: https://github.com/wandb/wandb/blob/main/wandb/sdk/lib/runid.py
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


class AnomalyDetectionBuffer:
    def __init__(self, max_size=10):
        self.size = max_size
        self.buffer = collections.deque(maxlen=max_size)
        self.latest_deviation = None

    def add(self, value, significance=3):
        if np.isfinite(value):  # Only add finite values to the buffer
            self.latest_deviation = self.calculate_deviation(value)
            self.buffer.append(value)
        else:
            self.latest_deviation = None

        return self.is_outlier(significance=significance)

    def calculate_deviation(self, value):
        if len(self.buffer) == self.buffer.maxlen:
            mean = np.mean(self.buffer)
            std = np.std(self.buffer)
            return np.abs(value - mean) / std if std != 0 else None
        else:
            return None  # Can't check for outliers if buffer isn't full yet

    def is_outlier(self, significance=3):
        if self.latest_deviation is None:
            return False
        else:
            return self.latest_deviation > significance

    def get_latest_deviation(self):
        return self.latest_deviation


class TrainingMode(Enum):
    NORMAL = 1
    HPO = 3


def signal_handler(sig, frame, successful, timeout=False):
    """
    This function is called when a signal is received. It finishes the wandb
    run either successfully or unsuccessfully. For example in case of a cluster
    timeout, the run is finished successfully to not lose the results in a sweep.
    """
    # Otherwise finish and exit.
    import wandb

    if timeout:
        print(
            "CANCEL DUE TO TIME LIMIT"
        )  # important for parsing of reason of cancellation

    exit_code = int(not successful)
    print(
        f'signal received - {sig} - finishing wandb - exit code {exit_code} - {datetime.datetime.now(pytz.timezone("Europe/Berlin")).strftime("%Y-%m-%d %H:%M:%S")}'
    )

    # Finish the wandb run
    wandb.finish(exit_code=exit_code)
    sys.exit(exit_code)


def catch_wandb_issue(func):
    """
    A simple decorator that catches a weird wandb error
    that occurs when a run is scheduled via submitit out
    of a notebook in which wandb has been used already.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        import wandb

        try:
            return func(*args, **kwargs)
        except wandb.sdk.wandb_manager.ManagerConnectionRefusedError as e:
            raise Exception(
                "You should not use wandb in the notebook from which you start this training."
            )

    return wrapper


class TrainingAgent:
    def __init__(
        self,
        task_type: str,
        device: str,
        splits_to_evaluate: None
        | tp.Dict[str, tp.List[datasets.TabularDataset] | None],
        base_path: str,
        wandb_key: str,
        wandb_project: None | str,
        wandb_entity: str,
        model_path: str,
        wandb_path: str,
        mode: tp.Literal["online", "offline", "disabled"],
        model_string: str,
        id: str | None = None,
        debug: bool = False,
        wandb_tags: list | None = None,
        renamed_gpu_runs: bool = False,
        remapped_time: bool = True,
        splits_per_dataset_to_evaluate: list[int] | None = None,
        task_settings_overwrite: dict[str, int] | None = None,
    ):
        """

        Parameters
        ----------
        :param task_type: task type of the problem
        :param device: device to run on
        :param splits_to_evaluate: dictionary of validation splits to use for the evaluation, you can either pass a list of datasets to evaluate for one split or None to use the default datasets,
        defaults are available for "valid", "test", and "valid_hard".
        An argument could look like this: `{"valid": [valid_dataset1, valid_dataset2], "test": None, "valid_hard": None, "my_test_set": [test_dataset1, test_dataset2]}`
        :param base_path: base path for the results
        :param wandb_key: wandb key for logging/uploading results
        :param wandb_project: wandb project to log to
        :param wandb_entity: wandb entity to log to
        :param model_path: path to the model that is to be run
        :param wandb_path: path to the wandb run
        :param mode: mode of the run (online, offline, disabled)
        :param model_string: string of the model
        :param id:
        :param debug: debug mode (True/False)
        :param wandb_tags: list of wandb tags to add to the upload
        :param renamed_gpu_runs: If False, no change in behavior. If True, it assumes that the result files were written to a different path and renames
            the path to look for them accordingly. Specifically, it follows the code in `tabular_evaluation.py` and prefixes the path with "gpu_" if True.
            This is passed to `tabular_evaluation.evaluate_and_score`'s `rename_gpu_runs` parameter.
        :param remapped_time: If False, no change in behavior. If True, we assume the time string was remapped according `tabular_evaluation.get_mapped_time`
            for the `BaselineTrainingAgent`. This is passed to `tabular_evaluation.evaluate_and_score`'s `allow_remap_time` parameter.
        :param splits_per_dataset_to_evaluate: list of splits (e.g. folds) to evaluate for each dataset. If None, a default number of splits is evaluated.
            This is passed to `tabular_evaluation.evaluate_and_score`'s `splits` parameter.
        :param task_settings_overwrite: dictionary to overwrite the default task settings (filter for datasets) to determine which datasets to run.
        """
        assert task_type in [
            "multiclass",
            "regression",
            "survival",
            "quantile_regression",
        ], f"task_type {task_type} not supported"
        assert mode in ["online", "offline", "disabled"], f"mode {mode} not supported"

        if wandb_project is None:
            wandb_project = get_wandb_project(task_type)
        # In case of distributed trainings, only rank 0 should be logged to wandb.
        # Otherwise runs are created that don't contain any metrics besides the system ones.
        if "LOCAL_RANK" in os.environ:
            self.rank = int(os.environ["LOCAL_RANK"])
            self.world_size = torch.cuda.device_count()
        elif "SLURM_PROCID" in os.environ and torch.cuda.device_count() > 1:
            self.rank = int(os.environ["SLURM_PROCID"])
            self.world_size = int(os.environ["SLURM_NTASKS"])
        else:
            self.rank = 0
            self.world_size = 1

        self.id = id if id is not None else generate_wandb_id()
        self.device = device
        self.task_type = task_type
        self.debug = debug

        if splits_to_evaluate is None:
            splits_to_evaluate = {"valid": None, "test": None}
        print("splits that are evaluated: ", splits_to_evaluate)
        if (
            splits_to_evaluate.get("test", None) is not None
            or splits_to_evaluate.get("valid", None) is not None
        ):  # they should either be empty ([]) to be disabled, or None to be loaded from scratch
            assert (
                mode == "disabled"
            ), "You can only provide validation/test datasets in disabled mode. Otherwise our scores on wandb are not comparable."

        def get_datasets(split):
            max_samples, max_features, _, max_classes = default_task_settings()

            if task_settings_overwrite is not None:
                max_samples = task_settings_overwrite.get("max_samples", max_samples)
                max_features = task_settings_overwrite.get("max_features", max_features)
                max_classes = task_settings_overwrite.get("max_classes", max_classes)

            if self.rank == 0:
                if split == "valid_hard":
                    # hard limit at 8 million cells
                    datasets, _ = get_benchmark_for_task(
                        task_type,
                        split,
                        return_as_lists=False,
                        max_classes=max_classes,
                        max_num_cells=3_000_000,  # this can be used to cap size for memory saving, without we are at something like 8.6GB for reg, 7.5 for cls(for the one model, I tested)
                        return_capped=True,
                        max_samples=max_samples,
                        max_features=max_features,
                    )
                else:
                    datasets, _ = get_benchmark_for_task(
                        task_type,
                        split,
                        return_as_lists=False,
                        max_samples=max_samples,
                        max_classes=max_classes,
                        max_features=max_features,
                    )
            else:
                datasets = []

            return datasets

        for valid_split in splits_to_evaluate:
            set_value = splits_to_evaluate[valid_split]
            if (
                valid_split
                not in [
                    "valid",
                    "test",
                    "valid_hard",
                    "gbdt_friendly_medium",
                    "tabzilla",
                ]
                and set_value is None
            ):
                raise ValueError(
                    f"Split {valid_split} is has no default datasets assigned to it."
                )
            if set_value is not None and not isinstance(set_value, list):
                raise ValueError(f"Split {valid_split} is not a list of datasets")
            if set_value is None:
                if self.debug:
                    splits_to_evaluate[valid_split] = get_datasets("debug")
                else:
                    splits_to_evaluate[valid_split] = get_datasets(valid_split)

        self.splits_to_evaluate = splits_to_evaluate

        self.wandb_key = wandb_key
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_tags = wandb_tags

        # In case of distributed trainings, only rank 0 should be logged to wandb.
        # For the other ranks, the wandb commands are a NOP, as indicated by 'disabled'.
        self.mode = mode if self.rank == 0 else "disabled"
        print(f"My rank is {self.rank} and my mode is {self.mode}")

        self.base_path = base_path

        self.model_string = model_string
        self.model_path = self.get_model_path(model_path)
        self.wandb_path = (
            os.path.join(self.base_path, "wandb") if wandb_path is None else wandb_path
        )

        self.valid_metrics = get_standard_eval_metrics(self.task_type)
        self.metric_used = get_main_eval_metric(self.task_type)

        self.renamed_gpu_runs = renamed_gpu_runs
        self.remapped_time = remapped_time
        self.splits_per_dataset_to_evaluate = splits_per_dataset_to_evaluate

    @property
    def files_prefix(self):
        # Truncate the model string to prevent reaching the limit of 255 bytes for a filename in linux (assume 8 bits per char here)
        # We choose 223 here to account for the 32 chars added to the string.
        return f"model_{self.model_string[:223]}_id_{self.id}"

    def get_model_path(self, model_path=None):
        """
        This should become a property soon, but for now we need to be able to overwrite it for backwards compatibility.
        """
        return (
            os.path.join(
                self.base_path,
                "results",
                "models_diff",
                f"{self.files_prefix}_epoch_-1.cpkt",
            )
            if model_path is None
            else model_path
        )

    def copy_wandb_run_history(self, new_run, previous_run_id, max_epoch=None):
        """
        This function is used to continue runs and have consistent logs that include the previous trainings logs
        based on the `previous_run_id`.

        The records will be copied until `max_epoch` is reached.
        """
        import wandb

        # initialize wandb
        api = wandb.Api()

        # load the previous run
        prev_run = api.run(
            f"{self.wandb_entity}/{self.wandb_project}/{previous_run_id}"
        )

        # get all the logged metrics for the run
        history = prev_run.scan_history()

        # replay the logged metrics from the previous run to the new run
        for log_msg in history:
            # Filter out all values that are None.
            log_msg = {key: val for key, val in log_msg.items() if val is not None}

            # log the metrics for the new run
            new_run.log(log_msg)

            # stop if the current epoch is greater than the stop epoch
            epoch = log_msg.get("epoch", -1)
            if max_epoch is not None and epoch is not None and epoch >= max_epoch:
                print(f"Epoch {max_epoch} of saved model reached!")
                break

    def init_signal_handler(self, use_signal_handler=True):
        if use_signal_handler:
            # we handle the SIGUSR2 signal separately. It can be seen as a warning, that the process will be killed soon.
            # It is sent by SLURM a few minutes before time out or `scancel --signal USR2`
            signal.signal(
                signal.SIGUSR2, partial(signal_handler, successful=False, timeout=True)
            )
            signal.signal(signal.SIGTERM, partial(signal_handler, successful=False))

    def wandb_login(self):
        if self.mode == "disabled":
            return

        import wandb

        if self.wandb_key is not None:
            wandb.login(key=self.wandb_key, relogin=True, force=True)
        else:
            # Don't use relogin here, as this requires the key being typed in interactively,
            # which is unfeasible for scheduled jobs. Therefore use the cached key, that is
            # being created on the first login to wandb through python.
            wandb.login()

    def init_wandb(self, name=None):
        import wandb

        tags = [*DEFAULT_TAGS]  # copies values to avoid modifying inplace
        tags += self.wandb_tags if self.wandb_tags is not None else []

        print("SETTING TAGS: ", tags)

        run = wandb.init(
            entity=self.wandb_entity,
            # Set the project where this run will be logged
            project=self.wandb_project,
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=name if name else self.model_string,
            # Whether wandb commands act as NOPs in case of rank > 1 ("disabled")
            # Or logs to wandb in case of rank == 0 ("online").
            mode=self.mode,
            # Don't log source code
            save_code=False,
            # The directory where the wandb files will be stored
            dir=self.wandb_path,
            # Default tags to e.g. identify the eval code version
            tags=tags,
            # self.id is either freshly generated in __init__ or loaded from a checkpoint
            id=self.id,
            # resume=allow means if the id exists, the run is continued, else a new run is created
            resume="allow",
        )

        return run


class TabPFNTrainingAgent(TrainingAgent):
    """
    This class is used to run our trainings with wandb logging.
    """

    def __init__(
        self,
        task_type: tp.Literal[
            "multiclass", "regression", "survival", "quantile_regression"
        ] = "multiclass",
        device: tp.Literal["cpu", "cuda"] = "cuda",
        wandb_key=None,
        wandb_project=None,  # defaults to get_wandb_project(task_type)
        wandb_entity=wandb_entity,
        base_path=base_path,
        splits_to_evaluate: None
        | tp.Dict[str, tp.List[datasets.TabularDataset] | None] = None,
        model_path=None,
        wandb_path=None,
        mode: tp.Literal["online", "offline", "disabled"] = "online",
        model_string=None,
        config_sample=None,
        continue_model_path=None,
        continue_id=None,
        save_model_enabled=True,
        epoch_time_threshold=float("inf"),
        early_stop_threshold=-float(
            "inf"
        ),  # if the metric is below this threshold, after the early_stop_epoch_start, the training is stopped
        early_stop_metric="valid/5_splits/mean_roc",  # {split_name}/{num_splits}_splits/{metric['aggregator']}_{metric['name']}
        early_stop_epoch_start=10,
        evaluation_storers: tp.Tuple[EvaluationStorer] | None = None,
        evaluation_plots: tp.List[tabular_results_plots.EvaluationPlot] | None = None,
        id=None,
        debug=False,
        save_evaluation_results=True,
        wandb_tags=None,
        perform_final_evaluation=True,
        run_full_eval_before_training=False,
    ):
        """
                Our main interface to train TabPFNs. This is a wrapper for the `scripts.model_builder.get_model` function, which
                adds support for wandb logging and checkpointing.


                How do I continue a training? There are two options, either i) you pretend the to be in the exact same run, or ii) you start a completely new run from the last state of another run.

                i) The recommendation is to sample your own id for this run with `scripts.utils.generate_wandb_id()`.
                If you specify this ID now and the run exists already on wandb it will be continued.
                This behaves as if the run was never stopped.

                Here is an example of how to setup a job that has 20 continuation runs scheduled:
                ```python
        from tabpfn.scripts import runner
        from tabpfn.scripts import model_configs


        task_type='multiclass'
        config = model_configs.load_config(task_type)

        wandb_id = runner.generate_wandb_id()
        name = 'super_strong_run'
        assert len(name) < 220, "code might break if you are above, please don't be"
        ex_parallel_test.submit_group(name, runner.wandb_get_model,
                        [{'config': {**config}, 'id': wandb_id, 'task_type': task_type, 'model_string': name} for _ in range(20)], max_parallel=1)
                ```


                ii) Alternatively, you can specify a `continue_model_path` to load a model from a checkpoint. This is useful, if you
                want to try out different continuations starting from the same model. It will write the checkpoints to a new
                location and create a separate wandb run. You can find the model path of a job in the wandb dashboard, under `model_path`.


                :param task_type:
                :param device: Specify the type of device in PyTorch style to run on. If you want to use multi-GPU, still use "cuda" here.
                :param wandb_key:
                :param wandb_project:
                :param wandb_entity:
                :param base_path:
                :param model_path:
                :param wandb_path:
                :param mode:
                :param model_string:
                :param config_sample:
                :param continue_model_path:
                :param save_model_enabled:
                :param epoch_time_threshold:
                :param early_stop_threshold:
                :param early_stop_metric:
                :param early_stop_epoch_start:
                :param evaluation_storers:
                :param evaluation_plots:
                :param id: The wandbid to use, if it already exists the run will be continued, else a new run is created.
                This id is also used to make our checkpoint paths `model_path` unique.
                :param debug:
                :param save_evaluation_results:
                :param wandb_tags:
                :param perform_final_evaluation:
                :param run_full_eval_before_training: Whether to run all seeds and not only one seed before training.
        """
        # In case of a continued model, the model, optimizer and scaler state are loaded from the checkpoint.
        self.loaded_checkpoint = None

        assert (
            config_sample is None
            or config_sample.get("task_type", task_type) == task_type
        ), f"task_type in config_sample ({config_sample.get('task_type', task_type)}) does not match task_type ({task_type})"

        if id is not None:
            assert (
                continue_id is None
            ), "Cannot set both the continue_id and fix a id, you need to set id=None and let a random id to be chosen."
            continue_id = id

        if continue_id is not None:
            try:
                loaded_model_path_based_on_id = configs.get_model_string_from_wandb(
                    wandb_id=continue_id, task_type=task_type
                )
                assert (
                    continue_model_path is None
                ), "You specified an existing wandb run id, but also a `continue_model_path`."
                continue_model_path = loaded_model_path_based_on_id
                if model_path is None and id == continue_id:
                    model_path = loaded_model_path_based_on_id
                print("Found model path for this id:", continue_model_path)
            except FileNotFoundError:
                print("No model found for this id, thus creating a new run.")

        if (
            model_path is not None
            and continue_model_path is None
            and os.path.isfile(model_path)
        ):
            print(
                "NOTE: model path already exists, thus continuing from the checkpoint."
            )
            continue_model_path = model_path

        if continue_model_path is not None:
            self.loaded_checkpoint: Checkpoint = Checkpoint.load(continue_model_path)
            loaded_config_sample = self.loaded_checkpoint.config
            # to avoid misunderstandings:
            self.loaded_checkpoint.config = None

            if config_sample is None:
                config_sample = loaded_config_sample
            else:
                config_sample = dict(loaded_config_sample, **config_sample)
                print("overwriting the following things in the config:")
                for key, value in loaded_config_sample.items():
                    if key in config_sample and config_sample[key] != value:
                        print(f"{key}: {value} -> {config_sample[key]}")
                print("added the following stuff to the config:")
                for key, value in config_sample.items():
                    if key not in loaded_config_sample:
                        print(f"{key}: {value}")
            print("Continuing from model path:", continue_model_path)
        else:
            assert (
                config_sample is not None
            ), "You need to specify a config_sample, if you do not continue a training. It can be an empty dict."

        # added such that it can be followed back on wandb what the previous run was
        config_sample["continue_model_path"] = continue_model_path

        self.continue_model_path = continue_model_path

        if model_string is None:
            model_string = "default"
            if continue_model_path is not None:
                match_model_string_from_path = re.findall(
                    ".*models_diff/model_(.*)_id_.*", continue_model_path
                )
                if len(match_model_string_from_path) > 0:
                    model_string = (
                        match_model_string_from_path[0] + "c"
                    )  # this marks it as a continuation
            elif "SLURM_JOB_NAME" in os.environ:
                model_string = (
                    os.environ.get("SLURM_JOB_NAME")
                    + "_"
                    + os.environ.get("SLURM_ARRAY_TASK_ID", "0")
                )

        super().__init__(
            task_type=task_type,
            splits_to_evaluate=splits_to_evaluate,
            device=device,
            base_path=base_path,
            wandb_key=wandb_key,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            model_path=model_path,
            wandb_path=wandb_path,
            mode=mode,
            model_string=model_string,
            id=id,
            debug=debug,
            wandb_tags=wandb_tags,
        )

        if continue_model_path and continue_id != id:
            self.model_string += f"_{id}"

        self.config_sample = config_sample

        self.save_model_enabled = save_model_enabled
        self.perform_final_evaluation = perform_final_evaluation

        self.step_logging_frequency = 100
        self.wandb_track_grads_freq = 200

        # We'd like to track the loss of the last 15 steps to track whether the training is broken or not.
        self.loss_buffer = AnomalyDetectionBuffer(15)
        # We also track the epoch times over the last 15 epochs to track whether the node is running slowly or not.
        self.epoch_time_threshold = epoch_time_threshold
        self.epoch_time_buffer = collections.deque(maxlen=15)

        self.early_stop_threshold = early_stop_threshold
        self.early_stop_metric = early_stop_metric
        self.early_stop_epoch_start = early_stop_epoch_start

        self.evaluation_plots = evaluation_plots
        self.evaluation_storers = evaluation_storers

        self.cached_evaluation_results = {
            split: {} for split in self.splits_to_evaluate
        }
        self.final_evaluation_model_types = [
            "single_fast",
            "single",
            # "rf_pfn",
            # "bagging",
        ]
        self.save_evaluation_results = save_evaluation_results
        self.run_full_eval_before_training = run_full_eval_before_training

        print("TabPFN TrainingAgent initialized")

    def early_stop_callback(self, log_msg):
        epoch = log_msg["epoch"]
        if self.early_stop_metric not in log_msg:
            print(
                f"Early stopping metric {self.early_stop_metric} not found in results."
            )
            return  # Early stopping is not enabled for this run

        metric_value = float(log_msg[self.early_stop_metric])

        print(
            f"Checking early stopping metric {self.early_stop_metric} with value {metric_value}"
            f" and threshold {self.early_stop_threshold} after epoch {epoch} (start epoch: {self.early_stop_epoch_start})"
        )

        if (
            metric_value < self.early_stop_threshold
            and epoch >= self.early_stop_epoch_start
        ):
            raise EarlyStoppingException(
                f"Early Stopped: {metric_value} ({self.metric_used}) < 0 ({self.early_stop_threshold}) after epoch {epoch} (start epoch: {self.early_stop_epoch_start})"
            )

    def run_custom_experiments(self, tabpfns, split, n_splits):
        import wandb

        log_msg = {}
        for model_type, (transformer_metric_with_model, tabpfn) in tabpfns.items():
            from tabpfn.scripts.benchmark import tabular

            max_samples, max_features, max_times, max_classes = default_task_settings()
            for split_name, function in [
                (
                    f"{split}_noise_features",
                    partial(
                        tabular.return_ds_with_uninformative_features, fraction=4.0
                    ),
                ),
                (
                    f"{split}_outliers",
                    partial(tabular.return_ds_with_outliers, outlier_factor=4_000),
                ),
            ]:
                log_msg_dataset_manipulation_experiment, _ = evaluate_and_score(
                    [
                        transformed_ds
                        for ds in (self.splits_to_evaluate.get(split, []))
                        if (transformed_ds := function(ds)).x.shape[1] <= max_features
                    ],
                    self.valid_metrics,
                    transformer_metric_with_model,
                    split_name=f"{model_type}/{split_name}",
                    metric_used=self.metric_used,
                    log_per_dataset_metrics=False,
                    max_time=-1,
                    num_splits=n_splits,
                    save=False,
                    overwrite=False,
                    fetch_only=False,
                    base_path=self.base_path,
                    path_interfix=self.task_type,
                )

                log_msg = {**log_msg, **log_msg_dataset_manipulation_experiment}
            if self.task_type == "regression":
                from tabpfn.scripts.benchmark.experiments import (
                    DoubleSlitRegressionExperiment,
                )

                fig = DoubleSlitRegressionExperiment().run(tabpfn)
                fig_path = "/tmp/doubleslit_experiment.png"
                fig.savefig(fig_path, dpi=200)
                log_msg[f"{model_type}__doubleslit_experiment"] = wandb.Image(fig_path)

        return log_msg

    def full_evaluation(
        self,
        inference_config_overwrite=None,
        model_types=None,
        **kwargs,
    ):
        self.init_wandb()

        import wandb

        wandb.log(
            {"continue_model_wandb_id": self.config_sample.get("wandb_run_id", "")}
        )

        def serialize(o):
            if isinstance(o, list):
                return [serialize(x) for x in o]
            elif isinstance(o, tuple):
                return tuple(serialize(x) for x in o)
            else:
                try:
                    return o.to_dict()
                except:
                    return o

        if inference_config_overwrite:
            wandb.log(
                {
                    f"inference_config_{k}": serialize(v)
                    for k, v in inference_config_overwrite.items()
                }
            )

        if model_types is None and kwargs.get("tabpfns", None) is None:
            model_types = self.final_evaluation_model_types

        self.setup_evaluation_storers()
        self.setup_evaluation_plots()

        evaluation_results, log_msg = self._full_evaluation(
            **kwargs,
            model_types=model_types,
            inference_config_overwrite=inference_config_overwrite,
        )

        wandb.log(log_msg)

        wandb.finish()

        return evaluation_results, log_msg

    def load_baseline_results(self, split, n_splits=10):
        # Load baseline results
        eval_config = get_evaluation_config_task_type(
            task_type=self.task_type, split=split
        )
        if self.debug:
            eval_config.splits = eval_config.splits[:1]
        else:
            assert (
                len(eval_config.splits) >= n_splits
            ), "Not enough splits ran for baseline evaluation"
            eval_config.splits = eval_config.splits[:n_splits]

        eval_config.set_methods_subset(
            subset=get_standard_eval_baselines(self.task_type)
        )
        if self.debug:
            methods_subset = {
                m: eval_config.methods[m]
                for m in eval_config.methods
                if "default" in m or m == "autogluon"
            }
            eval_config.methods["autogluon"] = list(eval_config.methods.values())[0]
            eval_config.set_methods_subset(subset=methods_subset)
        eval_config.max_time = 3600
        eval_config.max_times = [eval_config.max_time]

        evaluation_results = EvaluationResults(
            eval_config=eval_config,
            datasets=self.splits_to_evaluate[split],
            ignore_checks=True,  # TODO: change to True
        )

        return evaluation_results

    def evaluate_model_types_and_compare_to_baselines(
        self,
        tabpfns,
        split: Literal["test", "valid"],
        n_splits: int = 10,
    ) -> Tuple[EvaluationResults, dict]:
        """

        :param model:
        :param split:
        :param n_splits:
        :param model_types:
        :param inference_config_overwrite: Configs specififed here will overwrite any other config settings.
        :return:
            Evaluation Results
            Log message, each entry prepended by the split it was used in
        """
        log_msg = {}

        if n_splits not in self.cached_evaluation_results[split]:
            self.cached_evaluation_results[split][
                n_splits
            ] = self.load_baseline_results(split, n_splits=n_splits)
        evaluation_results = self.cached_evaluation_results[split][n_splits]

        for model_type in (pbar := tqdm.tqdm(tabpfns)):
            transformer_metric_with_model, tabpfn = tabpfns[model_type]
            pbar.set_description(f"Running inference for {model_type}")
            print("Running inference for", model_type)

            # Create TabPFN Results
            log_msg_scores, global_results = evaluate_and_score(
                evaluation_results.datasets,
                evaluation_results.eval_config.eval_metrics,
                transformer_metric_with_model,
                split_name=split,
                metric_used=evaluation_results.eval_config.metric_used,
                log_per_dataset_metrics=True,
                num_splits=len(evaluation_results.eval_config.splits),
            )
            log_msg_scores = {f"{model_type}/{k}": v for k, v in log_msg_scores.items()}
            log_msg = {**log_msg, **log_msg_scores}

            # Merge newly evaluated and existing results
            scoring_str = tabular_metrics.get_scoring_string(
                evaluation_results.eval_config.metric_used
            )
            evaluation_results.global_results[scoring_str][model_type] = {}
            for max_time in evaluation_results.eval_config.max_times:
                results = {
                    k: global_results[k].evaluations
                    for k in global_results
                    if type(global_results[k]).__name__
                    == DatasetEvaluationCollection.__name__
                }
                split_numbers = list(results.values())[0].keys()
                evaluation_results.global_results[scoring_str][model_type][max_time] = {
                    split: {
                        ds_name: results[ds_name][split] for ds_name in results.keys()
                    }
                    for split in split_numbers
                }

            # Need to add the predictor to the list of methods for evaluation
            evaluation_results.eval_config.methods.update(
                {model_type: transformer_metric_with_model}
            )
        # Recalculate results
        evaluation_results.recalculate_results()

        return evaluation_results, log_msg

    def _full_evaluation(
        self,
        model=None,
        tabpfns=None,
        split: Literal["test", "valid"] = "test",
        n_splits=NUM_SPLITS_FINAL,
        model_types=["single_fast"],
        inference_config_overwrite: dict = None,
        run_custom_experiments: bool = False,
        device=None,
    ):
        """
        This function runs the pfn with multiple decoding mechanisms, compares the model to the baselines,
         saves the results and creates plots based on these results.

        :param model: TabPFN BaseModel
        :param split:
        :param n_splits:
        :param model_types:
        :param run_custom_experiments: If True the function `run_custom_experiments` will be called and all experiments
        in there will be additionally done. This should yield an almost complete set of graphs and results, but uses
        a lot of disk.
        :param device: If None the default device will be used (cuda if available, cpu otherwise).
        :return:
        """
        # Ensemble models need multiple models at the same time which does not make sense when evaluating one model
        if model_types is not None:
            assert "ensemble" not in model_types and "best" not in model_types, (
                "Ensemble models need multiple models at the same time which does not make sense when evaluating one model."
                "Please use the evaluate_ensemble_models function instead."
            )
        assert (
            len(self.splits_to_evaluate[split]) > 0
        ), "No datasets to evaluate on. Please fill it."
        device = utils.get_default_device() if device is None else device
        if tabpfns is None:
            if model is None:
                model = get_model(
                    config=self.config_sample,
                    device=device,
                    should_train=False,
                    verbose=False,
                    state_dict=self.loaded_checkpoint.state_dict,
                ).model

            tabpfns = self.build_predictors(
                model, model_types, inference_config_overwrite
            )
        else:
            assert model is None, "If tabpfns is not None, model should be None."
            assert (
                model_types is None
            ), "If tabpfns is not None, model_types should be None."
            model_types = list(tabpfns.keys())

        # Load baseline results and evaluate model types
        (
            evaluation_results,
            log_msg,
        ) = self.evaluate_model_types_and_compare_to_baselines(
            tabpfns,
            split,
            n_splits,
        )
        print("evaluate and plot..")
        if self.evaluation_plots:
            # Create plots and store results
            evaluate_and_plot(
                evaluation_results=evaluation_results,
                evaluation_plots=self.evaluation_plots,
                evaluation_storers=self.evaluation_storers,
                incumbant_model_names=model_types,
                benchmark_name=f"final_{split}",
            )

            if self.save_evaluation_results:
                # Save entire evaluation results so plots can easily be adapted afterwards
                import wandb

                df_by_ds_pickle_path = os.path.join(
                    self.get_default_results_path(),
                    f"evaluation_results_{split}_df_by_ds.pickle",
                )

                with open(df_by_ds_pickle_path, "wb") as f:
                    pickle.dump(evaluation_results.df_by_ds, f)

                wandb.log(
                    {
                        f"evaluation_results_{split}_df_by_ds_pickle_path": df_by_ds_pickle_path
                    }
                )

                pickle_path = os.path.join(
                    self.get_default_results_path(),
                    f"evaluation_results_{split}.pickle",
                )

                with open(pickle_path, "wb") as f:
                    pickle.dump(evaluation_results, f)

                wandb.log({f"evaluation_results_{split}_pickle_path": pickle_path})

        if run_custom_experiments:
            log_msg = {
                **log_msg,
                **self.run_custom_experiments(tabpfns, split, n_splits),
            }
        return evaluation_results, log_msg

    def build_predictors(
        self, model, model_types=["single_fast"], inference_config_overwrite=None
    ):
        tabpfns = {}
        for model_type in model_types:
            tabpfn = get_best_tabpfn(
                self.task_type,
                model_type,
                model=model,
                model_config=self.config_sample,
                debug=self.debug,
                device=self.device,
                inference_config_overwrite=inference_config_overwrite,
            )

            tabpfn.show_progress = False
            tabpfn.seed = 1
            tabpfn.device = self.device
            transformer_metric_with_model = partial(
                transformer_metric, classifier=tabpfn
            )
            tabpfns[model_type] = transformer_metric_with_model, tabpfn

        return tabpfns

    def check_for_alerts(self, extra_infos):
        # Check in every callback whether the training is still running correctly.

        import wandb
        from wandb import AlertLevel

        if "mean_epoch_loss" in extra_infos:
            loss = extra_infos["mean_epoch_loss"]

            # Check if the loss is inf or nan
            if loss == float("inf") or loss == float("-inf") or loss == float("nan"):
                wandb.alert(
                    title="Loss is inf or nan",
                    text=f"The loss is {loss:.2f}, check whether the training is still running correctly.",
                    level=AlertLevel.WARN,
                    wait_duration=21600,  # every 6 hours
                )

            # Check if the loss is a lot higher than the average loss over the last steps
            if self.loss_buffer.add(loss, significance=7):
                wandb.alert(
                    title="Loss anomaly detected",
                    text=f"Current loss {loss:.2f} deviates by {self.loss_buffer.latest_deviation:.2f} from the average loss over the last {self.loss_buffer.size} steps.",
                    level=AlertLevel.WARN,
                    wait_duration=21600,  # every 6 hours
                )

        if "epoch_time" in extra_infos:
            epoch_time = extra_infos["epoch_time"]
            self.epoch_time_buffer.append(epoch_time)

            # If the buffer isn't full yet, report -inf as the mean to prevent sending an alert.
            epoch_mean = (
                np.mean(self.epoch_time_buffer)
                if len(self.epoch_time_buffer) == self.epoch_time_buffer.maxlen
                else float("-inf")
            )

            # If the mean of the epoch time over the last 15 epochs is higher than the threshold, send an alert.
            if epoch_mean > self.epoch_time_threshold:
                wandb.alert(
                    title="Epoch time alert",
                    text=f"The mean over the last 15 epochs took {epoch_mean:.2f} seconds which is above our threshold of {self.epoch_time_threshold:.2f} seconds.",
                    level=AlertLevel.WARN,
                    wait_duration=21600,  # every 6 hours
                )

    def save_callback(
        self, model, epoch, data_loader, scaler_state, optimizer_state, extra_infos=None
    ):
        import wandb

        extra_infos = {} if extra_infos is None else extra_infos
        save_callback_start_time = time.time()
        advanced_epoch = epoch > 0

        self.check_for_alerts(extra_infos)

        epochs = self.config_sample["epochs"]
        is_in_final_epoch = epoch == epochs

        log_msg = {}

        extra_infos = {} if extra_infos is None else extra_infos
        log_msg = {**log_msg, **extra_infos}

        if advanced_epoch:
            print(
                f"EPOCH CALLBACK ({epoch}/{epochs}), ESTIMATED TIME: "
                + str((((time.time() - self.start_time) / (epoch + 1)) * epochs) / 3600)
            )

        print("Saving model..")

        checkpoint = Checkpoint(
            state_dict=model.state_dict(),
            optimizer_state=optimizer_state,
            scaler_state=scaler_state,
            config=self.config_sample,
            trained_epochs_until_now=epoch,
        )
        checkpoint.save(self.model_path)

        print("Evaluating model..")

        # these are used as x-axis for wandb plots, if they are passed along
        add_to_log = {
            "epoch": epoch,
            "training_progress": epoch / epochs,
        }

        for split_name, datasets in self.splits_to_evaluate.items():
            if len(datasets) == 0:
                continue

            # Load baseline results and evaluate model types
            model_types = ["single_fast"]

            (
                evaluation_results,
                log_msg_scores,
            ) = self.evaluate_model_types_and_compare_to_baselines(
                self.build_predictors(model=model),
                split_name,
                n_splits=(
                    NUM_SPLITS
                    if advanced_epoch or self.run_full_eval_before_training
                    else 1
                ),
            )

            # Only calculates the aggregate results table for ranks, wins, normalized metrics
            evaluate_and_plot(
                evaluation_results=evaluation_results,
                evaluation_plots=[FullTabularResultsTablePlot()],
                evaluation_storers=[
                    WandbStorer(
                        logs_only=True,
                        add_to_log=add_to_log,
                    )  # Storing no figures, but only metrics aka logs to wandb
                ],
                incumbant_model_names=model_types,
                benchmark_name=split_name,
            )

            # Having more model types requires changing the log_msg_scores
            assert len(model_types) == 1, "Only one model type is supported here."
            # Result logs are prepended by the model type, we want to keep previous format
            delimiter = "/"
            log_msg_scores = {
                f"{delimiter.join(str.split(k, delimiter)[1:])}": log_msg_scores[k]
                for k in log_msg_scores
            }

            if advanced_epoch or self.run_full_eval_before_training:
                log_msg = {**log_msg, **log_msg_scores}

        log_msg = {
            **log_msg,
            **{
                f"model_path": f"{self.model_path}",
                "epoch_callback_time": time.time() - save_callback_start_time,
            },
            **add_to_log,
        }

        wandb.log(log_msg)

        if hasattr(model, "reset_save_peak_mem_factor"):
            print("Resetting peak memory factor")
            model.reset_save_peak_mem_factor()

        if is_in_final_epoch and self.perform_final_evaluation:
            for split in self.splits_to_evaluate:
                if len(self.splits_to_evaluate[split]) == 0:
                    continue
                self._full_evaluation(
                    model,
                    split=split,
                    model_types=self.final_evaluation_model_types,
                    n_splits=NUM_SPLITS_FINAL,
                )

        self.early_stop_callback(log_msg)
        print("Finished epoch callback")

    def step_callback(self, metrics, step):
        import wandb

        if step % self.step_logging_frequency == 0:
            metrics["step"] = step
            wandb.log(metrics)

    def get_default_results_path(self):
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(os.path.join(self.base_path, "evaluation_results"), exist_ok=True)

        path = os.path.join(
            self.base_path, "evaluation_results", f"{self.files_prefix}"
        )
        os.makedirs(path, exist_ok=True)
        return path

    def setup_evaluation_storers(self):
        import wandb

        if self.evaluation_storers is None:
            path = self.get_default_results_path()
            # file_system_storer = FilesystemStorer(path=path)
            self.evaluation_storers = [WandbStorer()]  # file_system_storer,
            # wandb.config.update({"evaluation_results_path": file_system_storer.path})

        wandb.config.update(
            {
                "evaluation_storers": [
                    storer.__class__.__name__ for storer in self.evaluation_storers
                ]
            }
        )

    def setup_evaluation_plots(self):
        if self.evaluation_plots is None:
            self.evaluation_plots = get_plotter_collection(self.task_type).plots

    @catch_wandb_issue
    def train(
        self,
        mode: TrainingMode = TrainingMode.NORMAL,
        sweep_id=None,
        use_signal_handler=True,
    ):
        """
        This function is used to train a model and supports currently three modes:
        - TrainingMode.NORMAL: Train a model from scratch using the provided config_sample.
        - TrainingMode.HPO: Combine the provided config_sample with the pulled hyperparameters
                            of the wandb sweep and train a model from scratch.

        The implementation supports distributed training and will only log to wandb from rank 0.
        In case of TrainingMode.HPO it pulls the sweep config only from rank 0 and broadcasts the
        config to all other ranks, which block until the config is ready.

        NOTE: In order for this to work, the wandb_path needs to be on a shared filesystem between all nodes
        involved in the distributed training. Also the model_string has to be unique to prevent conflicting
        config and sentinel files.

        Args:
            mode: The mode of this training. Defaults to TrainingMode.NORMAL.
            sweep_id: In case of a TrainingMode.HPO, the id of the sweep to use. Defaults to None.
            use_signal_handler: Whether to use the SLURM signal handler or not. Defaults to True.
            port: The port to use in case of distributed training. Defaults to 12355.
        """
        import wandb

        self.init_signal_handler(use_signal_handler)

        self.wandb_login()

        if mode == TrainingMode.HPO:
            print("Starting hpo training..")
            assert (
                sweep_id is not None
            ), "If you want to do a hpo search, you need to provide a sweep_id."

            self.sweep_id = sweep_id

            # Only run the wandb agent in rank 0
            if self.rank == 0:
                return wandb.agent(
                    sweep_id=self.sweep_id,
                    entity=self.wandb_entity,
                    project=self.wandb_project,
                    function=partial(self._train, mode=mode),
                    count=1,  # Not specifying count would run the sweep indefinitely
                )
        elif mode == TrainingMode.NORMAL:
            print("Starting training..")
        else:
            raise ValueError(f"Unknown training mode {mode}.")

        o = self._train(mode=mode)

        wandb.finish()

        return o

    def _train(self, mode: TrainingMode = TrainingMode.NORMAL):
        import wandb

        run = self.init_wandb()

        try:
            # Multi-GPU trainings pose some issues in the cases of hpo training as well as continue training.
            # In case of hpo training, only rank 0 pulls the sweep config and needs to broadcast it to the other ranks.

            # Pull the default config in case a config wasn't provided
            if self.config_sample is None:
                # TODO: max_classes & feats anschauen
                self.config_sample = load_config(
                    task_type=self.task_type,
                )

            # Always update the run id of this training since this could be continued at a later point as well
            self.config_sample["wandb_run_id"] = run.id

            # In case of hpo training, pull the sweep config.
            if mode == TrainingMode.HPO and self.rank == 0:
                # In case of hpo, we merge the sweep config with the local config.
                self.config_sample = set_from_dict(self.config_sample, wandb.config)

            self.config_sample = remap_config_for_hp_run(self.config_sample)

            # Preprocess the config, if we are starting a new run
            if self.continue_model_path is None:
                if self.rank == 0:
                    self.config_sample = preprocess_config(
                        self.config_sample, self.task_type
                    )  # This remaps parameters such as batch size and aggregate gradients
                print(
                    f"{self.device=}, {torch.cuda.is_available()=}, {torch.cuda.device_count()=}, {torch.cuda.device=}"
                )
                using_dist, _, _, _ = init_dist(
                    self.device if torch.cuda.is_available() else "cpu"
                )
                config_sample_ = [self.config_sample]
                if using_dist:
                    torch.distributed.broadcast_object_list(config_sample_)
                (self.config_sample,) = config_sample_
                print("ALL after broadcast_object_list")

            if self.mode == TrainingMode.HPO:
                print(
                    "Updating config, ignore WANDB warning for sweep parameter updates"
                    "These are modified accorinding to sweep."
                )

            wandb.config.update(self.config_sample, allow_val_change=True)
            # wandb.run.log_code("/home/hollmann/prior-fitting-mew/", include_fn=lambda path : path.endswith(".py") or path.endswith(".ipynb"))
            wandb.log(
                {
                    "valid_datasets": [
                        ds.name for ds in self.splits_to_evaluate.get("valid", [])
                    ]
                }
            )
            if "SLURM_JOB_ID" in os.environ:
                wandb.log({"SLURM_JOB_ID": os.environ["SLURM_JOB_ID"]})

            self.setup_evaluation_storers()
            self.setup_evaluation_plots()

            ### Start the training at this point ###

            self.start_time = time.time()

            print("going to save checkpoints at: ", self.model_path)

            epoch_callback_handler = (
                self.save_callback if self.rank == 0 else lambda *args, **kwargs: None
            )
            step_callback_handler = (
                self.step_callback if self.rank == 0 else lambda *args, **kwargs: None
            )

            model_loading_kwargs = {}
            if self.loaded_checkpoint is not None:
                model_loading_kwargs = {
                    "optimizer_state": self.loaded_checkpoint.optimizer_state,
                    "state_dict": self.loaded_checkpoint.state_dict,
                    "scaler_state": self.loaded_checkpoint.scaler_state,
                }

            print("ALL: lets get model")

            return get_model(
                self.config_sample,
                self.device,
                should_train=True,
                verbose=1,
                config_is_preprocessed=True,
                wandb_track_grads_freq=self.wandb_track_grads_freq,
                epoch_callback=epoch_callback_handler,
                step_callback=step_callback_handler,
                **model_loading_kwargs,
            )
        except Exception as e:
            import traceback

            print("Exception occured during training.")
            print(traceback.format_exc())
        finally:
            wandb.finish()


class BaselineTrainingAgent(TrainingAgent):
    def __init__(
        self,
        task_type="multiclass",
        device="cpu",
        splits_to_evaluate: None
        | tp.Dict[str, tp.List[datasets.TabularDataset] | None] = None,
        wandb_key=None,
        wandb_project=None,
        wandb_entity=wandb_entity,
        base_path=base_path,
        model_path=None,
        wandb_path=None,
        mode: tp.Literal["online", "offline", "disabled"] = "online",
        model_string="baseline_training",
        wandb_tags=None,
        renamed_gpu_runs: bool = False,
        remapped_time: bool = True,
        splits_per_dataset_to_evaluate: list[int] | None = None,
        task_settings_overwrite: dict[str, int] | None = None,
    ):
        super().__init__(
            task_type=task_type,
            splits_to_evaluate=splits_to_evaluate,
            device=device,
            base_path=base_path,
            wandb_key=wandb_key,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            model_path=model_path,
            wandb_path=wandb_path,
            mode=mode,
            model_string=model_string,
            wandb_tags=wandb_tags,
            renamed_gpu_runs=renamed_gpu_runs,
            remapped_time=remapped_time,
            splits_per_dataset_to_evaluate=splits_per_dataset_to_evaluate,
            task_settings_overwrite=task_settings_overwrite,
        )

        assert (
            self.world_size == 1
        ), "Baseline training is currently not supported for distributed trainings."

        print("Baseline TrainingAgent initialized")

    @catch_wandb_issue
    def train(
        self,
        method_name: str,
        max_time: int = 300,
        random_state: int = 0,
        use_signal_handler: bool = True,
        save: bool = False,
        overwrite: bool = False,
        fetch_only: bool = False,
        num_splits: int = NUM_SPLITS,
        method_kwargs: dict = {},
        datasets_part_of_parts: tp.Optional[tuple[int, int]] = None,
        metrics_for_eval: list[str] | None = None,
    ):
        """
        :param method_name: a method name that appears in get_clf_dict(self.task_type)
        :param max_time: maximum time in seconds to train the model, default 300
        :param random_state: random state for the training
        :param use_signal_handler: whether to handle SLURM signals gracefully
        :param save: whether to save the results in the base_path, dangerous
        :param overwrite: ~overwrite is pretty much equivalent to save
        :param fetch_only: whether to fetch the results from wandb and return them
        :param num_splits: number of splits to use for the evaluation
        :param method_kwargs: kwargs for the method, e.g. `{"model": TabPFNClassifier()}` for "transformer"
        :param datasets_part_of_parts: This option is useful to run baselines in parallel. This way you can start a baseline on a subset of the datasets.
        It works by specifying a tuple of (part, num_parts) to use for the evaluation. This is the current part of datasets to be evaluated (`part`) and the total number of parts (`num_parts`).
        This will then only use the `part`th part of the datasets split into `num_parts`.
        :param metrics_for_eval: list of metrics to evaluate, if None, all `self.valid_metrics` are computed.
        :return:
        """
        import wandb
        from .tabular_baselines import get_clf_dict

        print(
            f"Starting baseline (method: {method_name}, max_time: {max_time}) training.."
        )

        metric = get_clf_dict(self.task_type)[method_name]
        metric = partial(metric, **method_kwargs)
        assert (
            metric is not None
        ), f"Unknown method {method_name} for task type {self.task_type}."

        self.init_signal_handler(use_signal_handler)
        self.wandb_login()
        self.init_wandb(name=f"{self.model_string}_{method_name}_{max_time}_sec")

        # Report the config to wandb.
        wandb.config.update(
            {"method": method_name, "max_time": max_time}, allow_val_change=True
        )

        log_msg = {}
        for split_name, datasets in self.splits_to_evaluate.items():
            if len(datasets) == 0:
                continue

            print("evaluating split", split_name)

            if datasets_part_of_parts is not None and datasets_part_of_parts[1] != 1:
                assert (
                    self.mode == "disabled"
                ), "splitting datasets means that this run can't know the score for the full val or test set"

                def get_ith_part(lst, num_parts, i):
                    n = len(lst)
                    avg = n // num_parts
                    remainder = n % num_parts

                    start = i * avg + min(i, remainder)
                    end = start + avg + (1 if i < remainder else 0)

                    return lst[start:end]

                split, num_splits = datasets_part_of_parts
                datasets = get_ith_part(datasets, num_splits, split)

            log_msg_part, r = evaluate_and_score(
                datasets,
                [
                    m
                    for m in self.valid_metrics
                    if f"{m['aggregator']}_{m['name']}" in metrics_for_eval
                ]
                if metrics_for_eval is not None
                else self.valid_metrics,
                metric,
                split_name=split_name,
                metric_used=self.metric_used,
                log_per_dataset_metrics=True,
                max_time=max_time,
                num_splits=num_splits,
                random_state=random_state,
                save=save,
                overwrite=overwrite,
                fetch_only=fetch_only,
                base_path=self.base_path,
                path_interfix=self.task_type,
                method_name=method_name,
                rename_gpu_runs=self.renamed_gpu_runs,
                allow_remap_time=self.remapped_time,
                splits=self.splits_per_dataset_to_evaluate,
                evaluate_subsets=False,
            )
            log_msg = {**log_msg, **log_msg_part}

        out = (log_msg, r)

        wandb.log(log_msg)
        wandb.finish(exit_code=0)

        return out


def wandb_get_model(
    *args,
    train_kwargs=None,
    **kwargs,
):
    """
    A simple wrapper function that initializes the TabPFNTrainingAgent
    and starts a training in one step. That is particularly useful when
    submitting a Job to the SLURM Cluster as just this function can be
    passed along the arguments.
    """
    if train_kwargs is None:
        train_kwargs = {}

    if "config" in kwargs and not "config_sample" in kwargs:
        kwargs["config_sample"] = kwargs["config"]
        del kwargs["config"]

    agent = TabPFNTrainingAgent(
        *args,
        **kwargs,
    )

    return agent.train(**train_kwargs)


def wandb_get_baseline_results(
    *args,
    train_kwargs=None,
    **kwargs,
):
    """
    A simple wrapper function that initializes the BaselineTrainingAgent
    and starts a training in one step. That is particularly useful when
    submitting a Job to the SLURM Cluster as just this function can be
    passed along the arguments.
    """
    if train_kwargs is None:
        train_kwargs = {}

    agent = BaselineTrainingAgent(
        *args,
        **kwargs,
    )

    return agent.train(**train_kwargs)


"""
How to use this script for single gpu training:
```bash
python -m scripts.runner --config_path test_conf.json --adapt_config "c['hi'] = 'whats up'; c['nlayers'] = 3"
```
It is not allowed to create new config keys, only to change existing ones, this prevents errors due to typos.

For multi gpu you can do something like this
```bash
torchrun --nproc-per-node=<num gpus in your setup> scripts/runner.py --config_path config_for_reg_on_helix.yaml --task_type regression --adapt_config "c['recompute_layer']=True"
```
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--model_string",
        type=str,
        default=None,
        help="A string that is used to identify the model in the wandb project. If not given, the model is identified by the config.",
    )
    parser.add_argument("--task_type", type=str, default="regression")
    parser.add_argument("--save_model", type=bool, default=True)
    parser.add_argument("--base_path", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=wandb_entity)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument(
        "--adapt_config",
        type=str,
        default=None,
        help="Executable python code to adapt the config. The config dict is available as `c`.",
    )
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--no_validation", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    task_type = args.task_type
    if args.wandb_project is None:
        wandb_project = get_wandb_project(task_type)
    else:
        wandb_project = args.wandb_project
    save_model_enabled = args.save_model

    device = args.device
    sweep_id = args.sweep

    if sweep_id:
        train_kwargs = {"mode": TrainingMode.HPO, "sweep_id": sweep_id}
    else:
        train_kwargs = {}

    config = {
        "task_type": task_type,
        "device": device,
        "wandb_project": wandb_project,
        "wandb_entity": args.wandb_entity,
        "save_model_enabled": save_model_enabled,
        "train_kwargs": train_kwargs,
    }

    if args.model_string is not None:
        config["model_string"] = args.model_string

    if args.base_path is not None:
        config["base_path"] = args.base_path

    if args.disable_wandb:
        config["mode"] = "disabled"

    if args.debug:
        config["debug"] = True

    if args.no_validation:
        config["splilts_to_evaluate"] = {"valid": []}

    if args.config_path is not None:
        import yaml

        with open(args.config_path, "r") as f:
            config["config"] = yaml.load(f, Loader=yaml.FullLoader)

    if args.adapt_config is not None:
        assert "config" in config

        orig_keys = set(config["config"].keys())
        exec(args.adapt_config, {"c": config["config"]})
        assert len(config["config"]) == len(orig_keys), (
            "You created a new config entry with your `adapt_config` code. Only change existing entries. We do not allow creating new entries here to avoid typos."
            f"You created the new entries: {set(config['config'].keys()) - orig_keys}"
        )
    print(config)

    wandb_get_model(**config)
