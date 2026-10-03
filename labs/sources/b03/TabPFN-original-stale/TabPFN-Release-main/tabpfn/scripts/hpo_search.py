import os

import numpy as np

from datetime import datetime

from tabpfn.scripts.model_builder import get_model

from tabpfn.scripts.model_configs import *

from ConfigSpace import hyperparameters as CSH

from tabpfn.datasets import (
    load_openml_list,
    test_dids_classification,
    valid_large_classification,
)
from tabpfn.priors.utils import (
    uniform_int_sampler_f,
)

from tabpfn.scripts.tabular_evaluation import evaluate
from tabpfn.scripts.tabular_metrics import calculate_score_per_method

from tabpfn.scripts import tabular_metrics

import submitit

from tabpfn.priors.differentiable_prior import replace_differentiable_distributions
from tabpfn.scripts.model_configs import (
    fill_in_configsample,
    create_configspace_from_hierarchical,
)
import time

import sys

sys.path.append("DEHB/")
from dehb import DEHB


def postprocess_data(datasets):
    for ds in datasets:
        sel = [len(np.unique(ds[1][:, col])) > 1 for col in range(ds[1].shape[1])]
        # print(ds[0], ds[1][:, sel].shape, ds[1].shape)
        # print(ds[0], [np.unique(ds[1][:, col], return_counts=True) for col in range(ds[1].shape[1])])
        ds[1] = np.array(ds[1][:, sel])
        ds[1] = torch.tensor(ds[1])
        print(ds[0])


large_datasets = True
max_samples = 10000 if large_datasets else 5000
seq_len = 10000 if large_datasets else 3000


valid_large_classification = list(
    set(valid_large_classification) - {384, 387, 389, 396, 273}
)

test_datasets_multiclass, test_datasets_multiclass_df = load_openml_list(
    test_dids_classification,
    multiclass=True,
    shuffled=True,
    filter_for_nan=False,
    max_samples=max_samples,
    num_feats=100,
    return_capped=True,
)
valid_datasets_multiclass, valid_datasets_multiclass_df = load_openml_list(
    valid_large_classification,
    multiclass=True,
    shuffled=True,
    filter_for_nan=False,
    max_samples=max_samples,
    num_feats=100,
    return_capped=True,
)

# test_datasets_binary, test_datasets_binary_df = load_openml_list(test_dids_classification, multiclass=False, shuffled=False, filter_for_nan=False, max_samples = max_samples, num_feats=100, return_capped=True)
# valid_datasets_binary, valid_datasets_binary_df = load_openml_list(valid_large_classification, multiclass=False, shuffled=False, filter_for_nan=False, max_samples = max_samples, num_feats=100, return_capped=True)

postprocess_data(test_datasets_multiclass)
postprocess_data(valid_datasets_multiclass)
# postprocess_data(test_datasets_binary)
# postprocess_data(valid_datasets_binary)


def get_datasets(selector, task_type):
    if task_type == "binary":
        raise ValueError("binary not supported")
    else:
        ds = (
            valid_datasets_multiclass
            if selector == "valid"
            else test_datasets_multiclass
        )
    return ds


device = "cuda"
base_path = os.path.join("/work/dlclarge1/hollmann-PFN_Tabular/")
max_features = 100

maximum_runtime = 0


def get_prior_config_causal():
    config_general = get_general_config(max_features, 50, eval_positions=[30])
    config_general_real_world = {**config_general}

    config_flexible_categorical = get_flexible_categorical_config(max_features)
    config_flexible_categorical_real_world = {**config_flexible_categorical}
    config_flexible_categorical_real_world[
        "num_categorical_features_sampler_a"
    ] = -1.0  # Categorical features disabled by default

    config_gp = {}
    config_mlp = {}

    config_diff = get_diff_config()

    config = {
        **config_general_real_world,
        **config_flexible_categorical_real_world,
        **config_diff,
        **config_gp,
        **config_mlp,
    }

    return config


def get_prior_config_causal_only():
    config = get_prior_config_causal()
    config["differentiable_hyperparameters"]["prior_bag_exp_weights_1"] = {
        "distribution": "uniform",
        "min": 1000000.0,
        "max": 1000001.0,
    }  # Always select MLP
    return config


def get_prior_config_gp():
    config = get_prior_config_causal()
    config["differentiable_hyperparameters"]["prior_bag_exp_weights_1"] = {
        "distribution": "uniform",
        "min": 0.0,
        "max": 0.0000001,
    }  # Never select MLP
    return config


def get_prior_config(config_type):
    if config_type == "causal":
        return get_prior_config_causal()
    elif config_type == "gp":
        return get_prior_config_gp()
    elif config_type == "bnn":
        raise NotImplemented()
    elif config_type == "causal_only":
        return get_prior_config_causal_only()
    elif config_type == "bag_gp_bnn":
        raise NotImplemented()


def reload_config(
    config_type="causal",
    task_type="multiclass",
    longer=1,
    sample_differentiable_hyperparams_upfront=False,
):
    assert longer
    config = get_prior_config(config_type=config_type)

    config["prior_type"], config["differentiable"], config["flexible"] = (
        "prior_bag",
        True,
        True,
    )

    model_string = ""

    if longer == 1:
        config["seq_len"] = CSH.CategoricalHyperparameter("seq_len", [1024, 2048])
        # config['seq_len'] = hp.choice('seq_len', list(range(700, 1200)))
        config["batch_size"] = CSH.CategoricalHyperparameter(
            "batch_size", [2**i for i in range(2, 3)]
        )

        # Estimate how many gradients need to be aggregated
        # Depends on seq_len, nlayers, emsize and must batch_size > 1
        config["aggregate_k_gradients"] = config["batch_size"] * (
            (
                config["nlayers"]
                * config["emsize"]
                * config["seq_len"]
                * config["seq_len"]
            )
            / 10824640000
        )
        # config['aggregate_k_gradients'] = hp.choice('batch_size', [config['batch_size'], config['batch_size'] // 2, config['batch_size'] // 4])

        config["epochs"] = 600
        config["recompute_attn"] = True
        config["seq_len_extra_samples"] = None  # changed
        config["dynamic_batch_size"] = False

        # use_tokens = True
        # if use_tokens:
        # config['global']

        model_string = model_string + "_longer"
    elif longer == "hpo":
        config["seq_len"] = CSH.CategoricalHyperparameter("seq_len", [1024])
        # config['seq_len'] = hp.choice('seq_len', list(range(700, 1200)))
        config["batch_size"] = 16
        config["weight_decay"] = CSH.UniformFloatHyperparameter(
            "weight_decay", 0.00001, 0.1, log=True
        )

        # Estimate how many gradients need to be aggregated
        # Depends on seq_len, nlayers, emsize and must batch_size > 1
        config["aggregate_k_gradients"] = 4
        # config['aggregate_k_gradients'] = config['batch_size'] * ((config['nlayers'] * config['emsize'] * config['seq_len'] * config['seq_len']) / 10824640000)
        # config['aggregate_k_gradients'] = hp.choice('batch_size', [config['batch_size'], config['batch_size'] // 2, config['batch_size'] // 4])

        config["recompute_attn"] = True
        config["seq_len_extra_samples"] = None  # changed
        config["dynamic_batch_size"] = False

        # use_tokens = True
        # if use_tokens:
        # config['global']

        model_string = model_string + "_hpo"

    if sample_differentiable_hyperparams_upfront:
        model_string += "_nondiff"

    if task_type == "multiclass":
        config["max_num_classes"] = 10
        config["num_classes"] = uniform_int_sampler_f(2, config["max_num_classes"])
        # hp.choice('num_classes', [{'zipf' : zipf_sampler_f(1.1, 1, config['max_num_classes'])}
        #                         , {'uni': uniform_int_sampler_f(2, config['max_num_classes'])}])
        config["balanced"] = False
        config["multiclass_loss_type"] = CSH.CategoricalHyperparameter(
            "multiclass_loss_type", ["compatible"]
        )
        model_string = model_string + "_multiclass"

    model_string = (
        model_string
        + "_"
        + config_type
        + "_"
        + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        + "_sams"
    )

    return config, model_string


task_type, longer, device = "multiclass", 1, "cuda:0"
test_datasets, valid_datasets = get_datasets("test", task_type), get_datasets(
    "valid", task_type
)
config, model_string = reload_config(
    longer="hpo",
    task_type=task_type,
    sample_differentiable_hyperparams_upfront=True,
    config_type="causal_only",
)
replace_differentiable_distributions(config)
eval_addition = "seq_len1000"
eval_positions = [2000]
assert len(eval_positions) == 1
seq_len_test = 10000
N_draws = 0
N_grad_steps = 0
best_grad_steps = N_grad_steps


def objective(config_sample, budget=None):
    t = time.time()
    config_sample = fill_in_configsample(config, config_sample)

    config["epochs"] = budget

    train_res = get_model(config_sample, device, should_train=True, verbose=1)

    total_loss, total_positional_losses, model, dl = train_res

    # below line makes some trouble, commenting out, since it likely is not important
    # save_model(model, base_path, f'models_diff/prior_diff_real_checkpoint{model_string}_hpo_{uuid.uuid4().hex.upper()[0:6]}.ckpt', config_sample)

    max_features = config_sample["num_features"]
    params = {
        "max_features": max_features,
        "rescale_features": config_sample["normalize_by_used_features"],
        "normalize_to_ranking": config_sample["normalize_to_ranking"],
    }

    metrics_valid = evaluate(
        datasets=valid_datasets,
        model=model,
        method="transformer",
        device=device,
        overwrite=True,
        extend_features=True
        # just removed the style keyword but transformer is trained with style, just empty
        ,
        save=False,
        metric_used=tabular_metrics.cross_entropy,
        return_tensor=True,
        verbose=False,
        eval_positions=eval_positions,
        seq_len=seq_len_test,
        base_path=None,
        **params,
    )

    calculate_score_per_method(
        tabular_metrics.auc_metric, "roc", metrics_valid, valid_datasets, eval_positions
    )
    print(metrics_valid)

    assert len(eval_positions) == 1
    return {
        "fitness": metrics_valid[f"mean_roc_at_{eval_positions[0]}"],
        "cost": time.time() - t,
    }


def submitit_objective(ex, *args, **kwargs):
    return ex.submit(objective, *args, **kwargs).result()


if __name__ == "__main__":
    import argparse
    from functools import partial

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--partition", default="alldlc_gpu-rtx2080")
    parser.add_argument(
        "--total-cost", default=100, type=int, help="total time in hours"
    )
    args = parser.parse_args()
    cs = create_configspace_from_hierarchical(config)
    log_folder = os.path.join(base_path, "log_test_new/%j")
    ex = submitit.get_executor(
        folder=log_folder,
        slurm_partition=args.partition,
        timeout_min=24 * 60,
        slurm_job_name="pfn hpo",
    )
    dehb = DEHB(
        f=partial(submitit_objective, ex),
        dimensions=len(cs),
        cs=cs,
        min_budget=max(1, args.max_epochs // 10),  # change later
        max_budget=args.max_epochs,
        output_path="./temp",
        n_workers=args.n_workers,  # set to >1 to utilize parallel workers
    )
    dehb_result = dehb.run(
        total_cost=args.total_cost * 60 * 60, verbose=False, save_intermediate=True
    )
    print(dehb_result)
