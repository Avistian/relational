# Griffin

This is the official implementation of the paper "[Griffin: Towards a Graph-Centric Relational Database Foundation Model](https://arxiv.org/abs/2505.05568)".

---

## Table of Contents

- [Griffin](#griffin)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
  - [Dataset Preparation](#dataset-preparation)
    - [Using Provided Processed Datasets](#using-provided-processed-datasets)
    - [Processing Raw Data](#processing-raw-data)
  - [Pretraining \& Finetuning](#pretraining--finetuning)
    - [Using Provided Pretrained Checkpoints](#using-provided-pretrained-checkpoints)
    - [Pretraining from Scratch](#pretraining-from-scratch)
    - [Finetuning on Specific Datasets](#finetuning-on-specific-datasets)
    - [Transfer Experiments](#transfer-experiments)
      - [Overview](#overview)
      - [Data Splits](#data-splits)
      - [Pretrained Models](#pretrained-models)
      - [Run Command](#run-command)
      - [Notes](#notes)

---

## Getting Started

### Prerequisites

Install [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html) first.

Then install the following dependencies:

```bash
pip install datasets pqdm accelerate evaluate sentence_transformers einops torchmetrics seaborn
```

---

## Dataset Preparation

This section describes how to prepare the datasets for use with this project.

### Using Provided Processed Datasets

We provide already processed datasets for your convenience. You can download them from [Hugging Face RDB datasets collection](https://huggingface.co/datasets/yamboo/Griffin_datasets_joint_v65) and [Hugging Face Single datasets collection](https://huggingface.co/datasets/yamboo/Griffin_datasets_single_pretrain_v3).
The datasets are organized as follows:

```bash
./datasets/
├── joint-v65 # Major dataset, including all RDB datasets. Used for main experiments.
└── single-pretrain-v3 # Single-table dataset, including all single-table datasets. Used for pretraining.
```

### Processing Raw Data

If you wish to process the raw data yourself, the scripts and instructions are available in a separate branch: [processing_data](https://github.com/yanxwb/Griffin/tree/processing_data).

---

## Pretraining & Finetuning

This section covers how to use or pretrain your own models, and finetune them on specific datasets.

### Using Provided Pretrained Checkpoints

We provide pretrained model checkpoints to get you started quickly. You can download them from [Hugging Face checkpoints collection](https://huggingface.co/yamboo/Griffin_models).

The checkpoints are organized as follows:

```bash
./checkpoints/
├── single-completion # Pretrained single table completion model.
├── single-sft # Pretrained single table SFT model. Used in main experiments.
└── transfer # Pretrained transfer model. Used in transfer experiments.
    ├── commerce-1 # Split name.
       ├── FULL # Model name.
       ├── MIXED # Model name.
       └── LIMITED # Model name.
    ├── commerce-2 # Same as above.
       ├── FULL
       ├── MIXED
       └── LIMITED
    ├── others-1
       ├── FULL
       ├── MIXED
       └── LIMITED
    └── others-2
       ├── FULL
       ├── MIXED
       └── LIMITED
```

### Pretraining from Scratch

To pretrain the model from scratch, you can use the following command:

```bash
# Step 1: Completion Pretraining
accelerate launch --config_file hconfig.yaml hmaintask_completion.py datasets/single-pretrain-v3 logs/single-completion log --savepath checkpoints/single-completion --hop 0 --fanout 10 --fewshotfanout 0 --maxepoch 5 --batchsize 4096 --lr 0.00010178976613680036 --wd 0.008749895419888909 --num_mp 4 --use_rev True --use_gate True --eval_per_epoch 1 --hiddim 512

# Step 2: SFT Pretraining
accelerate launch --config_file hconfig.yaml hmaintask_combine.py datasets/single-pretrain-v3 logs/single-sft log --loadpath checkpoints/single-completion/best_checkpoint --savepath checkpoints/single-sft --task ALLTASK --hop 0 --fanout 10 --fewshotfanout 0 --maxepoch 40 --batchsize 4096 --lr 0.00042364843314963003 --wd 2.423189169972981e-05 --num_mp 4 --use_rev True --use_gate True --eval_per_epoch 2 --hiddim 512
```

### Finetuning on Specific Datasets

Once you have a pretrained model (either downloaded or trained by yourself), you can finetune it on specific downstream datasets.

To finetune the model, use the following command:

```bash
accelerate launch --config_file hconfig.yaml hmaintask_combine.py $dataset $log_path $log_name --loadpath $load_path --savepath $save_path --tasks $TASK --hop 2 --fanout 20 --maxepoch 50 --patience 15 --eval_per_epoch 2 --batchsize 256 --lr 3e-4 --wd 2e-4 --num_mp 4 --use_rev True --use_gate False --fewshotfanout 3 --hiddim 512 
```


### Transfer Experiments

#### Overview

The experiment involves four data splits and three types of pretrained models trained on each split. Each split contains six tasks, defined in the `task_names.yaml` file. The goal is to evaluate the transferability of models pretrained on one split to another.

#### Data Splits

The data splits are as follows:

- **commerce-1**
- **commerce-2**
- **others-1**
- **others-2**

#### Pretrained Models

The three pretrained models are:

- **FULL**
- **MIXED**
- **LIMITED**

#### Run Command

Use the following format to run the experiment:

```bash
bash transfer.sh $GPU $SPLIT_1 $SPLIT_2 $TASK $MODEL $EVAL_SAMPLE_RATIO $SEED_START $SEED_END
```

#### Notes

1. **Multiple GPUs:** You can use multiple GPUs to parallelize the experiments. For example, to use two GPUs with index 0 and 1:

```bash
bash transfer.sh 0,1 commerce-1 others-1 rel-f1-driver-dnf 1 42 43
```

2. **Model Indexing:**
   - `FULL`: index `1`
   - `MIXED`: index `2`
   - `LIMITED`: index `3`

3. **Evaluation Sample Ratio:**
   Evaluation sample ratio is set to 1 by default. This only applies to the validation phase and does not affect the final testing result. It is okay to set it to 1 for most tasks. If you want to evaluate on a smaller subset of the data to speed up the evaluation, you can set the evaluation sample ratio to a value less than 1 like 0.2. 

