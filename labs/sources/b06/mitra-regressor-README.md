---
license: apache-2.0
pipeline_tag: tabular-regression
---

# Mitra Regressor

Mitra regressor is a tabular foundation model that is pre-trained on purely synthetic datasets sampled from a mix of random regressors. 

## Architecture

Mitra is based on a 12-layer Transformer of 72 M parameters, pre-trained by incorporating an in-context learning paradigm.

## Usage

To use Mitra regressor, install AutoGluon by running:

```sh
pip install uv
uv pip install autogluon.tabular[mitra]   
```

A minimal example showing how to perform inference using the Mitra regressor:

```python
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# Load datasets
housing_data = fetch_california_housing()
housing_df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
housing_df['target'] = housing_data.target

print("Dataset shapes:")
print(f"California Housing: {housing_df.shape}")

# Create train/test splits (80/20)
housing_train, housing_test = train_test_split(housing_df, test_size=0.2, random_state=42)

print("Training set sizes:")
print(f"Housing: {len(housing_train)} samples")

# Convert to TabularDataset
housing_train_data = TabularDataset(housing_train)
housing_test_data = TabularDataset(housing_test)

# Create predictor with Mitra for regression
print("Training Mitra regressor on California Housing dataset...")
mitra_reg_predictor = TabularPredictor(
    label='target',
    path='./mitra_regressor_model',
    problem_type='regression'
)
mitra_reg_predictor.fit(
    housing_train_data.sample(1000), # sample 1000 rows
    hyperparameters={
        'MITRA': {'fine_tune': False}
    },
)

# Evaluate regression performance
mitra_reg_predictor.leaderboard(housing_test_data)
```

## License

This project is licensed under the Apache-2.0 License.

## Reference

```
@article{zhang2025mitra,
  title={Mitra: Mixed synthetic priors for enhancing tabular foundation models},
  author={Zhang, Xiyuan and Maddix, Danielle C and Yin, Junming and Erickson, Nick and Ansari, Abdul Fatir and Han, Boran and Zhang, Shuai and Akoglu, Leman and Faloutsos, Christos and Mahoney, Michael W and others},
  journal={arXiv preprint arXiv:2510.21204},
  year={2025}
}
```

Amazon Science blog: [Mitra: Mixed synthetic priors for enhancing tabular foundation models](https://www.amazon.science/blog/mitra-mixed-synthetic-priors-for-enhancing-tabular-foundation-models?utm_campaign=mitra-mixed-synthetic-priors-for-enhancing-tabular-foundation-models&utm_medium=organic-asw&utm_source=linkedin&utm_content=2025-7-22-mitra-mixed-synthetic-priors-for-enhancing-tabular-foundation-models&utm_term=2025-july)