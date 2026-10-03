# PFN Repo

## Installation

Simply run `pip install .` in the root directory of this repository or install it as a package from the repository.

## How to get a working TabPFN to play around with
To get our standard sklearn interfaces, simply use the following imports:
```python
from tabpfn import TabPFNClassifier, TabPFNRegressor
```

## Example usage
```python
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier
 
# Create a classifier
# The model will use a CUDA-enabled GPU if available, otherwise it will use the CPU.
# The most important parameters:
#   - The default is to `fit_at_predict_time` which means that the model will be fit at the time of prediction, this is useful for large datasets and when you only want to predict once.
#     If you want to predict multiple times for the same training set, set `fit_at_predict_time=False` and the model will be fit at the time of `fit` and save its state for later.
#   - You can also set the number of `n_estimators`, this is the easiest way to control the trade-off between speed and accuracy.
clf = TabPFNClassifier(fit_at_predict_time=True)

X, y = load_iris(return_X_y=True)
feature_names = load_iris()['feature_names']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf.fit(X_train, y_train)

preds = clf.predict_proba(X_test)  # <- all the compute happens in here
y_eval = np.argmax(preds, axis=1)

print('ROC AUC: ',  sklearn.metrics.roc_auc_score(y_test, preds, multi_class='ovr'), 'Accuracy', sklearn.metrics.accuracy_score(y_test, y_eval))
```
