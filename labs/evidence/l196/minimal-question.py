import pandas as pd
from rdblearn.preprocessing import TabularPreprocessor
train = pd.DataFrame({'category': ['b', 'c', 'd'] * 4,
                      'number': list(range(12))})
known = pd.DataFrame({'category': ['b', 'c', 'd'], 'number': [1, 2, 3]})
for unseen in ['a', 'z', '0', 'e']:
    pipeline = TabularPreprocessor().fit(train)
    support = pipeline.transform(train)
    before = pipeline.transform(known)
    pipeline.transform(pd.DataFrame({'category': [unseen], 'number': [1]}))
    after = pipeline.transform(known)
    print(unseen, before.category.tolist(), after.category.tolist())
