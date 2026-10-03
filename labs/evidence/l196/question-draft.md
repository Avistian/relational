# Draft: v0.1.2 known-category codes change after an unseen category

Destination: https://github.com/HKUSHXLab/rdblearn/issues
Status: DRAFT_ONLY — not posted. Search existing open/closed threads before posting.

I'm preparing a reproduction of RDBLearn's published RelBench experiments. At v0.1.2 commit b5b03ebf8091547285a6e06cba53d2d1a40cb171, I observed the following through the original full TabularPreprocessor:

```python
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
```

Observed: a and 0 change known codes [0,1,2] to [1,2,3]; z and e retain [0,1,2]. Numeric controls remain [1,2,3]. Cached support codes remain [0,1,2] repeated four times. A separate fresh-pipeline check also changes the code for query b from 0 alone to 1 when batched with a or 0.

My expectation was that known-category meanings would remain aligned with already encoded support. How is that alignment intended to be preserved when DynamicLabelEncoder.transform expands and sorts its vocabulary, and which preprocessing revision/environment should I use to reproduce the published experiments?

Environment: Python 3.12; NumPy 2.3.5; pandas 2.3.3; scikit-learn 1.7.2; AutoGluon features/common 1.5.0; pydantic 2.11.7; FastDFS 0.2.1. pip check passed. Full dependency freeze and all four cases are included in the companion reproduction packet.

This is a synthetic preprocessing observation. I have not measured benchmark prevalence, downstream score impact, or established the historical paper environment. I would appreciate a correction if my expectation or setup is mistaken.

Local attachments for review: reproducer.zip, diagnostic.json, requirements-diagnostic.txt, and protocol.json beside this draft. Attach the packet when posting; local course paths will not be accessible to maintainers. Preserve the old commit if a newer revision is suggested.
