"""Behavioral contract for the additive bandwidth diagnostic; tiny real-data run."""
import importlib.util
import numpy as np
from pathlib import Path

assert importlib.util.find_spec('_diagnose_l051'), 'Missing additive bandwidth diagnostic'
from _diagnose_l051 import smoothing_sweep
from relkit.bias_interventions import prepare_task

cache=Path(__file__).parent/'data/cache/l051'
result=smoothing_sweep('electricity',cache,max_rows=500,hs=(0.,.5,1e12),seeds=(0,),epochs=1,trees=5)
states,labels,metadata=prepare_task('electricity',cache,500)
assert result['test_y']==labels[2].tolist(), 'The diagnostic must retain raw held-out labels'
assert result['split_rows']==metadata['split_rows'], 'Bandwidth must not change split identity'
assert result['rows'][0]['changed_labels']==0, 'h=0 must recover original training labels'
for row in result['rows']:
    assert 0 <= row['positive_fraction'] <= 1
    for model,runs in row['runs'].items():
        for run in runs:
            assert run['accuracy']==np.mean((np.array(run['probability'])>=.5)==labels[2])
        assert row['effects'][model]['ci95'] is None, 'One seed has no seed interval'
        if row['h']==0: assert row['effects'][model]['mean']==0
assert result['rows'][1]['changed_labels']==metadata['changed_training_labels']
assert result['rows'][2]['status']=='NOT_FIT_ONE_CLASS', 'Report class collapse without inventing a classifier score'
assert not result['rows'][2]['runs'] and not result['rows'][2]['effects']
print('PASS: real-data bandwidth baseline, split, raw targets, probabilities and one-seed uncertainty')
