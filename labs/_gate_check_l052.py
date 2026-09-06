"""Fit smoke, resume without fitting, and reject a mutated live helper."""
import json,tempfile
from pathlib import Path
import relkit.tabr as model_module
from _paper_repro_l052 import reproduce,live_identity
from relkit.tabr_experiment import run_suite

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='l052-gate-') as directory:
        first=reproduce('smoke',directory)
        second=reproduce('smoke',directory)
        assert first['summary']==second['summary']
        original=model_module.eligible_mask
        def broken(query_ids,candidate_ids):
            raise AssertionError('Should reject contract before executing this helper')
        before=live_identity(model_module.TabRS,run_suite)
        model_module.eligible_mask=broken
        try:
            assert live_identity(model_module.TabRS,run_suite)!=before
            try:reproduce('smoke',directory)
            except ValueError as e:assert 'Changed code' in str(e)
            else:raise AssertionError('Mutated live helper was accepted')
        finally:model_module.eligible_mask=original
    result={'status':'PASS','fit':True,'completed_seed_resume':True,'changed_live_helper_rejected':True}
    (root/'_gate_l052_results.json').write_text(json.dumps(result,indent=2));print(result)
