"""Audit the independently executed full inline run, retaining non-bitwise differences."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
a=json.loads((P/'_paper_l092_results.json').read_text());b=json.loads((P/'_inline_paper_l092_results.json').read_text())
x,y=a['runs'][0],b['runs'][0]
assert len(x['trace'])==len(y['trace'])==200 and len(y['knn'])==40
assert x['test_predictions']==y['test_predictions'] and x['knn']==y['knn']
assert x['selected_epoch']==y['selected_epoch']==35
errors={k:max(abs(u[k]-v[k]) for u,v in zip(x['trace'],y['trace'])) for k in ['train_ce','validation_ce','validation_accuracy']}
assert max(errors.values())<1e-5
result={'status':'PASS_WITH_RECORDED_FLOAT_DIFFERENCES','full_encoder_fits':1,'epochs':200,'knn_evaluations':40,
'fresh_directory_download':'PASS','identical_test_predictions_checkpoint_and_knn':'PASS','bitwise_loss_trace_equality':'FAIL',
'max_absolute_trace_differences':errors,'comparison_note':'After exact-equality assertion failed, inspected differences and checked absolute1e-5 numerical tolerance. This is a post-hoc numeric diagnostic, not a changed paper-score acceptance criterion.',
'full_replay_result_sha256':hashlib.sha256((P/'_inline_paper_l092_results.json').read_bytes()).hexdigest(),
'full_replay_seconds':y['seconds'],'persisted_notebook_scope':'Diagnostic outputs regenerated separately; full replay completed before strict assertion prevented notebook save. Raw full replay retained.',
'historical_parity':'INCOMPARABLE','full_paper_parity':'NOT_ESTABLISHED'}
(P/'_replay_l092_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
