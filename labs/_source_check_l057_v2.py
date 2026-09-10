"""Compare pinned AutoGluon selector to every corrected measured OOF library."""
import json,hashlib,ast
from pathlib import Path
import numpy as np
from _source_check_l057 import LocalMetricAdapter,src,ns
from relkit.cross_ensemble import greedy_select,binary_loss,blend
ROOT=Path(__file__).resolve().parent

def main():
    # Diagnostic intervention only: disable upstream six-decimal score rounding.
    # This is explicitly a modified reference, not a claim of upstream parity.
    assert src.count('round_scores = True')==1
    modified=src.replace('round_scores = True','round_scores = False')
    namespace=dict(ns)
    classes=[n for n in ast.parse(modified).body if isinstance(n,ast.ClassDef)]
    exec(compile('from __future__ import annotations\n'+'\n\n'.join(ast.get_source_segment(modified,n) for n in classes),'<upstream rounding disabled>', 'exec'),namespace)
    class Unrounded(namespace['EnsembleSelection']):
        def _calculate_regret(self,y_true,y_pred_proba,metric,sample_weight=None):return binary_loss(y_true,y_pred_proba)
    cases=[]
    for item in json.loads((ROOT/'_data_l057_v2.json').read_text())['predictions']:
        path=ROOT/item['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        with np.load(path) as q:
            z,y=q['oof'],q['y'];w,trace=greedy_select(z,y,40)
            ref=LocalMetricAdapter(ensemble_size=40,problem_type='binary',metric=None).fit([z[:,j] for j in range(z.shape[1])],y)
            diagnostic=Unrounded(ensemble_size=40,problem_type='binary',metric=None).fit([z[:,j] for j in range(z.shape[1])],y)
            diagnostic_gap=float(np.max(np.abs(w-diagnostic.weights_)))
            assert diagnostic_gap<1e-12,(item['dataset'],item['seed'],diagnostic_gap)
            weights_gap=float(np.max(np.abs(w-ref.weights_)))
            pred_gap=float(np.max(np.abs(blend(z,w)-blend(z,ref.weights_))))
            cases.append(dict(dataset=item['dataset'],seed=item['seed'],weights_max_abs_error=weights_gap,
                oof_prediction_max_abs_error=pred_gap,local_weights=w.tolist(),reference_weights=ref.weights_.tolist(),
                local_best_loss=binary_loss(y,blend(z,w)),reference_best_loss=binary_loss(y,blend(z,ref.weights_)),
                rounding_disabled_weight_max_abs_error=diagnostic_gap,
                status='MATCH' if weights_gap<1e-12 else 'DIFFERENT_TIE_OR_ROUNDING_POLICY'))
    r=dict(cases=cases,scope='Actual corrected OOF libraries; pinned selector classes with local loss adapter; rounding/tie divergence preserved; separate modified-reference diagnostic disables only score rounding',
        source_sha256=hashlib.sha256((ROOT/'sources/l057/ensemble_selection.py').read_bytes()).hexdigest())
    (ROOT/'_source_check_l057_v2_results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
