"""Separate L053 measurement: remove only smooth clipping from numeric TD-S.

The historical comparison and larger-run operators are unchanged. This measures
the fixed-recipe intervention from paper Appendix B.2 on our local data protocol.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from relkit.realmlp import RealMLPS, RobustSmooth
from relkit.realmlp_experiment import run_suite, seed_interval

ROOT = Path(__file__).resolve().parent


class RobustOnly(RobustSmooth):
    """Same fitted medians/scales, including fallbacks; omit only smooth_clip."""
    def transform(self, x):
        return ((np.asarray(x) - self.median) * self.scale).astype('float32')


def paired_effect(baseline, intervention):
    """Positive difference means that removing clipping raised the test error."""
    by_seed = {r['seed']: r['error'] for r in baseline}
    assert len(by_seed) == len(baseline), 'Duplicate baseline seed'
    assert len({r['seed'] for r in intervention}) == len(intervention), 'Duplicate intervention seed'
    assert set(by_seed) == {r['seed'] for r in intervention}, 'Pair identical model seeds'
    differences = [r['error'] - by_seed[r['seed']] for r in intervention]
    return dict(differences=differences, **seed_interval(differences))


def measure():
    baseline = run_suite(RealMLPS, neural_only=True, data_root=ROOT/'data/cache/l052')
    changed = run_suite(RealMLPS, neural_only=True, prep_class=RobustOnly,
                        data_root=ROOT/'data/cache/l052')
    historical = json.loads((ROOT/'_verify_l053_results.json').read_text())
    paired = {}
    for name, data in baseline['results'].items():
        a = data['runs']['RealMLP-S']; b = changed['results'][name]['runs']['RealMLP-S']
        assert data['selection'] == changed['results'][name]['selection']
        assert data['hashes'] == changed['results'][name]['hashes']
        for new, old in zip(a, historical['results'][name]['runs']['RealMLP-S']):
            np.testing.assert_allclose(new['prediction'], old['prediction'], rtol=0, atol=0)
        paired[name] = dict(metric=data['metric'], **paired_effect(a, b))
    result = dict(status='MEASURED', paper_verdict='INCOMPARABLE',
        question='Fixed-recipe effect of removing smooth clipping; no learning-rate retuning',
        baseline=baseline, robust_only=changed, paired=paired,
        baseline_matches_historical_predictions=True,
        code_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                     ['_ablation_l053.py','relkit/realmlp.py','relkit/realmlp_experiment.py']},
        paper_source='https://arxiv.org/html/2407.04491v3#A2.SS2',
        deviations=['Three fixed capped TabR splits, not RealMLP meta-train benchmarks',
                    'Three model seeds, not ten random 60/20/20 splits',
                    'Width 64 and 64 epochs, not 256 and 256',
                    'Original-unit errors and paired seed intervals, not benchmark SGM'])
    (ROOT/'_ablation_l053_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(paired,indent=2))
    return result


if __name__ == '__main__':
    measure()
