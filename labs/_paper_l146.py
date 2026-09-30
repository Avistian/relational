"""Reconstruct the *reported* F1 table under explicit alternative selection rules."""
import json,hashlib
from pathlib import Path
from relkit.comparison_l146 import select_config
P=Path(__file__).resolve().parent;E=P/'evidence/l146';E.mkdir(parents=True,exist_ok=True)
configs=[f'L{l}-d{d}' for l in [1,4,8] for d in [.3,.4,.5]]
validation=[3.1897,3.1817,3.3257,3.1046,3.3352,3.1276,3.1589,3.2907,3.1843]
test=[4.942,5.6431,3.917,4.6316,4.0851,4.0042,5.5273,5.5569,4.6085]
v=select_config(configs,validation);t=configs[min(range(9),key=lambda i:test[i])]
r=dict(source='https://arxiv.org/html/2505.10960v1#A4',table=6,configs=configs,validation=validation,test=test,
       selected_by_displayed_validation=v,its_displayed_test=test[configs.index(v)],
       minimum_displayed_test_config=t,minimum_displayed_test=min(test),headline_test=3.917,rdl_test=4.022,
       validation_selected_relative_gain_percent=(4.022-test[configs.index(v)])/4.022*100,
       headline_relative_gain_percent=(4.022-3.917)/4.022*100,
       finding='Headline matches smallest displayed test; not the smallest displayed validation. Historical selection intent NOT_ESTABLISHED.',
       caveat='Printed rounded scores and stochastic final reevaluation cannot reconstruct the historical selection-time metrics.')
(E/'paper-table.json').write_text(json.dumps(r,indent=2))
files=['sources/l145/'+p.name for p in (P/'sources/l145').iterdir() if p.is_file()]
files+=['relkit/rdl_l117.py','_run_l117.py','_full_l145.py','relkit/relgt_l145.py']
ledger=dict(paper='https://arxiv.org/html/2505.10960v1',release='https://github.com/snap-stanford/relgt/tree/19e423ca3e7cac761130aba790857f2dc3a46ef7',
            reused_files={f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in files},
            released_checkpoint_selection='last tied minimum within fit',released_cross_config_selection='NOT_ESTABLISHED',
            course_protocol='docs/plans/2026-09-30-lesson-146-design.md',
            full_selected_reproduction='INCOMPLETE',full_relgt_search='NOT_RUN',fresh_canonical_rdl='NOT_RUN',
            reasons=['L145 full source tokens fail temporal ownership','L145 pilot projects USD80.41 for nine shallow-speed 100-epoch fits before overhead','Printed-table cross-config selection discrepancy unresolved'])
(E/'sources.json').write_text(json.dumps(ledger,indent=2));print(r)
