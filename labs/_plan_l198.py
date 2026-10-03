"""Freeze authored proposal specifications; these are not model execution configs."""
import json
from pathlib import Path
E=Path(__file__).resolve().parent/'evidence/l198'
cases=json.loads((E/'packet/evidence/l189/packet/cases.json').read_text())
tasks=['rel-f1/driver-dnf','rel-trial/study-outcome'];seeds=[0,1,2];draws=[0,1,2]
common=['Exact source/data hashes and target reconstruction','Full (entity_id, cutoff) train/validation/test manifests','One validation-only selection rule and fixed search budget','Per-task uncertainty resampling unit and coverage rationale','Timed all-in cost bound including retries and validation']
plans={
 'temporal':{
  'short_title':'Temporal-policy sensitivity',
  'matrices':{'model_evaluations':{'task':tasks,'model':['RDB-PFN','flattened-tree'],'policy':['timestamp','availability'],'seed':seeds}},
  'count_note':'24 model/policy/task/seed evaluations, each on complete validation and test populations. RDB-PFN is checkpoint inference; tree arms require fitting. Seed pairs share a frozen support schedule; all models use the same eligible labeled information within each policy.',
  'delta':'The new contribution would be a measured change in matched model contrasts under declared availability policies, not the discovery of temporal filtering or a cache repair.',
  'after_l197':'L197 provides a complete saved evidence inventory; it contains no results for these two availability policies. L196 shows why a source diagnostic alone cannot estimate accuracy harm.',
  'hypothesis_refined':'On each declared task, the absolute policy-induced change in the RDB-PFN minus tree AUROC contrast exceeds 0.01.',
  'falsifier_refined':'A justified paired interval strictly inside (-0.01, 0.01) rules out the proposed material change for that task. Entirely beyond either margin supports sensitivity. All boundary-crossing or boundary-touching intervals remain inconclusive.',
  'unresolved':common+['Availability lineage for every feature and support label; historical arrival times may be unavailable','Exact tree config, PFN checkpoint and shared label-budget/support policy','Stop if paired eligible supports cannot be maintained; do not silently drop queries'],
  'illustration':{'scores':{'a':.70,'b':.74,'c':.71,'d':.72},'mode':'interaction','interval':[-.04,-.02],'decision_mode':'sensitivity'},
  'arms':'A = timestamp/tree; B = timestamp/PFN; C = availability/tree; D = availability/PFN.',
  'next_artifact':'Availability ledger for all query and feature identities, distinguishing known, assumed and unknown observation times.',
 },
 'composite':{
  'short_title':'Prior × encoder interaction',
  'matrices':{
   'pretraining_checkpoints':{'prior':['single-table','relational'],'encoder':['conventional','composite'],'seed':seeds},
   'primary_prediction_batches':{'prior':['single-table','relational'],'encoder':['conventional','composite'],'seed':seeds,'task':tasks,'support_draw':draws},
   'released_prediction_batches':{'model':['released-DFS-RDB-PFN'],'task':tasks,'support_draw':draws},
   'tree_fit_prediction_batches':{'model':['flattened-tree'],'task':tasks,'seed':seeds,'support_draw':draws}},
  'count_note':'12 new checkpoints; 24 checkpoint/task combinations × 3 support draws = 72 primary batches. Also 6 frozen-checkpoint comparator batches and 18 tree fits/batches. Each batch scores the complete declared validation and test populations. Counts name different units and must not be summed as identical training jobs.',
  'delta':'Isolate the predictor-side composite operator and its interaction with a relational prior. RelGNN already supplies the operator, and RDB-PFN already supplies relational synthetic pretraining.',
  'after_l197':'L197 replays a small released-system advantage on study-outcome; that is neither a trained hybrid nor evidence that changing the encoder causes the advantage.',
  'hypothesis_refined':'The composite encoder gives a useful (>0.01 AUROC) conditional gain under the relational prior, and its gain exceeds its gain under the single-table prior.',
  'falsifier_refined':'For useful benefit, an upper interval endpoint below 0.01 rules out the proposed gain; a lower endpoint above 0.01 supports it. Separately test synergy against zero: an upper interaction endpoint ≤0 fails the positive-interaction claim. Uncertain or boundary-touching benefit intervals remain inconclusive.',
  'unresolved':common+['Trainable graph/head interface, finite gradients and compatible initialization','Exact generator/source exclusions, task curriculum, schedules and parameter/compute matching','Meaningful same-shape control for single-table prior; do not change both graph access and encoder capacity','Separate task/initialization uncertainty from repeated support draws'],
  'illustration':{'scores':{'a':.70,'b':.74,'c':.72,'d':.73},'mode':'conditional','interval':[-.005,.025],'decision_mode':'benefit'},
  'arms':'A = flat prior/conventional; B = flat prior/composite; C = relational prior/conventional; D = relational prior/composite.',
  'next_artifact':'Written tensor/interface contract and an explicitly proposed finite-gradient probe, followed by a complete training cost forecast. The probe itself is not run in L198.',
 },
 'transfer':{
  'short_title':'Compute-matched transfer',
  'matrices':{
   'source_pretraining_fits':{'holdout':['rel-f1','rel-trial'],'objective':['historical','future'],'seed':seeds},
   'target_fits':{'task':tasks,'arm':['scratch','extra-compute-scratch','historical-pretrained','future-pretrained'],'seed':seeds}},
  'count_note':'12 source pretraining fits plus 24 target fits. Each target fit evaluates its complete validation/test task. All source preprocessing excludes the entire target database; objectives cannot borrow target schema statistics.',
  'delta':'Evaluate temporal objectives under whole-database exclusion and matched end-to-end cost. Cross-database transfer and temporal pretraining each already exist.',
  'after_l197':'A supervised or published leaderboard advantage cannot isolate a pretraining benefit. Extra-compute scratch is the decisive comparator; source plus target cost belongs in the comparison.',
  'hypothesis_refined':'Each temporal objective improves the held-out task by more than 0.01 AUROC over extra-compute scratch at the same all-in budget.',
  'falsifier_refined':'For each objective/task, an upper paired interval endpoint below 0.01 rules out the useful gain; a lower endpoint above 0.01 supports it. Otherwise inconclusive. Database contamination or unmatched costs invalidate interpretation before statistical scoring.',
  'unresolved':common+['Explicit source database corpus, target holdouts and preprocessing exclusion audit','Exact shared backbone, adapters, objectives and learning schedules','Dollar and runtime matching policy, hardware prices and amortization convention','Multiplicity policy for two objective claims on each task'],
  'illustration':{'scores':{'a':.73,'b':.74},'mode':'gain','interval':[-.005,.025],'decision_mode':'benefit'},
  'arms':'A = extra-compute scratch; B = one temporal-pretrained arm. Ordinary scratch (illustrative 0.70) is a secondary control.',
  'next_artifact':'Database exclusion manifest and separate source/target cost envelope. A same-database result cannot substitute.',
 }}
for c in cases:
 c.update(plans[c['id']]);c['unresolved_execution_fields']=c.pop('unresolved')
 c['margin_auroc']=.01;c['protocol_state']='PROPOSAL_NOT_EXECUTION_READY';c['learner']='PENDING_WRITTEN_DEFENSE'
 c['evidence_policy']='Inherited receipt authenticates a source observation only; L197 replay supports its original scoped comparisons, not this new hypothesis.'
(E/'proposals.json').write_text(json.dumps(cases,indent=2)+'\n')
print('Prepared three proposal contracts with explicit unknowns')
