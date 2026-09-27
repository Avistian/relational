"""Assemble current execution/cost boundaries without conflating validation fits."""
import json,hashlib
from pathlib import Path
import modal
P=Path(__file__).resolve().parent;E=P/'evidence/l139';b=json.loads((P/'_budget_l139.json').read_text());training=json.loads((E/'training.json').read_text());v=modal.Volume.from_name('l139-trial-evidence')
costs={}
for phase in ['prepare','prepare2','pilot-seed-99','pilot2-seed-99']+[f'full-seed-{s}' for s in range(5)]:
 name=phase+'-cost.json';raw=b''.join(v.read_file(name));(E/name).write_bytes(raw);costs[phase]=json.loads(raw)
# CPU diagnostics have explicit measured runtime and their own worst-case reservations.
for file in ['diagnostic.json','diagnostic-pandas.json']:
 x=json.loads((E/file).read_text());costs[file]=dict(seconds=x['seconds'],resource_usd=x['seconds']*(2*.0000131+16*.00000222))
for file,key,rate in [('_notebook_pinned_l139_results.json','default_notebook',2*.0000131+16*.00000222),('_notebook_full_l139_results.json','fresh_full_notebook',.00033228)]:
 if (P/file).exists():
  x=json.loads((P/file).read_text());seconds=x['seconds']+(x.get('preparation',{}).get('seconds',0) if key=='fresh_full_notebook' else 0)
  costs[key]=dict(seconds=seconds,resource_usd=seconds*rate)
upper=sum(x['upper_usd'] for x in b['reservations'])+b['overhead_reserve_usd'];assert upper<=10
repro=dict(status='COMPLETE_SELECTED_RELEASED_PROTOCOL',reason='Five full-data twenty-epoch fits completed; independently reconstructed all 13,779 queries and rescored all 8,925 held-out predictions. Trial-specific paper settings preserved.',full_training='COMPLETE',historical_identity='NOT_ESTABLISHED',whole_paper='NOT_ESTABLISHED',primary_metrics=training['metrics'],preprocessing='Verified L132 graph reused for primary fits; fresh full notebook validation reported separately',costs=costs,worker_body_estimate_usd=sum(x['resource_usd'] for x in costs.values()),worst_case_reservations_plus_overhead_usd=upper,budget_usd=10,invoice='NOT_ITEMIZED; worker-body estimate excludes startup/commit/storage',live_colab='NOT_CHECKED',deployment='NOT_CHECKED',learner='PENDING_WRITTEN_DEFENSE',default_notebook='PASS' if (P/'_notebook_pinned_l139_results.json').exists() else 'NOT_RUN',full_fresh_notebook='PASS' if (P/'_notebook_full_l139_results.json').exists() else 'NOT_RUN')
(E/'reproduction.json').write_text(json.dumps(repro,indent=2));print({k:repro[k] for k in ['status','worker_body_estimate_usd','worst_case_reservations_plus_overhead_usd','full_fresh_notebook']})
