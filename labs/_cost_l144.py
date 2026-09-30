"""Pilot-based aggregate projection, with no silent tuning reduction."""
from pathlib import Path
import json,math
P=Path(__file__).resolve().parent;E=P/'evidence/l144';budget=json.loads((P/'_budget_l144.json').read_text());rate=.00026124;projections=[]
for arm in ['contextgnn','shallowrhsgnn']:
 paths=list(E.glob('pilot-'+arm+'*/result.json'));assert paths,'Missing completed timing pilot'
 r=json.loads(paths[-1].read_text());h=r['history'][0];bs=r['cfg']['batch_size'];full=r['scope']['full_query_counts']
 train_batches=min(math.ceil(full['train']/bs),2001);val_batches=math.ceil(full['val']/bs);test_batches=math.ceil(full['test']/bs)
 train_per_batch=h['train_seconds']/h['steps'];val_per_batch=h['validation_seconds']/4
 epoch_seconds=train_batches*train_per_batch+val_batches*val_per_batch
 projections.append({'arm':arm,'train_seconds_per_batch':train_per_batch,'validation_seconds_per_batch':val_per_batch,'released_epoch_train_batches':train_batches,'full_validation_batches':val_batches,'estimated_train_validation_epoch_seconds':epoch_seconds,'one_epoch_each_of_55_fits_usd':55*epoch_seconds*rate,'twenty_epochs_each_of_55_fits_usd':55*20*epoch_seconds*rate,'test_evaluation_excluded_from_projection':True})
minimum=sum(r['one_epoch_each_of_55_fits_usd'] for r in projections);maximum=sum(r['twenty_epochs_each_of_55_fits_usd'] for r in projections);reserved=sum(x['upper_usd'] for x in budget['reservations'])+budget['overhead_reserve_usd']
# This is a pilot-configuration scenario, NOT a lower bound across heterogeneous trials.
decision='STOP' if maximum+reserved>10 else 'PROCEED'
explanation=f'Timing-based scenario at the pilot configuration: both 50+5 arms cost about USD{minimum:.2f} even at one train/validation epoch per fit, or USD{maximum:.2f} at twenty; test passes and heterogeneous search widths can change costs. Reserved allocations plus overhead: USD{reserved:.2f}. These are projections, not a guaranteed lower bound or an invoice. Full search is not dispatched when its projected cost exceeds USD10.'
report={'decision':decision,'explanation':explanation,'projection_assumption':'every fit has pilot configuration throughput; pruning and different widths can alter runtime','one_epoch_scenario_usd':minimum,'twenty_epoch_scenario_usd':maximum,'reserved_plus_overhead_usd':reserved,'arms':projections,'full_selected_reproduction':'INCOMPLETE' if decision=='STOP' else 'NOT_RUN'}
(E/'cost-decision.json').write_text(json.dumps(report,indent=2));print(report)
