"""Read this session's actual allowance; stop new work with a delivery reserve."""
import os,json,sys
from pathlib import Path
sid=os.environ.get('CODEX_SESSION_ID') or os.environ.get('CODEX_THREAD_ID')
p=next(Path('/home/avist/.codex/sessions').rglob('*'+sid+'*.jsonl'))
latest=None
for line in p.open():
 e=json.loads(line)
 if e.get('type')=='event_msg' and e.get('payload',{}).get('type')=='token_count' and e['payload'].get('info'):
  latest=e
limits=latest['payload']['rate_limits'];primary=limits['primary'];remaining=100-primary['used_percent']
result={'timestamp':latest['timestamp'],'weekly_remaining_percent':remaining,'weekly_window_minutes':primary['window_minutes'],'total_token_usage':latest['payload']['info']['total_token_usage'],'credit_balance':limits['credits']['balance'],'stop_remaining_percent':5,'new_work_cutoff_remaining_percent':7}
out=Path(__file__).with_name('usage.jsonl')
with out.open('a') as f:f.write(json.dumps(result)+'\n')
print(json.dumps(result))
if '--guard' in sys.argv and remaining<=7:raise SystemExit('Reserve reached: finish and push current work, then stop at 5% remaining.')
