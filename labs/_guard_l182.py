"""Reserve before dispatch; unchanged sources and aggregate budget are mandatory."""
import fcntl,hashlib,json
from pathlib import Path
RATE=.00028372

def reserve(path,root,phase,seconds):
    with Path(path).open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f)
        if not phase or phase in [r['phase'] for r in b['reservations']]:raise ValueError('Duplicate/empty attempt')
        if type(seconds) is not int or seconds<=0:raise ValueError('Positive integer timeout required')
        for name,h in b['source_hashes'].items():
            if hashlib.sha256((Path(root)/name).read_bytes()).hexdigest()!=h:raise ValueError('Changed source '+name)
        billable=seconds+30;amount=billable*RATE
        if sum(r['billable_seconds_reserved'] for r in b['reservations'])+billable>b['max_worker_seconds']:raise ValueError('Aggregate runtime gate')
        if b['overhead_reserve_usd']+sum(r['upper_usd'] for r in b['reservations'])+amount>min(b['cap_usd'],b['planned_stop_usd']):raise ValueError('Aggregate spend gate')
        b['reservations'].append(dict(phase=phase,seconds=seconds,billable_seconds_reserved=billable,upper_usd=amount))
        f.seek(0);json.dump(b,f,indent=2);f.truncate()
    return amount
