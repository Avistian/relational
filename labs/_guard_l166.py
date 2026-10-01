"""Reserve every attempt before remote dispatch; no implicit retries."""
import fcntl,hashlib,json
from pathlib import Path
RATE=.00028372

def reserve(path,root,phase,seconds):
    with Path(path).open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f)
        if phase in [x['phase'] for x in b['reservations']]:raise ValueError('Duplicate attempt')
        if type(seconds) is not int or seconds<=0:raise ValueError('Invalid timeout')
        for name,sha in b['source_hashes'].items():
            if hashlib.sha256((Path(root)/name).read_bytes()).hexdigest()!=sha:raise ValueError('Changed source '+name)
        amount=(seconds+30)*RATE
        if b['overhead_reserve_usd']+sum(r['upper_usd'] for r in b['reservations'])+amount>b['cap_usd']:raise ValueError('Aggregate budget gate')
        b['reservations'].append(dict(phase=phase,seconds=seconds,billable_seconds_reserved=seconds+30,upper_usd=amount))
        f.seek(0);json.dump(b,f,indent=2);f.truncate()
    return amount
