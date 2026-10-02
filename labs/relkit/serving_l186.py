"""Relational serving simulation. Hypothetical timings; no model inference."""
import bisect
import hashlib
import itertools
import json
import math
import random
from collections import deque

CONFIG = dict(requests=10000, entities=20, rates=[10,50,100], seeds=[0,1,2],
              conditions=['normal','delayed','interrupted'],
              policies={'precomputed':dict(service_ms=1,refresh_ms=5000),
                        'cached':dict(service_ms=5,refresh_ms=1000),
                        'request':dict(service_ms=15,refresh_ms=0)},
              refresh_work_ms=20, event_interval_ms=500, drain_margin_ms=200000,
              deadline_ms=100, freshness_ms=2000, window=100, threshold=.1,
              label_delay_ms=30500)


def schedule(arrivals, service_ms):
    """Single FIFO worker, no cancellation: return every (start, finish)."""
    if type(service_ms) is not int or service_ms <= 0:
        raise ValueError('Positive integer service duration required')
    prior = -1
    free = 0
    result = []
    for arrival in arrivals:
        if type(arrival) is not int or arrival < 0 or arrival < prior:
            raise ValueError('Nonnegative ordered integer arrivals required')
        start = max(arrival, free)
        free = start + service_ms
        result.append((start, free))
        prior = arrival
    return result


def source_age(response_ms, dependency_events):
    """Age of the oldest observed dependency snapshot; unknown stays unknown."""
    if not dependency_events or type(response_ms) is not int or response_ms < 0:
        raise ValueError('Response time and nonempty dependencies required')
    for event in dependency_events:
        if event is not None and (type(event) is not int or event > response_ms):
            raise ValueError('Invalid or future dependency')
    if any(event is None for event in dependency_events):
        return None
    return response_ms - min(dependency_events)


def freshness_alert(stale, window=100, threshold=.1):
    """Only a complete trailing response window may trigger an alert."""
    if type(window) is not int or window <= 0 or not 0 <= threshold <= 1:
        raise ValueError('Invalid monitor parameters')
    if any(type(value) is not bool for value in stale):
        raise TypeError('Boolean stale flags required')
    return len(stale) >= window and sum(list(stale)[-window:]) / window > threshold


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',',':'), sort_keys=True).encode()).hexdigest()


def make_trace(rate, condition, seed, config=CONFIG):
    """Same requests across all policies/conditions; same underlying events across policies."""
    rng = random.Random(seed)
    arrivals, customers = [], []
    clock = 0
    for _ in range(config['requests']):
        clock += max(1, round(rng.expovariate(rate / 1000)))
        arrivals.append(clock)
        customers.append(rng.randrange(config['entities']))
    onset, end = clock // 3, 2 * clock // 3
    rng = random.Random(seed + 1000)
    events = []
    for customer in range(config['entities']):
        for table in range(2):
            events.append((-4900, customer, table, -5000))
            offset = (customer * 17 + table * 71) % 101
            for event in range(offset, clock+config['drain_margin_ms'], config['event_interval_ms']):
                arrival = event + rng.randrange(101)
                if table == 1 and onset <= event < end:
                    if condition == 'delayed': arrival += 2000
                    if condition == 'interrupted': arrival = max(arrival, end)
                events.append((arrival, customer, table, event))
    events.sort()
    return dict(arrivals=arrivals, customers=customers, events=events,
                onset_ms=None if condition=='normal' else onset, end_ms=end,
                sha256=digest([arrivals,customers,events]))


def summarize(rows, trace, config=CONFIG):
    """Nearest-rank latency quantiles and separate operational/label coverage."""
    def quantile(values, q):
        values = sorted(values)
        return values[math.ceil(len(values)*q)-1]
    onset = trace['onset_ms']
    alerts = [r['finish_ms'] for r in rows if r['alert']]
    transitions = [r['finish_ms'] for i,r in enumerate(rows) if r['alert'] and (i==0 or not rows[i-1]['alert'])]
    post = [] if onset is None else [t for t in alerts if t>=onset]
    edges = [] if onset is None else [t for t in transitions if t>=onset]
    before = [] if onset is None else [r for r in rows if r['finish_ms']<onset]
    latency = [r['latency_ms'] for r in rows]
    ages = [r['source_age_ms'] for r in rows]
    assert all(a is not None for a in ages), 'Full fixture requires all dependencies'
    return dict(requests=len(rows),p50_ms=quantile(latency,.5),p95_ms=quantile(latency,.95),
                p99_ms=quantile(latency,.99),deadline_misses=sum(t>config['deadline_ms'] for t in latency),
                stale_responses=sum(r['stale'] for r in rows),source_age_p95_ms=quantile(ages,.95),
                material_age_p95_ms=quantile([r['material_age_ms'] for r in rows],.95),
                max_queue_wait_ms=max(r['start_ms']-r['arrival_ms'] for r in rows),
                first_alert_ms=alerts[0] if alerts else None, alert_episodes=len(transitions),
                preincident_alert_responses=sum(r['alert'] for r in before),
                active_at_onset=bool(before and before[-1]['alert']) if onset is not None else None,
                first_post_onset_alert_ms=post[0] if post else None,
                post_onset_alert_delay_ms=post[0]-onset if post else None,
                first_new_episode_delay_ms=edges[0]-onset if edges else None,
                labels_available_at_last_response=sum(a+config['label_delay_ms']<=rows[-1]['finish_ms'] for a in trace['arrivals']),
                performance_monitor='NOT_SCORED',trace_sha256=trace['sha256'],response_sha256=digest(rows))


def simulate(trace, policy, config=CONFIG):
    """Advance arrivals only to each legal read time, including completed refreshes."""
    settings = config['policies'][policy]
    times = schedule(trace['arrivals'], settings['service_ms'])
    state = [[None,None] for _ in range(config['entities'])]
    index = 0
    events = trace['events']
    def advance(read_ms):
        nonlocal index
        while index < len(events) and events[index][0] <= read_ms:
            arrival, customer, table, event = events[index]
            assert event <= arrival
            old = state[customer][table]
            state[customer][table] = event if old is None else max(old,event)
            index += 1
    refresh = settings['refresh_ms']
    # Bootstrap snapshot is ready before time zero.
    advance(-4900)
    cached = [s[:] for s in state]
    materialized = -4880
    next_tick = 0
    flags = deque(maxlen=config['window'])
    rows = []
    for request_id, ((start,finish), arrival, customer) in enumerate(zip(times, trace['arrivals'],trace['customers'])):
        if refresh:
            while next_tick + config['refresh_work_ms'] <= start:
                advance(next_tick)
                cached = [s[:] for s in state]
                materialized = next_tick + config['refresh_work_ms']
                next_tick += refresh
            deps = cached[customer][:]
            read_ms = materialized-config['refresh_work_ms']
        else:
            advance(start)
            deps = state[customer][:]
            materialized = start
            read_ms = start
        age = source_age(finish, deps)
        stale = age is None or age > config['freshness_ms']
        flags.append(stale)
        rows.append(dict(request_id=request_id,customer=customer,arrival_ms=arrival,
                         start_ms=start,finish_ms=finish,read_ms=read_ms,
                         dependency_events=deps,source_age_ms=age,material_age_ms=finish-materialized,
                         latency_ms=finish-arrival,stale=stale,
                         alert=freshness_alert(flags,config['window'],config['threshold'])))
    return rows


def run_grid(config=CONFIG):
    results, witness = [], []
    for rate, condition, seed in itertools.product(config['rates'],config['conditions'],config['seeds']):
        trace = make_trace(rate,condition,seed,config)
        for policy in config['policies']:
            rows = simulate(trace,policy,config)
            results.append(dict(policy=policy,rate=rate,condition=condition,seed=seed,**summarize(rows,trace,config)))
            if (rate,condition,seed)==(50,'interrupted',0):
                onset = trace['onset_ms']
                picked = [r for r in rows if onset+1800<=r['arrival_ms']<=onset+2200][:3]
                witness.extend(dict(policy=policy,**r) for r in picked)
    return dict(experiment='L186 Relational Serving Contract',status='COMPLETE_COURSE_SIMULATION',
                cells=len(results),requests=sum(r['requests'] for r in results),config=config,
                results=results,witness=witness,boundaries=dict(timings='HYPOTHETICAL',
                live_production='NOT_RUN',fresh_model_inference='NOT_RUN',paper_result_reproduction='NOT_RUN',
                historical_arrival_identity='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE'))
