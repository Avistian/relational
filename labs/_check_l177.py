"""Live learner contracts: units, missing evidence, failed attempts and stopping."""
def checks(reservation_total, forecast_seconds, assess_plan):
    import math
    def reject(fn, *args):
        try: fn(*args)
        except (ValueError, TypeError): return
        raise AssertionError('Invalid evidence accepted')
    attempts=[dict(id='failed',seconds=900,lifecycle_seconds=30,rate=.00006172),
              dict(id='success',seconds=900,lifecycle_seconds=30,rate=.00006172)]
    assert math.isclose(reservation_total(attempts,3),3.1147992,abs_tol=1e-12)
    assert reservation_total([],0)==0
    reject(reservation_total,attempts+[attempts[0]],3)
    for bad in [-1,float('nan'),float('inf'),None,True]:
        reject(reservation_total,[dict(id='x',seconds=bad,lifecycle_seconds=0,rate=1)],0)
    assert forecast_seconds([1,2,8],10,3,5)==245
    assert forecast_seconds([1],0,1,5)==5
    for args in [([],2,3,0),([1],-1,3,0),([1],2.5,3,0),([1],2,.9,0),([float('nan')],2,3,0)]:
        reject(forecast_seconds,*args)
    assert assess_plan(5,600,1200,8,24,3600)['status']=='FEASIBLE_SCENARIO'
    assert assess_plan(5,3000,1200,8,24,3600)['status']=='EXCEEDS_SESSION'
    assert assess_plan(5,3000,1200,8,24,10800)['status']=='FEASIBLE_SCENARIO'
    assert assess_plan(9,300,200,8,24,3600)['status']=='OVER_BUDGET'
    assert assess_plan(5,300,200,25,24,3600)['status']=='MEMORY_LIMIT'
    assert assess_plan(5,300,None,8,24,3600)['status']=='INCOMPLETE_MEASUREMENT'
    assert assess_plan(5,300,200,None,24,3600)['status']=='INCOMPLETE_MEASUREMENT'
    # A known violation suffices to reject, even with missing measurements.
    assert assess_plan(9,None,None,None,24,3600)['status']=='OVER_BUDGET'
    assert assess_plan(5,300,200,8,24,3600,False)['status']=='SCIENTIFIC_STOP'
    assert assess_plan(8,3000,600,24,24,3600)['status']=='FEASIBLE_SCENARIO'
    reject(assess_plan,-1,300,200,8,24,3600)
    return 'PASS'

if __name__=='__main__':
    from relkit.compute_l177 import reservation_total,forecast_seconds,assess_plan
    print(checks(reservation_total,forecast_seconds,assess_plan))
