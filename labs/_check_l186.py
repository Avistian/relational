"""Behavioral contracts, including exact boundary and unavailable dependency cases."""
def checks(schedule, source_age, freshness_alert):
    assert schedule([0, 2, 30], 10) == [(0,10),(10,20),(30,40)]
    assert source_age(100, [80, 90]) == 20
    assert source_age(100, [80, None]) is None
    assert not freshness_alert([True]*99, 100, .1)
    assert not freshness_alert([True]*10+[False]*90, 100, .1)
    assert freshness_alert([True]*11+[False]*89, 100, .1)
    assert not freshness_alert([True]*100+[False]*100, 100, .1)
    for fn,args in [(schedule,([2,1],1)),(schedule,([0],0)),(source_age,(10,[11])),(source_age,(10,[])),(freshness_alert,([True],0,.1))]:
        try: fn(*args)
        except (ValueError,TypeError): pass
        else: raise AssertionError('Invalid input accepted')
    return 'PASS'

if __name__=='__main__':
    from relkit.serving_l186 import schedule, source_age, freshness_alert
    print(checks(schedule, source_age, freshness_alert))
