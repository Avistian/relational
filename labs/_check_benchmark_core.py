import numpy as np
from relkit.benchmark_core import choose_validation, paired_summary, null_search, validate_partitions

def checks():
    assert choose_validation([.4,.2,.3])==1
    for bad in ([.2,np.nan],[],[np.inf]):
        try:choose_validation(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid selection accepted')
    records=[dict(dataset=d,arm=a,seed=s,error=e,seconds=1.) for d,es in [('a',[1.,2.,3.]),('b',[3.,2.,1.])]
             for a,e in zip(['A','B','C'],es) for s in range(3)]
    r=paired_summary(records)
    assert all(v==2. for v in r['mean_ranks'].values())
    try:paired_summary(records[:-1])
    except ValueError:pass
    else:raise AssertionError('Missing seed silently dropped')
    try:validate_partitions({'train':[0,1],'val':[2],'test':[1,3]})
    except ValueError:pass
    else:raise AssertionError('Overlapping rows accepted')
    a=null_search(1,80,1000,seed=59);b=null_search(256,80,1000,seed=59)
    assert b['validation_error']<=a['validation_error']
    assert abs(b['test_error']-.5)<.07
    print('PASS: validation-only selection, complete paired blocks, partition identity, independent null test')

if __name__=='__main__':checks()
