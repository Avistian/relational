"""Behavioral contracts for the visible scaling lab; run before implementation."""
def check169(sample_fn, curve_fn, claim_fn):
    import hashlib
    import numpy as np
    for seed in range(10):
        token=f'rel-f1-dfs-2:driver-dnf:{seed}'
        expected=np.random.default_rng(int.from_bytes(hashlib.sha256(token.encode()).digest()[:4],'big')).choice(1500,128,replace=False)
        np.testing.assert_array_equal(sample_fn(1500,128,token),expected)
    for args in [(10,11,'x'),(10,0,'x'),(10,1.5,'x'),(10,True,'x')]:
        try:sample_fn(*args)
        except ValueError:pass
        else:raise AssertionError('Invalid context accepted')
    rows=[dict(context=k,seed=s,auc=.5+.01*i+.001*s) for i,k in enumerate([64,128,256,512,1024]) for s in range(10)]
    c=curve_fn(rows)
    np.testing.assert_allclose([r['mean_gain'] for r in c['doublings']],[.01]*4,atol=1e-14)
    assert all(r['positive_seeds']==10 for r in c['doublings'])
    assert c['levels'][0]['sample_sd']>0 and c['levels'][0]['mean']==np.mean([r['auc'] for r in rows[:10]])
    assert curve_fn(list(reversed(rows)))==c
    for bad in [rows[:-1],rows+[rows[0]],rows[:-1]+[dict(context=1024,seed=9,auc=float('nan'))],rows[:-1]+[dict(context=1024,seed=9,auc=2)]]:
        try:curve_fn(bad)
        except ValueError:pass
        else:raise AssertionError('Incomplete/invalid curve accepted')
    assert claim_fn('context',True,5,False)=='CONTEXT_RESPONSE_ONLY'
    assert claim_fn('parameters',False,5,False)=='CONFOUNDED'
    assert claim_fn('parameters',True,5,False)=='CONTROLLED_SWEEP_NOT_LAW'
    assert claim_fn('context',True,1,False)=='INSUFFICIENT_LEVELS'
    assert claim_fn('context',True,5,True)=='EXTRAPOLATION_NOT_ESTABLISHED'
    try:claim_fn('unknown',True,5,False)
    except ValueError:pass
    else:raise AssertionError('Unknown scaling axis accepted')
    return 'PASS'

if __name__=='__main__':
    from pathlib import Path
    assert (Path(__file__).parent/'relkit/scaling_l169.py').exists(),'Scaling contracts not implemented yet'
    from relkit.scaling_l169 import sample_context,scaling_curve,scaling_claim
    print(check169(sample_context,scaling_curve,scaling_claim))
