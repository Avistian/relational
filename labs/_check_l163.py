"""Behavioral checks shared by local code and the standalone notebooks."""
def check163(serialize, pool, choose, typed):
    import json, numpy as np
    row=dict(id='never_encode_001',price_usd=10.0,weight_kg=2.5,colour='red',condition='used',target=999999)
    s=serialize(row)
    assert 'never_encode' not in s and 'target' not in s and '999999' not in s
    assert json.loads(s)==dict(table='products',price_usd=10.0,weight_kg=2.5,colour='red',condition='used')
    assert list(json.loads(serialize(row,'reordered')))==['table','condition','colour','weight_kg','price_usd']
    assert json.loads(serialize(row,'renamed'))==dict(table='products',c0=10.0,c1=2.5,c2='red',c3='used')
    weird=dict(row,colour='red"; target=9\n雪');assert json.loads(serialize(weird))['colour']==weird['colour']
    assert row['target']==999999
    h=np.array([[[1.,3.],[3.,7.],[900.,900.]],[[8.,2.],[0.,0.],[0.,0.]]])
    m=np.array([[1,1,0],[1,0,0]])
    np.testing.assert_allclose(pool(h,m),[[2,5],[8,2]])
    np.testing.assert_allclose(pool(h[:,:2],m[:,:2]),[[2,5],[8,2]])
    assert choose([.01,.1,1],[4,2,3])==.1
    assert choose([.01,.1,1],[2,2,3])==.01
    a=dict(row,price_usd=0.,weight_kg=1.,colour='blue',condition='new')
    b=dict(row,price_usd=10.,weight_kg=3.)
    # The large held-out value must never refit training moments or vocabularies.
    x,meta=typed([a,b],[a,b,dict(row,price_usd=100.,weight_kg=5.,colour='unseen')])
    np.testing.assert_allclose(x[:,:2],[[-1,-1],[1,1],[19,3]])
    assert meta['categories']=={'colour':['blue','red'],'condition':['new','used']}
    np.testing.assert_array_equal(x[2,2:4],[0,0])
    assert typed([a,b],[a,b])[1]==meta
    for operation in [lambda:serialize(row,'oops'),lambda:serialize(dict(row,price_usd=float('nan'))),lambda:serialize({}),lambda:pool(h,np.zeros((2,3))),lambda:pool(h,[[1,2,0],[1,0,0]]),lambda:choose([1],[float('nan')]),lambda:choose([],[]),lambda:choose([1,2],[1]),lambda:typed([],[]),lambda:pool(h,[[1,1]])]:
        try:operation()
        except ValueError:pass
        else:raise AssertionError('Invalid input accepted')
    return 'PASS'

if __name__=='__main__':
    from relkit.rows_l163 import serialize_row,masked_mean,choose_alpha,typed_features
    print(check163(serialize_row,masked_mean,choose_alpha,typed_features))
