"""Hand oracles and interventions; no implementation imports until execution."""
import numpy as np

def check_functions(linear_readout,fit_scale,class_values):
    # phi(0)=1, phi(log(2))=1+log(2), phi(-log(2))=.5.
    # q=0, k=(0,1), v=(2,8) -> (2+16)/(1+2+eps).
    np.testing.assert_allclose(linear_readout([[0]],[[0],[1]],[[2],[8]]),[[18/(3+1e-6)]],rtol=1e-13)
    np.testing.assert_allclose(linear_readout([[0]],[[-np.log(2)],[0]],[[2],[8]],0),[[6]],rtol=1e-13)
    rng=np.random.default_rng(19);q=rng.normal(size=(5,4));k=rng.normal(size=(7,4));v=rng.normal(size=(7,3))
    # Scalar reference enumerates each support weight, independently of reassociation.
    phi=lambda x: float(np.exp(x)) if x<0 else float(x+1)
    ref=np.zeros((5,3))
    for i in range(5):
        weights=[sum(phi(q[i,d])*phi(k[j,d]) for d in range(4)) for j in range(7)]
        for c in range(3):ref[i,c]=sum(weights[j]*v[j,c] for j in range(7))/(sum(weights)+1e-6)
    out=linear_readout(q,k,v);np.testing.assert_allclose(out,ref,atol=1e-12)
    p=[5,0,3,2,6,1,4];np.testing.assert_allclose(linear_readout(q,k[p],v[p]),out,atol=1e-12)
    np.testing.assert_allclose(np.concatenate([linear_readout(q[:2],k,v),linear_readout(q[2:],k,v)]),out,atol=1e-12)
    mean,scale=fit_scale([[1,7],[3,7]]);np.testing.assert_equal(mean,[2,7]);np.testing.assert_equal(scale,[1,1])
    mean2,scale2=fit_scale([[1,7],[3,7]]);np.testing.assert_equal(mean,mean2)
    vals=class_values([20,10,20],[20,10],2);np.testing.assert_equal(vals,[[1,0],[0,1],[1,0]])
    invalid=[lambda:linear_readout([],[[0]],[[1]]),lambda:linear_readout([[0]],[[float('nan')]],[[1]]),lambda:linear_readout([[0]],[[0]],[[1]],-1),lambda:fit_scale([]),lambda:class_values([0,1,2],[0,1,2],2),lambda:class_values([5],[0,1],2),lambda:class_values([0],[0,0],2)]
    for f in invalid:
        try:f()
        except ValueError:pass
        else:raise AssertionError('Invalid input accepted')
    return {'status':'PASS','checks':'scalar oracle, signed inputs, support permutation, query batching, constant scale, class identity/capacity, invalid inputs'}

if __name__=='__main__':
    from relkit.scaling_b04a import linear_readout,fit_scale,class_values
    print(check_functions(linear_readout,fit_scale,class_values))
