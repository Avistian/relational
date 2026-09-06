"""Mathematical and information-boundary checks for L051."""
import importlib.util
import numpy as np
assert importlib.util.find_spec('relkit.bias_interventions'), 'L051 intervention implementation is missing'
from relkit.bias_interventions import smooth_targets, rotate_splits, add_noise_features, paired_effect

def check():
    x=np.array([[0.],[1.],[2.]])
    y=np.array([0,1,0])
    q,hard=smooth_targets(x,y,1.,np.eye(1))
    expected=np.exp(-.5)/(1+2*np.exp(-.5))
    assert np.isclose(q[1],1/(1+2*np.exp(-.5)))
    assert np.array_equal(hard,[0,0,0]), 'Classification must threshold strictly above .5'
    assert np.array_equal(smooth_targets(x,y,0.,np.eye(1))[1],y)
    assert np.allclose(smooth_targets(x,np.ones(3),.7,np.eye(1))[0],1)
    q2,_=smooth_targets(x*3,y,1.,np.eye(1)*9)
    assert np.allclose(q,q2), 'Covariance metric must compensate a change of units'
    r=np.array([[0.,-1.],[1.,0.]])
    a=np.array([[1.,2.],[-3.,4.]])
    rotated=rotate_splits([a,a+2],r)
    assert np.allclose(rotated[0]@r.T,a)
    assert np.allclose(np.linalg.norm(rotated[0][0]-rotated[0][1]),np.linalg.norm(a[0]-a[1]))
    noisy=add_noise_features([a,a+2],4,7)
    assert np.array_equal(noisy[0][:,:2],a) and noisy[0].shape==(2,6)
    assert np.array_equal(noisy[0],add_noise_features([a,a+2],4,7)[0])
    assert not np.array_equal(noisy[0][:,2:],noisy[1][:,2:]), 'Split noise must not be duplicated'
    s=paired_effect([.8,.7,.9],[.7,.7,.8])
    assert np.isclose(s['mean'],2/30) and s['sd']>0
    assert paired_effect([.8],[.7])['ci95'] is None
    print('PASS: smoothing, covariance units, rotation, noise independence and paired uncertainty')

if __name__=='__main__':check()
