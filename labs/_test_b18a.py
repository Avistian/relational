import unittest
import numpy as np
from relkit.state_b18a import eligible_support,state_key,replacement_contrast

class StateContract(unittest.TestCase):
    def test_temporal_labels(self):
        rows=[dict(id=1,event=1,available=2,label_available=5),dict(id=2,event=1,available=6,label_available=2),dict(id=3,event=7,available=1,label_available=1)]
        self.assertEqual([r['id'] for r in eligible_support(rows,5)], [1])
        with self.assertRaises(ValueError):eligible_support(rows+[rows[0]],10)
    def test_identity(self):
        x=np.array([[1.,np.nan],[2.,3.]])
        args=dict(ids=[8,9],x=x,y=[0,1],weights='abc',preprocessing={'scale':1},recipe={'seed':0})
        a=state_key(**args)
        self.assertEqual(a,state_key(**args))
        for key,val in [('ids',[9,8]),('y',[1,1]),('weights','def'),('preprocessing',{'scale':2}),('recipe',{'seed':1}),('x',np.nan_to_num(x))]:
            self.assertNotEqual(a,state_key(**dict(args,**{key:val})))
    def test_contrast(self):
        calls=[]
        def predict(x):calls.append(len(x));return x[:,0]+2*x[:,1]
        effect=replacement_contrast(predict,np.array([[4.,2.],[6.,1.]]),np.array([[1.,8.],[3.,9.]]),0)
        np.testing.assert_allclose(effect,[2,4]);self.assertEqual(sum(calls),6)

if __name__=='__main__':unittest.main()
