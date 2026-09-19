"""Independent fixtures for the load-bearing experimental design."""
import numpy as np
from relkit.ssl_regimes_l073 import nested_labels, paired_gains, crossing_brackets

def run():
    groups=nested_labels([8,2,7,9,5,4,1,0,3,6],[.2,.5,1.],73)
    assert list(map(len,groups))==[2,5,10]
    assert set(groups[0])<set(groups[1])<set(groups[2])
    assert groups==nested_labels([8,2,7,9,5,4,1,0,3,6],[.2,.5,1.],73)
    rows=[{'seed':2,'arm':'ssl','accuracy':.9},{'seed':1,'arm':'base','accuracy':.7},
          {'seed':1,'arm':'ssl','accuracy':.6},{'seed':2,'arm':'base','accuracy':.6}]
    assert np.allclose(paired_gains(rows,'ssl','base'),[-.1,.3])
    for bad in [rows[:-1],rows+[rows[0]]]:
        try:paired_gains(bad,'ssl','base')
        except ValueError:pass
        else:raise AssertionError('Missing/duplicate pair silently accepted')
    assert crossing_brackets([.1,.2,.4,1.],[.02,-.01,.03,-.01])==[[.1,.2],[.2,.4],[.4,1.]]
    assert crossing_brackets([.1,.2,1.],[.02,0.,-.01])==[]
    assert crossing_brackets([.1,.2,1.],[.02,.01,.01])==[]
    print('PASS: nesting, reproducibility, keyed pairing, missing/duplicate rejection, crossings and ties')
if __name__=='__main__':run()
