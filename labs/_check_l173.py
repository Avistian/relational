"""Behavioral contracts, including interventions and uneven task populations."""
import numpy as np
import torch

def check173(erase_target, task_weights, select_epoch):
    context=np.array([[1,2,3,0],[2,2,0,4]])
    actual=erase_target(context,np.array([2,4]))
    assert np.array_equal(actual,[[1,0,3,0],[2,2,0,0]])
    assert np.array_equal(context,[[1,2,3,0],[2,2,0,4]]),'Do not mutate context'
    for ids in [np.array([1]),np.array([-1,2])]:
        try: erase_target(context,ids)
        except ValueError: pass
        else: raise AssertionError('Invalid target identity accepted')
    counts=torch.tensor([9.,1.]); tasks=torch.tensor([0]*9+[1]); losses=torch.tensor([1.]*9+[5.])
    assert torch.allclose((losses*task_weights(tasks,counts,'cell')).mean(),torch.tensor(1.4))
    assert torch.allclose((losses*task_weights(tasks,counts,'task')).mean(),torch.tensor(3.))
    # The same example keeps its weight when a minibatch lacks the other task.
    assert task_weights(torch.tensor([1]),counts,'task').item()==5
    for counts_bad in [torch.tensor([0.,1.]),torch.tensor([1.,float('nan')])]:
        try: task_weights(tasks,counts_bad,'task')
        except ValueError: pass
        else: raise AssertionError('Invalid training population accepted')
    assert select_epoch([3.,2.,2.])==1,'Earliest validation minimum'
    for scores in [[],[1.,float('nan')]]:
        try: select_epoch(scores)
        except ValueError: pass
        else: raise AssertionError('Invalid validation result accepted')
    return 'PASS'

if __name__=='__main__':
    from relkit.multitask_l173 import erase_target,task_weights,select_epoch
    print(check173(erase_target,task_weights,select_epoch))
