"""Behavioral contracts: access, query identity, factorial controls and transfer admission."""
def checks(temporal_mask, keyed_mae, factorial_effect, transfer_gate):
    import copy, math
    def reject(fn):
        try: fn()
        except ValueError: return
        raise AssertionError('Malformed evidence was accepted')
    base=dict(event_time=5,available_at=6,is_label=False,label_end=None,query_target=False)
    cells=[base,dict(base,event_time=11),dict(base,available_at=11),dict(base,is_label=True,label_end=11),dict(base,query_target=True),dict(base,available_at=None),dict(base,is_label=True,label_end=10),dict(base,event_time=None)]
    assert temporal_mask(cells,10)==[True,False,False,False,False,False,True,False], 'Time, arrival, label horizon and target masking are distinct'
    assert temporal_mask([dict(base,event_time=10,available_at=10)],10)==[True]
    assert temporal_mask([],10)==[]
    reject(lambda:temporal_mask([{}],10))
    reject(lambda:temporal_mask([dict(base,is_label='False')],10))
    reject(lambda:temporal_mask([dict(base,event_time=float('nan'))],10))
    reject(lambda:temporal_mask([dict(base,is_label=True,label_end=4)],10))
    truth=[(7,10,2.),(7,20,5.),(8,10,3.)]
    pred=[(8,10,4.),(7,20,3.),(7,10,2.)]
    assert keyed_mae(truth,pred)==1., 'Join on entity AND cutoff, independent of row order'
    for bad in [pred[:-1],pred+[pred[0]],[(9,10,4.),*pred[1:]],[(8,10,float('inf')),*pred[1:]]]:
        reject(lambda bad=bad:keyed_mae(truth,bad))
    reject(lambda:keyed_mae([],[]))
    reject(lambda:keyed_mae(truth+[truth[0]],pred))
    arms={'mp_scratch':{0:4.,1:4.1,2:3.9},'mp_pretrained':{0:3.6,1:3.8,2:3.4},'gt_scratch':{0:3.9,1:4.,2:3.8},'gt_pretrained':{0:3.2,1:3.4,2:3.0}}
    effect=factorial_effect(arms)
    assert abs(effect['interaction_mean']-.3)<1e-12
    assert all(abs(x-.3)<1e-12 for x in effect['interaction_per_seed'])
    assert abs(effect['mp_gain_mean']-.4)<1e-12 and abs(effect['gt_gain_mean']-.7)<1e-12
    opposite=factorial_effect(arms,True)
    assert abs(opposite['interaction_mean']+.3)<1e-12
    no_extra=copy.deepcopy(arms);no_extra['gt_pretrained']={s:arms['gt_scratch'][s]-(arms['mp_scratch'][s]-arms['mp_pretrained'][s]) for s in range(3)}
    assert abs(factorial_effect(no_extra)['interaction_mean'])<1e-12
    for bad in [{k:v for k,v in arms.items() if k!='mp_pretrained'},dict(arms,gt_pretrained={0:3.2,1:3.4}),dict(arms,gt_pretrained={0:float('nan'),1:3.4,2:3.0})]:
        reject(lambda bad=bad:factorial_effect(bad))
    reject(lambda:factorial_effect(arms,higher_is_better='no'))
    ready=dict(checkpoint_architecture='PASS',parameter_shapes='PASS',schema_semantics='PASS',task_decoder='PASS',temporal_access='PASS',database_holdout='PASS',finite_gradients='PASS',validation_selection='PASS',full_cost='PASS')
    assert transfer_gate(ready)=={'status':'READY_FOR_SEPARATELY_AUTHORIZED_PROBE','blockers':[],'transfer_gain':'NOT_ESTABLISHED'}
    for name in ready:
        e=ready.copy();e[name]='NOT_CHECKED'
        assert name.upper() in transfer_gate(e)['blockers'] and transfer_gate(e)['transfer_gain']=='NOT_ESTABLISHED'
    assert len(transfer_gate({})['blockers'])==len(ready)
    return 'PASS'

if __name__=='__main__':
    from relkit.pretraining_l183 import temporal_mask,keyed_mae,factorial_effect,transfer_gate
    print(checks(temporal_mask,keyed_mae,factorial_effect,transfer_gate))
