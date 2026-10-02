"""Behavioral tests: visibility, source refit policy and full row/time identities."""
def checks(visible,baseline,score):
    import numpy as np,pandas as pd
    cols=['position','points','grid']
    assert visible(cols,'position',['points'],True,'global')==['grid']
    assert visible(cols,'position',['points'],False,'global')==['grid']
    assert visible(cols,'position',['points'],True,'seed_only')==['grid']
    assert visible(cols,'position',['points'],False,'seed_only')==cols
    def reject(fn,*args):
        try:fn(*args)
        except ValueError:return
        raise AssertionError('Invalid input admitted')
    reject(visible,cols,'position',[],True,'unknown')
    fit=pd.DataFrame({'entity':[1,1,2],'time':[1,2,1],'y':[2.,8.,11.]})
    q=pd.DataFrame({'entity':[2,3,1],'time':[3,3,3]})
    assert np.array_equal(baseline(fit,q,'entity_mean'),[11,0,5])
    assert np.array_equal(baseline(fit,q,'global_median'),[8,8,8])
    assert np.allclose(baseline(fit,q,'global_mean'),[7,7,7])
    assert np.array_equal(baseline(fit,q,'global_zero'),[0,0,0])
    assert np.array_equal(baseline(fit,q,'entity_median'),[11,0,5])
    refit=pd.concat([fit,pd.DataFrame({'entity':[3],'time':[2],'y':[19.]})],ignore_index=True)
    assert baseline(refit,q,'entity_mean')[1]==19 and baseline(fit,q,'entity_mean')[1]==0
    reject(baseline,fit,q,'invented')
    truth=pd.DataFrame({'entity':[1,1,2],'time':[1,2,2],'y':[1.,3.,5.]})
    pred=pd.DataFrame({'entity':[2,1,1],'time':[2,1,2],'pred':[4.,2.,3.]})
    got=score(truth,pred);assert got['n']==3 and abs(got['mae']-2/3)<1e-12 and abs(got['r2']-.75)<1e-12
    reject(score,truth,pred.iloc[:2]);reject(score,truth,pd.concat([pred,pred.iloc[:1]]));reject(score,pd.concat([truth,truth.iloc[:1]]),pred)
    bad=pred.copy();bad.loc[0,'time']=3;reject(score,truth,bad)
    bad=pred.copy();bad.loc[0,'pred']=np.nan;reject(score,truth,bad)
    return 'PASS'
if __name__=='__main__':
    from relkit.autocomplete_l181 import visible_columns,baseline_predictions,keyed_scores
    print(checks(visible_columns,baseline_predictions,keyed_scores))
