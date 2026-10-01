"""Feedback checks used by the portable learner notebook."""
def check166(prior_fn, dfs_fn, mask_fn):
    import numpy as np
    p=prior_fn(7,parents=4,children=12,strength=1.)
    q=prior_fn(7,parents=4,children=12,strength=0.)
    assert set(p)=={'parent_ids','latent','child_parent','noise','values'}
    assert set(p['child_parent'])<=set(p['parent_ids'])
    np.testing.assert_allclose(p['values']-q['values'],p['latent'][p['child_parent']])
    np.testing.assert_array_equal(p['child_parent'],q['child_parent'])
    got=dfs_fn([0,1,2],[1,0,1],[2.,10.,6.])
    np.testing.assert_allclose(got,[[1,10],[2,4],[0,0]])
    np.testing.assert_allclose(dfs_fn([0,1,2],[1,1,0],[6.,2.,10.]),got)
    try:dfs_fn([0,1],[2],[4.])
    except ValueError:pass
    else:raise AssertionError('Unknown foreign key accepted')
    expected=np.array([[1,1,0,0],[1,1,0,0],[1,1,0,0],[1,1,0,0]],dtype=bool)
    np.testing.assert_array_equal(mask_fn(4,2),expected)
    for rows,support in [(3,0),(3,4),(True,1)]:
        try:mask_fn(rows,support)
        except ValueError:pass
        else:raise AssertionError('Invalid support boundary accepted')
    return 'PASS'

if __name__=='__main__':
    from relkit.rdbpfn_l166 import relational_prior,dfs_summary,context_mask
    print(check166(relational_prior,dfs_summary,context_mask))
