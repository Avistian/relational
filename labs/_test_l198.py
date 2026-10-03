"""Behavioral contracts used unchanged by the student notebook and verifier."""
def check198(contrast, interval, matrix):
    import math
    def rejects(fn):
        try: fn()
        except ValueError: return
        raise AssertionError('Invalid research contract was accepted')
    assert math.isclose(contrast({'a':.70,'b':.74,'c':.72,'d':.73}, 'interaction'),-.03,abs_tol=1e-14)
    assert math.isclose(contrast({'a':.70,'b':.74,'c':.72,'d':.73}, 'conditional'),.01,abs_tol=1e-14)
    assert math.isclose(contrast({'a':.70,'b':.74}, 'gain'),.04,abs_tol=1e-14)
    rejects(lambda:contrast({'a':.7,'b':.8,'c':.9},'interaction'))
    rejects(lambda:contrast({'a':True,'b':.8},'gain'))
    rejects(lambda:contrast({'a':.7,'b':float('nan')},'gain'))
    rejects(lambda:contrast({'a':.7,'b':.8,'extra':.1},'gain'))
    assert interval(-.005,.005,.01,'sensitivity')=='BELOW_USEFUL_MARGIN'
    assert interval(-.03,-.02,.01,'sensitivity')=='MATERIAL_SENSITIVITY'
    assert interval(.02,.03,.01,'sensitivity')=='MATERIAL_SENSITIVITY'
    assert interval(.01,.02,.01,'sensitivity')=='INCONCLUSIVE'
    assert interval(-.02,.02,.01,'sensitivity')=='INCONCLUSIVE'
    assert interval(.02,.03,.01,'benefit')=='USEFUL_BENEFIT'
    assert interval(-.03,.009,.01,'benefit')=='BELOW_USEFUL_MARGIN'
    assert interval(.01,.03,.01,'benefit')=='INCONCLUSIVE'
    for args in [(1,0,.01,'benefit'),(0,1,0,'benefit'),(0,float('nan'),.01,'benefit'),(0,1,.01,'unknown'),(False,1,.01,'benefit')]:rejects(lambda:interval(*args))
    axes={'task':['f1','trial'],'arm':['A','B'],'seed':[0,1,2]}
    rows=matrix(axes)
    assert len(rows)==12 and len({tuple(r.values()) for r in rows})==12
    assert rows[0]=={'task':'f1','arm':'A','seed':0} and rows[-1]=={'task':'trial','arm':'B','seed':2}
    for bad in [{},{'seed':[]},{'seed':[0,0]},{'seed':'012'},{'seed':[True,1]},{'':[0]}]:rejects(lambda:matrix(bad))
    return {'contrasts':'PASS','strict_interval_boundaries':'PASS','complete_cartesian_matrix':'PASS'}

if __name__=='__main__':
    from relkit.proposals_l198 import paired_contrast,interval_decision,expand_matrix
    print(check198(paired_contrast,interval_decision,expand_matrix))
