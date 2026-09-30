"""Behavioral contracts; also inlined beside the student's live functions."""
import numpy as np

def check_keyed_losses(fn):
    keys=[(7,10),(7,20),(8,10)]
    out=fn(keys,[1,5,9],[(8,10),(7,10),(7,20)],[9,1,5],[8,3,5])
    np.testing.assert_array_equal(out,[2,0,1])
    cases=[(keys[:2],[1,5],[2,5]), ([(7,10),(7,10),(8,10)],[1,1,9],[2,2,8]),
           (keys,[1,6,9],[2,5,8]), (keys,[1,5,9],[2,float('nan'),8]),
           ([(7,10),(7,20),(9,10)],[1,5,9],[2,5,8])]
    for k,t,p in cases:
        try: fn(keys,[1,5,9],k,t,p)
        except ValueError: pass
        else: raise AssertionError('Reject missing/duplicate/extra keys, changed targets and nonfinite predictions')
    try: fn(keys+[keys[0]],[1,5,9,1],keys,[1,5,9],[2,5,8])
    except ValueError: pass
    else: raise AssertionError('Reference duplicates must fail')

def check_paired_summary(fn):
    got=fn({2:4,0:2,1:3},{1:2,2:3,0:3})
    np.testing.assert_allclose(got['differences'],[-1,1,1])
    assert abs(got['mean']-1/3)<1e-12
    assert abs(got['sample_sd']-np.sqrt(4/3))<1e-12
    for a,b in [({0:2},{1:3}),({0:2},{0:3}),({0:2,1:np.nan},{0:3,1:4})]:
        try: fn(a,b)
        except ValueError: pass
        else: raise AssertionError('Need matching seed sets, at least two seeds and finite scores')

def check_rank_questions(fn):
    rows=[dict(id='b',hours=2,usd=0,evidence=2,ready=True),
          dict(id='a',hours=1,usd=0,evidence=1,ready=True),
          dict(id='c',hours=1,usd=0,evidence=3,ready=False),
          dict(id='d',hours=2,usd=11,evidence=3,ready=True)]
    assert fn(rows,2,10)==['b','a']
    assert fn(rows,1,10)==['a']
    assert fn(rows,2,0)==['b','a']
    assert fn(rows,0,10)==[]
    assert fn(rows+[dict(id='z',hours=2,usd=0,evidence=2,ready=True)],2,10)==['b','z','a']
    try: fn(rows+[rows[0]],2,10)
    except ValueError: pass
    else: raise AssertionError('Duplicate question IDs must fail')
    try: fn([dict(id='x',hours=-1,usd=0,evidence=1,ready=True)],2,10)
    except ValueError: pass
    else: raise AssertionError('Negative planning cost must fail')

if __name__=='__main__':
    from relkit.survey_l147 import keyed_losses,paired_summary,rank_questions
    for check,fn in [(check_keyed_losses,keyed_losses),(check_paired_summary,paired_summary),(check_rank_questions,rank_questions)]:
        check(fn)
    print('PASS: key identity, paired seed uncertainty, feasibility-first ranking')
