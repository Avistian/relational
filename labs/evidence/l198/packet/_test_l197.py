"""Behavior contracts: missing results are not zero and checked writing is not mastery."""
def check(coverage, admit, readiness):
    import copy
    checks=0
    def reject(fn):
        nonlocal checks
        try:fn()
        except ValueError:checks+=1
        else:raise AssertionError('Invalid evidence accepted')
    rows=[{'task':'a','status':'NOT_RUN','score':None},{'task':'b','status':'MEASURED','score':0.0}]
    assert coverage(['a','b'],rows)=={'declared':2,'measured':1,'missing':['a']};checks+=1
    for broken in [rows[:1],rows+rows[:1],[dict(rows[0],score=0),rows[1]],[rows[0],dict(rows[1],score=float('nan'))],[rows[0],dict(rows[1],status='PASS')]]:
        reject(lambda broken=broken:coverage(['a','b'],broken))
    reject(lambda:coverage(['a','a'],rows))
    allowed={'published_table':{'published_comparison'},'saved_predictions':{'scoped_pipeline_comparison'},'source_diagnostic':{'implementation_observation'}}
    claims=['published_comparison','scoped_pipeline_comparison','implementation_observation','fresh_model_reproduction','architecture_cause','economic_undervaluation']
    for lane in allowed:
        for claim in claims:
            for auth in [True,False]:
                for complete in [True,False]:
                    want='ADMISSIBLE_SCOPED' if auth and complete and claim in allowed[lane] else 'NOT_ESTABLISHED'
                    assert admit(lane,claim,auth,complete)==want;checks+=1
    reject(lambda:admit('published_table','published_comparison','true',True))
    reject(lambda:admit('invented','published_comparison',True,True))
    reject(lambda:admit('published_table','invented',True,True))
    fields=['claim','evidence','warrant','limitation','revision']
    essay={k:'A concrete sentence.' for k in fields}
    assert readiness(essay)=={'state':'READY_FOR_REVIEW','missing':[],'mastery':'PENDING_WRITTEN_DEFENSE'};checks+=1
    for key in fields:
        incomplete=dict(essay,**{key:'   '})
        assert readiness(incomplete)['missing']==[key];checks+=1
    reject(lambda:readiness(dict(essay,evidence=3)))
    reject(lambda:readiness({}))
    return {'status':'PASS','behavior_checks':checks}

if __name__=='__main__':
    from relkit.landscape_l197 import coverage_report,admit_claim,essay_readiness
    print(check(coverage_report,admit_claim,essay_readiness))
