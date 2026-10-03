"""Full unmodified released preprocessor. Can run standalone beside packet/.
Usage: python _diagnostic_l196.py [evidence-directory]
"""
import hashlib,importlib.metadata,json,sys
from pathlib import Path

def run_diagnostic(evidence):
    evidence=Path(evidence);packet=evidence/'packet';source=packet/'source'
    ledger=json.loads((packet/'source-manifest.json').read_text())
    for name,digest in ledger['files'].items():
        if hashlib.sha256((packet/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Source hash mismatch: '+name)
    sys.path[:0]=[str(source/'rdblearn'),str(source)]
    from rdblearn.preprocessing import TabularPreprocessor
    import rdblearn.preprocessing as module
    import pandas as pd
    if Path(module.__file__).resolve()!=(source/'rdblearn/rdblearn/preprocessing.py').resolve():
        raise ValueError('Wrong source imported; use a fresh interpreter')
    train=pd.DataFrame({'category':['b','c','d']*4,'number':list(range(12))})
    known=pd.DataFrame({'category':['b','c','d'],'number':[1,2,3]})
    observations=[]
    for unseen in ['a','z','0','e']:
        pipeline=TabularPreprocessor().fit(train)
        support=pipeline.transform(train)
        before=pipeline.transform(known)
        probe=pd.DataFrame({'category':[unseen],'number':[1]})
        pipeline.transform(probe)
        after=pipeline.transform(known)
        fresh=TabularPreprocessor().fit(train)
        alone=fresh.transform(known.iloc[[0]])
        together=TabularPreprocessor().fit(train).transform(pd.concat([probe,known.iloc[[0]]],ignore_index=True))
        observations.append(dict(unseen=unseen,before=before.category.tolist(),after=after.category.tolist(),numeric_before=before.number.tolist(),numeric_after=after.number.tolist(),query_alone=alone.category.tolist(),query_with_other=together.category.tolist(),cached_support=support.category.tolist()))
    packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','autogluon.features','autogluon.common','pydantic','fastdfs']}
    return dict(status='COMPLETE_DIAGNOSTIC',commit=ledger['commit'],observations=observations,packages=packages,source_files=len(ledger['files']),model_evaluations=0,model_effect='NOT_ESTABLISHED',historical_environment='NOT_ESTABLISHED',full_model_reproduction='INCOMPLETE_SOURCE_PREPROCESSING_GATE')

if __name__=='__main__':
    E=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent/'evidence/l196'
    result=run_diagnostic(E)
    (E/'diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
