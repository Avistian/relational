"""Replay all frozen predictions; no predictive fitting and no test-based model selection."""
import argparse,gzip,hashlib,json,platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from relkit.comparison_l060 import audit_records,aggregate_panel
from relkit.decision_guide import matched_datasets,select_feasible

LAB=Path(__file__).resolve().parent

def run():
    manifest=json.loads((LAB/'_sources_l079.json').read_text())
    for name,expected in manifest['local_sha256'].items():
        assert hashlib.sha256((LAB/name).read_bytes()).hexdigest()==expected, 'Source drift: '+name
    raw=gzip.decompress((LAB/'data/l079/l060-v2.json.gz').read_bytes())
    assert hashlib.sha256(raw).hexdigest()==manifest['evidence_sha256']
    evidence=json.loads(raw);summary=audit_records(evidence)
    paired=matched_datasets(evidence['design']['panels'])
    matched={regime:aggregate_panel([r for r in evidence['records'] if r['dataset'] in ids],ids,evidence['design']['arms'],evidence['config']['seeds']) for regime,ids in paired.items()}
    # Independent rank oracle: count strictly better and tied seed-mean errors.
    for regime,panel in {**summary,**{'matched_'+k:v for k,v in matched.items()}}.items():
        for ds,ranks in panel['dataset_ranks'].items():
            means=[next(r['mean'] for r in panel['details'] if r['dataset']==ds and r['arm']==a) for a in panel['arms']]
            expected=[1+sum(x<y for x in means)+(sum(x==y for x in means)-1)/2 for y in means]
            np.testing.assert_allclose(ranks,expected)
    scenarios=[dict(model='Trees',validation_loss=.32,p95_ms=2),dict(model='TabM',validation_loss=.29,p95_ms=8),dict(model='ICL',validation_loss=.27,p95_ms=35)]
    return {'status':'PASS','scope':'Full frozen L060 v2 prediction audit; no new training','records':len(evidence['records']),'evidence_sha256':manifest['evidence_sha256'],'summary':summary,'matched':matched,'matched_ids':paired,'synthetic_choices':{str(b):select_feasible(scenarios,b) for b in [1,5,10,40]},'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__},'training_replay':'NOT_RUN','paper_reproduction':'NOT_RUN','live_colab':'NOT_CHECKED'}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=LAB/'_verify_l079_results.json');args=parser.parse_args()
    result=run();args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['summary','matched']},indent=2))
