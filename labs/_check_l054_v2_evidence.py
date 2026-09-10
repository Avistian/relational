"""Recompute saved corrected-v2 scores from predictions and verify source identities."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from relkit.realmlp_experiment import load_task,error
ROOT=Path(__file__).resolve().parent

def check():
    result=json.loads((ROOT/'_verify_l054_v2_results.json').read_text())
    for path,expected in result['source_hashes'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    count=0;means=[]
    for name,dataset in result['results'].items():
        data=load_task(name,1200,600,ROOT/'data/cache/l052')
        assert data['hashes']==dataset['hashes'] and data['selection']==dataset['selection']
        for arm,runs in dataset['runs'].items():
            for run in runs:
                observed=error(np.asarray(run['prediction']),data['y']['test'],data['regression'],data['target_std'])
                np.testing.assert_allclose(observed,run['error'],rtol=2e-6,atol=1e-8)
                if arm=='TabM-mini':
                    members=np.asarray(run['member_predictions'])
                    np.testing.assert_allclose(members.mean(1),run['prediction'],rtol=1e-6,atol=1e-7)
                    individual=[error(members[:,i],data['y']['test'],data['regression'],data['target_std']) for i in range(result['config']['k'])]
                    np.testing.assert_allclose(np.mean(individual),run['individual_mean'],rtol=2e-6)
                if 'history' in run:assert run['best_epoch']==int(np.argmin(run['history']))+1
                count+=1
            np.testing.assert_allclose(np.mean([r['error'] for r in runs]),dataset['summary'][arm]['mean'])
            np.testing.assert_allclose(np.std([r['error'] for r in runs],ddof=1),dataset['summary'][arm]['sd'])
        means.append([dataset['summary'][a]['mean'] for a in result['ranks']['arms']])
    ranks=np.stack([rankdata(row) for row in means])
    np.testing.assert_allclose(ranks,result['ranks']['per_dataset'])
    record=dict(status='PASS',saved_predictions_reconciled=count,member_aggregation=True,first_best_epochs=True,
                source_hashes=True,data_hashes_and_rows=True,seed_statistics_and_ranks=True)
    (ROOT/'_check_l054_v2_evidence_results.json').write_text(json.dumps(record,indent=2)+'\n');print(record)
if __name__=='__main__':check()
