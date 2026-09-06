import hashlib,json,os
from pathlib import Path
from relkit.tabr import TabRS
from relkit.tabr_experiment import run_suite
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    os.chdir(HERE)
    result=run_suite(TabRS)
    result['source_sha256']={p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in ['relkit/tabr.py','relkit/tabr_experiment.py','_data_l052.json']}
    (HERE/'_verify_l052_results.json').write_text(json.dumps(result,indent=2))
    print(result['stats'])
