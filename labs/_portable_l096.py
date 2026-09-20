"""Replay inline solution under the invoking Python, in a temporary directory."""
import hashlib,importlib.metadata as metadata,json,os,platform,sys,tempfile
from pathlib import Path
P=Path(__file__).resolve().parent
nb=json.loads((P/'solutions/0096-multi-relational-data.ipynb').read_text())
with tempfile.TemporaryDirectory(prefix='l096-portable-') as d:
    old=os.getcwd();os.chdir(d)
    try:
        ns={'__name__':'__main__'}
        for cell in nb['cells']:
            if cell['cell_type']=='code':exec(''.join(cell['source']),ns)
        fresh=json.loads(Path('l096-fresh.json').read_text())
    finally:os.chdir(old)
expected=json.loads((P/'_experiment_l096_results.json').read_text())
assert fresh['records']==expected['records']
report={'status':'PASS','python':platform.python_version(),'packages':{k:metadata.version(k) for k in ['numpy','torch','torch-geometric']},'fresh_databases':33,'all_records':'EXACT','live_colab':'NOT_CHECKED'}
(P/'_portable_l096_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
