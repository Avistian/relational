"""Replay the exact standalone solution cells without repository imports or Jupyter."""
import json,os,tempfile,sys,time
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
P=Path(__file__).resolve().parent;os.environ['L095_DATA_PATH']=str(P/'data/l095/ml-100k.zip')
nb=json.loads((P/'solutions/0095-bipartite-graphs.ipynb').read_text());env={'__name__':'__main__'};count=0;start=time.time()
with tempfile.TemporaryDirectory(prefix='l095-portable-') as folder:
    os.chdir(folder)
    for cell in nb['cells']:
        if cell['cell_type']=='code':exec(compile(''.join(cell['source']),'<standalone-notebook>','exec'),env);count+=1
    fresh=json.loads(Path('l095-fresh.json').read_text())
expected=json.loads((P/'_experiment_l095_results.json').read_text());assert fresh['audit']==expected['release_audit']
# Whole ranking order can be sensitive to BLAS rounding; verify actual exact parity, do not assume it.
assert fresh['experiment']['runs']==expected['runs']
assert fresh['experiment']['summary']==expected['summary']
import importlib.metadata as metadata
result={'status':'PASS','code_cells':count,'fresh_folds':5,'all_rankings_and_metrics':'EXACT','environment':{k:metadata.version(k) for k in ['numpy','torch','torch-geometric']},'python':sys.version,'seconds':time.time()-start,'live_colab':'NOT_CHECKED'}
(P/'_portable_l095_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
