"""Full selected reproduction: source SQL, independent labels, all baseline vectors."""
import ast,hashlib,importlib.util,json,platform,sys,time,types,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from relkit.f1_l109 import load_inputs,regenerate,run_reproduction,baseline_predict,ARMS,ARCHIVES
P=Path(__file__).resolve().parent

def source_types():
    root=P/'sources/l109';manifest=json.loads((root/'manifest.json').read_text())
    for name,meta in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==meta['sha256']
    pkg=types.ModuleType('l109_source');pkg.__path__=[str(root)];sys.modules[pkg.__name__]=pkg
    for name in ['table','database','dataset','task_base','task_entity']:
        spec=importlib.util.spec_from_file_location('l109_source.'+name,root/(name+'.py'));mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    from l109_source.table import Table
    from l109_source.database import Database
    from l109_source.dataset import Dataset
    from l109_source.task_entity import EntityTask
    from l109_source.task_base import TaskType
    src=(root/'task_f1.py').read_text();node=next(n for n in ast.parse(src).body if isinstance(n,ast.ClassDef) and n.name=='DriverPositionTask')
    ns={'duckdb':duckdb,'pd':pd,'Database':Database,'Table':Table,'EntityTask':EntityTask,'TaskType':TaskType,'r2':None,'mae':None,'rmse':None}
    exec(ast.get_source_segment(src,node),ns)
    return Table,Database,Dataset,ns['DriverPositionTask']

def verify():
    start=time.perf_counter()
    db_registry=json.loads((P/'sources/l109/hashes.json').read_text())
    task_registry=json.loads((P/'sources/l109/task_hashes.json').read_text())
    assert ARCHIVES['db.zip'][1]==db_registry['rel-f1/db.zip']
    assert ARCHIVES['driver-position.zip'][1]==task_registry['rel-f1/tasks/driver-position.zip']
    tables,tasks,metadata=load_inputs(P/'data/l109')
    Table,Database,Dataset,Task=source_types()
    db=Database({n:Table(df.copy(),**metadata[n]) for n,df in tables.items()})
    # Original Dataset.get_db reads the exact authenticated table bytes, no network package.
    folder=P/'data/l109/source-db';folder.mkdir(exist_ok=True)
    with zipfile.ZipFile(P/'data/l109/db.zip') as z:
        for name in z.namelist():
            if name.endswith('.parquet'):
                dest=folder/'db'/Path(name).name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(z.read(name))
    dataset=Dataset(str(folder));dataset.val_timestamp=pd.Timestamp('2005-01-01');dataset.test_timestamp=pd.Timestamp('2010-01-01')
    task=Task(dataset);source_counts={}
    for split in ['train','val','test']:
        original=task._get_table(split).df.sort_values(['date','driverId']).reset_index(drop=True)
        ours=regenerate(tables,metadata,split).sort_values(['date','driverId']).reset_index(drop=True)
        released=tasks[split].sort_values(['date','driverId']).reset_index(drop=True)
        for other in [original,released]:
            assert len(ours)==len(other),(split,len(ours),len(other))
            assert (ours.date.to_numpy()==other.date.to_numpy()).all()
            np.testing.assert_array_equal(ours.driverId,other.driverId)
            np.testing.assert_allclose(ours.position,other.position,rtol=0,atol=1e-12)
        source_counts[split]={'rows':len(ours),'max_label_error':float(np.max(np.abs(ours.position-original.position))),'source_and_cached_labels':'MATCH'}
    report=run_reproduction(P/'data/l109',P/'evidence/l109')
    assert all(row['verdict']=='MATCH' for row in report['results'])
    src=(P/'sources/l109/baseline_node.py').read_text();node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    ns={'np':np,'pd':pd,'Table':Table,'Dict':dict,'task':types.SimpleNamespace(target_col='position',evaluate=lambda pred,*args:pred)}
    exec(ast.get_source_segment(src,node),ns)
    for split in ['val','test']:
        fit=tasks['train'] if split=='val' else pd.concat([tasks['train'],tasks['val']])
        query=tasks[split][['date','driverId']]
        for arm in ARMS:
            original=ns['evaluate'](Table(fit,{'driverId':'drivers'}),Table(query,{'driverId':'drivers'}),arm)
            np.testing.assert_allclose(baseline_predict(fit,query,arm),original,rtol=0,atol=1e-12)
    report.update(source_label_comparison=source_counts,source_prediction_vectors=10,seconds=time.perf_counter()-start,environment={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'duckdb':duckdb.__version__},cloud_spend_usd=0)
    report['hashes']={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [P/'relkit/f1_l109.py',P/'relkit/database_l109.py',*sorted((P/'evidence/l109').glob('*'))] if f.is_file()}
    (P/'_verify_l109_results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':verify()
