"""Verify reusable full preprocessing and independently audit all task labels."""
import hashlib,json,shutil,time,urllib.request,zipfile
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()

def prepare(root,prior):
    import numpy as np,pandas as pd,torch
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task
    from relbench.base import Table
    from relkit.trial_l139 import trial_target
    root=Path(root);prior=Path(prior);start=time.perf_counter();torch.set_num_threads(1)
    (root/'cache/tasks').mkdir(parents=True,exist_ok=True)
    source=Path(__file__).parent/'sources/l139';identity={}
    for name,registry in [('db.zip','relbench__datasets__hashes.json'),('tasks/study-outcome.zip','relbench__tasks__hashes.json')]:
        path=root/'cache'/name
        if name=='db.zip':shutil.copyfile(prior/'cache/relbench/rel-trial/db.zip',path)
        else:urllib.request.urlretrieve('https://relbench.stanford.edu/download/rel-trial/'+name,path)
        expected=json.loads((source/registry).read_text())['rel-trial/'+name];actual=sha(path)
        identity[name]=dict(sha256=actual,historical=expected,match=actual==expected)
        assert actual==expected,'Archive drift: audit before running'
        with zipfile.ZipFile(path) as z:z.extractall(root/'unpacked')
    (root/'data_identity.json').write_text(json.dumps(identity,indent=2))
    dataset=get_dataset('rel-trial');dataset.cache_dir=str(root/'unpacked')
    db=dataset.get_db(upto_test_timestamp=False)
    task=get_task('rel-trial','study-outcome');task.cache_dir=str(root/'unpacked/study-outcome')
    studies=db.table_dict['studies'].df;outcomes=db.table_dict['outcomes'].df;analyses=db.table_dict['outcome_analyses'].df
    # Independent pandas join/filter/groupby, not the task SQL implementation.
    joined=analyses.merge(outcomes[['id','nct_id','outcome_type']],left_on='outcome_id',right_on='id',how='left',suffixes=('','_outcome'))
    joined=joined.merge(studies[['nct_id','start_date']],left_on='nct_id_outcome',right_on='nct_id',how='left',suffixes=('','_study'))
    ok=(joined.p_value_modifier.isna()|joined.p_value_modifier.ne('>'))&joined.p_value.between(0,1)&joined.outcome_type.eq('Primary')
    good=joined[ok];report=dict(status='PASS',splits={},raw_rows={k:len(v) for k,v in db.table_dict.items()},source_sql='PASS',label_mismatches=0,eligibility_mismatches=0)
    samples={};arrays={}
    for split in ['train','val','test']:
        table=Table.load(root/'unpacked/study-outcome'/f'{split}.parquet');df=table.df
        pieces=[]
        for t in sorted(df.timestamp.unique()):
            keep=good[(good.start_date<=t)&(good.date>t)&(good.date<=t+pd.Timedelta(days=365))]
            grouped=keep.groupby('nct_id').p_value.min().le(.05).astype(int).rename('outcome').reset_index();grouped['timestamp']=t;pieces.append(grouped)
        reconstructed=pd.concat(pieces).set_index(['timestamp','nct_id']).sort_index()
        expected=df.set_index(['timestamp','nct_id']).sort_index()
        assert reconstructed.index.equals(expected.index),(split,'eligibility')
        assert np.array_equal(reconstructed.outcome,expected.outcome),(split,'labels')
        original=task.make_table(db,pd.Series(sorted(df.timestamp.unique()))).df.set_index(['timestamp','nct_id']).sort_index()
        assert original.index.equals(expected.index) and np.array_equal(original.outcome,expected.outcome)
        report['splits'][split]=dict(rows=len(df),positives=int(df.outcome.sum()),prevalence=float(df.outcome.mean()),timestamps=[str(t) for t in sorted(df.timestamp.unique())])
        for field,col in [('study','nct_id'),('time','timestamp'),('target','outcome')]:arrays[split+'_'+field]=df[col].astype('int64').to_numpy()
        samples[split]=[]
        chosen=pd.concat([df[df.outcome==label].head(6) for label in [0,1]])
        for row in chosen.itertuples():
            subset=joined[joined.nct_id==row.nct_id];events=[]
            for a in subset.itertuples():
                if pd.isna(a.date):continue
                events.append([float((a.date-row.timestamp)/pd.Timedelta(days=1)),None if pd.isna(a.p_value) else float(a.p_value),None if pd.isna(a.p_value_modifier) else a.p_value_modifier,None if pd.isna(a.outcome_type) else a.outcome_type])
            s=float((studies.set_index('nct_id').loc[row.nct_id,'start_date']-row.timestamp)/pd.Timedelta(days=1))
            assert trial_target(s,events,0)==(True,int(row.outcome))
            samples[split].append(dict(study=int(row.nct_id),timestamp=str(row.timestamp),start=s,analyses=events,target=int(row.outcome)))
    np.savez_compressed(root/'queries.npz',**arrays)
    (root/'samples.json').write_text(json.dumps(samples,indent=2));(root/'task_audit.json').write_text(json.dumps(report,indent=2))
    # Prior graph was built on the same complete test-cutoff snapshot using seed42/GloVe.
    old=json.loads((prior/'prepared/prepared.json').read_text());snapshot=dataset.get_db()
    assert old['archive_hashes']['db.zip']==identity['db.zip']['sha256']
    assert old['rows']=={k:len(v) for k,v in snapshot.table_dict.items()}
    assert old['preprocessing_seed']==42
    spec=json.loads((source/'manifest.json').read_text())['text_model']
    assert all(old['text_model'][k]==v for k,v in spec.items())
    del db,snapshot,studies,outcomes,analyses,joined,good
    import gc;gc.collect()
    graph=root/'graph.pt';shutil.copyfile(prior/'prepared/graph.pt',graph)
    digest=sha(graph);assert digest==sha(prior/'prepared/graph.pt')
    data,stats=torch.load(graph,weights_only=False)
    assert {k:data[k].tf.num_rows for k in data.node_types}==old['rows']
    assert {'|'.join(k):v.shape[1] for k,v in data.edge_index_dict.items()}==old['edges']
    report=dict(status='COMPLETE',seconds=time.perf_counter()-start,archive_hashes=identity,graph_sha256=digest,graph_bytes=graph.stat().st_size,rows=old['rows'],edges=old['edges'],preprocessing_seed=42,text_model=spec,preprocessing='REUSED_FROM_L132; not fresh materialization',upstream_preparation=old,task='rel-trial/study-outcome')
    shutil.copyfile(prior/'prepared/stypes.json',root/'stypes.json')
    (root/'prepared.json').write_text(json.dumps(report,indent=2));return report
