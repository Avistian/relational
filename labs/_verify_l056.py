"""Audit frozen TabArena artifacts and validate ranking against pinned upstream code."""
import hashlib, importlib.util, importlib.metadata, json
from pathlib import Path
import numpy as np
import pandas as pd
from relkit.leaderboard import summarize, aligned_errors, rank_errors, macro_ranks

ROOT=Path(__file__).resolve().parent
ARMS=['CatBoost','LightGBM','RealMLP','TabM']

def load_snapshot():
    manifest=json.loads((ROOT/'_sources_l056.json').read_text())
    frames=[]
    for item in manifest['artifacts']:
        path=ROOT/'data/l056'/(item['method']+'.parquet')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],path
        frame=pd.read_parquet(path)
        assert set(frame.ta_suite)=={manifest['artifact_suite']}
        frame['arm']=item['method'].replace('_GPU','')
        frames.append(frame)
    return pd.concat(frames,ignore_index=True)

def verify():
    from _check_l056 import checks
    checks()
    manifest=json.loads((ROOT/'_sources_l056.json').read_text())
    for name,sha in manifest['source_files'].items():
        assert hashlib.sha256((ROOT/'sources/l056'/name).read_bytes()).hexdigest()==sha,name
    rows=load_snapshot()
    summary=summarize(rows,ARMS)
    spec=importlib.util.spec_from_file_location('l056_upstream',ROOT/'sources/l056/elo_utils.py')
    upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    helper=upstream.EloHelper(method_col='arm',task_col='dataset',split_col='fold')
    max_error=0.
    for regime in summary:
        x=rows.loc[rows.method_subtype==regime]
        wide=aligned_errors(x,ARMS)
        ranks=pd.DataFrame([rank_errors(v) for v in wide.to_numpy()],index=wide.index,columns=ARMS)
        expected=4-macro_ranks(ranks)
        names,actual=helper._rank_win_matrix(x)
        max_error=max(max_error,float(np.max(np.abs(actual-expected.loc[:,names].to_numpy().T))))
    assert max_error<1e-12
    # Lightweight view vs full evidence: no new training and no score selection.
    lite=summarize(rows.loc[rows.fold==0],ARMS)
    source_paths=['relkit/leaderboard.py','_verify_l056.py','_sources_l056.json']
    result=dict(scope='Reanalysis of frozen published measurements; no models retrained',
        rows=len(rows),versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','pyarrow']},
        snapshot_suite='tabarena-2025-06-12',summary=summary,lite=lite,
        parity=dict(operation='dataset-balanced rank-to-win totals',max_abs_error=max_error,regimes=3),
        source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths},
        original_elo_table='NOT_REPRODUCED',training='NOT_RUN',browser='NOT_CHECKED',live_colab='NOT_CHECKED')
    (ROOT/'_verify_l056_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for regime,s in summary.items():print(regime,s['mean_ranks'],s['tabm_minus_catboost_rank_gap'])
    print('Upstream parity max error:',max_error)
    return result

if __name__=='__main__':verify()
