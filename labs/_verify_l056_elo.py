"""Versioned measured evaluator extension; leaves original rank evidence untouched."""
import hashlib, importlib.util, json
from pathlib import Path
import numpy as np
from _verify_l056 import ROOT,ARMS,load_snapshot
from relkit.leaderboard import aligned_errors
from relkit.leaderboard_elo import paired_wins,fit_elo,rating_audit
from _check_l056_elo import checks


def verify():
    checks();rows=load_snapshot()
    spec=importlib.util.spec_from_file_location('upstream_l056',ROOT/'sources/l056/elo_utils.py')
    upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    helper=upstream.EloHelper(method_col='arm',task_col='dataset',split_col='fold')
    out={};differences=[]
    for regime in ['default','tuned','tuned_ensemble']:
        frame=rows.loc[rows.method_subtype==regime]
        errors=aligned_errors(frame,ARMS)
        out[regime]=rating_audit(errors)
        official=helper.compute_mle_elo_from_ranks(frame).reindex(ARMS).to_numpy()
        differences.append(float(np.abs(official-out[regime]['ratings']).max()))
        # The matchup frequency A/B is pool-independent; fitted A/B rating gap need not be.
        reduced=errors[['CatBoost','TabM']]
        reduced_elo=fit_elo(paired_wins(reduced))
        out[regime]['two_method_gap']=float(reduced_elo[1]-reduced_elo[0])
        print(regime,dict(zip(ARMS,np.round(out[regime]['ratings'],3))))
    assert max(differences)<1e-4,differences
    record=dict(scope='Fresh execution of four-method rating reanalysis, no training',results=out,
        source_parity_max_elo_error=max(differences),source_revision='e7cc6b049f9be11a6df29eb2560d9ccb2399d95c',
        source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ['relkit/leaderboard_elo.py','_verify_l056_elo.py','_sources_l056.json']},
        original_figure1='NOT_REPRODUCED',historical_rank_evidence='PRESERVED')
    (ROOT/'_verify_l056_elo_results.json').write_text(json.dumps(record,indent=2)+'\n')
    print('PASS: current upstream fitted-rating parity',max(differences))
    return record

if __name__=='__main__':verify()
