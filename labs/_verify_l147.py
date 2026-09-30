"""Fresh CPU reanalysis of reused L146 artifacts; no model training."""
import hashlib,json
from pathlib import Path
import numpy as np
from relkit.survey_l147 import keyed_losses,paired_summary,rank_questions
P=Path(__file__).resolve().parent;E=P/'evidence/l146';OUT=P/'evidence/l147'

def verify():
    hashes={}; bundle={}; runs=[]; means={split:{arm:{} for arm in ['gnn','relgt']} for split in ['val','test']}; count=0
    def record(p):hashes[str(p.relative_to(P))]=hashlib.sha256(p.read_bytes()).hexdigest()
    references={}
    for split in ['val','test']:
        p=E/f'prepared/{split}.npz';record(p)
        with np.load(p,allow_pickle=False) as z:
            references[split]={k:z[k].copy() for k in ['entity','cutoff','target']}
        for k,v in references[split].items():bundle[f'ref_{split}_{k}']=v
    for arm in ['gnn','relgt']:
        for seed in [0,1,2]:
            p=E/f'fit-{arm}-{seed}/predictions.npz';record(p)
            with np.load(p,allow_pickle=False) as z:
                for k in z.files:bundle[f'{arm}_{seed}_{k}']=z[k].copy()
                row=dict(arm=arm,seed=seed)
                for split,ref in references.items():
                    # An independent fixed permutation detects positional alignment.
                    order=np.random.default_rng(147+seed).permutation(len(ref['target']))
                    losses=keyed_losses(list(zip(ref['entity'],ref['cutoff'])),ref['target'],
                        list(zip(z[split+'_entity'][order],z[split+'_cutoff'][order])),
                        z[split+'_target'][order],z[split+'_pred'][order])
                    value=float(losses.mean());row[split]=value;means[split][arm][seed]=value;count+=len(losses)
                runs.append(row)
    summary_path=E/'summary.json';record(summary_path);old=json.loads(summary_path.read_text())
    paired={s:paired_summary(means[s]['gnn'],means[s]['relgt']) for s in means}
    for split in paired:
        np.testing.assert_allclose(paired[split]['differences'],old['paired'][split]['differences'],atol=1e-12,rtol=0)
        for arm in ['gnn','relgt']:
            vals=list(means[split][arm].values())
            assert abs(np.mean(vals)-old['arms'][arm][split]['mean'])<1e-12
            assert abs(np.std(vals,ddof=1)-old['arms'][arm][split]['sample_sd'])<1e-12
    assert count==7554
    questions=json.loads((OUT/'questions.json').read_text());record(OUT/'questions.json')
    report=dict(status='PASS',scope='Fresh reanalysis of REUSED L146 predictions; no training',verified_predictions=count,
        runs=runs,paired=paired,ranking=rank_questions(questions,4,10),ranking_one_hour=rank_questions(questions,1,10),
        full_selected_reproduction='INCOMPLETE',whole_paper='NOT_RUN',historical_identity='NOT_ESTABLISHED',
        cloud_spend_usd=0,learner='PENDING_WRITTEN_DEFENSE',reference_boundary='Prepared L146 targets reused; raw label reconstruction not rerun')
    np.savez_compressed(OUT/'audit-inputs.npz',**bundle)
    record(OUT/'audit-inputs.npz')
    for p in [P/'relkit/survey_l147.py',P/'_check_l147.py',P/'_verify_l147.py']:record(p)
    (OUT/'input-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (P/'_verify_l147_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':verify()
