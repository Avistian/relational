"""Complete saved-evidence replay. Never invokes a model or modifies L200 inputs."""
import hashlib,json,subprocess,sys,tempfile,zipfile
from pathlib import Path
import numpy as np
from relkit.comparison_b01 import compare_contracts,paired_effect,claim_gate

def replay(root,compare=compare_contracts,paired=paired_effect,gate=claim_gate):
    root=Path(root);lock=json.loads((root/'source-lock.json').read_text())
    for name,digest in lock['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Changed frozen source '+name)
    with tempfile.TemporaryDirectory(prefix='b01-replay-') as td:
        temp=Path(td)
        with zipfile.ZipFile(root/'l200-reproducer.zip') as z:
            if any(Path(n).is_absolute() or '..' in Path(n).parts for n in z.namelist()):
                raise ValueError('Invalid archive path')
            z.extractall(temp)
        e=temp/'labs/evidence/l200';original=json.loads((e/'report.json').read_text())
        for script in ['_audit_l200.py','_verify_l200.py']:
            subprocess.run([sys.executable,str(temp/'labs'/script)],cwd=temp,check=True,capture_output=True,text=True,timeout=120)
        measured=json.loads((e/'report.json').read_text())
        if measured!=original:raise ValueError('Original report differs from raw replay')
        verification=json.loads((temp/'labs/_verify_l200_results.json').read_text())
        if verification['status']!='PASS' or not verification['report_parity']:raise ValueError('Verifier failed')
        manifest=json.loads((e/'packet-manifest.json').read_text())
        # Hash the actual arrays, including dtype and shape, not a filename assertion.
        with np.load(e/'packet/prepared.npz') as data:
            def digest(names):
                h=hashlib.sha256()
                for name in names:
                    a=data[name];h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
                return h.hexdigest()
            keys=digest(['test_keys']);labels=digest(['y_test']);support=digest(['train_keys','support'])
        common=dict(task='rel-f1/driver-dnf; released orientation',query_keys=keys,
            split='11411 train / 566 validation / 702 test; original released split',labels=labels,
            features=manifest['files']['packet/prepared.npz'],support=support,
            preprocessing='Released support-only median and normalization protocol',
            visibility='Released materialized DFS features; no query labels as input',
            selection='Fixed released configurations; no test-based selection',metric='AUROC; one fixed test population',
            budget_policy='Fixed complete evaluation: 512 support; 10 paired draws; no cost-efficiency claim')
        contracts={arm:dict(common) for arm in measured['models']}
        comparison=compare(contracts['RDBPFN'],contracts['TabICLv1.1'])
        records=lambda arm:[dict(draw=i,auc=float(a)) for i,a in enumerate(measured['models'][arm]['per_seed'])]
        effect=paired(records('RDBPFN'),records('TabICLv1.1'))
        inherited=measured['paired_rdbpfn_minus_tabicl']
        if effect['per_draw']!=inherited['per_seed'] or effect['mean']!=inherited['mean']:
            raise ValueError('Paired report mismatch')
        point_in_time=dict(common,visibility='NOT_ESTABLISHED')
        return dict(experiment='B01-MATCHED-COMPARISON-AUDIT',execution='COMPLETE_REPLAY',runs=measured['runs'],predictions=measured['predictions'],
            source_archive_sha256=lock['files']['l200-reproducer.zip'],models=measured['models'],contracts=contracts,
            released_input_comparison=comparison,historical_availability_comparison=compare(point_in_time,point_in_time),
            paired_rdbpfn_minus_tabicl=effect,independent_verification=verification,
            claims={c:gate(comparison['status'],'COMPLETE_REPLAY',c) for c in ['selected_score','fresh_inference','architecture_cause','learner_mastery']},
            deviations=['Inherited released-label complement; checkpoint width 96 versus appendix 128',
                'No raw DFS regeneration or full feature-arrival verification',
                'Different priors, checkpoints and TabICL inference recipe; matching inputs does not isolate architecture',
                'Frozen configuration budgets are declared, not equal measured cost; no efficiency claim'],
            source_report_parity='EXACT',whole_paper_reproduction='NOT_RUN',fresh_training='NOT_RUN',year5_exit='INCOMPLETE')

if __name__=='__main__':
    root=Path(__file__).resolve().parent/'evidence/b01';r=replay(root)
    (root/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ['experiment','execution','runs','predictions','claims']},indent=2))
