"""Independent frozen oracles, Cartesian and Decimal arithmetic, adversarial contracts."""
import copy,hashlib,itertools,json,os,shutil,subprocess,sys,tempfile
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from _audit_l198 import run198,authenticate198
from _test_l198 import check198
from relkit.proposals_l198 import paired_contrast,interval_decision,expand_matrix
P=Path(__file__).resolve().parent;E=P/'evidence/l198';Q=E/'packet'
authenticate198(E)
r=json.loads((E/'report.json').read_text());assert run198(E)==r
independent={}
with tempfile.TemporaryDirectory(prefix='l198-independent-') as tmp:
    root=Path(tmp);shutil.copytree(Q,root,dirs_exist_ok=True)
    (root/'relkit/__init__.py').write_text('')
    for lesson in ['l189','l197']:
        proc=subprocess.run([sys.executable,str(root/('_verify_'+lesson+'.py'))],cwd=root,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
        assert proc.returncode==0,proc.stderr
        independent[lesson]=json.loads((root/('_verify_'+lesson+'_results.json')).read_text());assert independent[lesson]['status']=='PASS'
cases=json.loads((E/'proposals.json').read_text())
for row in r['ranking']['weight_grid']:
    truth={c['id']:Fraction(c['impact']*sum(a*b for a,b in zip(c['feasibility'],row['weights'])),sum(row['weights'])) for c in cases}
    assert row['scores']=={k:float(v) for k,v in truth.items()}
    assert row['leaders']==sorted(k for k,v in truth.items() if v==max(truth.values()))
expected_counts=[{'model_evaluations':24},{'pretraining_checkpoints':12,'primary_prediction_batches':72,'released_prediction_batches':6,'tree_fit_prediction_batches':18},{'source_pretraining_fits':12,'target_fits':24}]
for c,o,counts in zip(cases,r['proposals'],expected_counts):
    assert o['counts']==counts
    for name,axes in c['matrices'].items():
        expected=set(itertools.product(*axes.values()));actual={tuple(row[k] for k in axes) for row in o['matrices'][name]}
        assert actual==expected and len(o['matrices'][name])==len(expected)
    x={k:Decimal(str(v)) for k,v in c['illustration']['scores'].items()};mode=c['illustration']['mode']
    exact=(x['b']-x['a']) if mode=='gain' else (x['d']-x['c'])
    if mode=='interaction':exact-=x['b']-x['a']
    assert abs(o['illustrative_contrast']-float(exact))<1e-14
    assert o['budget']['status']=='NOT_ESTABLISHED' and o['future_experiment']=='NOT_RUN'
contracts=check198(paired_contrast,interval_decision,expand_matrix)
mutations=[(lambda s,m:max(s.values()),interval_decision,expand_matrix),(paired_contrast,lambda *args:'USEFUL_BENEFIT',expand_matrix),(paired_contrast,interval_decision,lambda a:expand_matrix(a)[:1])]
for funcs in mutations:
    try:check198(*funcs)
    except AssertionError:pass
    else:raise AssertionError('Wrong learner implementation passed')
rejected=0
with tempfile.TemporaryDirectory() as tmp:
    e=Path(tmp);shutil.copytree(E/'packet',e/'packet');shutil.copy(E/'input-manifest.json',e/'input-manifest.json')
    target=e/'packet/evidence/l189/packet/cases.json';data=target.read_bytes()
    for mode in ['missing','changed','extra']:
        target.write_bytes(data)
        if mode=='missing':target.unlink()
        elif mode=='changed':target.write_bytes(data+b' ')
        else:(e/'packet/extra.json').write_text('{}')
        try:authenticate198(e)
        except ValueError:rejected+=1
        else:raise AssertionError('Damaged input accepted')
result=dict(status='PASS',independent_predecessor_oracles=independent,full_report_parity='EXACT',fraction_weight_scenarios=27,complete_matrix_counts=expected_counts,decimal_contrasts=3,learner_contracts=contracts,wrong_learner_functions_rejected=3,packet_corruptions_rejected=rejected,new_model_experiments='NOT_RUN')
(P/'_verify_l198_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
