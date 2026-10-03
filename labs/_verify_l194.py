"""Independent Decimal/BeautifulSoup oracle plus semantic and byte-tamper checks."""
import ast,copy,hashlib,json,math,tempfile,shutil
from decimal import Decimal
from pathlib import Path
from bs4 import BeautifulSoup
from _replay_l194 import build_report
from relkit.report_l194 import oriented_gap,evidence_license,summarize_comparisons
P=Path(__file__).resolve().parent;E=P/'evidence/l194';manifest=json.loads((E/'input-manifest.json').read_text());report=json.loads((E/'report.json').read_text())
soup=BeautifulSoup((E/'packet/paper.html').read_text(),'html.parser');expected=[]
for table in soup.find_all('table'):
    rows=[[c.get_text(' ',strip=True) for c in r.find_all(['td','th'])] for r in table.find_all('tr')]
    header=next((r for r in rows if 'Dataset' in r and 'RDBLearn' in r),None)
    if header is None:continue
    dataset=''
    for r in rows[rows.index(header)+1:]:
        if len(r)!=len(header):continue
        dataset=r[0] or dataset
        a=Decimal(r[header.index('RDBLearn')]);b=Decimal(r[header.index('AutoGluon+DFS')]);metric='MAE' if 'MAE' in r[1] else 'AUROC';d=a-b if metric=='AUROC' else b-a
        expected.append((dataset.lower(),r[1].split(' (')[0].lower(),metric,a,b,d))
assert len(expected)==21
for row,oracle in zip(report['task_results'],expected):
    assert row['task'].endswith('/'+oracle[0]+'/'+oracle[1])
    assert row['metric']==oracle[2] and row['paper_model']==float(oracle[3]) and row['paper_comparator']==float(oracle[4])
    assert math.isclose(row['gap'],float(oracle[5]),abs_tol=1e-12)
assert [sum(x[5]>0 for x in expected),sum(x[5]<0 for x in expected),sum(x[5]==0 for x in expected)]==[17,3,1]
assert build_report(E/'packet',manifest,oriented_gap,evidence_license,summarize_comparisons)==report
assert all(r['fresh_gap'] is None and r['fresh_mean'] is None and r['fresh_sd'] is None and r['attribution']=='NOT_ESTABLISHED' for r in report['task_results'])
checks=0
def rejects(fn):
    global checks
    try:fn()
    except (ValueError,KeyError):checks+=1
    else:raise AssertionError('Invalid input accepted')
for metric in ['AUROC','MAE']:
    assert oriented_gap(metric,None,.5) is None
    for bad in [float('nan'),float('inf'),-1,True,'0.5']:rejects(lambda:oriented_gap(metric,bad,.5))
rejects(lambda:oriented_gap('AUROC',1.1,.5));rejects(lambda:oriented_gap('accuracy',.7,.5))
assert oriented_gap('MAE',4,5)==1 and oriented_gap('AUROC',.5,.6)<0
ids=[r['task'] for r in report['task_results']]
for i in range(21):
    subset=copy.deepcopy(report['task_results']);subset.pop(i);rejects(lambda:summarize_comparisons(subset,ids))
    missing=copy.deepcopy(report['task_results']);missing[i]['gap']=None;assert summarize_comparisons(missing,ids)['missing']==1
rejects(lambda:summarize_comparisons(report['task_results']+[report['task_results'][0]],ids))
rejects(lambda:evidence_license('PROVEN_BENCHMARK_CAUSE'))
for variant in ['bytes','score','seed','prior_score','comparator','diagnostic']:
    with tempfile.TemporaryDirectory() as tmp:
        q=Path(tmp);shutil.copytree(E/'packet',q,dirs_exist_ok=True);m=copy.deepcopy(manifest)
        name={'bytes':'tasks.json','score':'runs.json','seed':'runs.json','prior_score':'prior-report.json','comparator':'analysis-protocol.json','diagnostic':'preprocessing.json'}[variant];path=q/name
        if variant=='bytes':path.write_bytes(path.read_bytes()+b' ')
        else:
            obj=json.loads(path.read_text())
            if variant=='score':obj[0]['score']=.9
            elif variant=='seed':obj.pop()
            elif variant=='prior_score':obj['task_results'][0]['mean']=.9
            elif variant=='comparator':obj['comparator']='RelGT'
            elif variant=='diagnostic':obj['observations'][0]['after']=[0,1,2]
            path.write_text(json.dumps(obj));m['files'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
        rejects(lambda:build_report(q,m,oriented_gap,evidence_license,summarize_comparisons))
# Learner functions must affect the full result rather than be unused demonstrations.
for funcs in [(lambda *a:0,evidence_license,summarize_comparisons),(oriented_gap,lambda *a:'wrong',summarize_comparisons),(oriented_gap,evidence_license,lambda *a:{})]:
    assert build_report(E/'packet',manifest,*funcs)!=report
result=dict(status='PASS',independent_paper_rows=21,decimal_sign_counts=[17,3,1],rejected_invalid_cases=checks,missingness_cases=21,learner_function_mutations=3,source_diagnostic='SAVED_EVIDENCE_REPLAY',fresh_model_evaluations=0)
(P/'_verify_l194_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
