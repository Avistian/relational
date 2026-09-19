"""Reconcile every artifact with independent set-based scores and pairwise AUC."""
from pathlib import Path
import ast,hashlib,json,warnings
import numpy as np
import scipy.sparse as sp
from relkit.link_l087 import adjacency,drnl_subgraph,load_graph
LAB=Path(__file__).resolve().parent
manifest=json.loads((LAB/'_sources_l087.json').read_text())
for entry in manifest['files']:
    p=LAB/'sources/l087'/entry['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256']
paper=json.loads((LAB/'_paper_l087_results.json').read_text())
assert paper['environment']['implementation_sha256']==hashlib.sha256((LAB/'relkit/link_l087.py').read_bytes()).hexdigest()
checked=0
for r in paper['runs']:
    p=LAB/'results/l087/paper'/r['artifact'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['artifact_sha256'];a=np.load(p)
    full,n=load_graph(r['dataset'],manifest,LAB/'sources/l087');pos=set(map(tuple,full));tr=set(map(tuple,a['train']));te=set(map(tuple,a['test']))
    assert not tr&te and tr|te==pos
    tn=set(map(tuple,a['train_neg']));en=set(map(tuple,a['test_neg']));assert not tn&en and not (tn|en)&pos
    ns=[set() for _ in range(n)]
    for u,v in tr:ns[u].add(v);ns[v].add(u)
    pairs=np.vstack((a['test'],a['test_neg']));expected={k:[] for k in ['CN','AA','RA']}
    for u,v in pairs:
        common=ns[u]&ns[v];expected['CN'].append(len(common));expected['AA'].append(sum(1/np.log(len(ns[w])) for w in common));expected['RA'].append(sum(1/len(ns[w]) for w in common))
    for m in expected:
        np.testing.assert_allclose(a[m],expected[m],atol=1e-12)
        pscore=a[m][:len(te)];negative=a[m][len(te):]
        # Exact pairwise AUC independently counts wins and half-ties in bounded chunks.
        wins=sum(float((p[:,None]>negative).sum()+.5*(p[:,None]==negative).sum()) for p in np.array_split(pscore,32))
        assert abs(wins/(len(pscore)*len(negative))-r['auc'][m])<1e-12
    checked+=1
# Teaching artifacts: independent isolation/candidate/score/selection checks.
teach=json.loads((LAB/'_teaching_l087_results.json').read_text());e,n=load_graph('USAir',manifest,LAB/'sources/l087');truth=set(map(tuple,e))
for row in teach['runs']:
    path=LAB/f"results/l087/teaching-{row['seed']}.npz";assert hashlib.sha256(path.read_bytes()).hexdigest()==row['artifact_sha256'];a=np.load(path)
    parts=[set(map(tuple,a[k])) for k in ['context','train_pos','val_pos','test_pos']]
    assert set.union(*parts)==truth and all(not parts[i]&parts[j] for i in range(4) for j in range(i))
    negatives=[set(map(tuple,a[k])) for k in ['train_neg','val_neg','test_neg']]
    assert not set.union(*negatives)&truth and all(not negatives[i]&negatives[j] for i in range(3) for j in range(i))
    rp,rn=a['ranking_pos'],a['ranking_neg'];assert np.array_equal(rp,np.vstack((a['test_pos'],a['test_pos'][:,::-1])))
    assert np.all(rn[:,:,0]==rp[:,0,None])
    for query in rn:
        assert len(set(query[:,1]))==50
        assert all(u!=v and tuple(sorted((u,v))) not in truth for u,v in query)
    z=a['embeddings'];ps=np.einsum('ij,ij->i',z[rp[:,0]],z[rp[:,1]]);ns=np.einsum('ijk,ijk->ij',z[rn[:,:,0]],z[rn[:,:,1]])
    np.testing.assert_allclose(ps,a['positive_scores'],rtol=1e-6,atol=2e-6);np.testing.assert_allclose(ns,a['negative_scores'],rtol=1e-6,atol=2e-6)
    # Recorded float scores define exact ties; rankdata is independent of count formula.
    from scipy.stats import rankdata
    ranks=np.array([rankdata(-np.r_[p,neg],method='average')[0] for p,neg in zip(a['positive_scores'],a['negative_scores'])],dtype=np.float64)
    np.testing.assert_array_equal(ranks,row['ranking']['ranks']);assert float(np.mean(1/ranks))==row['ranking']['mrr']
    assert row['selected_epoch']==1+int(np.argmax([r['val_auc'] for r in row['trace']]))
source=(LAB/'sources/l087/Python/util_functions.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='node_label');scope={'np':np,'ssp':sp};exec(compile(ast.Module(body=[node],type_ignores=[]),'pinned-SEAL-node_label','exec'),scope)
rng=np.random.default_rng(87)
with warnings.catch_warnings():
    warnings.simplefilter('ignore',RuntimeWarning)
    for _ in range(30):
        e=np.array([(u,v) for u in range(8) for v in range(u+1,8) if rng.random()<.35]).reshape(-1,2)
        sub,lab,nodes=drnl_subgraph(adjacency(e,8),0,1,2)
        np.testing.assert_array_equal(lab,scope['node_label'](sub))
r={'status':'PASS','archived_files':len(manifest['files']),'full_artifacts_checked':checked,'independent_heuristic_and_pairwise_auc_checks':checked*3,'original_source_DRNL_cases':30,'teaching_artifact_audits':3,'MATLAB_runtime':'NOT_RUN','SEAL_training':'NOT_RUN'}
(LAB/'_audit_l087_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
