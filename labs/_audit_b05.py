"""Independent scalar retrieval and scoring; authenticate every episode and query."""
import hashlib,json,math,itertools
from pathlib import Path

def scalar_neighbors(support,queries,ids,k):
    f=len(support[0]);n=len(support)
    means=[sum(row[j] for row in support)/n for j in range(f)]
    scales=[math.sqrt(sum((row[j]-means[j])**2 for row in support)/n) or 1 for j in range(f)]
    return [[identity for _,identity in sorted((sum(((q[j]-row[j])/scales[j])**2 for j in range(f)),identity) for row,identity in zip(support,ids))[:k]] for q in queries]

def audit_course(root):
    root=Path(root);p=json.loads((root/'course-protocol.json').read_text());raw=(root/'inputs.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==p['input_sha256'],'Input hash'
    inputs=json.loads(raw);table=inputs['X'];records=json.loads((root/'episodes.json').read_text())
    expected=set(itertools.product(p['targets'],p['seeds'],p['episode_sizes']));seen=set();summary=[];total=0
    for r in records:
        key=(r['target'],r['seed'],r['size']);assert key in expected and key not in seen,'Missing/duplicate setting';seen.add(key)
        t,seed,size=key;x=[row[:t]+row[t+1:] for row in table]
        selected=scalar_neighbors(x,[x[seed]],list(range(len(x))),size)[0]
        assert r['selected_ids']==selected==r['target_intervention_ids'],'Selection identities'
        s=r['support_ids'];q=r['query_ids'];assert len(s)==size-8 and len(q)==8
        assert len(set(s+q))==size and set(s+q)==set(selected),'Partition keys'
        assert r['support_x']==[x[i] for i in s] and r['query_x']==[x[i] for i in q],'Feature identity'
        assert r['support_y']==[table[i][t] for i in s] and r['query_y']==[table[i][t] for i in q],'Label identity'
        nn=scalar_neighbors([x[i] for i in s],[x[i] for i in q],s,4);assert nn==r['neighbor_ids'],'Neighbor identity'
        pred=[sum(table[i][t] for i in ids)/4 for ids in nn]
        assert len(r['prediction'])==8 and all(math.isfinite(z) for z in r['prediction'])
        assert all(abs(a-b)<1e-10 for a,b in zip(pred,r['prediction'])),'Prediction values'
        mse=sum((a-table[i][t])**2 for a,i in zip(pred,q))/8;total+=8
        summary.append(dict(target=t,seed=seed,size=size,mse=mse))
    assert seen==expected,'Incomplete matrix'
    return dict(status='COMPLETE_COURSE_MECHANISM',episodes=len(records),predictions=total,independent_neighbors='PASS',all_keyed_predictions='PASS',target_interventions='18 selections unchanged',rows=summary,claims='No TabDPT training/inference, benchmark parity or architecture attribution')
if __name__=='__main__':
    e=Path(__file__).resolve().parent/'evidence/b05';r=audit_course(e);(e/'course-audit.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
