"""Replay complete diagnostic, mutation checks, source gate and learner oracles."""
import copy,hashlib,json
from pathlib import Path
from _audit_b04 import audit_records
from _source_b04 import source_gate
from _test_b04 import check_functions
from relkit.scalable_b04 import attention_readout,fit_missingness,transform_missingness

def verify(root):
    p=Path(root);e=p/'evidence/b04';protocol=json.loads((e/'diagnostic-protocol.json').read_text());raw_bytes=(e/'inputs.json').read_bytes()
    assert hashlib.sha256(raw_bytes).hexdigest()==protocol['inputs_sha256']
    raw=json.loads(raw_bytes);records=[json.loads((e/(c['name']+'.json')).read_text()) for c in protocol['configs']]
    digest=hashlib.sha256((e/'diagnostic-protocol.json').read_bytes()).hexdigest()
    assert all(r['protocol_sha256']==digest for r in records)
    report=audit_records(raw,protocol,records)
    mutations=[]
    for name,fn in [
        ('missing configuration',lambda r:r.pop()),
        ('duplicate configuration',lambda r:r.append(copy.deepcopy(r[0]))),
        ('changed row identity',lambda r:r[0]['query_ids'].__setitem__(0,r[0]['support_ids'][0])),
        ('changed support identity',lambda r:r[0]['support_ids'].__setitem__(0,r[0]['support_ids'][1])),
        ('nonfinite probability',lambda r:r[0]['probabilities'][0].__setitem__(0,float('nan'))),
        ('class-column reversal',lambda r:r[0].__setitem__('classes',[1,0])),
        ('changed checkpoint',lambda r:r[0].__setitem__('checkpoint_sha256','changed')),
        ('query-fitted mean',lambda r:r[0]['means'].__setitem__(0,r[0]['means'][0]+1)),
        ('missing indicator columns',lambda r:r[-1].__setitem__('transformed_features',30)),
        ('missing prediction row',lambda r:r[0]['probabilities'].pop())]:
        altered=copy.deepcopy(records);fn(altered)
        try:audit_records(raw,protocol,altered)
        except (AssertionError,ValueError):mutations.append(name)
        else:raise AssertionError('Corruption accepted: '+name)
    return dict(status='PASS',source=source_gate(p/'sources/b04'),diagnostic=report,mutations_rejected=mutations,learner_functions=check_functions(attention_readout,fit_missingness,transform_missingness))

if __name__=='__main__':
    p=Path(__file__).resolve().parent;r=verify(p)
    (p/'evidence/b04/diagnostic-audit.json').write_text(json.dumps(r['diagnostic'],indent=2)+'\n')
    (p/'_verify_b04_results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
