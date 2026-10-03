"""Complete independent replay and adversarial record checks."""
import copy,hashlib,json
from pathlib import Path
from _audit_b04a import audit_records
from _source_b04a import source_gate
P=Path(__file__).resolve().parent;E=P/'evidence/b04a'

def verify(root):
    root=Path(root);e=root/'evidence/b04a';protocol=json.loads((e/'protocol.json').read_text());raw_bytes=(e/'inputs.json').read_bytes()
    for path,digest in json.loads((e/'execution-contract.json').read_text()).items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=digest:raise ValueError('Execution identity changed: '+path)
    if hashlib.sha256(raw_bytes).hexdigest()!=protocol['inputs_sha256']:raise ValueError('Inputs corrupted')
    raw=json.loads(raw_bytes);records=[json.loads((e/'runs'/(c['name']+'.json')).read_text()) for c in protocol['configs']]
    report=audit_records(raw,protocol,records)
    if report!=json.loads((e/'audit.json').read_text()):raise ValueError('Saved audit does not match reconstruction')
    gate=source_gate(root/'sources/b04a')
    if gate!=json.loads((e/'source-gate.json').read_text()):raise ValueError('Source gate mismatch')
    rejected=[]
    for mutation in ['missing','duplicate','query_order','support_overlap','classes','probabilities','means','configuration','timing','inputs_hash']:
        bad=copy.deepcopy(records)
        if mutation=='missing':bad.pop()
        elif mutation=='duplicate':bad[-1]=copy.deepcopy(bad[0])
        elif mutation=='query_order':bad[0]['query_ids'].reverse()
        elif mutation=='support_overlap':bad[0]['support_ids'][0]=1024
        elif mutation=='classes':bad[0]['classes'].reverse()
        elif mutation=='probabilities':bad[0]['probabilities'][0]=[.999,.001]
        elif mutation=='means':bad[0]['mean'][0]+=1
        elif mutation=='configuration':bad[0]['config']['features']=7
        elif mutation=='timing':bad[0]['warm_pipeline_seconds']=[]
        elif mutation=='inputs_hash':bad[0]['inputs_sha256']='bad'
        try:audit_records(raw,protocol,bad)
        except ValueError:rejected.append(mutation)
        else:raise AssertionError('Corruption accepted: '+mutation)
    return {'status':'PASS','configurations':report['configurations'],'predictions':report['predictions'],'rejected_corruptions':rejected,'source_status':gate['status']}

if __name__=='__main__':print(json.dumps(verify(P),indent=2))
