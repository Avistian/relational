"""Independent audits plus corruption checks; no model execution."""
import copy,json,tempfile,shutil,hashlib
from pathlib import Path
from _audit_b03 import audit_sources,audit_predictions
P=Path(__file__).resolve().parent
if __name__=='__main__':
    r=audit_sources(P/'sources/b03');assert r['status']=='INCOMPLETE_SOURCE_PROTOCOL'
    assert r['missing_loader']=='tabpfn.datasets' and r['benchmark_runs']==0
    with tempfile.TemporaryDirectory() as td:
        for f in (P/'sources/b03').glob('*.zip'):shutil.copyfile(f,Path(td)/f.name)
        p=Path(td)/'TabPFN-main.zip';p.write_bytes(p.read_bytes()+b'corrupted')
        try:audit_sources(Path(td))
        except ValueError:pass
        else:raise AssertionError('Source corruption accepted')
    path=P/'evidence/b03/permutation-diagnostic.json'
    if path.exists():
        lock=json.loads((P/'evidence/b03/source-lock.json').read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==lock['diagnostic_record_sha256']
        assert hashlib.sha256((P/'_diagnostic_b03.py').read_bytes()).hexdigest()==lock['diagnostic_worker_sha256']
        raw=json.loads(path.read_text());report=audit_predictions(raw)
        changes=[lambda x:x['records'].pop(),lambda x:x['train_ids'].append(x['test_ids'][0]),lambda x:x['records'][1].update(permutation=[0,1,2]),lambda x:x['records'][0]['probabilities'][0].__setitem__(0,float('nan')),lambda x:x['test_ids'].reverse()]
        for change in changes:
            bad=copy.deepcopy(raw);change(bad)
            try:audit_predictions(bad)
            except (ValueError,AssertionError):pass
            else:raise AssertionError('Evidence corruption accepted')
        (P/'evidence/b03/diagnostic-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (P/'evidence/b03/source-gate.json').write_text(json.dumps(r,indent=2)+'\n')
    print('PASS: source gate, archive corruption and available diagnostic mutations')
