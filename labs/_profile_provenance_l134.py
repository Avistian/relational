"""Authenticate the historical implementation and scope the timing validation change."""
import ast,gzip,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
ARCHIVE=P/'sources/l134/scale_before_record_validation.py'

def _without_functions(code,names):
    tree=ast.parse(code)
    tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name not in names]
    return ast.dump(tree,include_attributes=False)

def verify_profile_provenance(current_notebook_code=None):
    old=ARCHIVE.read_bytes();digest=hashlib.sha256(old).hexdigest()
    for path,field in [('_sources_l134.json','files'),('_budget_l134.json','source_hashes')]:
        assert digest==json.loads((P/path).read_text())[field]['labs/relkit/scale_l134.py']
    current=(P/'relkit/scale_l134.py').read_text()
    assert _without_functions(old.decode(),{'profile_summary'})==_without_functions(current,{'profile_summary'})
    historical=gzip.decompress((P/'sources/l134/notebook_before_record_validation.py.gz').read_bytes()).decode()
    assert hashlib.sha256(historical.encode()).hexdigest()==json.loads((P/'_notebook_gpu_l134_results.json').read_text())['code_sha256']
    if current_notebook_code is not None:
        names={'profile_summary','check_summary'}
        assert _without_functions(historical,names)==_without_functions(current_notebook_code,names),'Unexpected change outside measurement validation and its checks'
    return dict(status='PASS',historical_module_sha256=digest,allowed_changes=['profile_summary','check_summary'],historical_gpu_run='UNCHANGED_ARCHIVE; not a fresh execution of revised validation')
