"""Run with a newly installed Python environment; download data to an empty directory.
This script checks exact full-seed replay, not just package importability.
"""
import hashlib,json,platform,tempfile
from pathlib import Path
import numpy,scipy,torch,matplotlib
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'relkit'))
from gat_l084 import load_cora,train_cora
LAB=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='l084-cora-clean-') as tmp:
 manifest=json.loads((LAB/'_sources_l084.json').read_text());data=load_cora(Path(tmp),manifest)
 actual=train_cora(data,0)
report=LAB/'_paper_l084_results.json'
expected=json.loads(report.read_text())['runs'][0] if report.exists() else json.loads((LAB/'runs/l084/seed-000.json').read_text());expected.pop('seconds',None)
assert actual==expected
result={'status':'PASS','python':platform.python_version(),'platform':platform.platform(),'torch':torch.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'data_downloaded_into_empty_directory':True,'verified_data_files':len(manifest['data']),'full_seed0_epochs':actual['epochs'],'test_accuracy':actual['test_accuracy'],'all_epoch_losses_and_score_match':True,'implementation_sha256':hashlib.sha256((LAB/'relkit/gat_l084.py').read_bytes()).hexdigest(),'scope':'Fresh CPU model/plotting dependencies; notebook tooling, Colab and other platforms are separate'}
(LAB/'_clean_environment_l084_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
