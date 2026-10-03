"""Pin upstream tracked bytes and record exact source gaps, with no model run."""
import ast,hashlib,json,shutil,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;S=P/'sources/b15';src=Path('/tmp/b15-rdblearn')
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip()
assert commit=='78561f0a9c1dd231d44659e761d5d85e18c82f6e'
files=subprocess.check_output(['git','ls-files','-z'],cwd=src).decode().split('\0')
for name in filter(None,files):
 dest=S/'rdblearn'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src/name,dest)
config=ast.parse((src/'rdblearn/config.py').read_text())
cls=next(n for n in config.body if isinstance(n,ast.ClassDef) and n.name=='RDBLearnConfig')
defaults={n.target.id:ast.literal_eval(n.value) for n in cls.body if isinstance(n,ast.AnnAssign) and isinstance(n.value,ast.Constant)}
assert defaults['enable_target_augmentation'] is False and defaults['random_seed'] is None
status=dict(experiment='B15-RDBLEARN11-TRIAL',paper='2607.05476v2',table=5,dataset='rel-trial',task='study-outcome',published_auroc=.7271,
 source_commit=commit,release_tag='v1.1',status='INCOMPLETE_SOURCE_PROTOCOL_GATE',execution='NOT_RUN',
 default_label_history=False,readme_default_label_history=True,
 resolved='Pinned config defaults history OFF; README default ON is inconsistent. Estimator guards history creation and augmentation with this flag. Paper description agrees with code default, not README.',
 missing=['Exact Table 5 trial-task candidate grid including backbones, depths, aggregation sets and categorical encoding',
 'Per-candidate seeds and training subsamples (released random_seed defaults to None)',
 'Selected checkpoint revision and SHA256 for each backbone',
 'Exact validation selection, final refit support and snapshot mapping to reported score',
 'Published prediction receipts and complete data/config/environment identity'],
 release_example='examples/rdblearn_relbench_example.py uses rel-f1/driver-dnf defaults; it is not the selected trial experiment',
 paid_execution_admitted=False,cloud_spend_usd=0,full_suite='NOT_RUN',backbone_pretraining='NOT_RUN',historical_identity='NOT_ESTABLISHED')
(P/'evidence/b15/paper-status.json').write_text(json.dumps(status,indent=2)+'\n')
manifest=dict(commit=commit,tag_object='46a725de458205962b9eda3ffcd52fe7482ae519',paper_url='https://arxiv.org/html/2607.05476v2',source_url='https://github.com/HKUSHXLab/rdblearn/tree/'+commit,
 files={str(f.relative_to(S)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(S.rglob('*')) if f.is_file() and f.name!='manifest.json'})
(S/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(status['status'],len(manifest['files']),'source files authenticated')
