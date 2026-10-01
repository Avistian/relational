"""Assemble source-only release; immutable experiment manifest precedes training."""
import hashlib,json,shutil
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;D=P/'releases/l157-f1-audit';D.mkdir(parents=True,exist_ok=True)
files=['_run_l156.py','_run_l152.py','_run_l117.py','relkit/rdl_l117.py','relkit/batch_audit_l123.py','relkit/regression_l152.py','relkit/temporal_audit_l156.py','relkit/fe_experiment_l129.py','relkit/contribution_l157.py','_check_l157.py','requirements-l117-runtime.txt']
for name in files:
 dest=D/'labs'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(P/name,dest)
for name in ['_preflight_l156.py','_audit_fe_l156.py','_replay_l156.py']:
 text=(P/name).read_text().replace("evidence/l156","evidence/l157").replace("L156 full F1 temporal audit","L157 isolated full F1 temporal audit").replace('Lesson156 ·','Lesson157 ·')
 (D/'labs'/name).write_text(text)
for src in (P/'sources/l117').iterdir():
 if src.is_file():
  dest=D/'labs/sources/l117'/src.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
for name in ['data-manifest.json','frame/LICENSE']:
 src=P/'sources/l129'/name
 if src.exists():
  dest=D/'labs/sources/l129'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
protocol=json.loads((P/'evidence/l156/protocol.json').read_text());protocol.update(name='L157 — RelBench F1 temporal-audit reproducibility release',approval='2026-10-01 chat',fresh_fits=10,prior_test_exposure=True)
(D/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
(D/'correction-protocol.json').write_bytes((P/'evidence/l156/correction-protocol.json').read_bytes())
(D/'requirements-audit.txt').write_text('numpy==1.26.4\npandas==2.2.3\npyarrow==18.1.0\nduckdb==1.1.3\njinja2==3.1.6\n')
(D/'requirements-notebook.txt').write_text('numpy==1.26.4\nnbformat==5.10.4\nnbclient==0.10.2\nipykernel==6.29.5\nnbconvert==7.16.6\n')
(D/'THIRD_PARTY_NOTICES.md').write_text('''# Third-party notices and provenance

RelBench source files in labs/sources/l117 are MIT licensed; preserve their adjacent LICENSE. Pinned commit: 9aa346267c2e1c560bd92da07d6f4ad1ca2f0639. SQL comes from relbench-user-study commit445bb7a3b1230f49f8e5890ae81754d3e365680f; no license was found at its pinned root, so it is downloaded on demand and is not bundled or relicensed. The inherited Frame adapter notice is retained in labs/sources/l129/frame/LICENSE. Course implementations retain their provenance docstrings and are provided under the root MIT license. Dependencies keep their respective licenses.

Raw database/task archives and pretrained text weights are downloaded from their original hosts, never bundled or relicensed here. SHA256 and a text-model revision identify the expected artifacts. Availability of those upstream hosts is required for fresh runs. Code licensing does not establish permission to redistribute every upstream dataset. The compact evidence is derived benchmark output, not the raw database.

Primary paper: Robinson et al., RelBench: A Benchmark for Deep Learning on Relational Databases, NeurIPS2024, https://arxiv.org/abs/2407.20060. Released baseline target: Table7 basic GNN, rel-f1/driver-position. The fixed2005 intervention is course-specific.
''')
print(D)
