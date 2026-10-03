"""Freeze approved inputs; never modify the L193 experiment."""
import hashlib,json,shutil
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l194';Q=E/'packet';old=P/'evidence/l193'
Q.mkdir(parents=True,exist_ok=True)
m=json.loads((old/'input-manifest.json').read_text())
for name,h in m['files'].items():
    data=(old/'packet'/name).read_bytes()
    assert hashlib.sha256(data).hexdigest()==h,name
    (Q/name).write_bytes(data)
shutil.copyfile(old/'report.json',Q/'prior-report.json')
shutil.copyfile(old/'input-manifest.json',Q/'prior-manifest.json')
contract=dict(name='L194 complete evidence analysis of L193 RDBLearn v1',comparator='AutoGluon+DFS',paper='https://arxiv.org/html/2602.18495v1',tasks=21,cloud_usd=0,local_cap_seconds=1800,source_diagnostic='RECORDED_OBSERVATIONS_REPLAY',fresh_inference='NOT_RUN',approval='User approved lesson and report; no source repair or paid inference')
(Q/'analysis-protocol.json').write_text(json.dumps(contract,indent=2)+'\n')
manifest=dict(files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Q.iterdir())},origin='L193 packet authenticated against its input manifest; prior report copied as recorded evidence')
(E/'input-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Frozen',len(manifest['files']),'files')
