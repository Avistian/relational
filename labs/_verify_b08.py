"""Independent numerical audit, primitive parity and corruption rejection."""
import copy,json,shutil,tempfile,subprocess,sys
from pathlib import Path
import torch
from _audit_b08 import audit,E,P
from _reproduce_b08 import source_audit,released
from relkit.limix_b08 import Attention
results={}
for script in ['_test_b08.py','_model_test_b08.py']:
 subprocess.run([sys.executable,str(P/script)],check=True);results[script]='PASS'
audit_result=audit();(E/'course-audit.json').write_text(json.dumps(audit_result,indent=2)+'\n')
torch.manual_seed(19);a=Attention(8);q=torch.randn(2,3,8);kv=torch.randn(2,4,8);mask=torch.tensor([[1,1,0,0]]*3,dtype=torch.bool)
expected=a.out(torch.nn.functional.scaled_dot_product_attention(a.q(q).unsqueeze(1),a.k(kv).unsqueeze(1),a.v(kv).unsqueeze(1),attn_mask=mask,dropout_p=0).squeeze(1))
torch.testing.assert_close(a(q,kv,mask),expected,rtol=1e-5,atol=1e-6);results['attention_torch_reference']='PASS'
rejections=[]
for mutation in ['missing_arm','duplicate_arm','alter_prediction','alter_protocol','unpaired_init']:
 with tempfile.TemporaryDirectory() as td:
  root=Path(td);shutil.copy(E/'course-protocol.json',root);shutil.copytree(E/'runs',root/'runs')
  file=root/'runs/results.json';r=json.loads(file.read_text())
  if mutation=='missing_arm':r['records'].pop()
  elif mutation=='duplicate_arm':r['records'].append(r['records'][0])
  elif mutation=='alter_prediction':
   f=root/'runs'/r['records'][0]['file'];f.write_bytes(f.read_bytes()+b'x')
  elif mutation=='alter_protocol':
   f=root/'course-protocol.json';f.write_text(f.read_text()+' ')
  else:r['records'][1]['initial_sha256']='incorrect'
  file.write_text(json.dumps(r))
  try:audit(root)
  except AssertionError:rejections.append(mutation)
  else:raise AssertionError('Accepted corruption '+mutation)
g=source_audit();results['source']=g['status']
import _reproduce_b08 as repro
with tempfile.TemporaryDirectory() as td:
 root=Path(td);bad=copy.deepcopy(g);bad['checkpoint']['sha256']='wrong';(root/'source-gate.json').write_text(json.dumps(bad))
 original=repro.E;repro.E=root
 try:
  try:repro.source_audit()
  except AssertionError as ex:assert 'checkpoint metadata' in str(ex)
  else:raise AssertionError('Accepted altered checkpoint identity')
 finally:repro.E=original
results['source_gate_corruption']='REJECTED'

r=subprocess.run([sys.executable,str(P/'_reproduce_b08.py'),'--lane','paper'],capture_output=True,text=True)
assert r.returncode==2 and 'INCOMPLETE_SOURCE_PROTOCOL' in r.stdout;results['paper_dispatch_blocked']='PASS'
# Verify initial released-packet gates without fetching weights or running a model.
with tempfile.TemporaryDirectory() as td:
 root=Path(td);data=root/'data.npz';data.write_bytes(b'not a packet');ck=root/'ckpt';ck.write_bytes(b'wrong weights');spec=root/'packet.json'
 spec.write_text(json.dumps(dict(file='data.npz',sha256='wrong')))
 try:released(spec,ck,root/'out.npz')
 except ValueError as ex:assert 'Packet hash' in str(ex)
 else:raise AssertionError('Accepted altered packet')
 import hashlib
 config=dict(file='data.npz',sha256=hashlib.sha256(data.read_bytes()).hexdigest())
 spec.write_text(json.dumps(config))
 try:released(spec,ck,root/'out.npz')
 except ValueError as ex:assert 'Missing dataset_version' in str(ex)
 else:raise AssertionError('Accepted unprovenanced packet')
 for field in ['dataset_version','split_provenance','mask_provenance','scaler_provenance','feature_types','row_identity_provenance']:config[field]='test'
 spec.write_text(json.dumps(config))
 try:released(spec,ck,root/'out.npz')
 except ValueError as ex:assert 'Wrong checkpoint' in str(ex)
 else:raise AssertionError('Accepted wrong weights')
results['released_packet_initial_guards']='PASS: altered bytes, missing provenance, wrong weights; full inference NOT_RUN'
results.update(status='PASS' ,corruption_rejections=rejections,course_arms=9)
(P/'_verify_b08_results.json').write_text(json.dumps(results,indent=2)+'\n');print(results)
