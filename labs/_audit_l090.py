"""Audit provenance, declared protocol, execution coverage and current source identity."""
from pathlib import Path
import ast,hashlib,json
import numpy as np
from relkit.gcn_l082 import load_cora
R=Path(__file__).resolve().parent
manifest=json.loads((R/'_sources_l078.json').read_text())
for record in manifest['files']:
 assert hashlib.sha256((R/record['path']).read_bytes()).hexdigest()==record['sha256'],record['path']
x,s,y,tr,va,te=load_cora(R)
assert x.shape==(2708,1433) and (len(tr),len(va),len(te))==(140,500,1000)
assert not(set(tr.tolist())&set(va.tolist()) or set(tr.tolist())&set(te.tolist()) or set(va.tolist())&set(te.tolist()))
for lane,n in [('paper',100),('inductive',3)]:
 r=json.loads((R/f'_{lane}_l090_results.json').read_text())
 assert len(r['runs'])==n and [q['seed'] for q in r['runs']]==list(range(n))
 assert abs(np.mean([q['test_accuracy'] for q in r['runs']])-r['mean'])<1e-12
 for path,sha in r['sha256'].items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==sha,path
 if lane=='paper':
  for run in r['runs']:
   history=run['validation_loss'];assert len(history)==run['epochs']<=200
   for i in range(11,len(history)-1):assert history[i]<=np.mean(history[i-10:i])
   if len(history)<200:assert history[-1]>np.mean(history[-11:-1])
# Byte-for-byte visible PROVIDED model bodies, excluding intentional student tasks.
import nbformat
nb=nbformat.read(R/'0090-gnn-checkpoint.ipynb',as_version=4)
inline={}
for c in nb.cells:
 if c.cell_type=='code':
  for node in ast.parse(c.source).body:
   if isinstance(node,(ast.FunctionDef,ast.ClassDef)):inline[node.name]=ast.dump(node,include_attributes=False)
for file in ['gcn_l082.py','checkpoint_l090.py']:
 for node in ast.parse((R/'relkit'/file).read_text()).body:
  if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name not in ['propagate','masked_objective','eligible_neighbors','masked_mean','verdict']:
   assert inline[node.name]==ast.dump(node,include_attributes=False),node.name
r={'status':'PASS','pinned_upstream_files':len(manifest['files']),'data_shape':[2708,1433],'split':[140,500,1000],'fresh_paper_runs':100,'inductive_runs':3,'source_hashes_current':True,'inline_provided_ast_matches_canonical':True,'historical_framework_parity':'INCOMPARABLE'}
(R/'_audit_l090_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
