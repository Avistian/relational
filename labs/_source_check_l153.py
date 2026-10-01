"""Pin integrity plus full class AST identity and passive training instrumentation."""
import ast,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;D=P/'sources/l153'
m=json.loads((D/'manifest.json').read_text())
for name,info in m['sources'].items():assert hashlib.sha256((D/name).read_bytes()).hexdigest()==info['sha256'],name
ours={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((P/'relkit/recommendation_model_l153.py').read_text()).body if isinstance(n,ast.ClassDef)}
checked=[]
for name in ['nn.py','model.py']:
 for n in ast.parse((D/name).read_text()).body:
  if isinstance(n,ast.ClassDef):assert ours[n.name]==ast.dump(n,include_attributes=False);checked.append(n.name)
s=(P/'_run_l153.py').read_text();node=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='instrument');ns={};exec(ast.get_source_segment(s,node),ns)
for pilot in [False,True]:
 transformed=ns['instrument']((D/'gnn_link.py').read_text(),pilot);compile(transformed,'instrumented','exec')
 assert 'if val_metrics[tune_metric] >= best_val_metric:' in transformed
 assert 'F.softplus(-diff_score).mean()' in transformed
 assert ('test_pred = test(' in transformed)==(not pilot)
 assert ('if steps >= 32:' in transformed)==pilot
 assert ('if steps > args.max_steps_per_epoch:' in transformed)==(not pilot)
r=dict(status='PASS',source_files=len(m['sources']),identical_classes=checked,pilot_test_evaluation='ABSENT',release_selection_ties='LATEST',release_training_limit=2001)
(P/'_source_check_l153_results.json').write_text(json.dumps(r,indent=2));print(r)
