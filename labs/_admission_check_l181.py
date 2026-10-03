"""Test both admission entry points without making a Modal API call."""
import ast,json,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
result=subprocess.run([sys.executable,str(P/'_run_l181.py')],capture_output=True,text=True)
assert result.returncode==2,result.stderr
receipt=json.loads(result.stdout);assert receipt['admission']=='BLOCKED' and receipt['cloud_usd']==0
wrapper=R/'modal/l181_paper_repro.py';tree=ast.parse(wrapper.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');fn.decorator_list=[]
namespace={'Path':Path,'subprocess':subprocess,'sys':sys,'__file__':str(wrapper)}
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(wrapper),'exec'),namespace)
try:namespace['main']()
except SystemExit as e:assert e.code==2
else:raise AssertionError('Wrapper admitted a blocked run')
(P/'_admission_l181_results.json').write_text(json.dumps(dict(status='PASS',direct_gate_exit=2,modal_local_body_exit=2,modal_service_invocation='NOT_RUN',paid_dispatch='NOT_RUN',cloud_usd=0),indent=2)+'\n')
print('Both local gate bodies rejected; Modal service not invoked')
