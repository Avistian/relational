"""Export authored graph semantics without changing the desktop or notebook drawing."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'labs'))
import _solution_map_content as solution
import _architecture_071_090 as early
routes={}
for lesson,maps in solution.MAPS.items():
 for m in maps:
  routes['solution-'+m.key]=dict(title=m.title,scope=m.scope,nodes=[dict(id=n['key'],title=n['title'],body=' · '.join(n['lines']),role=n['role']) for n in m.nodes],edges=[dict(a=x['a'],b=x['b'],label=x['label'],dashed=x['control']) for x in m.edges])
def collect():
 for key,(title,subtitle,nodes,edges,footer) in early.SPECS.items():
  routes[key]=dict(title=title,scope=subtitle+' '+footer,nodes=[dict(id=n[0],title=n[3],body=n[4],role=n[5]) for n in nodes],edges=[dict(a=x[0],b=x[1],label=x[2] if len(x)>2 else '',dashed=len(x)>2) for x in edges])
collect()
import _architecture_091_100
collect()
files=['labs/_solution_map_content.py','labs/_architecture_071_090.py','labs/_architecture_091_100.py']
result=dict(sources={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},routes=routes)
p=ROOT/'assets/architecture-routes.json';data=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
if '--check' in sys.argv:assert p.read_text()==data,'Architecture route snapshot is stale'
else:p.write_text(data)
print(len(routes),'authored graphs exported')
