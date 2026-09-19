"""Final integrity audit: source bytes, full-run hashes, notebook TODOs and delivery files."""
import argparse,ast,hashlib,json,re
from pathlib import Path
import nbformat
LAB=Path(__file__).resolve().parent;ROOT=LAB.parent
parser=argparse.ArgumentParser();parser.add_argument('--fresh-cli',type=Path);args=parser.parse_args()
manifest=json.loads((LAB/'_sources_l086.json').read_text())
for row in manifest['files']:
 p=LAB/'sources/l086'/row['upstream'].replace('/','_')
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
run=json.loads((LAB/'_paper_l086_results.json').read_text())
assert len(run['runs'])==100 and [r['seed'] for r in run['runs']]==list(range(100))
for path,digest in run['sha256'].items():assert hashlib.sha256((LAB/path).read_bytes()).hexdigest()==digest
student=nbformat.read(LAB/'0086-pyg-fundamentals.ipynb',as_version=4)
solution=nbformat.read(LAB/'solutions/0086-pyg-fundamentals.ipynb',as_version=4)
assert sum('raise NotImplementedError' in c.source for c in student.cells if c.cell_type=='code')==3
assert not any('raise NotImplementedError' in c.source for c in solution.cells if c.cell_type=='code')
for nb in [student,solution]:
 nbformat.validate(nb)
 for c in nb.cells:
  if c.cell_type=='code':ast.parse(c.source)
assert all(not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
for n in ['_verify','_execution','_delivery','_clean_environment']:
 assert json.loads((LAB/(n+'_l086_results.json')).read_text())['status']=='PASS'
if args.fresh_cli:
 assert json.loads(args.fresh_cli.read_text())['runs']==[run['runs'][0]]
workflow=(ROOT/'.github/workflows/pages.yml').read_text();section=workflow.split('# L086 PyG',1)[1].split('# L085',1)[0]
for path in re.findall(r'labs/[\w/.-]+',section):
 assert (ROOT/path).exists(),path
source_paths=sorted(set([p for p in LAB.glob('*l086*') if p.is_file() and p.name!='_audit_l086_results.json']+[LAB/'relkit/pyg_l086.py']))
r={'status':'PASS','historical_source_files_verified':len(manifest['files']),'full_run_seeds_verified':100,'run_implementation_hashes_verified':True,'fresh_cli_seed0_exact':True if args.fresh_cli else 'NOT_CHECKED','student_live_todos':3,'solution_execution':'PASS','pages_copy_sources_exist':True,'local_sha256':{str(p.relative_to(LAB)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}}
(LAB/'_audit_l086_results.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k!='local_sha256'})
