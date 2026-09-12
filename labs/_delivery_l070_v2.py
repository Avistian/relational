"""Read-only L070 final package and current-execution identity checks."""
import hashlib,json,re
from pathlib import Path
import nbformat
from bs4 import BeautifulSoup
from _check_quality_audit import check_lesson,plain
from _build_l070 import TASKS,source,parts,markdown2html_mistune
ROOT=Path(__file__).parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check():
 r=check_lesson(70,require_review=True)
 student=nbformat.read(ROOT/'0070-foundation-model-checkpoint.ipynb',4);teacher=nbformat.read(ROOT/'solutions/0070-foundation-model-checkpoint.ipynb',4)
 manuscript=plain(BeautifulSoup((ROOT.parent/'lessons/0070-foundation-model-checkpoint.html').read_text(),'html.parser'))
 markdown=[c.source for c in student.cells if c.cell_type=='markdown'];fragments=0
 for fragment in parts():
  if re.fullmatch(r'<!--(?:figure:\w+|results-table|analysis-text)-->',fragment):continue
  if not fragment.strip():continue
  assert fragment.replace('../labs/','').strip() in markdown,'Canonical manuscript absent from portable notebook'
  assert plain(BeautifulSoup(markdown2html_mistune(fragment),'html.parser')) in manuscript,'Canonical manuscript absent from lesson'
  fragments+=1
 for name,*_ in TASKS:
  a=[c for c in student.cells if c.cell_type=='code' and c.source.startswith('# TODO — '+name+'\n')]
  b=[c for c in teacher.cells if c.cell_type=='code' and c.source.startswith('# TODO — '+name+'\n')]
  assert len(a)==len(b)==1 and 'raise NotImplementedError' in a[0].source
  assert b[0].source=='# TODO — '+name+'\n'+source(name)
 execution=json.loads((ROOT/'_execution_l070_v2_results.json').read_text());assert execution['status']=='PASS'
 assert execution['notebook_sha256']==sha(execution['notebook_path'])
 assert execution['executor_sha256']==sha(ROOT/'_execute_l070.py')
 assert execution['notebook_code_sha256']==hashlib.sha256(json.dumps([c.source for c in teacher.cells if c.cell_type=='code'],sort_keys=True).encode()).hexdigest()
 assert execution['exit_sha256']==sha(execution['exit_path'])
 live=json.loads(Path(execution['exit_path']).read_text());assert live['status']=='COMPLETE'
 assert live['input_preparation']['helper_sha256']==sha(ROOT/'_prepare_l070_v2.py')
 assert live['input_preparation']['manifest_sha256']==sha(ROOT/'data/l070-v2/manifest.json')
 r.update(canonical_fragments=fragments,live_tasks=len(TASKS),execution=execution,scope='Final current files, notebook prose and live EXIT identities; no live Colab or publication claim')
 return r
if __name__=='__main__':
 r=check();(ROOT/'_delivery_l070_v2_results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
