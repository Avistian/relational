"""Browser interactions, notebook portability and actual copied Pages-stage links."""
import ast,json,os,re,subprocess,tempfile
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0089-sampling-at-scale'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
errors=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto((ROOT/'lessons'/f'{SLUG}.html').as_uri())
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900})
  for q,n,e in [(1,2,1),(2,4,3),(3,6,5)]:
   page.locator('#cluster-sampling select').select_option(str(q))
   output=page.locator('#cluster-sampling output').inner_text()
   assert f'{n} nodes' in output and f'{e} retained edges' in output
   assert ('= 2.5.' if q==1 else '= 3.') in output
   assert page.locator('#cluster-sampling svg circle').count()==6
   assert page.locator('#cluster-sampling svg').evaluate('(svg)=>Array.from(svg.querySelectorAll("text,circle")).every(e=>{const b=e.getBBox();return b.x>=0 && b.y>=0 && b.x+b.width<=600 && b.y+b.height<=125;})')
   page.locator('#cluster-sampling').screenshot(path=f'/tmp/l089-q{q}-{width}.png')
  page.locator('#cluster-sampling button').click();assert page.locator('#cluster-sampling select').input_value()=='1'
  page.locator('#cluster-sampling select').focus();page.keyboard.press('ArrowDown');assert page.locator('#cluster-sampling select').input_value()=='2'
  page.evaluate("document.querySelector('#cluster-sampling').clusterAPI.setState(99)");assert page.locator('#cluster-sampling select').input_value()=='3'
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'Page overflows viewport'
  page.screenshot(path=f'/tmp/l089-page-{width}.png',full_page=False)
 for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete && e.naturalWidth>0')
 assert page.locator('#warmup').inner_text().strip()
 assert page.locator('#prediction').inner_text().strip()
 assert page.locator('#teachback textarea').count()==1
 page.goto((LAB/'html'/f'{SLUG}.html').as_uri())
 count=page.locator('img[src^="data:image/png;base64,"]').count();assert count==4
 for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete && e.naturalWidth>0')
 assert not errors,errors;browser.close()
nb=nbformat.read(LAB/f'{SLUG}.ipynb',as_version=4);sol=nbformat.read(LAB/'solutions'/f'{SLUG}.ipynb',as_version=4)
assert sum('raise NotImplementedError' in c.source for c in nb.cells if c.cell_type=='code')==3
assert not any('from relkit' in c.source for c in nb.cells if c.cell_type=='code')
assert not any('raise NotImplementedError' in c.source for c in sol.cells if c.cell_type=='code')
assert 'attachment:' not in json.dumps(nb)
canonical=ast.parse((LAB/'relkit/cluster_gcn_l089.py').read_text())
solution_nodes={n.name:ast.dump(n,include_attributes=False) for c in sol.cells if c.cell_type=='code' for n in ast.parse(c.source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for n in canonical.body:
 if isinstance(n,(ast.FunctionDef,ast.ClassDef)): assert solution_nodes[n.name]==ast.dump(n,include_attributes=False),n.name
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
(LAB/'_delivery_l089_results.json').write_text(json.dumps({'status':'RUNNING'})+'\n')
workflow=(ROOT/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
lines=[x[10:] for x in block.splitlines() if x.startswith('          ')]
lines=[x for x in lines if not x.startswith(('VER=','sed -i'))]
with tempfile.TemporaryDirectory(prefix='l089-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=ROOT,check=True,capture_output=True,text=True)
 checked=0
 for path in [stage/'lessons'/f'{SLUG}.html',stage/'reference/cluster-gcn.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   dest=(path.parent/unquote(part.path)).resolve();assert dest.exists(),(path,url)
   assert not dest.is_symlink();checked+=1
 assert (stage/'labs/solutions'/f'{SLUG}.ipynb').exists()
 assert (stage/'labs/relkit/cluster_gcn_l089.py').exists()
 assert (stage/'labs/sources/l089/run_ppi.sh').exists()
 traces=len(list((stage/'labs/results/l089/teaching').glob('*/test_predictions.npz')));assert traces==12
r={'status':'PASS','browser_widths':[1200,375],'induced_union_arithmetic_all_states':'PASS','keyboard_reset_clamp':'PASS','portable_figures':count,'student_TODOs':3,'actual_pages_copy_commands':'PASS','copied_local_links':checked,'prediction_traces_staged':traces,'page_errors':errors,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(LAB/'_delivery_l089_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
