"""Real Chromium WL interactions, portable notebook and actual copied Pages staging."""
import json,os,re,subprocess,tempfile
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0088-graph-classification'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
errors=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto((ROOT/'lessons'/f'{SLUG}.html').as_uri())
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900})
  for mode in ['path','collision']:
   page.locator('#wl-refinement select').select_option(mode)
   for step in range(5):
    slider=page.locator('#wl-refinement input');slider.fill(str(step));slider.dispatch_event('input')
    expected='different histograms' if mode=='path' and step>0 else 'same histograms'
    assert expected in page.locator('#wl-refinement output').inner_text()
    assert page.locator('#wl-refinement svg').count()==2
   page.locator('#wl-refinement').screenshot(path=f'/tmp/l088-{mode}-{width}.png')
  page.locator('#wl-refinement button').click();assert page.locator('#wl-refinement input').input_value()=='0'
  slider.focus();page.keyboard.press('ArrowRight');assert slider.input_value()=='1'
  page.evaluate("document.querySelector('#wl-refinement').wlAPI.setState('path',99)");assert slider.input_value()=='4'
  page.locator('#wl-refinement button').click()
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'page overflow'
  page.screenshot(path=f'/tmp/l088-page-{width}.png',full_page=False)
 for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete && e.naturalWidth>0')
 assert page.locator('#warmup').inner_text().strip()
 assert page.locator('#prediction').inner_text().strip() and page.locator('#teachback textarea').count()==1
 page.goto((LAB/'html'/f'{SLUG}.html').as_uri())
 count=page.locator('img[src^="data:image/png;base64,"]').count();assert count==4
 for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete && e.naturalWidth>0')
 assert not errors,errors;browser.close()
nb=nbformat.read(LAB/f'{SLUG}.ipynb',as_version=4)
assert sum('raise NotImplementedError' in c.source for c in nb.cells if c.cell_type=='code')==3
assert not any('from relkit' in c.source for c in nb.cells if c.cell_type=='code')
assert 'attachment:' not in json.dumps(nb)
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
(LAB/'_delivery_l088_results.json').write_text(json.dumps({'status':'RUNNING'})+'\n')
workflow=(ROOT/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
lines=[x[10:] for x in block.splitlines() if x.startswith('          ')]
lines=[x for x in lines if not x.startswith(('VER=','sed -i'))]
with tempfile.TemporaryDirectory(prefix='l088-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=ROOT,check=True,capture_output=True,text=True)
 checked=0
 for path in [stage/'lessons'/f'{SLUG}.html',stage/'reference/graph-classification.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   dest=(path.parent/unquote(part.path)).resolve();assert dest.exists(),(path,url)
   assert not dest.is_symlink();checked+=1
 assert (stage/'labs/solutions'/f'{SLUG}.ipynb').exists()
 assert (stage/'labs/relkit/gin_l088.py').exists()
 assert (stage/'labs/data/l088/MUTAG.txt').exists()
 assert (stage/'labs/sources/l088/models/graphcnn.py').exists()
 count_traces=len(list((stage/'labs/results/l088/paper').glob('*.npz')))
r={'status':'PASS','browser_widths':[1200,375],'joint_WL_rounds_both_pairs':'PASS','keyboard_reset_and_clamp':'PASS','portable_figures':count,'student_TODOs':3,'actual_pages_copy_commands':'PASS','copied_local_links':checked,'raw_prediction_traces_staged':count_traces,'page_errors':errors,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(LAB/'_delivery_l088_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
