"""Real Chromium interactions and copied staging using the actual Pages copy commands."""
import hashlib,json,os,re,subprocess,tempfile
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0087-link-prediction'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
errors=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto((ROOT/'lessons'/f'{SLUG}.html').as_uri())
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900})
  assert 'H = [0,1,0]' in page.locator('#edge-leak output').inner_text()
  page.locator('#edge-leak input').check();assert 'H = [0,1,1]' in page.locator('#edge-leak output').inner_text()
  page.locator('#edge-leak button').click();assert not page.locator('#edge-leak input').is_checked()
  slider=page.locator('#edge-rank input');expected=[4,3.5,3,3,3,2.5,1.5,1,1]
  for i,r in enumerate(expected):
   slider.fill(str(i));slider.dispatch_event('input');assert f'· rank {r} ·' in page.locator('#edge-rank output').inner_text()
  page.locator('#edge-rank button').click();assert 'reciprocal 0.400' in page.locator('#edge-rank output').inner_text()
  slider.focus();page.keyboard.press('ArrowRight');assert slider.input_value()=='6'
  page.locator('#edge-rank button').click()
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'page overflow'
  page.locator('#edge-leak').screenshot(path=f'/tmp/l087-leak-{width}.png');page.locator('#edge-rank').screenshot(path=f'/tmp/l087-rank-{width}.png')
  page.screenshot(path=f'/tmp/l087-page-{width}.png',full_page=False)
 for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete && e.naturalWidth>0')
 assert page.locator('#warmup').inner_text().strip()
 assert page.locator('#prediction').inner_text().strip() and page.locator('#teachback textarea').count()==1
 page.goto((LAB/'html'/f'{SLUG}.html').as_uri());assert page.locator('img[src^="data:image/png;base64,"]').count()==6
 for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete && e.naturalWidth>0')
 assert not errors,errors;browser.close()
nb=nbformat.read(LAB/f'{SLUG}.ipynb',as_version=4)
assert sum('raise NotImplementedError' in c.source for c in nb.cells if c.cell_type=='code')==3
assert not any('from relkit' in c.source for c in nb.cells if c.cell_type=='code')
assert 'attachment:' not in json.dumps(nb)
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
# Stage fresh copies using the actual build shell. Skip only version-in-place edits.
(LAB/'_delivery_l087_results.json').write_text(json.dumps({'status':'RUNNING'})+'\n')
workflow=(ROOT/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
lines=[x[10:] for x in block.splitlines() if x.startswith('          ')]
lines=[x for x in lines if not x.startswith(('VER=','sed -i'))]
with tempfile.TemporaryDirectory(prefix='l087-pages-') as tmp:
 stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
 subprocess.run(['bash','-c',script],cwd=ROOT,check=True,capture_output=True,text=True)
 checked=0
 for path in [stage/'lessons'/f'{SLUG}.html',stage/'reference/link-prediction.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   dest=(path.parent/unquote(part.path)).resolve();assert dest.exists(),(path,url)
   assert not dest.is_symlink();checked+=1
 assert (stage/'labs/solutions'/f'{SLUG}.ipynb').exists()
 assert len(list((stage/'labs/results/l087/paper').glob('*.npz')))==80
r={'status':'PASS','browser_widths':[1200,375],'leak_intervention_reset':'PASS','all_nine_rank_states_keyboard_reset':'PASS','portable_figures':6,'student_TODOs':3,'actual_pages_copy_commands':'PASS','copied_local_links':checked,'raw_split_artifacts_staged':80,'page_errors':errors,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(LAB/'_delivery_l087_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
