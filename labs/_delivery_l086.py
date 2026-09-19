"""Exercise browser controls and validate L086 links in copied Pages trees."""
import json,os,shutil,tempfile
from pathlib import Path
from urllib.parse import urlsplit,unquote
from html.parser import HTMLParser
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
errors=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
 page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto((ROOT/'lessons/0086-pyg-fundamentals.html').as_uri())
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900})
  slider=page.locator('#normalized input');slider.fill('20');slider.dispatch_event('input')
  assert '10.314796' in page.locator('#normalized output').inner_text()
  page.locator('#normalized button').click();assert '5.415816' in page.locator('#normalized output').inner_text()
  assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1'),'page overflow'
  page.screenshot(path=f'/tmp/l086-{width}.png',full_page=True)
 for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete && e.naturalWidth>0')
 page.goto((LAB/'html/0086-pyg-fundamentals.html').as_uri())
 assert page.locator('img[src^="data:image/png;base64,"]').count()==1
 for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete && e.naturalWidth>0')
 assert not errors,errors;browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
with tempfile.TemporaryDirectory(prefix='l086-pages-') as tmp:
 stage=Path(tmp)
 for folder in ['assets','lessons','reference']:shutil.copytree(ROOT/folder,stage/folder)
 (stage/'labs/solutions').mkdir(parents=True);(stage/'labs/html').mkdir();(stage/'labs/figures').mkdir()
 shutil.copytree(LAB/'figures/l086',stage/'labs/figures/l086')
 for f in ['index.html','notebooks.html']:shutil.copy2(ROOT/f,stage/f)
 for src in list(LAB.glob('*l086*'))+[LAB/'0086-pyg-fundamentals.ipynb']:
  if src.is_file():shutil.copy2(src,stage/'labs'/src.name)
 for folder in ['solutions','html']:shutil.copy2(LAB/folder/('0086-pyg-fundamentals.'+('ipynb' if folder=='solutions' else 'html')),stage/'labs'/folder)
 checked=0
 for path in [stage/'lessons/0086-pyg-fundamentals.html',stage/'reference/pyg-fundamentals.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   assert (path.parent/unquote(part.path)).resolve().exists(),(path,url)
   checked+=1
r={'status':'PASS','browser_widths':[1200,375],'widget_intervention_and_reset':'PASS','portable_notebook_image':'PASS','copied_pages_local_links':checked,'page_errors':errors,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(LAB/'_delivery_l086_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
