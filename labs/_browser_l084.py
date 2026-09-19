"""Test scalar intervention, keyboard reset, desktop/mobile and portable notebook figures."""
import json,os,threading
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from functools import partial
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
errors=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
 for width in [1100,375]:
  page=browser.new_page(viewport={'width':width,'height':950});page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto((ROOT/'lessons/0084-gat.html').as_uri())
  out=page.locator('#attention output');assert 'Weighted output: 1.575' in out.inner_text()
  page.locator('#attention input[type=checkbox]').uncheck();assert 'Weighted output: 0.731' in out.inner_text()
  page.locator('#attention button').click();assert 'Weighted output: 1.575' in out.inner_text()
  slider=page.locator('#attention input[type=range]');slider.fill('-2');slider.dispatch_event('input');assert 'Scores: [0.00, 1.00, -0.40]' in out.inner_text()
  slider.focus();page.keyboard.press('ArrowRight');assert slider.input_value()=='-1.9'
  page.locator('#attention button').click();assert slider.input_value()=='2'
  for img in page.locator('figure img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
  assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1'),'page overflow'
  page.locator('#attention').scroll_into_view_if_needed();page.screenshot(path=f'/tmp/l084-intervention-{width}.png')
  page.locator('img[alt^="Full Cora"]').scroll_into_view_if_needed();page.screenshot(path=f'/tmp/l084-architecture-{width}.png')
  page.emulate_media(media='print');assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1')
  page.close()
 page=browser.new_page(java_script_enabled=False);page.goto((ROOT/'labs/html/0084-gat.html').as_uri())
 figures=page.locator('img[src^="data:image/png;base64,"]').count();assert figures>=3
 for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
 page.goto((ROOT/'notebooks.html').as_uri());assert page.locator('#lab-84 a').count()==4
 class QuietHandler(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),partial(QuietHandler,directory=str(ROOT)))
 threading.Thread(target=server.serve_forever,daemon=True).start()
 page=browser.new_page()
 try:
  page.goto(f'http://127.0.0.1:{server.server_port}/index.html')
  tile=page.locator('#lesson-nav a[href="lessons/0084-gat.html"]');tile.wait_for(state='attached')
  assert 'GAT: learn which neighbors to weight' in tile.inner_text()
  tile.evaluate('(e)=>e.closest("details").open=true')
  tile.click();assert page.locator('h1').inner_text()=='GAT: learn which neighbors to weight'
 finally:
  server.shutdown();server.server_close()
 browser.close()
assert not errors,errors
r={'status':'PASS','widths':[1100,375],'controls':['edge deletion renormalizes','negative score slope','keyboard slider','reset'],'portable_figures':figures,'no_js_gallery':True,'http_manifest_tile_navigation':True,'page_errors':errors,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(ROOT/'labs/_browser_l084_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
