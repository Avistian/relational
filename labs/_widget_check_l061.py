"""Exercise the actual L061 GP widget at desktop and mobile widths."""
import functools,json,math,threading
from pathlib import Path
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(ROOT)));threading.Thread(target=server.serve_forever,daemon=True).start()
results=[];errors=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True)
 for width in [900,375]:
  page=browser.new_page(viewport={'width':width,'height':1200});page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto(f'http://127.0.0.1:{server.server_port}/lessons/0061-prior-data-fitted-networks.html',wait_until='networkidle')
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
  for location in [-2,0,.2,.8,1,3]:
   state=page.evaluate('(x)=>window.l061GP.setQuery(x)',location);x=min(1,max(0,location));k=math.exp(-(x-.2)**2/(2*.6**2))
   assert abs(state['mean']-k/1.0001)<1e-12 and abs(state['variance']-(1.0001-k*k/1.0001))<1e-12
   results.append(dict(width=width,**state))
  page.locator('#gp-reset').click();assert page.locator('#gp-query').input_value()=='0.8'
  page.locator('#gp-query').focus();page.keyboard.press('ArrowLeft');assert page.locator('#gp-query').input_value()=='0.79'
  page.locator('#gp-reset').click();page.locator('#gp-conditioning-viz').screenshot(path=f'/tmp/l061-widget-{width}.png')
  page.locator('.arch-atlas').screenshot(path=f'/tmp/l061-architecture-{width}.png')
  page.emulate_media(media='print');assert page.locator('.arch-atlas').is_visible()
  page.close()
 browser.close()
server.shutdown();assert not errors,errors
result=dict(status='PASS',states=results,keyboard='PASS',reset='PASS',page_overflow='NONE',console_errors=errors,print='architecture visible; no live Colab claim')
(ROOT/'labs/_widget_l061_v2_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'PASS','states':len(results)}))
