"""Fresh browser verification for one lesson, used before advancing to the next."""
from pathlib import Path
import sys,json,functools,threading
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
slug=sys.argv[1];file=next((ROOT/'lessons').glob(slug+'*.html'))
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(ROOT)))
threading.Thread(target=server.serve_forever,daemon=True).start()
rows=[]
with sync_playwright() as pw:
 browser=pw.chromium.launch(args=['--disable-dev-shm-usage'])
 for width in [375,1200]:
  page=browser.new_page(viewport={'width':width,'height':950});errors=[]
  page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto(f'http://127.0.0.1:{server.server_port}/lessons/{file.name}')
  page.locator('img').evaluate_all('es=>es.forEach(e=>e.loading="eager")')
  page.wait_for_function('Array.from(document.images).every(i=>i.complete)')
  assert page.locator('img').evaluate_all('es=>es.every(i=>i.naturalWidth)'),file.name
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2'),file.name
  assert not page.locator('.retrieval-bank,[id="warmup"],[id$="-warmup"]').count()
  assert not errors,errors
  page.screenshot(path=str(OUT/f'{slug}-{width}.png'))
  rows.append({'width':width,'status':'PASS','script_errors':errors,'images':'PASS','overflow':'none','opening_reviews':'absent'})
  page.close()
 browser.close()
server.shutdown();(OUT/f'{slug}-browser.json').write_text(json.dumps(rows,indent=2)+'\n');print(slug,rows)
