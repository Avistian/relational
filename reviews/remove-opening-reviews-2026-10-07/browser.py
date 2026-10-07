from pathlib import Path
import functools,json,threading,sys
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'));from refresh_lesson_visuals import selected
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(ROOT)))
threading.Thread(target=server.serve_forever,daemon=True).start()
rows=[]
with sync_playwright() as pw:
 browser=pw.chromium.launch(args=['--disable-dev-shm-usage'])
 page=browser.new_page(viewport={'width':375,'height':950})
 for i,p in enumerate(selected()):
  errors=[];listener=lambda e:errors.append(str(e));page.on('pageerror',listener)
  page.goto(f'http://127.0.0.1:{server.server_port}/lessons/{p.name}',wait_until='load')
  assert page.locator('.retrieval-bank,[id="warmup"],[id$="-warmup"]').count()==0,p.name
  assert not errors,(p.name,errors)
  rows.append({'lesson':p.name,'opening_review':'absent','script_errors':errors})
  if p.name.split('-')[0] in ['0019','0055','0081','0082','0121','b20']:
   page.screenshot(path=str(OUT/(p.stem+'-375.png')))
  page.remove_listener('pageerror',listener)
  if (i+1)%40==0:print('Checked',i+1,'lessons',flush=True)
 browser.close()
server.shutdown();(OUT/'browser.json').write_text(json.dumps(rows,indent=2)+'\n')
print('PASS:',len(rows),'lessons without opening reviews or script errors')
