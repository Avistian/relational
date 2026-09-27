"""Final HTTP rendering checks for every reviewed lesson, including the new L111."""
import functools,json,os,threading
from pathlib import Path
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from playwright.sync_api import sync_playwright
from _review_pages_103_135 import stage_site,audit
from _gallery_delivery import reveal_gallery_link
R=Path(__file__).resolve().parents[1];D=R/'reviews/lessons-103-135'
os.environ['LD_LIBRARY_PATH']='/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu:'+os.environ.get('LD_LIBRARY_PATH','')
stage=stage_site();assert audit(stage)['status']=='PASS'
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();base=f'http://127.0.0.1:{server.server_port}'
rows=[];errors=[];http_errors=[];screens=Path('/tmp/review-103-135-final');screens.mkdir(exist_ok=True)
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
  page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('response',lambda r:http_errors.append([r.status,r.url]) if r.url.startswith(base) and r.status>=400 else None)
  page.goto(base+'/index.html')
  for n in range(103,136):
   path=next((stage/'lessons').glob(f'{n:04d}-*.html'));reveal_gallery_link(page,f'a[href="lessons/{path.name}"]')
  for n in range(103,136):
   path=next((stage/'lessons').glob(f'{n:04d}-*.html'));url=base+'/lessons/'+path.name
   for width in [1200,375]:
    page.set_viewport_size(dict(width=width,height=900));page.goto(url);page.wait_for_timeout(70)
    assert page.locator('h1').is_visible();assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),(n,width,'overflow')
    imgs=page.locator('figure img');assert imgs.evaluate_all('(xs)=>xs.every(x=>x.complete&&x.naturalWidth>0)'),(n,width,'images')
    reminder=page.locator('.sequence-context summary')
    if reminder.count():reminder.focus();page.keyboard.press('Enter');assert page.locator('.sequence-context details').get_attribute('open') is not None
    if n in [103,111,117,125,132,135]:page.screenshot(path=str(screens/f'{n}-{width}.png'))
    rows.append(dict(lesson=n,width=width,status='PASS',figures=imgs.count(),no_page_overflow=True))
   page.emulate_media(media='print');assert page.locator('h1').is_visible();assert page.locator('figure img').evaluate_all('(xs)=>xs.every(x=>x.complete&&x.naturalWidth>0)');page.emulate_media(media='screen')
   context=browser.new_context(java_script_enabled=False,viewport=dict(width=375,height=900));p=context.new_page();p.goto(url)
   assert p.locator('h1').is_visible();assert not p.evaluate('document.documentElement.scrollWidth>innerWidth+1'),(n,'noJS overflow');context.close()
  browser.close()
finally:server.shutdown();server.server_close();thread.join()
report=dict(status='PASS' if not errors and not http_errors else 'FAIL',views=rows,print_lessons=33,nojs_lessons=33,manifest_links=33,javascript_errors=errors,http_errors=http_errors,stage=str(stage),screenshots=str(screens))
(D/'browser.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='views'})
assert report['status']=='PASS'
