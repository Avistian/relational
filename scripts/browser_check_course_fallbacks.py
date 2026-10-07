"""Verify whole-course no-JS reading and printable computation figures."""
import sys,json,threading,functools
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[1]
from refresh_lesson_visuals import selected,key_for
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*a):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(root)));threading.Thread(target=server.serve_forever,daemon=True).start()
out=root/'reviews/course-visual-quality-2026-10-07';rows=[]
with sync_playwright() as p:
 browser=p.chromium.launch();page=browser.new_page(java_script_enabled=False,viewport={'width':375,'height':1000})
 for path in selected():
  page.emulate_media(media='screen');page.set_viewport_size({'width':375,'height':1000});page.goto(f'http://127.0.0.1:{server.server_port}/lessons/{path.name}');cv=page.locator('.course-visual');maps=page.locator('.responsive-map');errors=[]
  if cv.count() and not cv.is_visible():errors.append('no-JS trace hidden')
  if any(not x.is_visible() for x in maps.all()):errors.append('no-JS mobile map hidden')
  if page.evaluate('document.documentElement.scrollWidth>innerWidth+2'):errors.append('no-JS overflow')
  if key_for(path) in ['16','43','99','143','175','b24'] and cv.count():cv.screenshot(path=str(out/'after'/f'{key_for(path)}-nojs.png'))
  page.set_viewport_size({'width':1200,'height':1000});page.emulate_media(media='print')
  if cv.count():
   if not cv.locator('.cv-print-answer').is_visible():errors.append('print answer hidden')
   if key_for(path) in ['43','143']:cv.screenshot(path=str(out/'after'/f'{key_for(path)}-print.png'))
  for fig in page.locator('figure[data-responsive-map]').all():
   if not fig.locator('img,svg').first.is_visible():errors.append('print architecture hidden')
  rows.append(dict(lesson=path.name,trace=cv.count(),maps=maps.count(),issues=errors))
  if len(rows)%40==0:print(len(rows),'fallbacks checked',flush=True)
 browser.close()
(out/'fallbacks.json').write_text(json.dumps(rows,indent=2)+'\n');print('ISSUES',[(x['lesson'],x['issues']) for x in rows if x['issues']]);server.shutdown()
assert not any(x['issues'] for x in rows)
