from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
import functools,threading,json
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[2];out=Path(__file__).parent
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(root)));threading.Thread(target=server.serve_forever,daemon=True).start()
with sync_playwright() as pw:
 browser=pw.chromium.launch(args=['--disable-dev-shm-usage']);page=browser.new_page(viewport={'width':375,'height':950})
 page.goto(f'http://127.0.0.1:{server.server_port}/lessons/0002-design-matrix-leakage.html')
 panel=page.locator('.leak-viz');assert '85' in panel.inner_text();panel.get_by_role('checkbox').check();assert '55' in panel.inner_text();panel.get_by_role('button',name='E — days_since_last_order').click();assert '25' in panel.inner_text();panel.get_by_role('checkbox').uncheck();assert '8' in panel.inner_text()
 panel.screenshot(path=str(out/'0002-widget-mobile.png'))
 for prefix,selector in [('0050','#pairing-trace'),('0051','#smooth')]:
  path=next((root/'lessons').glob(prefix+'*.html'));page.goto(f'http://127.0.0.1:{server.server_port}/lessons/{path.name}');page.locator(selector).screenshot(path=str(out/f'{prefix}-trace-mobile.png'))
 browser.close()
server.shutdown();(out/'changed-widgets.json').write_text(json.dumps({'status':'PASS','L002':'delay checkbox changes spend 85 to 55 and recency 8 to 25','captures':['0050-trace-mobile.png','0051-trace-mobile.png']},indent=2)+'\n')
