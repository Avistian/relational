"""Inspect revised computation figures and portable previews at reading widths."""
from pathlib import Path
import functools,json,threading
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
class Quiet(SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(ROOT)))
threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}'
rows=[]
with sync_playwright() as pw:
    browser=pw.chromium.launch(args=['--disable-dev-shm-usage'])
    for lesson,slug,figure in [(81,'0081-mpnn-framework','trace'),(82,'0082-gcn','channels')]:
        for width in [375,1200]:
            page=browser.new_page(viewport={'width':width,'height':950},reduced_motion='reduce')
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(base+'/lessons/'+slug+'.html');page.locator('img').evaluate_all('es=>es.forEach(e=>e.loading="eager")')
            page.wait_for_function('Array.from(document.images).every(i=>i.complete && i.naturalWidth)')
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2')
            image=page.locator(f'img[src$="l{lesson:03}/{figure}.png"]');box=image.locator('xpath=ancestor::figure')
            box.screenshot(path=str(OUT/f'l{lesson}-{width}.png'))
            if width==375:
                native=box.locator('.mpnn-mobile-trace')
                assert native.is_visible()
                assert native.evaluate('e=>e.scrollWidth<=e.clientWidth+2')
                assert native.locator('td').evaluate_all('es=>es.every(e=>parseFloat(getComputedStyle(e).fontSize)>=15)')
            # No JavaScript / print readers still have the worked calculation.
            page.emulate_media(media='print');box.screenshot(path=str(OUT/f'l{lesson}-print-{width}.png'))
            assert image.evaluate('e=>e.clientWidth<=e.parentElement.clientWidth+2')
            page.emulate_media(media='screen')
            page.goto(base+'/labs/html/'+slug+'.html')
            page.wait_for_function('Array.from(document.images).every(i=>i.complete && i.naturalWidth)')
            assert page.locator('img[src^="data:image/png"]').count()>0
            assert not errors,errors
            rows.append({'lesson':lesson,'width':width,'images':'PASS','native_mobile_trace':'PASS' if width==375 else 'not applicable','print_fit':'PASS','portable_preview':'PASS','browser_errors':errors})
            page.close()
        page=browser.new_page(java_script_enabled=False,viewport={'width':375,'height':950})
        page.goto(base+'/lessons/'+slug+'.html')
        assert page.locator(f'img[src$="l{lesson:03}/{figure}.png"]').count()==1
        page.close()
    browser.close()
server.shutdown();(OUT/'rendering.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
