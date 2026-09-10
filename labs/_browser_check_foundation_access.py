"""Check the actual lesson → lab → exercise path, with JavaScript unavailable."""
import functools,json,threading
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from playwright.sync_api import sync_playwright
from _foundation_config import SLUGS
from _check_foundation_access import check as static_check

ROOT=Path(__file__).resolve().parents[1]


def check(stage=Path('/tmp/relational-foundation-pages')):
    static_check(stage)
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    base=f'http://127.0.0.1:{server.server_port}'
    shots=Path('/tmp/relational-access-screenshots');shots.mkdir(exist_ok=True)
    checked=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        for width in [1100,375]:
            context=browser.new_context(java_script_enabled=False,viewport={'width':width,'height':850})
            page=context.new_page()
            for n in range(58,71):
                slug=f'{n:04}-{SLUGS[n]}'
                page.goto(f'{base}/lessons/{slug}.html')
                launcher=page.locator('[data-lab-launch]')
                read=launcher.get_by_role('link',name='Open lab',exact=True)
                assert read.bounding_box()['y']<650,(n,width,'Lab link outside opening screen')
                assert page.locator('body').evaluate('(el)=>el.scrollWidth<=window.innerWidth+1'),(n,width)
                if n==58:page.screenshot(path=str(shots/f'lesson58-{width}.png'))
                read.click()
                assert page.url==f'{base}/labs/html/{slug}.html'
                toolbar=page.locator('[data-lab-launch]')
                assert toolbar.get_by_role('link',name='Run in Colab').is_visible()
                toolbar.get_by_role('link',name='Jump to exercises').click()
                assert page.url.endswith('#lab-exercises')
                assert page.locator('#lab-exercises').bounding_box()['y']<200,(n,width,'Exercise jump failed')
                if n==58:page.screenshot(path=str(shots/f'exercises58-{width}.png'))
                checked.append(dict(lesson=n,width=width,journey='PASS',javascript=False))
            page.goto(f'{base}/notebooks.html')
            assert page.locator('#foundation-materials a').bounding_box()['y']<650
            for n in range(58,71):assert page.locator(f'#lab-{n} a').count()>=3
            page.goto(f'{base}/labs/html/foundation-sequence.html')
            assert page.locator('.lab-package').count()==13
            assert page.get_by_role('link',name='Open lab 58',exact=True).bounding_box()['y']<650
            assert page.locator('body').evaluate('(el)=>el.scrollWidth<=window.innerWidth+1')
            page.screenshot(path=str(shots/f'directory-{width}.png'))
            context.close()
        # A fetch failure must preserve the statically generated notebook list.
        page=browser.new_page()
        page.route('**/lessons/manifest.json*',lambda route:route.abort())
        page.goto(f'{base}/notebooks.html',wait_until='networkidle')
        for n in range(58,71):assert page.locator(f'#lab-{n} a').count()>=3
        browser.close()
    server.shutdown()
    result=dict(status='PASS',journeys=checked,gallery_without_javascript='PASS',
                gallery_with_failed_manifest_request='PASS',package_directory='PASS',live_colab='NOT_RUN')
    (ROOT/'labs/_access_foundation_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: 26 lesson-to-lab-to-exercise journeys; static gallery; failed-fetch fallback')


if __name__=='__main__':check()
