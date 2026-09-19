"""Headless Chromium interaction and actual layout checks at desktop and mobile sizes."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];SLUG='0071-vime-masked-tabular-ssl'
if __name__=='__main__':
    errors=[];screens=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
        for width in [1000,375]:
            page=browser.new_page(viewport={'width':width,'height':900})
            page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto((ROOT/'lessons'/f'{SLUG}.html').as_uri())
            assert page.locator('#mask-trace input').count()==3
            assert 'actually changed 1/3' in page.locator('.vime-readout').inner_text()
            page.locator('#mask-trace input').nth(2).check()
            assert 'actually changed 2/3' in page.locator('.vime-readout').inner_text()
            page.locator('#mask-trace button').click()
            assert 'actually changed 1/3' in page.locator('.vime-readout').inner_text()
            slider=page.locator('#consistency-trace input')
            for value,expected in [('0','0.20'),('.4','0.04'),('1','0.40')]:
                slider.evaluate('(e,value)=>{e.value=value;e.dispatchEvent(new Event("input"));}',value)
                assert 'Clean-anchor MSE = '+expected in page.locator('#consistency-trace output').inner_text()
            slider.focus();page.keyboard.press('ArrowLeft')
            assert 'Clean logit = 0.9' in page.locator('#consistency-trace output').inner_text()
            assert page.locator('#warmup').inner_text().strip()
            assert page.locator('#prediction').inner_text().strip()
            assert page.locator('#teachback textarea').count()==1
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1'), 'Page overflow'
            for el in page.locator('figure img').all():
                assert el.evaluate('(e)=>e.complete && e.naturalWidth>0')
            for selector,name in [('#mask-trace','corruption'),('#consistency-trace','consistency'),('figure','architecture')]:
                loc=page.locator(selector).nth(1 if name=='architecture' else 0)
                loc.scroll_into_view_if_needed()
                dest=f'/tmp/l071-{name}-{width}.png';page.screenshot(path=dest);screens.append(dest)
            page.emulate_media(media='print')
            assert page.locator('figure img').count()==5
            page.close()
        # Static course access remains when JavaScript is unavailable.
        page=browser.new_page(java_script_enabled=False)
        for filename in ['index.html','notebooks.html']:
            page.goto((ROOT/filename).as_uri())
            if filename=='index.html':assert page.locator('#lesson-nav').count()==1
            else:assert page.locator(f'a[href="lessons/{SLUG}.html"]').count()>=1
        page.goto((ROOT/'labs/html'/f'{SLUG}.html').as_uri())
        assert page.locator('#lab-exercises').count()==1
        assert page.locator('img[src^="data:image/png;base64,"]').count()==5
        # Verify the actual manifest-driven course navigation over HTTP as well.
        from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
        from functools import partial
        from threading import Thread
        server=ThreadingHTTPServer(('127.0.0.1',0),partial(SimpleHTTPRequestHandler,directory=str(ROOT)))
        worker=Thread(target=server.serve_forever,daemon=True);worker.start()
        try:
            live=browser.new_page()
            live.goto(f'http://127.0.0.1:{server.server_port}/index.html')
            live.locator(f'#lesson-nav a[href="lessons/{SLUG}.html"]').wait_for()
            live.goto(f'http://127.0.0.1:{server.server_port}/notebooks.html')
            live.locator(f'#nb-list a[href="labs/{SLUG}.ipynb"]').first.wait_for()
            live.close()
        finally:
            server.shutdown();server.server_close()
        browser.close()
    assert not errors,errors
    out=dict(status='PASS',widths=[1000,375],page_errors=errors,screenshots=screens,controls='collision toggles, reset, slider extremes, keyboard',no_javascript_navigation='PASS',http_manifest_navigation='PASS',live_colab='NOT_CHECKED')
    (ROOT/'labs/_browser_l071_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
