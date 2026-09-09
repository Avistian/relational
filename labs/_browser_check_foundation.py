"""Real Chromium checks against the copied Pages tree (requires Playwright)."""
import functools, json, threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from playwright.sync_api import sync_playwright
from _foundation_config import SLUGS


def check(stage=Path('/tmp/relational-foundation-pages')):
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(QuietHandler,directory=str(stage)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    records=[];shots=Path('/tmp/relational-foundation-screenshots');shots.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        for width in [1100,375]:
            page=browser.new_page(viewport={'width':width,'height':850},device_scale_factor=1)
            errors=[];missing=[]
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.on('response',lambda r:missing.append(r.url) if r.url.startswith(base) and r.status>=400 else None)
            for n in range(58,71):
                slug=f'{n:04}-{SLUGS[n]}'
                page.goto(f'{base}/lessons/{slug}.html',wait_until='networkidle')
                assert page.locator('h1').count()==1
                assert page.locator('#prediction .predict-options button').count()==3
                assert page.locator('#prediction .predict-reveal').is_disabled()
                page.locator('#prediction .predict-options button').first.click()
                page.locator('#prediction .predict-reveal').click()
                assert page.locator('#prediction .predict-outcome').inner_text()
                slider=page.locator('.foundation-exhibit input[type=range]')
                baseline=page.locator('.foundation-exhibit output').inner_text()
                states=[]
                for edge in ['min','max']:
                    slider.evaluate('(el,edge)=>{el.value=el[edge];el.dispatchEvent(new Event("input",{bubbles:true}))}',edge)
                    states.append(page.locator('.foundation-exhibit output').inner_text())
                assert states[0]!=states[1],(n,'control did not change computation')
                page.locator('.foundation-exhibit button').click()
                assert page.locator('.foundation-exhibit output').inner_text()==baseline
                page.evaluate('document.querySelectorAll("img").forEach(x=>x.loading="eager")')
                page.wait_for_function('Array.from(document.images).every(i=>i.complete && i.naturalWidth>0)')
                assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1'),(n,width,'page overflow')
                if n in [60,62,64,68,70]:
                    page.locator('.foundation-exhibit').screenshot(path=str(shots/f'l{n:03}-{width}-exhibit.png'))
                    if page.locator('.foundation-route').count():page.locator('.foundation-route').screenshot(path=str(shots/f'l{n:03}-{width}-architecture.png'))
                records.append(dict(lesson=n,width=width,widget='PASS',prediction='PASS',layout='PASS'))
            for n in range(58,71):
                slug=f'{n:04}-{SLUGS[n]}'
                page.goto(f'{base}/labs/html/{slug}.html',wait_until='domcontentloaded')
                assert page.locator('img').count()>=2
                page.wait_for_function('Array.from(document.images).every(i=>i.complete && i.naturalWidth>0)')
                assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1'),(n,width,'notebook page overflow')
            assert not errors,errors
            assert not missing,missing
            page.close()
        browser.close()
    server.shutdown()
    report=dict(status='PASS',browser='Chromium via Playwright',records=records,
                prepared_labs='13 pages at 1100px and 375px; all PNGs decoded; no page overflow',
                errors=[],missing_local_requests=[],live_colab='NOT_RUN')
    out=Path(__file__).resolve().parent/'_browser_foundation_results.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':check()
