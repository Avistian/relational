"""Readability, interaction and asset checks for every revised lesson and lab."""
import functools,json,threading
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
from playwright.sync_api import sync_playwright
from _architecture_revision import ROOT,PANELS


def check(stage=Path('/tmp/relational-foundation-pages')):
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)))
    threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}'
    records=[];errors=[];missing=[];shots=Path('/tmp/relational-depth-integrated');shots.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        for width in [1100,375]:
            page=browser.new_page(viewport={'width':width,'height':1000},device_scale_factor=1)
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.on('response',lambda r:missing.append(r.url) if r.url.startswith(base) and r.status>=400 else None)
            for n in range(47,71):
                lesson=next((stage/'lessons').glob(f'{n:04}-*.html'));slug=lesson.stem
                page.goto(f'{base}/lessons/{lesson.name}',wait_until='networkidle')
                assert page.locator('[data-lesson-depth]').count()>=4
                assert page.locator('.arch-atlas').count()==len(PANELS.get(n,[]))
                page.evaluate('document.querySelectorAll("img").forEach(x=>x.loading="eager")')
                page.wait_for_function('Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)')
                # Exercise existing native controls in the actual integrated lesson.
                controls=0
                for control in page.locator('input[type=range]').all():
                    original=control.input_value()
                    for edge in ['min','max']:
                        control.evaluate('(e,k)=>{e.value=e[k];e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}))}',edge);controls+=1
                    control.evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}))}',original)
                for select in page.locator('select').all():
                    original=select.input_value()
                    for value in select.locator('option').evaluate_all('(es)=>es.filter(x=>!x.disabled).map(x=>x.value)'):
                        select.select_option(value);controls+=1
                    select.select_option(original)
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),(n,width,'lesson overflow')
                for panel in PANELS.get(n,[]):
                    node=page.locator(f'#architecture-{panel["key"]}')
                    assert node.is_visible()
                    overflow=node.evaluate('''root=>[...root.querySelectorAll('h3,h4,p,li,td,th,code,.aa-op,.aa-equation')].filter(e=>{let r=e.getBoundingClientRect(),b=root.getBoundingClientRect();return r.left<b.left-1||r.right>b.right+1||e.scrollWidth>e.clientWidth+2}).map(e=>e.textContent.slice(0,80))''')
                    assert not overflow,(n,width,overflow)
                    node.screenshot(path=str(shots/f'{n:04}-{panel["key"]}-{width}.png'))
                page.locator('[data-lesson-depth]').first.screenshot(path=str(shots/f'{n:04}-prose-{width}.png'))
                # Prepared notebooks retain readable horizontally scrollable figures.
                page.goto(f'{base}/labs/html/{slug}.html',wait_until='domcontentloaded')
                page.wait_for_function('Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)')
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),(n,width,'prepared notebook overflow')
                assert page.locator('img').count()>=2
                responsive=page.get_by_role('link',name='Open the responsive diagram',exact=True)
                assert responsive.count()==len(PANELS.get(n,[])),(n,'notebook responsive diagram links')
                for i,panel in enumerate(PANELS.get(n,[])):
                    assert responsive.nth(i).get_attribute('href')==f'https://avistian.github.io/relational/labs/html/architecture-review/{n:04}-{panel["key"]}.html'
                assert 'NotImplementedError' in page.locator('body').inner_text() or '____' in page.locator('body').inner_text()
                if n in [48,61,66]:page.screenshot(path=str(shots/f'{n:04}-notebook-{width}.png'))
                records.append(dict(lesson=n,width=width,control_states=controls,lesson_page='PASS',prepared_lab='PASS'))
            page.close()
        # Static architecture gallery and a keyboard path to a complete study.
        page=browser.new_page(viewport={'width':375,'height':900},java_script_enabled=False)
        page.goto(f'{base}/labs/html/architecture-review/index.html')
        assert page.locator('article').count()==16
        page.locator('article h2 a').first.focus();page.keyboard.press('Enter')
        page.wait_for_url('**/0047-saint.html');assert page.locator('.arch-atlas').is_visible()
        assert not errors,errors
        assert not missing,missing
        browser.close()
    server.shutdown()
    report=dict(status='PASS',records=records,architecture_panels=16,errors=errors,missing_local_assets=missing,
                screenshots=str(shots),gallery='JavaScript-disabled and keyboard navigation PASS',
                live_colab='NOT_RUN',coverage='24 lessons and 24 prepared labs at desktop and mobile; all native range/select states exercised')
    (ROOT/'labs/_depth_browser_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


if __name__=='__main__':check()
