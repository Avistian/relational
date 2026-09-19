"""Desktop/mobile behavior and readability capture for the row graph."""
import json,os
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];SLUG='0074-carte-cross-table-transfer'
def run():
    libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
    if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
    errors=[];screens=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
        for width in [1100,375]:
            page=browser.new_page(viewport={'width':width,'height':950})
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto((ROOT/'lessons'/f'{SLUG}.html').as_uri())
            out=page.locator('#cell-graph output')
            assert 'center [10, 0]' in out.inner_text()
            page.locator('#cell-graph input[type=range]').focus();page.keyboard.press('ArrowRight')
            assert 'center [14.5, 0.5]' in out.inner_text()
            page.locator('#cell-graph input[type=checkbox]').check();assert 'center [2, -2]' in out.inner_text()
            page.locator('#cell-graph button').click();assert 'center [10, 0]' in out.inner_text()
            page.locator('#cell-graph select').select_option('1,3');assert 'center [2, 8]' in out.inner_text()
            page.locator('#cell-graph button').click()
            assert page.locator('#warmup').inner_text().strip();assert page.locator('#prediction').inner_text().strip()
            assert page.locator('#teachback textarea').count()==1
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'Page overflow'
            if width==375:
                region=page.locator('#cell-graph .figure-scroll');region.focus();page.keyboard.press('ArrowRight');page.wait_for_timeout(250)
                assert region.evaluate('(e)=>e.scrollLeft>0')
                region.evaluate('(e)=>e.scrollLeft=0')
            for img in page.locator('figure img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
            for selector,name in [('#cell-graph','widget'),('figure','graph'),('figure:nth-of-type(2)','architecture')]:
                page.locator(selector).first.scroll_into_view_if_needed();dest=f'/tmp/l074-{name}-{width}.png';page.screenshot(path=dest);screens.append(dest)
            page.emulate_media(media='print');assert page.locator('figure img').count()==5
            page.close()
        page=browser.new_page(java_script_enabled=False)
        for filename in ['index.html','notebooks.html']:
            page.goto((ROOT/filename).as_uri());assert page.locator(f'a[href="lessons/{SLUG}.html"]').count()>=1
        page.goto((ROOT/'labs/html'/f'{SLUG}.html').as_uri());assert page.locator('img[src^="data:image/png;base64,"]').count()==5
        assert page.locator('#lab-exercises').count()==1
        browser.close()
    assert not errors,errors
    report={'status':'PASS','widths':[1100,375],'javascript_errors':errors,'screenshots':screens,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
    (ROOT/'labs/_browser_l074_results.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':run()
