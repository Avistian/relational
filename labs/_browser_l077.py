"""Actual desktop/mobile arithmetic controls, portable figures and print checks."""
import json,os
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];SLUG='0077-single-table-ceiling'
def run():
    libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
    if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
    errors=[];screens=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
        for width in [1100,375]:
            page=browser.new_page(viewport={'width':width,'height':950});page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto((ROOT/'lessons'/f'{SLUG}.html').as_uri())
            w=page.locator('#bound-viz');out=w.locator('output')
            assert 'Ceiling 50%' in out.inner_text()
            w.locator('input').focus();page.keyboard.press('End');assert 'Ceiling 90%' in out.inner_text()
            page.keyboard.press('Home');assert 'Ceiling 90%' in out.inner_text()
            for n in range(1,10):assert page.evaluate('(n)=>CeilingViz.bound(10-n,n)',n)==max(10-n,n)/10
            w.locator('button').click();assert 'Ceiling 50%' in out.inner_text()
            w=page.locator('#repair-viz');w.locator('select').select_option('delta')
            assert 'ceiling 100%' in w.locator('output').inner_text()
            w.locator('button').click();assert 'ceiling 50%' in w.locator('output').inner_text()
            w=page.locator('#reach-viz')
            for hops,text in [('0','0 hops:'),('1','1 hop:'),('2','2 hops:')]:
                w.locator('select').select_option(hops);assert text in w.locator('output').inner_text()
            w.locator('button').click();assert '1 hop:' in w.locator('output').inner_text()
            recall=page.locator('#flatten-recall');assert recall.locator('svg').count()>0
            before=recall.inner_text();recall.locator('button').first.click();assert recall.inner_text()!=before
            recall.locator('button').first.click()
            assert page.locator('#warmup').inner_text().strip()
            assert page.locator('#prediction').inner_text().strip()
            assert page.locator('#teachback textarea').count()==1
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'Page overflow'
            for img in page.locator('figure img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
            for selector,name in [('#bound-viz','bound'),('#repair-viz','repair'),('#reach-viz','reach'),('#flatten-recall','recall'),('figure:first-of-type','collision'),('figure:last-of-type','results')]:
                page.locator(selector).scroll_into_view_if_needed();dest=f'/tmp/l077-{name}-{width}.png';page.screenshot(path=dest);screens.append(dest)
            if width==375:
                scroll=page.locator('figure>div').first;scroll.focus();page.keyboard.press('ArrowRight');page.wait_for_timeout(250);assert scroll.evaluate('(e)=>e.scrollLeft>0')
            page.emulate_media(media='print');assert page.locator('figure img').count()==4
            page.close()
        page=browser.new_page(java_script_enabled=False);page.goto((ROOT/'labs/html'/f'{SLUG}.html').as_uri())
        assert page.locator('img[src^="data:image/png;base64,"]').count()==4
        assert page.locator('#lab-exercises').count()==1
        browser.close()
    assert not errors,errors
    result={'status':'PASS','widths':[1100,375],'javascript_errors':errors,'screenshots':screens,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
    (ROOT/'labs/_browser_l077_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':run()
