"""Real Chromium interaction, layout, figures, and notebook-preview checks."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];SLUG='0072-scarf-subtab-contrastive-views'
def run():
    errors=[];screens=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
        for width in [1100,375]:
            page=browser.new_page(viewport={'width':width,'height':950})
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto((ROOT/'lessons'/f'{SLUG}.html').as_uri())
            assert '0.3133' in page.locator('#loss-trace output').inner_text()
            page.locator('#loss-trace select').select_option('subtab')
            assert '0.5514' in page.locator('#loss-trace output').inner_text()
            for tau in ['0.1','2']:
                page.locator('#loss-trace input').evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input"));}',tau)
                assert f'τ = {float(tau):.1f}' in page.locator('#loss-trace output').inner_text()
            page.locator('#loss-trace button').click()
            slider=page.locator('#loss-trace input');slider.focus();page.keyboard.press('ArrowRight')
            assert 'τ = 1.1' in page.locator('#loss-trace output').inner_text()
            page.locator('#loss-trace button').click()
            assert '(3.00, 4.00)' in page.locator('#subset-trace output').inner_text()
            page.locator('#subset-trace input').nth(2).uncheck()
            assert '(3.00, 5.00)' in page.locator('#subset-trace output').inner_text()
            page.locator('#subset-trace input').nth(1).uncheck()
            page.locator('#subset-trace input').nth(0).click()
            assert page.locator('#subset-trace input:checked').count()==1
            page.locator('#subset-trace input').nth(1).check();page.locator('#subset-trace input').nth(2).check()
            assert page.locator('#warmup').inner_text().strip()
            assert page.locator('#prediction').inner_text().strip()
            assert page.locator('#teachback textarea').count()==1
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1'), 'Page has horizontal overflow'
            for img in page.locator('figure img').all():assert img.evaluate('(e)=>e.complete && e.naturalWidth>0')
            for selector,name in [('figure','scarf'),('figure','subtab'),('#loss-trace','loss'),('#subset-trace','subset')]:
                loc=page.locator(selector).nth(1 if name=='subtab' else 0);loc.scroll_into_view_if_needed()
                dest=f'/tmp/l072-{name}-{width}.png';page.screenshot(path=dest);screens.append(dest)
            page.emulate_media(media='print');assert page.locator('figure img').count()==4
            page.close()
        page=browser.new_page(java_script_enabled=False)
        for filename in ['index.html','notebooks.html']:
            page.goto((ROOT/filename).as_uri())
            if filename=='index.html':assert page.locator('#lesson-nav').count()==1
            else:assert page.locator(f'a[href="lessons/{SLUG}.html"]').count()>=1
        page.goto((ROOT/'labs/html'/f'{SLUG}.html').as_uri())
        assert page.locator('img[src^="data:image/png;base64,"]').count()==4
        assert page.locator('#lab-exercises').count()==1
        browser.close()
    assert not errors,errors
    report={'status':'PASS','widths':[1100,375],'javascript_errors':errors,'screenshots':screens,'live_colab':'NOT_CHECKED','deployed_site':'NOT_CHECKED'}
    (ROOT/'labs/_browser_l072_results.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':run()
