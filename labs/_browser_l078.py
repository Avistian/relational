"""Actual browser controls, mobile geometry and notebook image checks."""
import json,os
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];SLUG='0078-message-passing-preview'
def run():
    libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
    if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
    errors=[];screens=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
        for width in [1100,375]:
            page=browser.new_page(viewport={'width':width,'height':950});page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto((ROOT/'lessons'/f'{SLUG}.html').as_uri())
            for selector,baseline,changed in [('#message-viz','4.50','7.50'),('#reach-viz','3.75','5.25'),('#normalization-viz','5.415816','10.314796')]:
                w=page.locator(selector);assert baseline in w.locator('output').inner_text()
                w.locator('input').focus();page.keyboard.press('End');assert changed in w.locator('output').inner_text(),w.inner_text()
                page.keyboard.press('Home');assert w.locator('.value').inner_text()=='0'
                w.locator('button').click();assert w.locator('.value').inner_text()=='8'
                w.scroll_into_view_if_needed();dest=f'/tmp/l078-{selector[1:]}-{width}.png';page.screenshot(path=dest);screens.append(dest)
            assert page.locator('#warmup').inner_text().strip()
            assert page.locator('#prediction').inner_text().strip()
            assert page.locator('#teachback textarea').count()==1
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'Page overflow'
            for img in page.locator('figure img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
            for i,img in enumerate(page.locator('figure').all()):
                img.scroll_into_view_if_needed();dest=f'/tmp/l078-figure{i}-{width}.png';page.screenshot(path=dest);screens.append(dest)
            page.emulate_media(media='print');assert page.locator('figure img').count()==4
            page.close()
        page=browser.new_page(java_script_enabled=False);page.goto((ROOT/'labs/html'/f'{SLUG}.html').as_uri());assert page.locator('img[src^="data:image/png;base64,"]').count()==4
        for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
        browser.close()
    assert not errors,errors
    result={'status':'PASS','widths':[1100,375],'javascript_errors':errors,'screenshots':screens,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
    (ROOT/'labs/_browser_l078_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':run()
