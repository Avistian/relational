"""Exercise every L079 control and inspect desktop/mobile and portable figures."""
import json,os
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];SLUG='0079-neural-tabular-decision-guide'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
errors=[];screens=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
    for width in [1100,375]:
        page=browser.new_page(viewport={'width':width,'height':950});page.on('pageerror',lambda e:errors.append(str(e)));page.on('requestfailed',lambda e:errors.append(e.url))
        page.goto((ROOT/'lessons'/f'{SLUG}.html').as_uri())
        w=page.locator('#budget-viz');assert 'Choose Trees' in w.locator('output').inner_text()
        w.locator('input').focus();page.keyboard.press('End');assert 'Choose ICL' in w.locator('output').inner_text()
        page.keyboard.press('Home');assert 'No feasible' in w.locator('output').inner_text()
        for _ in range(9):page.keyboard.press('ArrowRight')
        assert 'Choose TabM' in w.locator('output').inner_text()
        w.locator('button').click();assert w.locator('input').input_value()=='5'
        w.scroll_into_view_if_needed();dest=f'/tmp/l079-budget-{width}.png';page.screenshot(path=dest);screens.append(dest)
        c=page.locator('#cohort-viz');assert '1.818' in c.inner_text();c.locator('select').select_option('matched');assert '1.000' in c.inner_text();assert 'Matched names' in c.locator('output').inner_text()
        c.locator('select').select_option('all');assert 'Unequal populations' in c.locator('output').inner_text()
        assert page.locator('#warmup').inner_text().strip();assert page.locator('#teachback textarea').count()==1
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'Horizontal page overflow'
        for i,img in enumerate(page.locator('figure img').all()):
            assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0');img.scroll_into_view_if_needed();dest=f'/tmp/l079-figure{i}-{width}.png';page.screenshot(path=dest);screens.append(dest)
        page.emulate_media(media='print');assert page.locator('figure img').count()==3
        if width==1100:page.pdf(path='/tmp/l079-print.pdf')
        page.close()
    page=browser.new_page(java_script_enabled=False);page.goto((ROOT/'labs/html'/f'{SLUG}.html').as_uri())
    assert page.locator('img[src^="data:image/png;base64,"]').count()==3
    for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
    page.goto((ROOT/'notebooks.html').as_uri());assert page.locator('#lab-79 a').count()==3
    browser.close()
assert not errors,errors
r={'status':'PASS','widths':[1100,375],'javascript_errors':errors,'all_budget_states':True,'both_cohorts':True,'no_javascript_gallery':True,'screenshots':screens,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(ROOT/'labs/_browser_l079_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
