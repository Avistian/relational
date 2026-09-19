"""Real desktop/mobile interaction and portable notebook image checks."""
import json,os
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
    for width in [1100,375]:
        page=browser.new_page(viewport={'width':width,'height':950});page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto((ROOT/'lessons/0082-gcn.html').as_uri())
        assert page.locator('#warmup').inner_text().strip()
        assert '5.415816' in page.locator('#normalized output').inner_text()
        control=page.locator('#normalized input');control.fill('20');control.dispatch_event('input')
        assert '10.314796' in page.locator('#normalized output').inner_text()
        control.focus();page.keyboard.press('ArrowLeft');assert control.input_value()=='19'
        page.locator('#normalized button').click();assert control.input_value()=='8'
        assert page.locator('#teachback textarea').count()==1
        assert page.locator('#predict button').count()>=3
        for img in page.locator('figure img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1'),'page overflow'
        page.locator('#normalized').scroll_into_view_if_needed();page.screenshot(path=f'/tmp/l082-interaction-{width}.png')
        page.locator('img[alt^="Complete Cora"]').scroll_into_view_if_needed();page.screenshot(path=f'/tmp/l082-architecture-{width}.png');page.close()
    page=browser.new_page(java_script_enabled=False)
    page.goto((ROOT/'labs/html/0082-gcn.html').as_uri())
    assert page.locator('img[src^="data:image/png;base64,"]').count()==4
    for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
    page.goto((ROOT/'notebooks.html').as_uri());assert page.locator('#lab-82 a').count()==4
    browser.close()
assert not errors,errors
report={'status':'PASS','widths':[1100,375],'normalized_states':[8,20,19,8],'notebook_portable_images':4,'no_js_gallery':True,'page_errors':errors,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(ROOT/'labs/_browser_l082_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
