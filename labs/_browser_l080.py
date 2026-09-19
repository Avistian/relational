"""Verify exam controls, mobile rendering and no-JS lab access."""
import os,json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];SLUG='0080-year-2-exit-exam'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
errors=[];screens=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
    for width in [1100,375]:
        page=browser.new_page(viewport={'width':width,'height':950});page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto((ROOT/'lessons'/f'{SLUG}.html').as_uri())
        assert page.locator('#warmup').inner_text().strip()
        assert '0/4' in page.locator('#exam-gates output').inner_text()
        for box in page.locator('#exam-gates input').all():box.check()
        assert 'Submission ready' in page.locator('#exam-gates output').inner_text()
        page.locator('#exam-gates input').first.uncheck();assert '3/4' in page.locator('#exam-gates output').inner_text()
        page.locator('details summary').click()
        for img in page.locator('figure img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
        assert page.locator('#teachback textarea').count()==1
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'Page overflow'
        page.locator('details').scroll_into_view_if_needed();dest=f'/tmp/l080-results-{width}.png';page.screenshot(path=dest);screens.append(dest)
        page.emulate_media(media='print')
        if width==1100:page.pdf(path='/tmp/l080-print.pdf')
        page.close()
    page=browser.new_page(java_script_enabled=False);page.goto((ROOT/'labs/html'/f'{SLUG}.html').as_uri())
    assert page.locator('img[src^="data:image/png;base64,"]').count()==2
    for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
    page.goto((ROOT/'notebooks.html').as_uri());assert page.locator('#lab-80 a').count()==3
    browser.close()
assert not errors,errors
report={'status':'PASS','widths':[1100,375],'gate_states_checked':[0,4,3],'images':2,'no_javascript_gallery':True,'javascript_errors':errors,'screenshots':screens,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(ROOT/'labs/_browser_l080_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
