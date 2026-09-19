"""Actual desktop/mobile rendering, keyboard interventions and image delivery."""
import json,os
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];SLUG='0075-pytorch-frame-row-encoder'
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
   scope=page.locator('#frame-scope');out=scope.locator('output')
   assert 'Current fitted mean: 20.' in out.inner_text()
   scope.locator('input[type=checkbox]').check();assert 'Current fitted mean: 265.' in out.inner_text()
   scope.locator('input[type=range]').focus();page.keyboard.press('Home');assert 'Current fitted mean: 15.' in out.inner_text()
   page.keyboard.press('End');assert 'Current fitted mean: 265.' in out.inner_text()
   scope.locator('button').click();assert 'Current fitted mean: 20.' in out.inner_text()
   widget=page.locator('#frame-numeric');out=widget.locator('output')
   assert '→ [2.5, -0.5].' in out.inner_text()
   widget.locator('input[type=range]').focus();page.keyboard.press('Home');assert '→ [-3.5, 2.5].' in out.inner_text()
   page.keyboard.press('End');assert '→ [4.5, -1.5].' in out.inner_text()
   widget.locator('input[type=checkbox]').check();assert '→ [0.5, 0.5].' in out.inner_text()
   widget.locator('button').click();assert 'Current x=30' in out.inner_text()
   assert page.locator('#warmup').inner_text().strip()
   assert page.locator('#prediction').inner_text().strip()
   assert page.locator('#teachback textarea').count()==1
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'Page overflow'
   for img in page.locator('figure img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
   if width==375:
    region=page.locator('figure .figure-scroll').first;region.focus();page.keyboard.press('ArrowRight');page.wait_for_timeout(250)
    assert region.evaluate('(e)=>e.scrollLeft>0')
   for selector,name in [('#frame-numeric','numeric'),('#frame-scope','scope'),('figure:last-of-type','architecture')]:
    page.locator(selector).scroll_into_view_if_needed();dest=f'/tmp/l075-{name}-{width}.png';page.screenshot(path=dest);screens.append(dest)
   page.emulate_media(media='print');assert page.locator('figure img').count()==4
   page.close()
  page=browser.new_page(java_script_enabled=False)
  page.goto((ROOT/'labs/html'/f'{SLUG}.html').as_uri())
  assert page.locator('img[src^="data:image/png;base64,"]').count()==4
  assert page.locator('#lab-exercises').count()==1
  browser.close()
 assert not errors,errors
 report={'status':'PASS','widths':[1100,375],'javascript_errors':errors,'screenshots':screens,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
 (ROOT/'labs/_browser_l075_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':run()
