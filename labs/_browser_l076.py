"""Actual desktop/mobile controls, arithmetic, figures, notebook and print."""
import json,os
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];SLUG='0076-encoder-predictor-stack'
def run():
 libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
 if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
 errors=[];screens=[]
 with sync_playwright() as p:
  browser=p.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
  for width in [1100,375]:
   page=browser.new_page(viewport={'width':width,'height':950});page.on('pageerror',lambda e:errors.append(str(e)))
   page.goto((ROOT/'lessons'/f'{SLUG}.html').as_uri())
   widget=page.locator('#time-viz');out=widget.locator('output')
   assert '1, 2, 0, 0' in out.inner_text()
   widget.locator('input').focus();page.keyboard.press('End');assert '2, 2, 1, 0' in out.inner_text()
   page.keyboard.press('Home');assert 'Cutoff day 3' in out.inner_text() and '1, 2, 0, 0' in out.inner_text()
   widget.locator('button').click();assert 'Cutoff day 10' in out.inner_text()
   widget=page.locator('#aggregate-viz');out=widget.locator('output');assert 'mean [4,3]' in out.inner_text()
   widget.locator('select').select_option('sum');assert 'sum [8,6]' in out.inner_text()
   widget.locator('button').click();assert 'mean [4,3]' in out.inner_text()
   # Exhaustive widget arithmetic oracle across every allowed cutoff and reduction.
   for cutoff in range(3,13):
    for mode in ['sum','mean']:
     rows=page.evaluate('([t,m])=>RDLStackViz.compute(t,m)',[cutoff,mode]);n42=2 if cutoff>=12 else 1;n99=1 if cutoff>=11 else 0
     assert [r['n'] for r in rows]==[n42,2,n99,0]
     assert rows[1]['value']==([4,3] if mode=='mean' else [8,6])
   assert page.locator('#warmup').inner_text().strip()
   assert page.locator('#prediction').inner_text().strip()
   assert page.locator('#teachback textarea').count()==1
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'Page overflow'
   for img in page.locator('figure img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
   for selector,name in [('#time-viz','time'),('#aggregate-viz','aggregate'),('figure:first-of-type','architecture'),('figure:nth-of-type(2)','routing'),('figure:last-of-type','gradients')]:
    page.locator(selector).scroll_into_view_if_needed();dest=f'/tmp/l076-{name}-{width}.png';page.screenshot(path=dest);screens.append(dest)
   if width==375:
    scroll=page.locator('.rdl-figure-scroll').first;scroll.focus();page.keyboard.press('ArrowRight');page.wait_for_timeout(250);assert scroll.evaluate('(e)=>e.scrollLeft>0')
   page.emulate_media(media='print');assert page.locator('figure img').count()==4
   page.close()
  page=browser.new_page(java_script_enabled=False);page.goto((ROOT/'labs/html'/f'{SLUG}.html').as_uri())
  assert page.locator('img[src^="data:image/png;base64,"]').count()==4
  assert page.locator('#lab-exercises').count()==1
  browser.close()
 assert not errors,errors
 result={'status':'PASS','widths':[1100,375],'javascript_errors':errors,'screenshots':screens,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
 (ROOT/'labs/_browser_l076_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':run()
