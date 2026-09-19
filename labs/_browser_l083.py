"""Exercise every L083 mechanism control on desktop/mobile and inspect portable notebook images."""
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
  page.goto((ROOT/'lessons/0083-graphsage.html').as_uri())
  assert page.locator('#warmup').inner_text().strip()
  assert 'Mean: [3,2]' in page.locator('#mean output').inner_text()
  control=page.locator('#mean input');control.fill('9');control.dispatch_event('input')
  assert 'Mean: [5,2]' in page.locator('#mean output').inner_text()
  control.focus();page.keyboard.press('ArrowLeft');assert control.input_value()=='8'
  control.fill('1');control.dispatch_event('input');assert 'Mean: [1,2]' in page.locator('#mean output').inner_text()
  page.locator('#mean button').click();assert control.input_value()=='5'
  assert '261 occurrences/root' in page.locator('#budget output').inner_text()
  for first,second in [(1,1),(25,25),(10,25)]:
   for name,value in [('first',first),('second',second)]:
    c=page.locator(f'#budget input[name="{name}"]');c.fill(str(value));c.dispatch_event('input')
   assert f'{1+first+first*second} occurrences/root' in page.locator('#budget output').inner_text()
  page.locator('#budget button').click()
  assert page.locator('#predict button').count()>=3
  assert page.locator('#teachback textarea').count()==1
  for img in page.locator('figure img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
  assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1'),'page overflow'
  for section in ['mean','budget']:
   page.locator('#'+section).scroll_into_view_if_needed();page.screenshot(path=f'/tmp/l083-{section}-{width}.png')
  page.locator('img[alt^="Complete released"]').scroll_into_view_if_needed();page.screenshot(path=f'/tmp/l083-architecture-{width}.png')
  page.emulate_media(media='print');assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1')
  page.close()
 page=browser.new_page(java_script_enabled=False);page.goto((ROOT/'labs/html/0083-graphsage.html').as_uri())
 assert page.locator('img[src^="data:image/png;base64,"]').count()==4
 for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0')
 page.screenshot(path='/tmp/l083-notebook.png')
 page.goto((ROOT/'notebooks.html').as_uri());assert page.locator('#lab-83 a').count()==4
 browser.close()
assert not errors,errors
r={'status':'PASS','widths':[1100,375],'mean_states':[5,9,8,1,5],'fanout_states':[[10,25],[1,1],[25,25],[10,25]],'portable_figures':4,'no_js_gallery':True,'page_errors':errors,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(ROOT/'labs/_browser_l083_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
