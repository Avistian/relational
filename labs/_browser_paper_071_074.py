"""Inspect the visible reproduction sections at desktop and mobile widths."""
import os,json
from pathlib import Path
from playwright.sync_api import sync_playwright
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
root=Path(__file__).resolve().parents[1];records=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
 for n in range(71,75):
  path=next((root/'labs/html').glob(f'00{n}-*.html'))
  for width in [1100,375]:
   page=browser.new_page(viewport={'width':width,'height':900},java_script_enabled=False)
   page.goto(path.as_uri()+'#paper-reproduction')
   assert page.locator('#paper-reproduction').count()==1
   assert page.get_by_text('Author-run reproduction evidence',exact=False).count()>=1
   assert page.get_by_text('RUN_PAPER_REPRO',exact=False).count()>=1
   shot=f'/tmp/l{n}-paper-{width}.png';page.screenshot(path=shot)
   records.append(dict(lesson=n,width=width,anchor='PASS',visible_source_blocks=page.locator('.highlight').count(),screenshot=shot))
   page.close()
 browser.close()
(root/'labs/_paper_browser_results.json').write_text(json.dumps(dict(status='PASS',records=records),indent=2))
print('Paper-section anchors and source blocks passed on desktop/mobile for all four previews.')
