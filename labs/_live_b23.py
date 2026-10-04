"""Verify deployed commit's bytes and actual Pages navigation/widget behavior."""
import base64,concurrent.futures,hashlib,io,json,os,subprocess,sys,time,zipfile
from pathlib import Path
import requests
from playwright.sync_api import sync_playwright
P=Path(__file__).resolve().parent;R=P.parent;V=R/'reviews/lesson-b23';S='b23-declared-comparison';BASE='https://avistian.github.io/relational/'

def main():
 started=time.monotonic();sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();run_id=sys.argv[1]
 run=json.loads(subprocess.check_output(['gh','run','view',run_id,'--json','headSha,conclusion,url'],cwd=R,text=True))
 assert run['headSha']==sha and run['conclusion']=='success',run
 seal=json.loads((P/'evidence/b23/artifact-manifest.json').read_text())
 paths=[x['path'] for x in seal['files'] if not x['path'].startswith(('reviews/','lessons/content/'))]+['labs/evidence/b23/artifact-manifest.json','lessons/manifest.json']
 paths+=[f'lessons/{slug}.html' for slug in ['b20-curriculum-order','b21-structural-robustness']]
 def check(rel):
  expected=subprocess.check_output(['git','show',sha+':'+rel],cwd=R)
  response=requests.get(BASE+rel,params={'b23':sha[:12]},timeout=60);response.raise_for_status()
  assert hashlib.sha256(response.content).digest()==hashlib.sha256(expected).digest(),rel
  return dict(path=rel,sha256=hashlib.sha256(response.content).hexdigest(),bytes=len(response.content))
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:checks=list(pool.map(check,paths))
 # Authenticate the embedded source/evidence packet from the live downloadable student notebook.
 book=requests.get(BASE+f'labs/{S}.ipynb',params={'b23':sha[:12]},timeout=60).json()
 import ast
 source=next(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code' and ''.join(c['source']).startswith('packet=base64.b64decode'))
 tree=ast.parse(source);encoded=ast.literal_eval(tree.body[0].value.args[0]);raw=base64.b64decode(encoded)
 assert raw==(P/'evidence/b23/portable-packet.zip').read_bytes()
 manifest=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='manifest')
 with zipfile.ZipFile(io.BytesIO(raw)) as z:
  for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
 libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
 if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
 errors=[];states=0
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
  page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
  page.on('response',lambda r:errors.append(r.url) if r.url.startswith(BASE) and r.status>=400 else None)
  for width in [1200,375]:
   page.set_viewport_size({'width':width,'height':950});page.goto(BASE+f'lessons/{S}.html?b23='+sha[:12],wait_until='networkidle')
   board=page.locator('[data-comparison-board]')
   for arm in ['RDBPFN_single','TabICLv1.1','Logistic']:
    board.locator('[name=reference]').select_option(arm)
    expected=json.loads((P/'evidence/b23/report.json').read_text())
    delta=expected['models']['RDBPFN']['per_seed'][0]-expected['models'][arm]['per_seed'][0]
    assert abs(float(board.locator('output').get_attribute('data-delta'))-delta)<1e-9;states+=1
   board.locator('[name=missing]').check();assert board.locator('output').get_attribute('data-state')=='INCOMPLETE';states+=1
   board.locator('button').click();assert board.locator('output').get_attribute('data-state')=='COMPLETE'
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2')
   assert page.locator('.comparison-figure img').count()==3 and page.evaluate('Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)')
   page.screenshot(path=str(V/f'live-{width}.png'),full_page=True);board.screenshot(path=str(V/f'live-widget-{width}.png'))
  page.goto(BASE+'?b23='+sha[:12]);page.wait_for_selector('a[href="lessons/'+S+'.html"]')
  page.goto(BASE+'notebooks.html?b23='+sha[:12]);page.wait_for_selector('a[href="labs/'+S+'.ipynb"]')
  page.goto(BASE+f'labs/html/{S}.html?b23='+sha[:12]);assert page.locator('img[src^="data:image/png"]').count()==3
  browser.close()
 assert not errors,errors
 out=dict(status='LIVE_VERIFIED',site=BASE+f'lessons/{S}.html',commit=sha,workflow=run,live_files=checks,embedded_packet_files=len(manifest),browser_states=states,desktop_and_mobile=True,navigation=True,rendered_notebook=True,browser_errors=errors,seconds=time.monotonic()-started,selected_reproduction='COMPLETE_SELECTED_RELEASE_EVALUATION',historical_identity='NOT_ESTABLISHED',reserved_usd_including_overhead=3.6909712)
 (V/'deployment.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k not in ['live_files']},'files',len(checks))
if __name__=='__main__':main()
