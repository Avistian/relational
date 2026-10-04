"""Verify the deployed commit's bytes, embedded packet and actual browser behavior."""
import ast,base64,concurrent.futures,hashlib,io,json,os,subprocess,sys,time,zipfile
from pathlib import Path
import requests
from playwright.sync_api import sync_playwright
P=Path(__file__).resolve().parent;R=P.parent;V=R/'reviews/lesson-b24';S='b24-architecture-thesis-defense';BASE='https://avistian.github.io/relational/'
def main():
 started=time.monotonic();sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();run_id=sys.argv[1]
 run=json.loads(subprocess.check_output(['gh','run','view',run_id,'--json','headSha,conclusion,url'],cwd=R,text=True))
 assert run['headSha']==sha and run['conclusion']=='success',run
 seal=json.loads((P/'evidence/b24/artifact-manifest.json').read_text())
 paths=[x['path'] for x in seal['files'] if not x['path'].startswith(('reviews/','lessons/content/'))]+['labs/evidence/b24/artifact-manifest.json','labs/evidence/b24/local-budget.json','lessons/manifest.json','lessons/b23-declared-comparison.html','assets/retrieval-pool.js','assets/paper-deck.js','reference/glossary.html']
 def check(rel):
  expected=subprocess.check_output(['git','show',sha+':'+rel],cwd=R)
  response=requests.get(BASE+rel,params={'b24':sha[:12]},timeout=45);response.raise_for_status()
  assert hashlib.sha256(response.content).digest()==hashlib.sha256(expected).digest(),rel
  return dict(path=rel,sha256=hashlib.sha256(response.content).hexdigest(),bytes=len(response.content))
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:checks=list(pool.map(check,paths))
 book=requests.get(BASE+f'labs/{S}.ipynb',params={'b24':sha[:12]},timeout=45).json()
 source=next(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code' and ''.join(c['source']).startswith('packet=base64.b64decode'))
 tree=ast.parse(source);raw=base64.b64decode(ast.literal_eval(tree.body[0].value.args[0]));assert raw==(P/'evidence/b24/portable-packet.zip').read_bytes()
 manifest=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='manifest')
 with zipfile.ZipFile(io.BytesIO(raw)) as z:
  for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
  assert z.read('b23-packet.zip')==(P/'evidence/b23/portable-packet.zip').read_bytes()
 libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
 if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
 errors=[];states=0
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
  page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None);page.on('response',lambda r:errors.append(r.url) if r.url.startswith(BASE) and r.status>=400 else None)
  for width in [1200,375]:
   page.set_viewport_size({'width':width,'height':950});page.goto(BASE+f'lessons/{S}.html?b24='+sha[:12],wait_until='networkidle')
   b=page.locator('[data-defense-board]');assert b.locator('output').get_attribute('data-eligible')=='false';b.locator('[name=reproduction]').check();b.locator('[name=leakage]').uncheck();assert b.locator('output').get_attribute('data-eligible')=='true';b.locator('[name=protocol]').select_option('0');assert b.locator('output').get_attribute('data-eligible')=='false';b.locator('button').click();states+=3
   f=page.locator('[data-falsification-board]');assert f.locator('output').get_attribute('data-state')=='REVISE';f.locator('[name=matched]').uncheck();assert f.locator('output').get_attribute('data-state')=='INCOMPARABLE';f.locator('button').click();states+=2
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2');assert page.locator('.defense-figure img').count()==3 and page.evaluate('Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)')
   page.screenshot(path=str(V/f'live-{width}.png'),full_page=True);b.screenshot(path=str(V/f'live-gate-{width}.png'))
  page.goto(BASE+'?b24='+sha[:12]);page.wait_for_selector('a[href="lessons/'+S+'.html"]')
  page.goto(BASE+'notebooks.html?b24='+sha[:12]);page.wait_for_selector('a[href="labs/'+S+'.ipynb"]')
  page.goto(BASE+f'labs/html/{S}.html?b24='+sha[:12]);assert page.locator('img[src^="data:image/png"]').count()==3
  page.goto(BASE+'reference/b24-proposal-template.html?b24='+sha[:12]);assert page.locator('h2').count()==5
  page.goto(BASE+'lessons/b23-declared-comparison.html?b24='+sha[:12]);assert page.locator('a[href="'+S+'.html"]').count()==1
  browser.close()
 assert not errors,errors
 out=dict(status='LIVE_VERIFIED',site=BASE+f'lessons/{S}.html',commit=sha,workflow=run,live_files=checks,embedded_packet_files=len(manifest),original_b23_packet_identical=True,browser_states=states,desktop_and_mobile=True,navigation=True,rendered_notebook=True,proposal_template=True,browser_errors=errors,seconds=time.monotonic()-started,reproduction='COMPLETE_SAVED_EVIDENCE_REPLAY',historical_identity='NOT_ESTABLISHED',cloud_usd=0,learner='PENDING_WRITTEN_DEFENSE')
 (V/'deployment.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='live_files'},'files',len(checks))
if __name__=='__main__':main()
