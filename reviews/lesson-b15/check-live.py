"""Fetch deployed B15 bytes and exercise actual live desktop/mobile controls."""
import argparse,ast,base64,concurrent.futures,hashlib,io,json,os,time,urllib.request,zipfile
from pathlib import Path
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[2];V=R/'reviews/lesson-b15';S='b15-parameter-free-encoders';base='https://avistian.github.io/relational/'
p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--run',required=True);args=p.parse_args()
manifest=json.loads((R/'labs/evidence/b15/artifact-manifest.json').read_text())['files']
hidden={n:h for n,h in manifest.items() if any(part.startswith('.') for part in Path(n).parts)}
files={n:h for n,h in manifest.items() if n not in hidden and n.startswith(('labs/','lessons/','reference/','assets/')) and not n.startswith('lessons/content/')}
for name in ['index.html','notebooks.html','lessons/manifest.json','assets/retrieval-pool.js','assets/retrieval-bank.js','assets/predict.js','assets/teachback.js','assets/lesson.css']:
 files[name]=hashlib.sha256((R/name).read_bytes()).hexdigest()
for path in (R/'lessons').glob('b*.html'):files[str(path.relative_to(R))]=hashlib.sha256(path.read_bytes()).hexdigest()
def check(item):
 name,digest=item
 request=urllib.request.Request(base+name+'?b15='+args.commit,headers={'User-Agent':'B15-publication-check'})
 try:
  with urllib.request.urlopen(request,timeout=45) as response:data=response.read()
 except Exception as exc: raise RuntimeError(name+': '+str(exc)) from exc
 actual=hashlib.sha256(data).hexdigest()
 if actual!=digest:raise AssertionError('Live hash mismatch '+name)
 return name
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:checked=list(pool.map(check,files.items()))
# Pages rejects dot-paths. Authenticate the same source bytes inside the live,
# downloadable notebook packet instead; never silently drop them from the audit.
with urllib.request.urlopen(base+'labs/'+S+'.ipynb?b15='+args.commit,timeout=45) as response:
 live_notebook=json.load(response)
blobs=[]
for cell in live_notebook['cells']:
 if cell['cell_type']=='code':
  for node in ast.walk(ast.parse(''.join(cell['source']))):
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and isinstance(node.args[0],ast.Constant):blobs.append(node.args[0].value)
assert len(blobs)==1
with zipfile.ZipFile(io.BytesIO(base64.b64decode(blobs[0]))) as packet:
 for name,digest in hidden.items():assert hashlib.sha256(packet.read(name.removeprefix('labs/'))).hexdigest()==digest,name
 source_manifest=json.loads((R/'labs/sources/b15/manifest.json').read_text())
 for name,digest in source_manifest['files'].items():assert hashlib.sha256(packet.read('sources/b15/'+name)).hexdigest()==digest,name
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':950});page.goto(base+'lessons/'+S+'.html?b15='+args.commit,wait_until='networkidle')
  assert page.locator('h1').inner_text().startswith('B15')
  for world in [0,1]:
   r=page.locator('[data-label-rules]');r.locator('[name=world]').select_option(str(world))
   for access in ['local','expanded']:
    r.locator('[name=access]').select_option(access);assert float(r.locator('output').get_attribute('data-probability'))==(.5 if access=='local' else 1-world);states+=1
  r.locator('button').click()
  v=page.locator('[data-label-visibility]');v.locator('[name=cutoff]').select_option('11');assert v.locator('output').get_attribute('data-future')=='true';v.locator('button').click();assert v.locator('output').get_attribute('data-future')=='false';states+=2
  c=page.locator('[data-label-columns]');c.locator('[name=examples]').select_option('expanded');c.locator('[name=column]').select_option('1');assert c.locator('output').get_attribute('data-probability')=='1';c.locator('button').click();states+=1
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2')
  assert page.locator('#b15-teachback textarea').count()==1
  page.screenshot(path=str(V/f'live-{width}.png'),full_page=True)
 page.goto(base);page.wait_for_selector('a[href="lessons/'+S+'.html"]')
 page.goto(base+'notebooks.html');page.wait_for_timeout(500);assert S in page.content()
 browser.close()
assert not errors,errors
result=dict(status='PASS',commit=args.commit,workflow_run=args.run,live_url=base+'lessons/'+S+'.html',sha256_matched_files=len(checked),files=checked,hidden_source_paths_verified_in_live_notebook=list(hidden),embedded_source_files_verified=len(source_manifest['files']),browser_states=states,desktop_mobile='PASS',browser_errors=errors,checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),paper_inference='NOT_RUN',cloud_spend_usd=0,live_colab='NOT_CHECKED')
(V/'deployment.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
