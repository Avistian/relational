"""Verify browser states, actual portable notebook, source parity and deterministic build."""
import ast,functools,hashlib,json,os,subprocess,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0176-few-shot-icl-evaluation'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
solcode='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code');stucode='\n\n'.join(c.source for c in student.cells if c.cell_type=='code')
assert 'NotImplementedError' not in solcode and stucode.count('NotImplementedError')==3
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
soldefs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(solcode).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for filename in ['relkit/few_shot_l176.py','relkit/scaling_l169.py','_check_l176.py','_audit_l169.py','_audit_l176.py','sources/l166/upstream/model_pretrain/src/models.py','_run_l176.py','_fetch_l176.py']:
 for n in ast.parse((P/filename).read_text()).body:
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)):assert soldefs[n.name]==ast.dump(n,include_attributes=False),(filename,n.name)
execution=json.loads((P/'_execution_l176_results.json').read_text());assert execution['executed_code_sha256']==hashlib.sha256(solcode.encode()).hexdigest()
r=json.loads((P/'evidence/l176/report.json').read_text());errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda msg:errors.append(msg.text) if msg.type=='error' else None);page.on('requestfailed',lambda req:errors.append(req.url))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/(S+'.html')).as_uri());host=page.locator('#support-explorer')
  for db in ['rel-f1','rel-trial']:
   host.locator('[data-db]').select_option(db)
   for arm in ['RDBPFN','RDBPFN_single','TabICLv1.1']:
    host.locator('[data-arm]').select_option(arm)
    for k in [64,128,256,512,1024]:
     host.locator('[data-k]').select_option(str(k))
     for seed in range(10):
      host.locator('[data-seed]').select_option(str(seed));curve=r['curves'][db+'/'+arm];base=curve['levels'][0]['per_seed'][seed];level=next(x for x in curve['levels'] if x['context']==k);score=level['per_seed'][seed];text=host.locator('output').inner_text()
      assert host.get_attribute('data-state')==f'{db}/{arm}/{k}/{seed}'
      assert f'AUROC {score:.6f}' in text and f'{score-base:+.6f}' in text and f'added {k-64}' in text
      assert f'{base:.6f}' in host.locator('[data-baseline]').inner_text()
      assert host.locator('.fs-support span').count()==k//16 and host.locator('.fs-support .old').count()==4;states+=1
  host.locator('[data-reset]').click();assert host.get_attribute('data-state')=='rel-f1/RDBPFN/64/0'
  host.locator('[data-k]').focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert host.get_attribute('data-state')=='rel-f1/RDBPFN/128/0';host.locator('[data-reset]').click()
  host.screenshot(path=f'/tmp/l176-explorer-{width}.png')
  pred=page.locator('#predict');assert pred.locator('.predict-reveal').is_disabled();assert len(set(len(x.split()) for x in pred.locator('.predict-option').all_inner_texts()))==1
  pred.locator('[data-value=no]').click();pred.locator('.predict-reveal').click();assert 'preprocessing' in pred.locator('.predict-outcome').inner_text()
  assert page.locator('#warmup button').count()>0 and page.locator('#teachback textarea').count()==1
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===3&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),f'Overflow {width}'
  page.evaluate('scrollTo(0,0)');page.screenshot(path=f'/tmp/l176-top-{width}.png')
  for i in range(3):page.locator('figure').nth(i).screenshot(path=f'/tmp/l176-figure-{i}-{width}.png')
 page.emulate_media(media='print');assert host.locator('.fs-controls').evaluate('(x)=>getComputedStyle(x).display')=='none'
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri());assert pg.locator('noscript').count()==1 and '458,100' in pg.locator('body').inner_text();assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
 page.emulate_media(media='screen');page.set_viewport_size({'width':1050,'height':900});page.goto((P/'html'/(S+'.html')).as_uri());assert page.locator('img[src^="data:image/png"]').count()==3
 assert 'All600evaluations rescored' in page.locator('body').inner_text();page.locator('img[src^="data:image/png"]').first.screenshot(path='/tmp/l176-notebook-figure.png');browser.close()
assert not errors,errors
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start()
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 for path,href in [('index.html','lessons/'+S+'.html'),('notebooks.html','labs/html/'+S+'.html')]:
  page.goto(f'http://127.0.0.1:{server.server_port}/'+path);reveal_gallery_link(page,'a[href="'+href+'"]')
 browser.close()
server.shutdown()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
count=0
for path in [R/'lessons'/(S+'.html'),R/'reference/few-shot-icl-evaluation.html']:
 parser=Links();parser.feed(path.read_text())
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),url;count+=1
paths=[R/'lessons'/(S+'.html'),R/'reference/few-shot-icl-evaluation.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'evidence/l176/report.md',R/'assets/l176-evidence.js']+sorted((P/'figures/l176').iterdir())
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
for script in ['_figures_l176.py','_build_l176.py']:subprocess.run([str(R/'.venv/bin/python'),str(P/script)],check=True,capture_output=True)
assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],'Nondeterministic artifacts'
r=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,keyboard_reset='PASS',no_js='PASS',print='PASS',inline_source_parity='PASS',portable_figures=3,local_links=count,deterministic_build='PASS',manifest_galleries='PASS',javascript_errors=errors,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l176_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
