"""Check actual UI, portable implementation, deterministic build and five-page PDF."""
import ast,functools,hashlib,io,itertools,json,os,subprocess,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from pypdf import PdfReader,PdfWriter
from _gallery_delivery import reveal_gallery_link
from relkit.checkpoint_l190 import claim_gate,rank_cases
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0190-research-gap-checkpoint';E=P/'evidence/l190'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
solcode='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code');stucode='\n\n'.join(c.source for c in student.cells if c.cell_type=='code')
assert 'NotImplementedError' not in solcode and stucode.count('NotImplementedError')==3
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
soldefs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(solcode).body if isinstance(n,ast.FunctionDef)}
for path in [P/'relkit/checkpoint_l190.py',P/'_replay_l190.py']:
 for n in ast.parse(path.read_text()).body:
  if isinstance(n,ast.FunctionDef):assert soldefs[n.name]==ast.dump(n,include_attributes=False),n.name
assert json.loads((P/'_execution_l190_results.json').read_text())['executed_code_sha256']==hashlib.sha256(solcode.encode()).hexdigest()
cases=json.loads((E/'packet/cases.json').read_text());ranked=rank_cases(cases);errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':950});page.goto((R/'lessons'/(S+'.html')).as_uri());host=page.locator('#claim-explorer')
  for bits in itertools.product([False,True],repeat=4):
   evidence=dict(zip(['authenticated','complete','metric_checked','comparable'],bits))
   for k,v in evidence.items():host.locator('[data-check="'+k+'"]').set_checked(v)
   for kind in ['saved_metric','fresh_training','novelty','general_superiority']:
    host.locator('[data-kind]').select_option(kind);assert host.get_attribute('data-verdict')==claim_gate(kind,evidence);assert 'Checkpoint INCOMPLETE' in host.locator('.baseline').inner_text();states+=1
  host.locator('[data-reset]').click();assert host.get_attribute('data-verdict')=='SUPPORTED_REPLAY'
  host.locator('[data-check="complete"]').focus();page.keyboard.press('Space');assert host.get_attribute('data-verdict')=='BLOCKED';host.locator('[data-reset]').click()
  rank=page.locator('#ranking-explorer')
  for row in ranked:
   for i,v in enumerate(row['weights']):rank.locator('[data-weight="'+str(i)+'"]').select_option(str(v))
   assert rank.get_attribute('data-leaders')==','.join(row['leaders']);assert json.loads(rank.get_attribute('data-scores'))==row['scores'];states+=1
  rank.locator('[data-reset]').click();assert json.loads(rank.get_attribute('data-scores'))==ranked[0]['scores']
  assert page.locator('#checkpoint-quiz button').count()==0
  assert page.locator('figure img').evaluate_all('(xs)=>xs.length===2&&xs.every(x=>x.complete&&x.naturalWidth>0)')
  assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),f'Overflow {width}'
  host.screenshot(path=f'/tmp/l190-claim-{width}.png');rank.screenshot(path=f'/tmp/l190-rank-{width}.png');page.evaluate('scrollTo(0,0)');page.screenshot(path=f'/tmp/l190-top-{width}.png')
 page.emulate_media(media='print');assert host.locator('.controls').evaluate('(x)=>getComputedStyle(x).display')=='none'
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri());assert '21,060' in pg.locator('body').inner_text();assert pg.locator('noscript').count()==2;assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
 page.emulate_media(media='screen');page.goto((P/'html'/(S+'.html')).as_uri());assert page.locator('img[src^="data:image/png"]').count()==2;page.locator('img[src^="data:image/png"]').first.screenshot(path='/tmp/l190-notebook.png')
 page.goto((R/'reference/research-gap-document.html').as_uri());page.set_viewport_size({'width':1100,'height':1000});assert page.locator('.sheet').count()==5
 for i in range(5):page.locator('.sheet').nth(i).screenshot(path=f'/tmp/l190-dossier-{i+1}.png')
 page.emulate_media(media='print')
 page.set_viewport_size({'width':680,'height':1000})
 for i in range(5):page.locator('.sheet').nth(i).screenshot(path=f'/tmp/l190-print-{i+1}.png')
 raw=page.pdf(format='A4',print_background=True,prefer_css_page_size=True)
 reader=PdfReader(io.BytesIO(raw));assert len(reader.pages)==5,f'Expected five pages, got {len(reader.pages)}'
 for i,p in enumerate(reader.pages):
  text=p.extract_text();assert f'Page {i+1} / 5' in text,(i,text[-300:]);assert len(text)>900
 writer=PdfWriter()
 for p in reader.pages:writer.add_page(p)
 writer.add_metadata({'/Title':'L190 Research-gap document','/Author':'Relational teaching workspace','/CreationDate':'D:20261002000000Z','/ModDate':'D:20261002000000Z'})
 with (R/'reference/research-gap-document.pdf').open('wb') as file:writer.write(file)
 page.set_viewport_size({'width':375,'height':900});page.emulate_media(media='screen');page.goto((R/'reference/research-gap-document.html').as_uri());assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1');page.locator('.sheet').first.screenshot(path='/tmp/l190-dossier-mobile.png')
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(SimpleHTTPRequestHandler,directory=str(R)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  for name in ['index.html','notebooks.html']:
   page.goto(f'http://127.0.0.1:{server.server_port}/'+name);reveal_gallery_link(page,'a[href*="'+S+'"]')
 finally:server.shutdown()
 assert not errors,errors;browser.close()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
links=0
for name in ['lessons/'+S+'.html','reference/research-gap-checkpoint.html','reference/research-gap-document.html']:
 path=R/name;parser=Links();parser.feed(path.read_text())
 for url in parser.links:
  part=urlsplit(url)
  if part.scheme or not part.path:continue
  assert (path.parent/unquote(part.path)).resolve().is_file(),url;links+=1
outputs=[R/'lessons'/(S+'.html'),R/'reference/research-gap-checkpoint.html',R/'reference/research-gap-document.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb')]+list((P/'figures/l190').glob('*'))
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs};subprocess.run([str(R/'.venv/bin/python'),str(P/'_build_l190.py')],check=True)
subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True)
assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},'Nondeterministic builder'
result=dict(status='PASS',browser_states=states,widths=[1200,375],keyboard_reset=True,no_js=True,print=True,pdf_pages=5,inline_source_parity=True,deterministic_build=True,local_links=links,galleries=True,portable_figures=2,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l190_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
