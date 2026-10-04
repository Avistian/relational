"""Behavioral learner, source, browser and link checks for B19."""
import base64,copy,functools,http.server,json,os,re,shutil,subprocess,sys,tempfile,threading,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from playwright.sync_api import sync_playwright
P=Path(__file__).resolve().parent;R=P.parent;V=R/'reviews/lesson-b19';S='b19-benchmark-evidence';start=time.monotonic()
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
r=json.loads((P/'evidence/b19/diagnostic.json').read_text())
student=nbformat.read(P/f'{S}.ipynb',4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
indices=[i for i,c in enumerate(student.cells) if c.cell_type=='code' and 'raise NotImplementedError("Implement' in c.source]
assert len(indices)==3 and all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
images=re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)','\n'.join(c.source for c in student.cells));assert len(images)==3
assert all(base64.b64decode(x).startswith(b'\x89PNG\r\n\x1a\n') for x in images)
for index in indices:
    n=copy.deepcopy(student)
    for j in indices:
        if j!=index:n.cells[j].source=solution.cells[j].source
    with tempfile.TemporaryDirectory(prefix='b19-blank-') as td:
        try:NotebookClient(n,timeout=90,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
        except CellExecutionError as e:assert 'NotImplementedError' in str(e)
        else:raise AssertionError('Blank learner function passed')
import _test_b19 as tests
for name,test,bad in [('split_audit','test_split',lambda *a,**k:dict(train_n=2,test_n=2,overlap_groups=[],group_disjoint=True)),('paired_scores','test_pairs',lambda *a,**k:dict(keys=[['x',0],['y',0]],means={'A':.1})),('dataset_summary','test_datasets',lambda rows:dict(n_datasets=len(rows),mean=.05,se_dataset=.1))]:
 old=getattr(tests,name);setattr(tests,name,bad)
 try:tests.Boundaries(test).debug()
 except (AssertionError,KeyError):pass
 else:raise AssertionError('Wrong learner function passed')
 finally:setattr(tests,name,old)
with tempfile.TemporaryDirectory(prefix='b19-source-') as td:
 t=Path(td);shutil.copytree(P/'sources/b19',t/'sources/b19');shutil.copytree(P/'evidence/b19',t/'evidence/b19');shutil.copy(P/'_reproduce_b19.py',t/'_reproduce_b19.py')
 good=subprocess.run([sys.executable,str(t/'_reproduce_b19.py'),'--phase','audit'],capture_output=True,text=True);assert good.returncode==0,good.stderr
 blocked=subprocess.run([sys.executable,str(t/'_reproduce_b19.py'),'--phase','paper'],capture_output=True,text=True);assert blocked.returncode!=0 and 'NOT_RUN:' in blocked.stderr
 (P/'evidence/b19/paper-refusal.txt').write_text(blocked.stdout+blocked.stderr)
 f=t/'sources/b19/paper.html';f.write_bytes(f.read_bytes()+b'corrupted')
 bad=subprocess.run([sys.executable,str(t/'_reproduce_b19.py'),'--phase','audit'],capture_output=True,text=True);assert bad.returncode!=0 and 'SOURCE_HASH_MISMATCH' in bad.stderr
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}/';errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None);page.on('requestfailed',lambda req:errors.append(req.url));page.on('response',lambda response:errors.append(response.url) if response.url.startswith(base) and response.status>=400 else None)
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':950});page.goto(base+f'lessons/{S}.html')
  board=page.locator('[data-split-board]')
  for seed in range(3):
   for regime in ['random','grouped']:
    board.locator('[name=seed]').select_option(str(seed));board.locator('[name=regime]').select_option(regime)
    expected=next(a for a in r['arms'] if a['seed']==seed and a['regime']==regime and a['model']=='group_memory')
    assert abs(float(board.locator('output').get_attribute('data-memory'))-expected['brier'])<1e-12
    assert board.locator('.evidence-group').count()==24
    assert board.locator('.mixed').count()==(24 if regime=='random' else 0)
    assert board.locator('.held').count()==(8 if regime=='grouped' else 0);states+=1
  board.locator('button').click();assert board.locator('output').get_attribute('data-key')=='0/grouped'
  board.locator('[name=seed]').focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert board.locator('[name=seed]').input_value()=='1';board.locator('button').click()
  for policy in ['measured','rf']:
   panel=page.locator('[data-missing-board]');panel.locator('select').select_option(policy);assert panel.locator('output').get_attribute('data-winner')==('A' if policy=='measured' else 'B');states+=1
  panel.locator('button').click();assert panel.locator('select').input_value()=='measured'
  for replicas in ['1','2']:
   panel=page.locator('[data-uncertainty-board]');panel.locator('select').select_option(replicas);assert panel.locator('output').get_attribute('data-rows')==str(9*int(replicas));states+=1
  panel.locator('button').click();assert panel.locator('select').input_value()=='1'
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2'),'Horizontal page overflow'
  assert page.locator('#b19-warmup').inner_text().strip();assert page.locator('#b19-predict').inner_text().strip();assert page.locator('#b19-teachback textarea').count()==1
  page.screenshot(path=str(V/f'lesson-{width}.png'),full_page=True)
  for selector,name in [('[data-split-board]','split'),('[data-missing-board]','missing'),('[data-uncertainty-board]','uncertainty'),('[aria-label="Evaluation computation"]','pipeline'),('[aria-label="Versioned evidence"]','versions')]:page.locator(selector).screenshot(path=str(V/f'{name}-{width}.png'))
  for i,fig in enumerate(page.locator('.evidence-figure').all()):
   if width==375:
    assert fig.evaluate('(el)=>el.scrollWidth>el.clientWidth')
    fig.focus();page.keyboard.press('End');fig.evaluate('(el)=>el.scrollLeft=0')
   fig.screenshot(path=str(V/f'figure-{i}-{width}.png'))
 page.emulate_media(media='print');assert not board.locator('button').is_visible();assert board.locator('output').is_visible();page.screenshot(path=str(V/'print.png'),full_page=True);page.emulate_media(media='screen')
 nojs=browser.new_page(java_script_enabled=False);nojs.goto(base+f'lessons/{S}.html');assert nojs.locator('noscript').first.is_visible();assert nojs.locator('table').count()>=3;nojs.close()
 page.goto(base+'index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]')
 page.goto(base+'notebooks.html');page.wait_for_selector('a[href="labs/'+S+'.ipynb"]')
 page.set_viewport_size({'width':1000,'height':950});page.goto(base+f'labs/html/{S}.html');assert page.locator('img[src^="data:image/png"]').count()==3
 page.locator('img[src^="data:image/png"]').first.screenshot(path=str(V/'notebook-figure.png'))
 browser.close()
server.shutdown();assert not errors,errors
class Links(HTMLParser):
 def __init__(self):super().__init__();self.urls=[]
 def handle_starttag(self,tag,attrs):self.urls.extend(v for k,v in attrs if k in ('href','src'))
links=0
for path in [R/'lessons'/f'{S}.html',R/'reference'/f'{S}.html']:
 parser=Links();parser.feed(path.read_text())
 for url in parser.urls:
  u=urlsplit(url)
  if u.scheme:continue
  target=(path.parent/unquote(u.path)).resolve() if u.path else path
  assert target.is_file(),url;links+=1
out=dict(status='PASS',blank_tasks_rejected=3,wrong_functions_rejected=3,corrupted_source_rejected=True,paper_refusal=True,desktop_mobile_states=states,keyboard_reset=True,nojs_print=True,portable_figures=3,local_links=links,browser_errors=errors,seconds=time.monotonic()-start,live_colab='NOT_CHECKED')
(P/'_delivery_b19_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
