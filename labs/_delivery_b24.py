"""Test actual learner wiring, desktop/mobile controls, notebook figures and site links."""
import copy,functools,http.server,itertools,json,os,tempfile,threading,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from playwright.sync_api import sync_playwright
P=Path(__file__).resolve().parent;R=P.parent;V=R/'reviews/lesson-b24';S='b24-architecture-thesis-defense';start=time.monotonic()
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/f'{S}.ipynb',4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',4);indices=[i for i,c in enumerate(student.cells) if c.metadata.get('learner_function')];assert len(indices)==3
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code');assert sum(c.source.count('data:image/png;base64,') for c in student.cells)==3
wrong={'paired_effect':"def paired_effect(*args):\n    return dict(positive=10,mean=.1)",'defense_gate':"def defense_gate(*args):\n    return dict(eligible=True,learner='PASS')",'falsification_contract':"def falsification_contract(*args):\n    return dict(ready=True,tests=2,execution='NOT_RUN')"}
for blank in [True,False]:
 for i in indices:
  n=copy.deepcopy(solution);name=n.cells[i].metadata['learner_function'];n.cells[i].source=student.cells[i].source if blank else wrong[name]
  with tempfile.TemporaryDirectory(prefix='b24-learner-') as td:
   try:NotebookClient(n,timeout=90,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
   except CellExecutionError as e:assert ('NotImplementedError' if blank else 'AssertionError') in str(e)
   else:raise AssertionError('Wrong learner accepted '+name)
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}/';errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None);page.on('requestfailed',lambda req:errors.append(req.url));page.on('response',lambda r:errors.append(r.url) if r.url.startswith(base) and r.status>=400 else None)
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':950});page.goto(base+f'lessons/{S}.html');board=page.locator('[data-defense-board]')
  for vals in [[2,2,2,2,2],[2,2,2,2,0],[2,2,2,1,1],[1,1,1,1,1]]:
   for el,v in zip(board.locator('[data-score]').all(),vals):el.select_option(str(v))
   for repro,leak in itertools.product([False,True],repeat=2):
    board.locator('[name=reproduction]').set_checked(repro);board.locator('[name=leakage]').set_checked(leak)
    expected=sum(vals)>=8 and min(vals)>0 and repro and not leak
    assert board.locator('output').get_attribute('data-eligible')==str(expected).lower();assert 'pending written defense' in board.locator('output').inner_text();states+=1
  board.locator('button').click();assert board.locator('output').get_attribute('data-total')=='10';assert not board.locator('[name=reproduction]').is_checked();assert board.locator('[name=leakage]').is_checked()
  board.locator('[name=reproduction]').focus();page.keyboard.press('Space');assert board.locator('[name=reproduction]').is_checked();board.locator('button').click()
  f=page.locator('[data-falsification-board]')
  for effect,threshold,matched in itertools.product([-.03,-.005,0,.005,.02,.03],[0,.005,.02],[False,True]):
   for name,value in [('effect',effect),('threshold',threshold)]:f.locator(f'[name={name}]').evaluate('(el,v)=>{el.value=v;el.dispatchEvent(new Event("input",{bubbles:true}));}',str(value))
   f.locator('[name=matched]').set_checked(matched);expected='INCOMPARABLE' if not matched else 'REVISE' if effect<=threshold else 'SURVIVES_THIS_TEST';assert f.locator('output').get_attribute('data-state')==expected;states+=1
  f.locator('button').click();assert f.locator('[name=effect]').input_value()=='-0.005';assert f.locator('[name=threshold]').input_value()=='0';assert f.locator('[name=matched]').is_checked()
  f.locator('[name=effect]').focus();page.keyboard.press('ArrowRight');assert f.locator('[name=effect]').input_value()=='0';assert f.locator('output').get_attribute('data-state')=='REVISE';f.locator('button').click()
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2'),'Lesson page overflow'
  assert page.locator('#b24-warmup').inner_text().strip();page.screenshot(path=str(V/f'lesson-{width}.png'),full_page=True);board.screenshot(path=str(V/f'gate-{width}.png'));f.screenshot(path=str(V/f'falsification-{width}.png'))
  for i,fig in enumerate(page.locator('.defense-figure').all()):
   if width==375:
    assert fig.evaluate('(el)=>el.scrollWidth>el.clientWidth');fig.focus();page.keyboard.press('ArrowRight');fig.evaluate('(el)=>el.scrollLeft=0')
   fig.screenshot(path=str(V/f'figure-{i}-{width}.png'))
  for choice in ['yes','no']:
   page.goto(base+f'lessons/{S}.html');pred=page.locator('#b24-predict');assert pred.locator('.predict-reveal').is_disabled();pred.locator('[data-value='+choice+']').click();pred.locator('.predict-reveal').click();assert pred.locator('.predict-outcome').inner_text().strip();states+=1
  teach=page.locator('#b24-teachback');teach.locator('textarea').fill('B23 is a released flat-input comparison with a small positive mean and six of ten positive support effects. I would test graph utility against tuned trees and a strong flat model with matched legal information. I will reserve an untouched task and preregister a reversal rule. Historical availability remains open and my writing still needs assessment.');teach.locator('button').first.click();assert 'graph-specific operation' in teach.inner_text()
  page.goto(base+f'labs/html/{S}.html');assert page.locator('img[src^="data:image/png"]').count()==3;assert page.evaluate('Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)')
  for i,fig in enumerate(page.locator('.defense-figure').all()):fig.screenshot(path=str(V/f'notebook-figure-{i}-{width}.png'))
  page.goto(base+'reference/b24-proposal-template.html');assert page.locator('h2').count()==5;assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2');page.screenshot(path=str(V/f'template-{width}.png'),full_page=True)
 page.goto(base+'index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]');page.goto(base+'notebooks.html');page.wait_for_selector('a[href="labs/'+S+'.ipynb"]')
 page.goto(base+f'lessons/{S}.html');page.emulate_media(media='print');assert page.locator('.defense-controls').first.is_hidden();assert page.locator('.defense-figure img').first.evaluate('(el)=>getComputedStyle(el).minWidth')=='0px';page.screenshot(path=str(V/'print.png'),full_page=True)
 page.goto(base+'reference/b24-proposal-template.html');assert page.locator('h2').nth(1).evaluate('(el)=>getComputedStyle(el).breakBefore')=='page';page.screenshot(path=str(V/'template-print.png'),full_page=True)
 ctx=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':950});q=ctx.new_page();q.goto(base+f'lessons/{S}.html');assert 'NOT ELIGIBLE' in q.locator('[data-defense-board] output').inner_text();assert q.locator('noscript').count()==2;q.screenshot(path=str(V/'nojs.png'),full_page=True);browser.close()
server.shutdown();assert not errors,errors
class Links(HTMLParser):
 def __init__(self):super().__init__();self.urls=[]
 def handle_starttag(self,tag,attrs):self.urls.extend(v for k,v in attrs if k in ('href','src'))
links=0
for file in [R/'lessons'/f'{S}.html',R/'reference'/f'{S}.html',R/'reference/b24-proposal-template.html']:
 parser=Links();parser.feed(file.read_text())
 for url in parser.urls:
  u=urlsplit(url)
  if u.scheme:continue
  target=(file.parent/unquote(u.path)).resolve() if u.path else file
  assert target.is_file(),url;links+=1
out=dict(status='PASS',blank_tasks_rejected=3,wrong_notebook_tasks_rejected=3,browser_states=states,desktop_and_375px=True,keyboard_reset=True,no_js=True,print=True,portable_figures=3,template_sections=5,local_links=links,browser_errors=errors,seconds=time.monotonic()-start)
(P/'_delivery_b24_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
