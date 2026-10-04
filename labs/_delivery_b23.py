"""Actual notebook misuse tests, browser interaction, portable images and local links."""
import copy,functools,http.server,json,os,re,tempfile,threading,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from playwright.sync_api import sync_playwright
P=Path(__file__).resolve().parent;R=P.parent;V=R/'reviews/lesson-b23';S='b23-declared-comparison';start=time.monotonic()
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/f'{S}.ipynb',4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',4);indices=[i for i,c in enumerate(student.cells) if c.metadata.get('learner_function')];assert len(indices)==3
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code');assert sum(c.source.count('data:image/png;base64,') for c in student.cells)==3
wrong={'admit_grid':"def admit_grid(records):\n    return dict(runs=40,predictions=28080,paper_runs=30,course_runs=10)",'paired_comparison':"def paired_comparison(*args):\n    return dict(positive=10,mean=.1,per_seed=[.1]*10,seeds=list(range(10)))",'claim_status':"def claim_status(*args):\n    return dict(numerical='CLOSE',historical='ESTABLISHED',learner='PASS')"}
for blank in [True,False]:
 for i in indices:
  n=copy.deepcopy(solution);name=n.cells[i].metadata['learner_function'];n.cells[i].source=student.cells[i].source if blank else wrong[name]
  with tempfile.TemporaryDirectory(prefix='b23-learner-') as td:
   try:NotebookClient(n,timeout=90,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
   except CellExecutionError as e:assert ('NotImplementedError' if blank else 'AssertionError') in str(e)
   else:raise AssertionError('Wrong learner accepted '+name)
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}/';errors=[];states=0;report=json.loads((P/'evidence/b23/report.json').read_text())
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None);page.on('requestfailed',lambda req:errors.append(req.url));page.on('response',lambda r:errors.append(r.url) if r.url.startswith(base) and r.status>=400 else None)
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':950});page.goto(base+f'lessons/{S}.html');board=page.locator('[data-comparison-board]')
  for arm in ['RDBPFN_single','TabICLv1.1','Logistic']:
   board.locator('[name=reference]').select_option(arm)
   for seed in range(10):
    board.locator('[name=seed]').evaluate('(el,v)=>{el.value=v;el.dispatchEvent(new Event("input",{bubbles:true}));}',str(seed))
    for missing in [False,True]:
     board.locator('[name=missing]').set_checked(missing)
     expected=report['models']['RDBPFN']['per_seed'][seed]-report['models'][arm]['per_seed'][seed]
     assert abs(float(board.locator('output').get_attribute('data-delta'))-expected)<1e-9
     assert board.locator('output').get_attribute('data-state')==('INCOMPLETE' if missing else 'COMPLETE');states+=1
  board.locator('button').click();assert board.locator('[name=seed]').input_value()=='0';assert board.locator('[name=reference]').input_value()=='TabICLv1.1';assert not board.locator('[name=missing]').is_checked()
  board.locator('[name=seed]').focus();page.keyboard.press('ArrowRight');assert board.locator('[name=seed]').input_value()=='1';board.locator('button').click()
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2'),'Page overflow'
  assert page.locator('#b23-warmup').inner_text().strip();page.screenshot(path=str(V/f'lesson-{width}.png'),full_page=True);board.screenshot(path=str(V/f'widget-{width}.png'))
  for i,fig in enumerate(page.locator('.comparison-figure').all()):
   if width==375:assert fig.evaluate('(el)=>el.scrollWidth>el.clientWidth');fig.focus();page.keyboard.press('ArrowRight');fig.evaluate('(el)=>el.scrollLeft=0')
   fig.screenshot(path=str(V/f'figure-{i}-{width}.png'))
  for choice in ['yes','no']:
   page.goto(base+f'lessons/{S}.html');pred=page.locator('#b23-predict');assert pred.locator('.predict-reveal').is_disabled();pred.locator('[data-value='+choice+']').click();pred.locator('.predict-reveal').click();assert pred.locator('.predict-outcome').inner_text().strip();states+=1
  teach=page.locator('#b23-teachback');teach.locator('textarea').fill('I reproduce the released Table9 driver-dnf comparison,512supports and ten paired draws. All three means agree, but historical availability remains unestablished. The fixed logistic baseline is weaker here. I would use an untouched task and tuned trees to try to falsify the thesis.');teach.locator('button').first.click();assert 'complete selected release evaluation' in teach.inner_text()
  page.goto(base+f'labs/html/{S}.html');assert page.locator('img[src^="data:image/png"]').count()==3;assert page.evaluate('Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)')
  for i,fig in enumerate(page.locator('.comparison-figure').all()):fig.screenshot(path=str(V/f'notebook-figure-{i}-{width}.png'))
 page.goto(base+'index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]');page.goto(base+'notebooks.html');page.wait_for_selector('a[href="labs/'+S+'.ipynb"]')
 page.goto(base+f'lessons/{S}.html');page.emulate_media(media='print');assert page.locator('.comparison-controls').is_hidden();assert page.locator('.comparison-figure img').first.evaluate('(el)=>getComputedStyle(el).minWidth')=='0px';page.screenshot(path=str(V/'print.png'),full_page=True)
 ctx=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':950});q=ctx.new_page();q.goto(base+f'lessons/{S}.html');assert '0.022700' in q.locator('output').inner_text();assert q.locator('noscript').inner_text();q.screenshot(path=str(V/'nojs.png'),full_page=True);browser.close()
server.shutdown();assert not errors,errors
class Links(HTMLParser):
 def __init__(self):super().__init__();self.urls=[]
 def handle_starttag(self,tag,attrs):self.urls.extend(v for k,v in attrs if k in ('href','src'))
links=0
for file in [R/'lessons'/f'{S}.html',R/'reference'/f'{S}.html']:
 parser=Links();parser.feed(file.read_text())
 for url in parser.urls:
  u=urlsplit(url)
  if u.scheme:continue
  target=(file.parent/unquote(u.path)).resolve() if u.path else file
  assert target.is_file(),url;links+=1
out=dict(status='PASS',blank_tasks_rejected=3,wrong_notebook_tasks_rejected=3,browser_states=states,desktop_and_375px=True,keyboard_reset=True,no_js=True,print=True,portable_figures=3,local_links=links,browser_errors=errors,seconds=time.monotonic()-start)
(P/'_delivery_b23_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
