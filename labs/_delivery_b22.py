"""Test real notebook tasks, historical gate, live widgets and rendered layouts."""
import copy,functools,http.server,json,os,re,shutil,subprocess,sys,tempfile,threading,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from playwright.sync_api import sync_playwright
P=Path(__file__).resolve().parent;R=P.parent;V=R/'reviews/lesson-b22';S='b22-support-state-refinement';started=time.monotonic()
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/f'{S}.ipynb',4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
indices=[i for i,c in enumerate(student.cells) if c.cell_type=='code' and c.metadata.get('learner_function')];assert len(indices)==3
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert len(re.findall('data:image/png;base64,','\n'.join(c.source for c in student.cells)))==4
wrong={'attention_read':'def attention_read(query, keys, values):\n    return np.asarray(query) @ np.asarray(keys).T @ np.asarray(values)',
'replace_support':"def replace_support(before, after, n_support, mode):\n    return np.asarray(before).copy() if mode=='skip' else np.asarray(after).copy()",
'paired_effect':"def paired_effect(baseline_logits, changed_logits, labels):\n    return dict(delta_ce=0.,delta_accuracy_pp=0.)"}
for blank in [True,False]:
 for index in indices:
  n=copy.deepcopy(solution);name=n.cells[index].metadata['learner_function']
  n.cells[index].source=student.cells[index].source if blank else wrong[name]
  with tempfile.TemporaryDirectory(prefix='b22-learner-') as td:
   try:NotebookClient(n,timeout=90,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
   except CellExecutionError as e:assert ('NotImplementedError' if blank else 'AssertionError') in str(e)
   else:raise AssertionError('Invalid learner implementation accepted: '+name)
with tempfile.TemporaryDirectory(prefix='b22-gate-') as td:
 t=Path(td);shutil.copytree(P/'sources/b22',t/'sources/b22');shutil.copy(P/'_reproduce_b22.py',t/'_reproduce_b22.py')
 proc=subprocess.run([sys.executable,str(t/'_reproduce_b22.py'),'--fresh'],capture_output=True,text=True);assert proc.returncode!=0 and 'NOT_RUN: INCOMPLETE_SOURCE_PROTOCOL_GATE' in proc.stderr
 f=t/'sources/b22/paper.html';f.write_bytes(f.read_bytes()+b'altered')
 proc=subprocess.run([sys.executable,str(t/'_reproduce_b22.py')],capture_output=True,text=True);assert proc.returncode!=0 and 'SOURCE_HASH_MISMATCH' in proc.stderr
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}/';errors=[];states=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
 page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None);page.on('requestfailed',lambda req:errors.append(req.url));page.on('response',lambda r:errors.append(r.url) if r.url.startswith(base) and r.status>=400 else None)
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':950});page.goto(base+f'lessons/{S}.html');board=page.locator('[data-support-board]')
  for mode in ['normal','identity','skip','permute']:
   board.locator('[name=mode]').select_option(mode)
   for a in [0,.25,.5,.75,1]:
    for gate in [0,.25,.5,.75,1]:
     for name,value in [('attention',a),('gate',gate)]:board.locator(f'[name={name}]').evaluate('(el,v)=>{el.value=v;el.dispatchEvent(new Event("input",{bubbles:true}));}',str(value))
     v=[1,3] if mode=='skip' else [3,4] if mode=='permute' else [2,5]
     baseline=4+gate*(a*2+(1-a)*5);changed=4+gate*(a*v[0]+(1-a)*v[1])
     assert abs(float(board.locator('output').get_attribute('data-delta'))-(changed-baseline))<1e-6
     assert '4 (preserved)' in board.locator('[data-changed]').inner_text()
     shown=float(re.search(r'Final scalar state: ([0-9.]+)',board.locator('[data-changed]').inner_text()).group(1));assert abs(shown-changed)<=.000500001;states+=1
  board.locator('button').click();assert board.locator('[name=mode]').input_value()=='skip';assert float(board.locator('[name=attention]').input_value())==.75;assert float(board.locator('[name=gate]').input_value())==.5
  board.locator('[name=attention]').focus();page.keyboard.press('ArrowLeft');assert float(board.locator('[name=attention]').input_value())==.5;board.locator('button').click()
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2'),'Horizontal page overflow'
  assert page.locator('#b22-warmup').inner_text().strip()
  page.screenshot(path=str(V/f'lesson-{width}.png'),full_page=True);board.screenshot(path=str(V/f'widget-{width}.png'))
  for i,fig in enumerate(page.locator('.support-figure').all()):
   if width==375:
    assert fig.evaluate('(el)=>el.scrollWidth>el.clientWidth');fig.focus();page.keyboard.press('ArrowRight');fig.evaluate('(el)=>el.scrollLeft=0')
   fig.screenshot(path=str(V/f'figure-{i}-{width}.png'))
  for choice in ['no','yes']:
   page.goto(base+f'lessons/{S}.html');pred=page.locator('#b22-predict');assert pred.locator('.predict-reveal').is_disabled();pred.locator('[data-value='+choice+']').click();pred.locator('.predict-reveal').click();assert pred.locator('.predict-outcome').inner_text().strip();states+=1
  teach=page.locator('#b22-teachback');teach.locator('textarea').fill('I restore support input only, preserve query output, and run later blocks on the changed memory. Identity must leave predictions unchanged. A random write need not help; the trained checkpoint, exact episodes and paired metric recipe are still needed for the historical claim.');teach.locator('button').first.click();assert 'I branch after' in teach.inner_text()
  page.goto(base+f'labs/html/{S}.html');assert page.locator('img[src^="data:image/png"]').count()>=4;assert page.evaluate('Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)')
  for i,fig in enumerate(page.locator('.support-figure').all()):fig.screenshot(path=str(V/f'notebook-figure-{i}-{width}.png'))
 page.goto(base+'index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]');page.goto(base+'notebooks.html');page.wait_for_selector('a[href="labs/'+S+'.ipynb"]')
 page.goto(base+f'lessons/{S}.html');page.emulate_media(media='print');assert page.locator('.support-controls').is_hidden();assert page.locator('.support-figure img').first.evaluate('(el)=>getComputedStyle(el).minWidth')=='0px';page.screenshot(path=str(V/'print.png'),full_page=True)
 context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':950});p=context.new_page();p.goto(base+f'lessons/{S}.html');assert '5.375' in p.locator('output').inner_text();assert p.locator('noscript').inner_text();p.screenshot(path=str(V/'nojs.png'),full_page=True);browser.close()
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
out=dict(status='PASS',blank_tasks_rejected=3,wrong_notebook_tasks_rejected=3,source_corruption_rejected=True,historical_gate=True,browser_states=states,desktop_and_375px=True,keyboard_reset=True,no_js=True,print=True,portable_figures=4,local_links=links,browser_errors=errors,seconds=time.monotonic()-started)
(P/'_delivery_b22_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
