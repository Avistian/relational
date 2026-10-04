"""Portable learner/source gates and exhaustive browser controls, desktop/mobile."""
import base64,copy,functools,http.server,json,os,re,shutil,subprocess,sys,tempfile,threading,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from playwright.sync_api import sync_playwright
P=Path(__file__).resolve().parent;R=P.parent;V=R/'reviews/lesson-b16';S='b16-autograble-graph-selection';start=time.monotonic();V.mkdir(parents=True,exist_ok=True)
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
r=json.loads((P/'evidence/b16/diagnostic.json').read_text())
student=nbformat.read(P/f'{S}.ipynb',4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
indices=[i for i,c in enumerate(student.cells) if c.cell_type=='code' and 'raise NotImplementedError("Complete' in c.source]
assert len(indices)==3 and all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
images=re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)','\n'.join(c.source for c in student.cells));assert len(images)==4
assert all(base64.b64decode(x).startswith(b'\x89PNG\r\n\x1a\n') for x in images)
# Each TODO must be live; solve the other two and leave this one blank.
for index in indices:
 n=copy.deepcopy(student)
 for j in indices:
  if j!=index:n.cells[j].source=solution.cells[j].source
 with tempfile.TemporaryDirectory(prefix='b16-blank-') as td:
  try:NotebookClient(n,timeout=90,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
  except CellExecutionError as e:assert 'NotImplementedError' in str(e)
  else:raise AssertionError('A blank task passed')
# Source authentication and refusal must be real, including corruption.
with tempfile.TemporaryDirectory(prefix='b16-gate-') as td:
 t=Path(td);shutil.copytree(P/'sources/b16',t/'sources/b16');(t/'evidence/b16').mkdir(parents=True)
 shutil.copy(P/'evidence/b16/paper-status.json',t/'evidence/b16/paper-status.json');shutil.copy(P/'_reproduce_b16.py',t/'_reproduce_b16.py')
 good=subprocess.run([sys.executable,str(t/'_reproduce_b16.py'),'--phase','audit'],capture_output=True,text=True);assert good.returncode==0
 blocked=subprocess.run([sys.executable,str(t/'_reproduce_b16.py'),'--phase','paper'],capture_output=True,text=True);assert blocked.returncode!=0 and 'NOT_RUN:' in blocked.stderr
 (P/'evidence/b16/paper-refusal.txt').write_text(blocked.stdout+blocked.stderr)
 f=t/'sources/b16/autograble/src/autograble/selection.py';f.write_text(f.read_text()+'\n# corruption\n')
 bad=subprocess.run([sys.executable,str(t/'_reproduce_b16.py'),'--phase','audit'],capture_output=True,text=True);assert bad.returncode!=0 and 'SOURCE_HASH_MISMATCH' in bad.stderr
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}/';errors=[];states=0;geometry=0
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None);page.on('requestfailed',lambda req:errors.append(req.url));page.on('response',lambda response:errors.append(response.url) if response.url.startswith(base) and response.status>=400 else None)
 for width in [1200,375]:
  page.set_viewport_size({'width':width,'height':950});page.goto(base+f'lessons/{S}.html')
  board=page.locator('[data-partition-selection]')
  for x in [x for x in r['subsets'] if x['world']=='signal']:
   board.locator('[name=subset]').select_option(','.join(x['cols']));board.locator('[name=penalty]').select_option(str(x['penalty']))
   assert abs(float(board.locator('output').get_attribute('data-j'))-x['J'])<1e-12
   assert board.locator('.ps-block').count()==len(x['sizes']);states+=1
   board.screenshot(path=str(V/f'partition-{width}-{x["penalty"]}-{"-".join(x["cols"]) or "empty"}.png'))
  board.locator('button').click();assert board.locator('[name=subset]').input_value()=='A'
  board.locator('[name=penalty]').focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert board.locator('[name=penalty]').input_value()=='1';board.locator('button').click()
  search=page.locator('[data-greedy-search]')
  for x in r['selections']:
   search.locator('[name=world]').select_option(x['world']);search.locator('[name=penalty]').select_option(str(x['penalty']));search.locator('[name=method]').select_option(x['method'])
   assert json.loads(search.locator('output').get_attribute('data-selected'))==x['selected'];assert search.locator('[data-selected=true]').count()==1;states+=1
   search.screenshot(path=str(V/f'search-{width}-{x["world"]}-{x["penalty"]}-{x["method"]}.png'))
  search.locator('button').click();assert search.locator('[name=world]').input_value()=='xor'
  graph=page.locator('[data-incidence-boundary]')
  for identity in ['typed','anonymous']:
   for feature in ['none','B']:
    graph.locator('[name=identity]').select_option(identity);graph.locator('[name=features]').select_option(feature)
    assert int(graph.locator('output').get_attribute('data-blocks'))==({'typed':2,'anonymous':1}[identity]*(2 if feature=='B' else 1));states+=1
    # Filled graph shapes and all text must stay inside the authored viewBox.
    assert graph.locator('svg').evaluate('''svg=>{const v=svg.viewBox.baseVal;return [...svg.querySelectorAll('text,rect,circle')].every(e=>{const b=e.getBBox();return b.x>=v.x-.1&&b.y>=v.y-.1&&b.x+b.width<=v.x+v.width+.1&&b.y+b.height<=v.y+v.height+.1})}'''),'SVG bounds'
    assert graph.locator('svg').evaluate('''svg=>{const r=[...svg.querySelectorAll('rect,circle')].map(e=>e.getBBox());return r.every((a,i)=>r.slice(i+1).every(b=>!(a.x<b.x+b.width&&a.x+a.width>b.x&&a.y<b.y+b.height&&a.y+a.height>b.y)))}'''),'Node overlap';geometry+=1
    graph.screenshot(path=str(V/f'graph-{width}-{identity}-{feature}.png'))
  graph.locator('button').click();assert graph.locator('[name=features]').input_value()=='none'
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2'),'Horizontal overflow'
  assert page.locator('#b16-warmup').inner_text().strip();assert page.locator('#b16-predict').inner_text().strip();assert page.locator('#b16-teachback textarea').count()==1
  page.screenshot(path=str(V/f'lesson-{width}.png'),full_page=True)
  for name,b in [('partition',board),('search',search),('graph',graph)]:b.screenshot(path=str(V/f'{name}-default-{width}.png'))
 page.emulate_media(media='print');assert not graph.locator('button').is_visible();assert graph.locator('output').is_visible();page.screenshot(path=str(V/'print.png'),full_page=True);page.emulate_media(media='screen')
 nojs=browser.new_page(java_script_enabled=False);nojs.goto(base+f'lessons/{S}.html');assert nojs.locator('noscript').first.is_visible();assert 'A: risk 0' in nojs.locator('[data-partition-selection] output').inner_text();nojs.close()
 page.goto(base+'index.html');page.wait_for_selector('a[href="lessons/'+S+'.html"]')
 page.goto(base+'notebooks.html');page.wait_for_selector('a[href="labs/'+S+'.ipynb"]')
 page.set_viewport_size({'width':1000,'height':900});page.goto(base+f'labs/html/{S}.html');assert page.locator('img').count()==4
 for i in range(4):page.locator('img').nth(i).screenshot(path=str(V/f'notebook-{i}.png'))
 browser.close()
server.shutdown();assert not errors,errors
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.ids=set()
 def handle_starttag(self,t,a):
  self.links.extend(v for k,v in a if k in ('href','src'));self.ids.update(v for k,v in a if k in ('id','name'))
count=0
for name in ['lessons/'+S+'.html','reference/'+S+'.html']:
 path=R/name;p=Links();p.feed(path.read_text())
 for url in p.links:
  u=urlsplit(url)
  if u.scheme:continue
  target=(path.parent/unquote(u.path)).resolve() if u.path else path
  assert target.is_file(),url;count+=1
  if u.fragment and target.suffix=='.html':
   other=Links();other.feed(target.read_text());assert u.fragment in other.ids,url
out=dict(status='PASS',live_blank_tasks_rejected=3,embedded_pngs=4,portable_source_gate='PASS',corrupt_source_rejected=True,paper_attempt='REFUSED_AS_REQUIRED',browser_states=states,svg_geometry_states=geometry,keyboard_reset='PASS',desktop_mobile='PASS',print_nojs='PASS',local_links=count,browser_errors=errors,seconds=time.monotonic()-start,live_colab='NOT_CHECKED',deployment='NOT_REQUESTED')
(P/'_delivery_b16_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
