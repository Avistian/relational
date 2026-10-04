"""Exercise every browser mode, live learner task and fail-closed evidence gate."""
import base64,copy,functools,http.server,json,os,subprocess,sys,tempfile,threading,time,re,shutil
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from playwright.sync_api import sync_playwright
from _verify_b17 import verify
import _test_b17 as primitive
P=Path(__file__).resolve().parent;R=P.parent;V=R/'reviews/lesson-b17';S='b17-reusable-representations';start=time.monotonic()
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
r=json.loads((P/'evidence/b17/diagnostic.json').read_text());student=nbformat.read(P/f'{S}.ipynb',4);solution=nbformat.read(P/'solutions'/f'{S}.ipynb',4)
indices=[i for i,c in enumerate(student.cells) if c.cell_type=='code' and 'raise NotImplementedError("Complete' in c.source]
assert len(indices)==3 and all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
assert all('from relkit' not in c.source for c in student.cells)
images=re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)','\n'.join(c.source for c in student.cells));assert len(images)==3
for value in images:assert base64.b64decode(value).startswith(b'\x89PNG')
for index in indices:
    n=copy.deepcopy(student)
    for j in indices:
        if j!=index:n.cells[j].source=solution.cells[j].source
    with tempfile.TemporaryDirectory(prefix='b17-blank-') as td:
        try:NotebookClient(n,timeout=90,kernel_name='python3',resources={'metadata':{'path':td}}).execute(start_new_session=True)
        except CellExecutionError as e:assert 'NotImplementedError' in str(e)
        else:raise AssertionError('Blank learner function passed')
mutations=[('attention','test_attention',lambda q,k,v:v[:1].expand_as(q)),('aggregate_layers','test_aggregation',lambda rows,projections,final:final(projections[0](rows[0]))),('cache_identity','test_cache',lambda *args:'same-shape')]
for name,test,bad in mutations:
    original=getattr(primitive,name);setattr(primitive,name,bad)
    try:
        try:primitive.Boundaries(test).debug()
        except AssertionError:pass
        else:raise AssertionError('Mutant passed: '+name)
    finally:setattr(primitive,name,original)
for kind in ['prediction','embedding','id','support-label','coverage','cache-key','query-label-coverage']:
    bad=copy.deepcopy(r);item=bad['seeds'][0]['records'][0]
    if kind=='prediction':item['probability'][0]+=.01
    elif kind=='embedding':item['embedding'][0][0]+=.01
    elif kind=='id':item['ids'][0]='forged'
    elif kind=='support-label':item['support_labels'][0]=1
    elif kind=='coverage':bad['seeds'][0]['records'].pop()
    elif kind=='cache-key':item['cache_key']='forged'
    else:bad['seeds'][0]['checks'][0]['external_query_labels']=[1,1]
    try:verify(bad)
    except AssertionError:pass
    else:raise AssertionError('Corrupt evidence passed: '+kind)
with tempfile.TemporaryDirectory(prefix='b17-gate-') as td:
    t=Path(td);shutil.copytree(P/'sources/b17',t/'sources/b17');shutil.copy(P/'_reproduce_b17.py',t/'_reproduce_b17.py')
    good=subprocess.run([sys.executable,str(t/'_reproduce_b17.py'),'--phase','audit'],capture_output=True,text=True);assert good.returncode==0
    blocked=subprocess.run([sys.executable,str(t/'_reproduce_b17.py'),'--phase','paper'],capture_output=True,text=True);assert blocked.returncode!=0 and 'NOT_RUN' in blocked.stderr
    (P/'evidence/b17/paper-refusal.txt').write_text(blocked.stdout+blocked.stderr)
    f=t/'sources/b17/flextab-v2.html';f.write_text(f.read_text()+'\ncorrupt')
    bad=subprocess.run([sys.executable,str(t/'_reproduce_b17.py'),'--phase','audit'],capture_output=True,text=True);assert bad.returncode!=0 and 'SOURCE_HASH_MISMATCH' in bad.stderr
class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*a):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}/';errors=[];states=0
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('requestfailed',lambda req:errors.append(req.url));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None);page.on('response',lambda resp:errors.append(resp.url) if resp.url.startswith(base) and resp.status>=400 else None)
    for width in [1200,375]:
        page.set_viewport_size(dict(width=width,height=950));page.goto(base+f'lessons/{S}.html');board=page.locator('[data-representation-boundary]');att=page.locator('[data-single-key]')
        for seed in r['seeds']:
            for x in seed['records']:
                for name,value in [('seed',str(seed['seed'])),('task',x['task']),('mode',x['mode'])]:board.locator(f'[name={name}]').select_option(value)
                out=board.locator('output');assert abs(float(out.get_attribute('data-embedding-delta'))-x['embedding_delta'])<1e-12;assert abs(float(out.get_attribute('data-prediction-delta'))-x['prediction_delta'])<1e-12
                assert out.get_attribute('data-cache-hit')==str(x['cache_hit']).lower();states+=1
        board.locator('button').click();assert board.locator('[name=mode]').input_value()=='baseline'
        board.locator('[name=mode]').focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter');assert board.locator('[name=mode]').input_value()=='support-labels'
        board.screenshot(path=str(V/f'boundary-{width}.png'))
        for mode,expected in [('one',(2,-1)),('two',(.5,2))]:
            att.locator('select').select_option(mode);assert tuple(float(att.locator('output').get_attribute('data-'+k)) for k in ['first','second'])==expected;states+=1
        att.screenshot(path=str(V/f'attention-{width}.png'));att.locator('button').click();assert att.locator('select').input_value()=='one'
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2'),'Horizontal overflow'
        assert page.locator('#b17-warmup').inner_text().strip();assert page.locator('#b17-predict').inner_text().strip();assert page.locator('#b17-teachback textarea').count()==1
        for label,name in [('FlexTab model architecture','flextab'),('GTAlign model architecture','gtalign')]:
            page.locator(f'[aria-label="{label}"]').screenshot(path=str(V/f'{name}-{width}.png'))
        page.screenshot(path=str(V/f'lesson-{width}.png'),full_page=True)
    page.emulate_media(media='print');assert not board.locator('button').is_visible();assert board.locator('output').is_visible();page.screenshot(path=str(V/'print.png'),full_page=True);page.emulate_media(media='screen')
    nojs=browser.new_page(java_script_enabled=False);nojs.goto(base+f'lessons/{S}.html');assert nojs.locator('noscript').first.is_visible();assert 'Maximum embedding' in nojs.locator('[data-representation-boundary] output').inner_text();nojs.close()
    page.goto(base+'index.html');page.wait_for_selector(f'a[href="lessons/{S}.html"]');page.goto(base+'notebooks.html');page.wait_for_selector(f'a[href="labs/{S}.ipynb"]')
    page.set_viewport_size(dict(width=1000,height=950));page.goto(base+f'labs/html/{S}.html');assert page.locator('img').count()==3
    for i in range(3):page.locator('img').nth(i).screenshot(path=str(V/f'notebook-{i}.png'))
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
            q=Links();q.feed(target.read_text());assert u.fragment in q.ids,url
out=dict(status='PASS',blank_tasks_rejected=3,wrong_implementations_rejected=3,corrupt_results_rejected=7,source_corruption_rejected=True,paper_gate='REFUSED_AS_REQUIRED',browser_states=states,desktop_mobile='PASS',keyboard_reset='PASS',print_nojs='PASS',local_links=count,embedded_figures=3,browser_errors=errors,seconds=time.monotonic()-start,live_colab='NOT_CHECKED',deployment='NOT_REQUESTED')
(P/'_delivery_b17_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
