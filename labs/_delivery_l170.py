"""Real browser, notebook portability, canonical source and publication checks."""
import ast,functools,hashlib,json,os,re,subprocess,tempfile,threading
from pathlib import Path
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
import nbformat
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
from relkit.design_l170 import claim_gate
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0170-fm-design-checkpoint'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
student=nbformat.read(P/(S+'.ipynb'),4);solution=nbformat.read(P/'solutions'/(S+'.ipynb'),4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
assert all(not c.outputs for c in student.cells if c.cell_type=='code')
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in solution.cells if c.cell_type=='code')
assert sum(c.source.count('data:image/png;base64,') for c in solution.cells)==3
assert not any(any('def '+name+'(' in c.source for name in ['paired_design','adaptation_mode','claim_gate']) for c in student.cells if c.cell_type=='code' and 'raise NotImplementedError' not in c.source)
code='\n\n'.join(c.source for c in solution.cells if c.cell_type=='code')
assert hashlib.sha256(code.encode()).hexdigest()==json.loads((P/'_execution_l170_results.json').read_text())['executed_code_sha256']
function_cells={ast.dump(n,include_attributes=False) for c in solution.cells if c.cell_type=='code' for n in ast.parse(c.source).body if isinstance(n,ast.FunctionDef)}
for file in ['relkit/design_l170.py','relkit/scaling_l169.py','_audit_l169.py','_check_l170.py','_replay_l170.py']:
    for node in ast.parse((P/file).read_text()).body:
        if isinstance(node,ast.FunctionDef):assert ast.dump(node,include_attributes=False) in function_cells,(file,node.name)
for source in json.loads((P/'sources/l170/source-ledger.json').read_text())['sources']:
    assert hashlib.sha256((P/'sources/l170'/(source['name']+'.html')).read_bytes()).hexdigest()==source['sha256']
# Every input byte must still match the pinned evidence after authoring.
manifest=json.loads((P/'evidence/l170/input-manifest.json').read_text())
for name,digest in manifest['files'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,name
keys=['artifacts','metrics','dfs','temporal','exposure','fresh_training','matched_pipeline','heldout_databases','repeatability'];claims=['saved_replay','full_pipeline','fresh_pretraining','general_advantage','exact_repeatability']
expected=[claim_gate(claim,{k:bool(mask&(1<<i)) for i,k in enumerate(keys)}) for claim in claims for mask in range(512)]
errors=[];states=0
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
    page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None);page.on('requestfailed',lambda req:errors.append(req.url))
    for width in [1200,375]:
        page.set_viewport_size({'width':width,'height':900});page.goto((R/'lessons'/(S+'.html')).as_uri());host=page.locator('#design-explorer')
        assert host.get_attribute('data-verdict')=='NOT_ESTABLISHED'
        assert host.get_attribute('data-missing')=='matched_pipeline,heldout_databases,temporal,exposure'
        actual=host.evaluate('''(host,args)=>{
            const out=[],select=host.querySelector('[data-claim]');
            for(const claim of args.claims){select.value=claim;
                for(let mask=0;mask<512;mask++){
                    args.keys.forEach((key,i)=>host.querySelector('[data-key="'+key+'"]').checked=!!(mask&(1<<i)));
                    select.dispatchEvent(new Event('change',{bubbles:true}));
                    out.push({status:host.dataset.verdict,missing:host.dataset.missing?host.dataset.missing.split(','):[]});
                    if(!host.querySelector('output').textContent.includes(host.dataset.verdict))throw Error('Readout stale');
                    if(!host.querySelector('[data-baseline]').textContent.includes('Authenticated baseline'))throw Error('Baseline missing');
                }
            }return out;
        }''',dict(keys=keys,claims=claims))
        assert actual==expected;states+=len(actual)
        host.locator('[data-reset]').click();assert host.locator('[data-claim]').input_value()=='general_advantage'
        host.locator('[data-claim]').select_option('saved_replay');assert host.get_attribute('data-verdict')=='READY_FOR_REVIEW'
        check=host.locator('[data-key="artifacts"]');check.focus();page.keyboard.press('Space');assert not check.is_checked() and host.get_attribute('data-verdict')=='NOT_ESTABLISHED'
        host.locator('[data-reset]').click();assert check.is_checked()
        assert all(host.locator('[data-key="'+k+'"]').is_checked()==(k in ['artifacts','metrics']) for k in keys)
        predict=page.locator('#predict');assert predict.locator('.predict-reveal').is_disabled()
        labels=predict.locator('.predict-option').all_inner_texts();assert len(set(len(x.split()) for x in labels))==1
        predict.locator('[data-value="scoped"]').click();predict.locator('.predict-reveal').click();assert 'two tasks' in predict.locator('.predict-outcome').inner_text()
        assert page.locator('#warmup button').count()==0 and page.locator('#teachback textarea').count()==1
        assert page.locator('figure img').evaluate_all('(xs)=>xs.length===3&&xs.every(x=>x.complete&&x.naturalWidth>0)')
        assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
        page.evaluate('scrollTo(0,0)');page.screenshot(path=f'/tmp/l170-top-{width}.png');host.screenshot(path=f'/tmp/l170-intervention-{width}.png')
        for i in range(3):page.locator('figure').nth(i).screenshot(path=f'/tmp/l170-figure-{i}-{width}.png')
    page.emulate_media(media='print');assert page.locator('.design-controls').evaluate('(x)=>getComputedStyle(x).display')=='none'
    context=browser.new_context(java_script_enabled=False,viewport={'width':375,'height':900});pg=context.new_page();pg.goto((R/'lessons'/(S+'.html')).as_uri());assert pg.locator('noscript').count()==1;assert '229,050' in pg.locator('body').inner_text();assert not pg.evaluate('document.documentElement.scrollWidth>innerWidth+1');context.close()
    page.emulate_media(media='screen');page.set_viewport_size({'width':1000,'height':900});page.goto((P/'html'/(S+'.html')).as_uri());assert page.locator('img[src^="data:image/png"]').count()==3
    assert 'COMPLETE_SAVED_EVIDENCE_REPLAY' in page.locator('body').inner_text()
    page.locator('img[src^="data:image/png"]').first.screenshot(path='/tmp/l170-notebook-figure.png');browser.close()
assert not errors,errors
class Quiet(SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R)));threading.Thread(target=server.serve_forever,daemon=True).start()
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page()
    for path,selector in [('index.html','a[href="lessons/'+S+'.html"]'),('notebooks.html','a[href="labs/html/'+S+'.html"]')]:
        page.goto('http://127.0.0.1:'+str(server.server_port)+'/'+path);reveal_gallery_link(page,selector)
    browser.close()
server.shutdown();server.server_close()
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
lines=[l[10:] for l in block.splitlines() if l.startswith('          ')];lines=[l for l in lines if not l.startswith(('VER=','sed -i'))];count=0
with tempfile.TemporaryDirectory(prefix='l170-pages-') as tmp:
    stage=Path(tmp)/'public';script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),l) for l in lines)
    subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
    for path in [stage/'lessons'/(S+'.html'),stage/'reference/fm-design-checkpoint.html']:
        parser=Links();parser.feed(path.read_text())
        for url in parser.links:
            part=urlsplit(url)
            if part.scheme or not part.path:continue
            dest=(path.parent/unquote(part.path)).resolve();assert dest.exists(),str(dest);count+=1
paths=[R/'lessons'/(S+'.html'),R/'reference/fm-design-checkpoint.html',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'evidence/l170/report.json',P/'evidence/l170/report.md',R/'assets/l170-evidence.js']+sorted((P/'figures/l170').iterdir())
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
subprocess.run([str(R/'.venv/bin/python'),str(P/'_figures_l170.py')],check=True,capture_output=True)
subprocess.run([str(R/'.venv/bin/python'),str(P/'_build_l170.py')],check=True,capture_output=True)
subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/refresh_lesson_visuals.py')],check=True,capture_output=True)
after=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
assert before==after,'Builder changed: '+str([str(p) for p,a,b in zip(paths,before,after) if a!=b])
r=dict(status='PASS',browser_widths=[1200,375],interactive_states=states,keyboard_reset='PASS',no_js='PASS',print='PASS',inline_source_parity='PASS',primary_reading_snapshots_checked=4,portable_figures=3,notebook_code_cells=sum(c.cell_type=='code' for c in solution.cells),copied_pages_links=count,deterministic_build='PASS',manifest_galleries='PASS',javascript_errors=errors,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_delivery_l170_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
