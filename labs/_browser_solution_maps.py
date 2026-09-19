"""Real Chromium checks in a copied Pages tree (no symlinks, no model execution)."""
import functools,json,os,re,subprocess,threading
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
STAGE=Path('/tmp/relational-solution-pages')
SHOTS=Path('/tmp/relational-solution-screenshots')

def stage():
    workflow=(ROOT/'.github/workflows/pages.yml').read_text()
    block=workflow.split('      - name: Build site\n',1)[1].split('      - name: Setup Pages',1)[0]
    commands=[]
    for line in block.splitlines():
        if not line.startswith('          '):continue
        command=line[10:]
        if command.startswith(('VER=','sed -i')):continue
        commands.append(re.sub(r'\bpublic\b',str(STAGE),command))
    subprocess.run(['bash','-e','-c','\n'.join(commands)],cwd=ROOT,check=True,capture_output=True,text=True)

def check():
    stage();SHOTS.mkdir(exist_ok=True)
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(STAGE)))
    threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}'
    records=[];errors=[];missing=[];geometry=[];overflows=[];links=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
        for width in [1100,375]:
            page=browser.new_page(viewport={'width':width,'height':1000},reduced_motion='reduce')
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.on('response',lambda r:missing.append(r.url) if r.url.startswith(base) and r.status>=400 else None)
            for n in range(49,71):
                path=next((STAGE/'lessons').glob(f'{n:04}-*.html'))
                page.goto(base+'/lessons/'+path.name,wait_until='networkidle')
                if not page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'):overflows.append([n,width,'lesson'])
                for anchor in page.locator('.solution-route a').all():
                    assert page.locator('[id="'+anchor.get_attribute('href')[1:]+'"]').count()==1
                for f in page.locator('.solution-map').all():
                    bad=f.evaluate('''root=>[...root.querySelectorAll('[data-node]')].flatMap(g=>{let box=g.querySelector('rect').getBBox();return [...g.querySelectorAll('text')].filter(t=>{let r=t.getBBox();return r.x<box.x||r.x+r.width>box.x+box.width-5||r.y+r.height>box.y+box.height}).map(t=>({text:t.textContent,node:g.dataset.node}))})''')
                    if bad:geometry.append(dict(lesson=n,width=width,errors=bad))
                    summary=f.locator('summary');summary.focus();page.keyboard.press('Enter')
                    assert f.locator('details').get_attribute('open') is not None
                    page.keyboard.press('Enter')
                    f.locator('.sm-scroll').focus()
                    if width==375:
                        f.locator('.sm-scroll').evaluate('e=>e.scrollLeft=100')
                        assert f.locator('.sm-scroll').evaluate('e=>e.scrollLeft')>0
                        f.locator('.sm-scroll').evaluate('e=>e.scrollLeft=0')
                    if n in [49,52,55,64,66,70]:
                        page.evaluate('document.activeElement.blur()')
                        f.evaluate('e=>e.scrollIntoView()')
                        page.screenshot(path=str(SHOTS/f'{n:04}-{f.get_attribute("data-solution-map")}-{width}.png'),animations='disabled',timeout=20000)
                # Existing widgets still mount; run each native range at both extrema.
                exercised=0
                for control in page.locator('input[type=range]').all():
                    original=control.input_value()
                    for edge in ['min','max']:
                        control.evaluate('(e,k)=>{e.value=e[k];e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}))}',edge);exercised+=1
                    control.evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}))}',original)
                for select in page.locator('select').all():
                    original=select.input_value()
                    for value in select.locator('option').evaluate_all('(es)=>es.filter(x=>!x.disabled).map(x=>x.value)'):
                        select.select_option(value);exercised+=1
                    select.select_option(original)
                # Check new links as deployed; old notebook source citations are unchanged.
                targets=page.locator('.solution-story a,.solution-map a,.solution-handoff a').evaluate_all('(as)=>as.map(a=>a.href)')
                for url in set(targets):
                    if not url.startswith(base):continue
                    from urllib.parse import urlsplit,unquote
                    u=urlsplit(url);dest=STAGE/unquote(u.path.lstrip('/'))
                    if not dest.exists():links.append(url)
                    elif u.fragment and dest.suffix=='.html':
                        from bs4 import BeautifulSoup
                        if not BeautifulSoup(dest.read_text(),'html.parser').find(id=unquote(u.fragment)):links.append(url)
                page.goto(base+'/labs/html/'+path.name,wait_until='domcontentloaded')
                page.wait_for_function('Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)')
                if not page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'):overflows.append([n,width,'prepared lab'])
                records.append(dict(lesson=n,width=width,control_states=exercised))
                print(json.dumps(records[-1]),flush=True)
            page.close()
        page=browser.new_page(viewport={'width':1100,'height':1000},java_script_enabled=False)
        page.goto(base+'/reference/solution-map-atlas.html');assert page.locator('article').count()==22
        page.locator('article h3 a').first.focus();page.keyboard.press('Enter');page.wait_for_url('**/0049-excelformer-trompt.html')
        assert page.locator('.solution-map').count()==2
        page.emulate_media(media='print')
        assert page.locator('.solution-map svg').first.evaluate('e=>parseFloat(getComputedStyle(e).minWidth)')==0
        page.locator('.solution-map').first.evaluate('e=>e.scrollIntoView()')
        page.screenshot(path=str(SHOTS/'0049-print.png'),animations='disabled',timeout=20000)
        browser.close()
    server.shutdown()
    result=dict(status='PASS' if not any([errors,missing,geometry,overflows,links]) else 'FAIL',records=records,js_errors=errors,missing_local_assets=sorted(set(missing)),node_geometry=geometry,page_overflow=overflows,broken_new_links=sorted(set(links)),screenshots=str(SHOTS),keyboard_and_no_js='PASS',print_layout='PASS',live_colab='NOT_CHECKED',deployment='NOT_RUN')
    (ROOT/'labs/_solution_maps_browser_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
    assert result['status']=='PASS'
if __name__=='__main__':check()
