"""Independently trace displayed row destinations and validation selection."""
import functools,itertools,json,threading
from pathlib import Path
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
class Quiet(SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(ROOT)))
threading.Thread(target=server.serve_forever,daemon=True).start();records=[];errors=[]
with sync_playwright() as w:
    browser=w.chromium.launch(headless=True)
    for width in [900,375]:
        page=browser.new_page(viewport={'width':width,'height':1600});page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(f'http://127.0.0.1:{server.server_port}/lessons/0065-tabpfn-query-embeddings.html',wait_until='networkidle')
        folds=[1,0,1,0,2,2]
        for fold,bug in itertools.product(range(3),[False,True]):
            root=page.locator('#l065-scatter-widget');root.locator('select').select_option(str(fold));root.locator('input').set_checked(bug)
            q=[i for i,f in enumerate(folds) if f==fold];c=[i for i,f in enumerate(folds) if f!=fold]
            dest=list(range(sum(f<fold for f in folds),sum(f<=fold for f in folds))) if bug else q
            final=[i for f in range(3) for i,v in enumerate(folds) if v==f] if bug else list(range(6))
            out=root.locator('output').inner_text()
            for label,values in [('Context positions',c),('Query positions',q),('Destination for this call',dest),('Final first coordinate',final)]:
                assert label+': ['+','.join(map(str,values))+']' in out,(out,label,values)
            records.append(dict(width=width,fold=fold,bug=bug,destination=dest))
        root.locator('button').click();assert root.locator('select').input_value()=='0' and not root.locator('input').is_checked()
        root=page.locator('#l065-selection-widget')
        for v,t in itertools.product(range(70,91),[70,89,95]):
            root.evaluate('(e,values)=>{[...e.querySelectorAll("input")].forEach((x,i)=>x.value=values[i]);e.querySelector("input").dispatchEvent(new Event("input",{bubbles:true}));}',[v,t])
            selected=12 if v>80 else 6;score=t/100 if selected==12 else .84
            out=root.locator('output').inner_text();assert f'Selected layers [{selected}]' in out and f'Selected test accuracy: {score:.2f}' in out
            records.append(dict(width=width,validation=v,test=t,selected=selected))
        root.locator('button').click();assert [i.input_value() for i in root.locator('input').all()]==['78','89']
        root.locator('input').first.focus();page.keyboard.press('ArrowRight');assert 'validation 0.79' in root.locator('output').inner_text()
        page.close()
    browser.close()
server.shutdown();assert not errors
report=dict(status='PASS',states=records,keyboard='PASS',reset='PASS',console_errors=errors,scope='138 actual displayed states at 900 and 375 pixels: row identity scatter/concatenation, all validation slider values with three independent test outcomes')
(ROOT/'reviews/lesson-quality-audit-047-070/065-widget.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status='PASS',states=len(records))))
