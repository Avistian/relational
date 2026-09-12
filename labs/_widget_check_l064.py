"""Check L064 displayed attention work against explicit receiver/sender counts."""
import functools,itertools,json,re,threading
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
        page.goto(f'http://127.0.0.1:{server.server_port}/lessons/0064-tabpfn-v2.html',wait_until='networkidle')
        for c,q,f in itertools.product([20,80,160],[10,20,80],[2,3,20,39,40]):
            positions=list(range(0,f,2))+['target'];rows=range(c+q)
            # Each row-position receiver sees every position in that row for
            # feature attention, then exactly the context rows for row attention.
            feature=sum(len(positions) for _ in rows for __ in positions)
            row=sum(len(range(c)) for _ in positions for __ in rows)
            actual=page.evaluate('([c,q,f])=>L064.counts(c,q,f)',[c,q,f])
            assert actual==dict(groups=len(positions)-1,feature=feature,row=row)
            if f%2==0:
                page.locator('#l064-cost').evaluate('(e,vals)=>{[...e.querySelectorAll("input")].forEach((x,i)=>x.value=vals[i]);e.querySelector("input").dispatchEvent(new Event("input",{bubbles:true}));}',[c,q,f])
                text=page.locator('#l064-cost output').inner_text();numbers=[int(v.replace(',','')) for v in re.search(r'Feature attention: ([\d,]+) score pairs; row attention: ([\d,]+)',text).groups()]
                assert numbers==[feature,row] and '12,100 feature and 88,000 row pairs' in text
            records.append(dict(width=width,context=c,query=q,features=f,feature_scores=feature,row_scores=row))
        page.locator('#l064-cost button').click();assert [x.input_value() for x in page.locator('#l064-cost input').all()]==['80','20','20']
        first=page.locator('#l064-cost input').first;first.focus();page.keyboard.press('ArrowRight');assert first.input_value()=='100'
        assert 'C=100' in page.locator('#l064-cost output').inner_text()
        page.close()
    browser.close()
server.shutdown();assert not errors
report=dict(status='PASS',states=records,keyboard='PASS',reset='PASS',console_errors=errors,scope='90 API states at two widths, including odd feature counts; 54 real displayed control states compared with explicit receiver/sender counting')
(ROOT/'reviews/lesson-quality-audit-047-070/064-widget.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status='PASS',states=len(records))))
