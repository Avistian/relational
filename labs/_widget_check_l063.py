"""Check displayed L063 interventions and posterior arithmetic in actual Chromium."""
import functools,json,re,threading
from pathlib import Path
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
class Quiet(SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(ROOT)))
threading.Thread(target=server.serve_forever,daemon=True).start()
records=[];errors=[]
with sync_playwright() as w:
    browser=w.chromium.launch(headless=True)
    for width in [900,375]:
        page=browser.new_page(viewport={'width':width,'height':1500})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(f'http://127.0.0.1:{server.server_port}/lessons/0063-synthetic-scm-prior.html',wait_until='networkidle')
        for tick in range(31):
            edge=tick/10
            page.locator('#scm-edge-weight').evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input",{bubbles:true}))}',edge)
            text=page.locator('#scm-edge-viz .scm-readout').inner_text()
            value=float(re.search(r'Current H₂=.*?=([\d.−-]+)\.',text)[1].replace('−','-'))
            delta=float(re.search(r'Change=([\d.−-]+)\.',text)[1].replace('−','-'))
            assert abs(value-(edge*3.6+.1))<.00501 and abs(delta-(edge-2)*3.6)<.00501
            assert 'fixed H₁=3.6' in text
            records.append(dict(width=width,mode='edge',input=edge,value=value,delta=delta))
        for tick in range(1,100):
            r=tick/100
            page.locator('#scm-query-likelihood').evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input",{bubbles:true}))}',r)
            text=page.locator('#scm-posterior-viz .scm-readout').inner_text()
            weights=[float(x) for x in re.search(r'Posterior weights \(([^)]+)\)',text)[1].split(',')]
            probability=float(re.search(r'P\(class 1\) = .* = ([\d.]+)\.$',text)[1])
            # Posterior odds = context odds 3/2 times query-feature likelihood ratio.
            odds=1.5*r/(1-r); w1=odds/(1+odds)
            assert abs(weights[1]-w1)<.0000501 and abs(weights[0]-(1-w1))<.0000501
            assert abs(probability-(.1+.8*w1))<.0000501
            records.append(dict(width=width,mode='posterior',input=r,weights=weights,probability=probability))
        for root,slider,default in [('scm-edge-viz','scm-edge-weight','2'),('scm-posterior-viz','scm-query-likelihood','0.8')]:
            page.locator('#'+root+' button').click();assert page.locator('#'+slider).input_value()==default
            page.locator('#'+slider).focus();page.keyboard.press('ArrowLeft');assert page.locator('#'+slider).input_value()!=default
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        page.close()
    browser.close()
server.shutdown();assert not errors,errors
report=dict(status='PASS',states=records,reset='PASS',keyboard='PASS',console_errors=errors,
            scope='All 31 edge coefficients and 99 query likelihoods at desktop/mobile; displayed rounded values checked against separate arithmetic.')
(ROOT/'reviews/lesson-quality-audit-047-070/063-widget.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(status='PASS',states=len(records))))
