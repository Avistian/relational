"""Check actual L062 browser controls against the declared attention arithmetic."""
import functools,json,math,threading
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
        page.goto(f'http://127.0.0.1:{server.server_port}/lessons/0062-tabpfn-v1.html',wait_until='networkidle')
        for self_edge in [False,True]:
            page.locator('#v1-self').set_checked(self_edge)
            for context_value in [0,2,10]:
                for unrelated_value in [0,100]:
                    for selector,value in [('#v1-context-value',context_value),('#v1-query-value',unrelated_value)]:
                        page.locator(selector).evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input",{bubbles:true}))}',value)
                    actual=float(page.locator('#v1-attention-viz').get_attribute('data-output'))
                    expected=(math.e*context_value+(6 if self_edge else 0))/(1+math.e+int(self_edge))
                    assert abs(actual-expected)<1e-12
                    records.append(dict(width=width,self_edge=self_edge,context_value=context_value,unrelated_value=unrelated_value,output=actual))
        page.locator('#v1-reset').click()
        assert page.locator('#v1-context-value').input_value()=='2' and page.locator('#v1-query-value').input_value()=='90'
        assert not page.locator('#v1-self').is_checked()
        page.locator('#v1-query-value').focus();page.keyboard.press('ArrowLeft')
        assert page.locator('#v1-query-value').input_value()=='80'
        assert abs(float(page.locator('#v1-attention-viz').get_attribute('data-output'))-2*math.e/(1+math.e))<1e-12
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        page.locator('#v1-attention-viz').screenshot(path=f'/tmp/l062-widget-{width}.png')
        page.close()
    browser.close()
server.shutdown();assert not errors,errors
result=dict(status='PASS',states=records,reset='PASS',keyboard='PASS',console_errors=errors)
(ROOT/'reviews/lesson-quality-audit-047-070/062-widget.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status='PASS',states=len(records))))
