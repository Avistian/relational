"""Verify actual L066 displayed arithmetic independently with NumPy."""
import functools,json,re,threading
from pathlib import Path
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import numpy as np
from scipy.special import softmax
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
class Quiet(SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(ROOT)))
threading.Thread(target=server.serve_forever,daemon=True).start();errors=[];states=[]
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True)
    for width in [900,375]:
        page=browser.new_page(viewport={'width':width,'height':1800});page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(f'http://127.0.0.1:{server.server_port}/lessons/0066-tabicl-column-row-attention.html',wait_until='networkidle')
        for name,values,default in [('inducing',np.arange(-4,4.5,.5),1),('rope',range(9),1),('cost',range(40,1001,40),400)]:
            root=page.locator('#l066-'+name+'-viz');slider=root.locator('input')
            for value in values:
                value=float(value);slider.evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input",{bubbles:true}));}',value)
                actual=root.locator('svg text').all_text_contents()
                if name=='inducing':
                    context=np.array([[-1,0],[0,1],[2,1]],float);memory=softmax(context.T/np.sqrt(2),axis=1)@context
                    weights=softmax(np.array([value,-1])@memory.T/np.sqrt(2));out=weights@memory
                    numbers=[float(v) for v in re.findall(r'-?\d+\.\d+',actual[-2].split('weights')[-1])];assert np.allclose(numbers,weights,atol=5.1e-5)
                    numbers=[float(v) for v in re.findall(r'-?\d+\.\d+',actual[-1])];assert np.allclose(numbers,out,atol=5.1e-5)
                    assert all(f'{v:.4f}' in actual[1] for v in memory.ravel())
                elif name=='rope':
                    rotation=np.array([[np.cos(value),-np.sin(value)],[np.sin(value),np.cos(value)]]);vector=rotation@np.array([1,0])
                    assert all(f'{v:.4f}' in actual[1] for v in vector)
                    assert f'{vector[0]:.4f}' in actual[2] and '1.0000' in actual[3]
                else:
                    c=int(value);n=c+100
                    per_column=sum(c for _ in range(128))+sum(128 for _ in range(n))
                    expected=[per_column*3*4*8,sum(12 for _ in range(12))*n*3*8,sum(c for _ in range(n))*12*4]
                    displayed=[int(actual[i].replace(',','')) for i in [1,3,5]];assert displayed==expected
                states.append(dict(width=width,widget=name,value=value))
            root.locator('button').click();assert float(slider.input_value())==default
            slider.focus();page.keyboard.press('ArrowRight');assert float(slider.input_value())>default
        page.close()
    browser.close()
server.shutdown();assert not errors
report=dict(status='PASS',states=states,keyboard='PASS',reset='PASS',console_errors=errors,scope='All 102 actual slider states at two widths; inducing weights/outputs from independent NumPy matrices, RoPE from explicit rotation matrix, score counts from reader/sender enumeration. SVG text and keyboard/reset checked.')
(ROOT/'reviews/lesson-quality-audit-047-070/066-widget.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status='PASS',states=len(states))))
