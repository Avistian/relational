"""Check displayed L067 distance and sharing arithmetic across every control state."""
import functools, json, re, threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args): pass
server = ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(ROOT)))
threading.Thread(target=server.serve_forever, daemon=True).start()
states = []; errors = []
with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True)
    for width in [900, 375]:
        page = browser.new_page(viewport={'width': width, 'height': 1200})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(f'http://127.0.0.1:{server.server_port}/lessons/0067-local-pfn-retrieval-finetuning.html', wait_until='networkidle')
        cost = page.locator('#l067-cost-viz'); slider = cost.locator('input'); select = cost.locator('select')
        for k in range(20, 301, 10):
            slider.evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input",{bubbles:true}));}', k)
            for b in [1, 2, 4, 8, 16, 32]:
                select.select_option(str(b)); shown = cost.locator('.l067-readout').inner_text()
                exact = sum(k for _ in range(32*(k+1)))
                shared = sum(k for _ in range(b*k+32))
                numbers = [int(v.replace(',', '')) for v in re.findall(r'(?:Exact|Shared): ([\d,]+)', shown)]
                assert numbers == [exact, shared]
                assert f'ratio {exact/shared:.2f}×' in shown
                assert f'{32//b} queries per context' in shown
                states.append(dict(width=width, widget='cost', context=k, batch=b))
        cost.locator('button').click(); assert slider.input_value() == '100' and select.input_value() == '2'
        slider.focus(); page.keyboard.press('ArrowRight'); assert slider.input_value() == '110'
        select.focus(); page.keyboard.press('ArrowDown'); assert select.input_value() == '4'
        geometry = page.locator('#l067-geometry-viz'); slider = geometry.locator('input')
        for weight in range(1, 26):
            slider.evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input",{bubbles:true}));}', weight)
            shown = geometry.locator('.l067-readout').inner_text()
            distance_a = sum([0**2, weight*2**2]); distance_b = sum([3**2, weight*0**2])
            assert f'= {distance_a}.' in shown and '=9.' in shown
            assert 'Nearest: '+('A' if distance_a < distance_b else 'B')+'.' in shown
            states.append(dict(width=width, widget='geometry', weight=weight))
        geometry.locator('button').click(); assert slider.input_value() == '1'
        slider.focus(); page.keyboard.press('ArrowRight'); assert slider.input_value() == '2'
        page.close()
    browser.close()
server.shutdown(); assert not errors
report = dict(status='PASS', states=states, console_errors=errors, keyboard='PASS', reset='PASS',
              scope='All 398 actual displayed cost/distance control combinations across 900/375; independent score-pair enumeration and weighted coordinate distances; reset and keyboard controls')
(ROOT/'reviews/lesson-quality-audit-047-070/067-widget.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(dict(status='PASS', states=len(states))))
