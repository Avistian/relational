"""Check one lesson's visual states at desktop and mobile reading widths."""
import argparse
import functools,json,threading
from pathlib import Path
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('lesson',type=int)
parser.add_argument('--widgets',nargs='+',required=True)
parser.add_argument('--architecture')
args=parser.parse_args()
OUT=Path(f'/tmp/quality-l{args.lesson:03}-browser');OUT.mkdir(exist_ok=True)
lesson=next((ROOT/'lessons').glob(f'{args.lesson:04}-*.html'))
class Quiet(SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(ROOT)))
threading.Thread(target=server.serve_forever,daemon=True).start()
base=f'http://127.0.0.1:{server.server_port}'
errors=[];missing=[];records=[];problems=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    for width in [900,375]:
        page=browser.new_page(viewport={'width':width,'height':3000},device_scale_factor=1)
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('response',lambda r:missing.append(r.url) if r.url.startswith(base) and r.status>=400 else None)
        page.goto(base+'/lessons/'+lesson.name,wait_until='networkidle')
        page.add_style_tag(content='html { scroll-behavior: auto !important; }')
        page.evaluate('document.querySelectorAll("img").forEach(x=>x.loading="eager")')
        page.wait_for_function('Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)')
        if not page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'):problems.append([width,'page overflow'])
        split_numbers=page.locator('td').evaluate_all('''cells=>cells.flatMap(c=>{
          const value=c.textContent.trim();if(!/^[+\\-−]?\\d[\\d,]*(?:\\.\\d+)?%?$/.test(value))return [];
          const range=document.createRange();range.selectNodeContents(c);
          const lines=new Set([...range.getClientRects()].filter(r=>r.width&&r.height).map(r=>Math.round(r.top)));
          return lines.size>1?[{value,table:c.closest('table').textContent.trim().slice(0,100)}]:[];
        })''')
        if split_numbers:problems.append([width,'numeric table values wrap across lines',split_numbers])
        split_headers=page.locator('th').evaluate_all('''cells=>cells.flatMap(c=>{
          const issues=[], walker=document.createTreeWalker(c,NodeFilter.SHOW_TEXT);
          while(walker.nextNode()){
            const node=walker.currentNode;
            for(const match of node.textContent.matchAll(/[A-Za-z0-9]{3,}/g)){
              const range=document.createRange();range.setStart(node,match.index);range.setEnd(node,match.index+match[0].length);
              const lines=new Set([...range.getClientRects()].filter(r=>r.width&&r.height).map(r=>Math.round(r.top)));
              if(lines.size>1)issues.push({word:match[0],header:c.textContent.trim()});
            }
          }return issues;
        })''')
        if split_headers:problems.append([width,'table header words split across lines',split_headers])
        for name in args.widgets:
            root=page.locator('#'+name)
            def inspect(state):
                issues=root.evaluate('''r=>[...r.querySelectorAll('svg')].flatMap(s=>{
                  let v=s.viewBox.baseVal;if(!v.width)return [];
                  return [...s.querySelectorAll('text')].flatMap(t=>{let b=t.getBBox();return b.x<v.x-1||b.x+b.width>v.x+v.width+1||b.y<v.y-1||b.y+b.height>v.y+v.height+1?[t.textContent]:[]})
                })''')
                if issues:problems.append([width,name,state,issues])
                root.evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
                clip=root.evaluate('(e)=>{let r=e.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height}}')
                page.screenshot(path=str(OUT/f'{width}-{name}-{state}.png'),clip=clip)
                records.append(dict(width=width,widget=name,state=state,text=root.inner_text()[-1200:]))
            inspect('default')
            for i in range(root.locator('button').count()):
                root.locator('button').nth(i).click();inspect('button-'+str(i))
            for i in range(root.locator('details').count()):
                details=root.locator('details').nth(i)
                original=details.evaluate('(e)=>e.open')
                summary=details.locator(':scope > summary')
                summary.click()
                assert details.evaluate('(e)=>e.open')!=original,(width,name,'details click')
                inspect('details-'+str(i)+'-toggled')
                summary.focus();page.keyboard.press('Space')
                assert details.evaluate('(e)=>e.open')==original,(width,name,'keyboard details')
                inspect('details-'+str(i)+'-keyboard-restored')
            for i in range(root.locator('input[type=checkbox]').count()):
                checkbox=root.locator('input[type=checkbox]').nth(i)
                original=checkbox.is_checked()
                checkbox.set_checked(not original);inspect('checkbox-'+str(i)+'-toggled')
                checkbox.focus();page.keyboard.press('Space')
                assert checkbox.is_checked()==original,(width,name,'keyboard checkbox')
                inspect('checkbox-'+str(i)+'-keyboard-restored')
            for i in range(root.locator('svg rect[style*="cursor:pointer"]').count()):
                root.locator('svg rect[style*="cursor:pointer"]').nth(i).click();inspect('svg-choice-'+str(i))
            for i in range(root.locator('select').count()):
                select=root.locator('select').nth(i);original=select.input_value()
                values=select.locator('option').evaluate_all('(es)=>es.filter(e=>!e.disabled).map(e=>e.value)')
                for j,value in enumerate(values):
                    select.select_option(value);inspect('select-'+str(i)+'-'+str(j))
                select.select_option(original)
            for i in range(root.locator('input[type=range]').count()):
                slider=root.locator('input[type=range]').nth(i)
                original=slider.input_value()
                for edge in ['min','max']:
                    slider.evaluate('(e,k)=>{e.value=e[k];e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}))}',edge)
                    inspect('slider-'+str(i)+'-'+edge)
                slider.focus();before=slider.input_value();page.keyboard.press('ArrowLeft');after=slider.input_value()
                assert before!=after,(width,name,'keyboard slider')
                slider.evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input",{bubbles:true}))}',original)
        if args.architecture:
            page.locator('#'+args.architecture).screenshot(path=str(OUT/f'{width}-atlas.png'))
        page.close()
    browser.close()
server.shutdown()
report=dict(status='PASS' if not errors and not missing and not problems else 'FAIL',errors=errors,missing=missing,problems=problems,records=records,screenshots=str(OUT),scope='Live local Chromium; specified visual widgets, numeric table wrapping, native buttons/selects/checkboxes/details, clickable SVG choices, slider endpoints and keyboard at 900/375; not live Colab or deployed site')
(ROOT/f'reviews/lesson-quality-audit-047-070/{args.lesson:03}-browser.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
