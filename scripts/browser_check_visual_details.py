"""Inspect every added diagram, including keyboard, no-JS and print views."""
import argparse, functools, json, threading
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright
from refresh_lesson_visuals import ROOT, selected, key_for
from visual_details import DETAILS

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base-url');parser.add_argument('--report-dir',type=Path,default=ROOT/'reviews/lesson-visual-refinement-2026-10-07');args=parser.parse_args()
    out=args.report_dir/('live' if args.base_url else 'local');out.mkdir(parents=True,exist_ok=True)
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
    server=None
    if args.base_url:base=args.base_url.rstrip('/')
    else:
        server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(ROOT)))
        threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}'
    records=[];errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--disable-dev-shm-usage'])
        page=browser.new_page(viewport={'width':1000,'height':620})
        for key in [k+suffix for k in DETAILS for suffix in ['', '-mobile']]:
            page.goto(base+f'/assets/visual-details/{key}.svg')
            result=page.locator('svg').evaluate('''svg=>{const vb={width:svg.getBoundingClientRect().width,height:svg.getBoundingClientRect().height};const texts=[...svg.querySelectorAll('text')].map(e=>({t:e.textContent,b:e.getBoundingClientRect()}));const outside=texts.filter(({b})=>b.x<0||b.y<0||b.x+b.width>vb.width||b.y+b.height>vb.height).map(e=>e.t);const overlaps=[];for(let i=0;i<texts.length;i++)for(let j=i+1;j<texts.length;j++){const a=texts[i].b,b=texts[j].b;if(a.x<b.x+b.width&&a.x+a.width>b.x&&a.y<b.y+b.height&&a.y+a.height>b.y)overlaps.push([texts[i].t,texts[j].t]);}const panelOverflow=[];svg.querySelectorAll(':scope > g > g').forEach(g=>{const r=g.querySelector(':scope > rect');if(!r)return;const p=r.getBoundingClientRect();g.querySelectorAll('text').forEach(e=>{const t=e.getBoundingClientRect();if(t.x<p.x-1||t.right>p.right+1||t.y<p.y-1||t.bottom>p.bottom+1)panelOverflow.push(e.textContent)});});return {outside,overlaps,panelOverflow}}''')
            if result['outside'] or result['overlaps'] or result['panelOverflow']:errors.append(dict(key=key,geometry=result))
            page.locator('svg').screenshot(path=str(out/f'{key}-drawing.png'))
        page.close()
        for width in [1200,375]:
            page=browser.new_page(viewport={'width':width,'height':1000},reduced_motion='reduce')
            page.on('pageerror',lambda e:errors.append({'javascript':str(e)}))
            for path in selected():
                key=key_for(path)
                if key not in DETAILS:continue
                response=page.goto(base+'/lessons/'+path.name,wait_until='load');assert response.status==200
                detail=page.locator('.visual-detail');detail.scroll_into_view_if_needed()
                assert detail.count()==1
                assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),(key,width)
                img=detail.locator('img');img.evaluate('i=>i.loading="eager"');page.wait_for_function('(src)=>[...document.images].some(i=>i.src.endsWith(src)&&i.complete&&i.naturalWidth>0)',arg=f'visual-details/{key}.svg')
                assert img.evaluate('e=>e.getBoundingClientRect().width<=e.parentElement.clientWidth+1'),(key,width,'image overflow')
                assert img.evaluate('e=>e.currentSrc').endswith(f'{key}-mobile.svg' if width<701 else f'{key}.svg'),(key,width,'layout')
                detail.screenshot(path=str(out/f'{key}-{width}.png'))
                button=detail.locator('.vs-figure-tools button');button.focus();page.keyboard.press('Enter')
                assert page.locator('dialog[open]').count()==1
                page.wait_for_function('document.querySelector(".vs-dialog img").complete')
                page.keyboard.press('Escape');assert button.evaluate('e=>e===document.activeElement')
                summary=detail.locator('.vd-question summary');before=detail.locator('.vd-question').evaluate('d=>d.open');summary.focus();page.keyboard.press('Enter');assert detail.locator('.vd-question').evaluate('d=>d.open')!=before,(key,width,'answer toggle')
                if key in ['b09','b18a','b19b']:
                    archive=page.locator('.vd-reference');assert not archive.evaluate('d=>d.open')
                    archive.locator('summary').focus();page.keyboard.press('Enter');assert archive.evaluate('d=>d.open')
                    page.keyboard.press('Enter');assert not archive.evaluate('d=>d.open')
                records.append(dict(key=key,width=width,whole_route_visible=True,enlarge=True,answer=True))
            page.close()
        page=browser.new_page(java_script_enabled=False,viewport={'width':1200,'height':1000})
        for path in selected():
            key=key_for(path)
            if key not in DETAILS:continue
            page.goto(base+'/lessons/'+path.name);detail=page.locator('.visual-detail');detail.scroll_into_view_if_needed()
            assert detail.locator('img').is_visible()
            detail.locator('summary').click();assert detail.locator('details').get_attribute('open') is not None
            page.emulate_media(media='print')
            assert detail.locator('.vs-print-answer').is_visible()
            assert detail.locator('img').evaluate('e=>e.getBoundingClientRect().width<=e.parentElement.getBoundingClientRect().width+1')
            if key in ['174','b09','b19b','b18a']:detail.screenshot(path=str(out/f'{key}-print.png'))
            page.emulate_media(media='screen')
        page.close()
        page=browser.new_page(java_script_enabled=False,viewport={'width':375,'height':1000})
        for path in selected():
            key=key_for(path)
            if key not in DETAILS:continue
            page.goto(base+'/lessons/'+path.name);detail=page.locator('.visual-detail');detail.scroll_into_view_if_needed()
            image=detail.locator('picture img');image.evaluate('i=>i.loading="eager"')
            page.wait_for_function('(src)=>[...document.images].some(i=>i.currentSrc.endsWith(src)&&i.complete&&i.naturalWidth>0)',arg=f'{key}-mobile.svg')
            assert image.evaluate('e=>e.currentSrc').endswith(f'{key}-mobile.svg')
            assert image.evaluate('e=>e.getBoundingClientRect().width<=innerWidth')
            assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
        page.close()
        browser.close()
    (out/'browser.json').write_text(json.dumps(dict(drawings=2*len(DETAILS),pages=records,nojs_print=len(DETAILS),nojs_mobile=len(DETAILS),errors=errors),indent=2)+'\n')
    if server:server.shutdown()
    print(json.dumps(dict(drawings=2*len(DETAILS),page_visits=len(records),nojs_print=len(DETAILS),nojs_mobile=len(DETAILS),errors=errors)))
    assert not errors
if __name__=='__main__':main()
