"""Inspect every added diagram, including keyboard, no-JS and print views."""
import argparse, functools, json, threading
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright
from refresh_lesson_visuals import ROOT, selected, key_for
from visual_details import DETAILS

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base-url');args=parser.parse_args()
    out=ROOT/'reviews/lesson-visuals-second-pass-2026-10-07'/('live' if args.base_url else 'local');out.mkdir(parents=True,exist_ok=True)
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
        for key in DETAILS:
            page.goto(base+f'/assets/visual-details/{key}.svg')
            result=page.locator('svg').evaluate('''svg=>{const texts=[...svg.querySelectorAll('text')].map(e=>({t:e.textContent,b:e.getBBox()}));const outside=texts.filter(({b})=>b.x<0||b.y<0||b.x+b.width>1000||b.y+b.height>620).map(e=>e.t);const overlaps=[];for(let i=0;i<texts.length;i++)for(let j=i+1;j<texts.length;j++){const a=texts[i].b,b=texts[j].b;if(a.x<b.x+b.width&&a.x+a.width>b.x&&a.y<b.y+b.height&&a.y+a.height>b.y)overlaps.push([texts[i].t,texts[j].t]);}return {outside,overlaps}}''')
            if result['outside'] or result['overlaps']:errors.append(dict(key=key,geometry=result))
            page.screenshot(path=str(out/f'{key}-drawing.png'))
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
                scroll=detail.locator('.vd-scroll');scroll.focus();page.keyboard.press('ArrowRight');page.wait_for_timeout(180)
                assert scroll.evaluate('e=>e.scrollWidth<=e.clientWidth||e.scrollLeft>0'),(key,width,'scroll')
                scroll.evaluate('e=>e.scrollLeft=0')
                detail.screenshot(path=str(out/f'{key}-{width}.png'))
                button=detail.locator('.vs-figure-tools button');button.focus();page.keyboard.press('Enter')
                assert page.locator('dialog[open]').count()==1
                page.wait_for_function('document.querySelector(".vs-dialog img").complete')
                page.keyboard.press('Escape');assert button.evaluate('e=>e===document.activeElement')
                summary=detail.locator('summary');before=detail.locator('details').evaluate('d=>d.open');summary.focus();page.keyboard.press('Enter');assert detail.locator('details').evaluate('d=>d.open')!=before,(key,width,'answer toggle')
                records.append(dict(key=key,width=width,scroll=True,enlarge=True,answer=True))
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
        browser.close()
    (out/'browser.json').write_text(json.dumps(dict(drawings=len(DETAILS),pages=records,nojs_print=len(DETAILS),errors=errors),indent=2)+'\n')
    if server:server.shutdown()
    print(json.dumps(dict(drawings=len(DETAILS),page_visits=len(records),nojs_print=len(DETAILS),errors=errors)))
    assert not errors
if __name__=='__main__':main()
