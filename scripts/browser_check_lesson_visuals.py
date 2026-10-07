"""Exercise the published visual interface across all later lessons."""
import argparse, functools, json, os, threading
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright
from refresh_lesson_visuals import selected, key_for, ROOT

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--base-url');parser.add_argument('--sample',action='store_true');parser.add_argument('--report-dir',type=Path,default=ROOT/'reviews/lesson-visuals-2026-10-07');args=parser.parse_args()
    report_dir=args.report_dir;report_dir.mkdir(parents=True,exist_ok=True)
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
    server=None
    if args.base_url:base=args.base_url.rstrip('/')
    else:
        server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(args.root)))
        threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}'
    errors=[];records=[];shots=[]
    samples={'52','54','64','66','74','82','102','145','164','184','b10','b17','b22','166','191','b08'}
    paths=[p for p in selected() if not args.sample or key_for(p) in samples]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--disable-dev-shm-usage'])
        for width in [1200,375]:
            page=browser.new_page(viewport={'width':width,'height':900},reduced_motion='reduce')
            for path in paths:
                key=key_for(path);page_errors=[]
                def capture(e):page_errors.append(str(e))
                page.on('pageerror',capture)
                response=page.goto(base+'/lessons/'+path.name,wait_until='load')
                page.wait_for_timeout(50)
                assert response.status==200,(key,response.status)
                assert page.locator('[data-visual-reading]').count()==1,key
                # New component never causes document-level horizontal overflow.
                overflow=page.evaluate('document.documentElement.scrollWidth > innerWidth + 1')
                if overflow:errors.append(dict(lesson=key,width=width,kind='document-overflow'))
                geometry=page.locator('.visual-story svg text').evaluate_all('''es=>es.flatMap(e=>{const b=e.getBBox();return b.x<-.1||b.x+b.width>300.1||b.y<-.1||b.y+b.height>225.1?[e.textContent]:[]})''')
                if geometry:errors.append(dict(lesson=key,width=width,kind='scene-text-bounds',details=geometry))
                for a in page.locator('[data-visual-reading] a').all():
                    assert page.locator('[id="'+a.get_attribute('href')[1:]+'"]').count()==1,(key,a.get_attribute('href'))
                stories=page.locator('.visual-story');states=0
                for story in stories.all():
                    for i,button in enumerate(story.locator('.vs-controls button').all()):
                        button.focus();page.keyboard.press('Enter');states+=1
                        assert button.get_attribute('aria-pressed')=='true'
                        assert story.locator('.vs-active').count()==(0 if i==0 else 1)
                        if i:assert story.locator('.vs-active').get_attribute('data-visual-step')==str(i)
                    story.locator('.vs-controls button').first.click()
                    summary=story.locator('details summary');summary.focus();page.keyboard.press('Enter')
                    assert story.locator('details').evaluate('(d)=>d.open');page.keyboard.press('Enter')
                    if key in samples:
                        target=report_dir/f'{story.get_attribute("id")}-{width}.png';story.screenshot(path=str(target));shots.append(target.name)
                tools=page.locator('.vs-figure-tools button:visible')
                if tools.count():
                    tools.first.focus();page.keyboard.press('Enter');states+=1
                    assert page.locator('dialog[open]').count()==1,(key,width,'dialog did not open')
                    page.locator('.vs-dialog-canvas').focus();page.keyboard.press('ArrowRight')
                    page.wait_for_function('document.querySelector(".vs-dialog img").complete')
                    assert page.locator('.vs-dialog img').evaluate('(i)=>i.naturalWidth>0'),key
                    if width==375:
                        assert page.locator('.vs-dialog-canvas').evaluate('e=>e.scrollWidth>e.clientWidth')
                    page.keyboard.press('Escape');assert not page.locator('dialog[open]').count()
                    assert tools.first.evaluate('e=>e===document.activeElement')
                page.locator('figure img[loading=lazy]').evaluate_all('(es)=>es.forEach(e=>e.loading="eager")')
                page.wait_for_function('Array.from(document.querySelectorAll("figure img")).every(i=>i.complete)')
                missing=page.locator('figure img').evaluate_all('(es)=>es.filter(e=>!e.complete||e.naturalWidth===0).map(e=>e.src)')
                if missing:errors.append(dict(lesson=key,width=width,kind='missing-existing-image',details=missing))
                if page_errors:errors.append(dict(lesson=key,width=width,kind='javascript',details=page_errors))
                records.append(dict(lesson=key,width=width,story=bool(stories.count()),interaction_states=states))
                page.remove_listener('pageerror',capture)
            page.close()
            print(f'Checked {len(paths)} lessons at {width}px',flush=True)
            (report_dir/'browser-progress.json').write_text(json.dumps(dict(complete=False,states=records,errors=errors),indent=2)+'\n')
        # Static fallback and print are tested on every authored story.
        page=browser.new_page(viewport={'width':1200,'height':900},java_script_enabled=False)
        for path in paths:
            page.goto(base+'/lessons/'+path.name,wait_until='load')
            if page.locator('.visual-story').count():
                assert page.locator('.vs-stages svg').count()==3*page.locator('.visual-story').count()
                assert page.locator('.vs-explanation').count()==3*page.locator('.visual-story').count()
                assert page.locator('.vs-controls').count()==0
                page.emulate_media(media='print')
                assert page.locator('.visual-story').first.evaluate('e=>getComputedStyle(e).transform')=='none'
                assert page.locator('.vs-print-answer').first.evaluate('e=>e.getBoundingClientRect().height>0')
                page.emulate_media(media='screen')
        page.close();browser.close()
    if server:server.shutdown()
    result=dict(base=base,lessons=len(paths),states=records,errors=errors,screenshots=shots)
    (report_dir/('live-browser.json' if args.base_url else 'browser.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(lessons=len(paths),pages=len(records),errors=errors)))
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
