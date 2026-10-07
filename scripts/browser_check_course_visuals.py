"""Whole-course resource/render audit plus authored route and viewer interactions."""
from pathlib import Path
import argparse,functools,json,threading
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright
from refresh_lesson_visuals import ROOT,selected,key_for
from course_visual_specs import SPECS

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base-url');ap.add_argument('--export',action='store_true');ap.add_argument('--quick',action='store_true');ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=228);args=ap.parse_args()
 out=ROOT/'reviews/course-visual-quality-2026-10-07'/('live' if args.base_url else 'after');out.mkdir(parents=True,exist_ok=True)
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*a):pass
 server=None
 if args.base_url:base=args.base_url.rstrip('/')
 else:
  server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(ROOT)));threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}'
 rows=[];errors=[]
 with sync_playwright() as p:
  browser=p.chromium.launch(args=['--disable-dev-shm-usage'])
  for width in [1200,375]:
   page=browser.new_page(viewport={'width':width,'height':1000},reduced_motion='reduce')
   for path in selected()[args.start:args.stop]:
    key=key_for(path);issues=[];listener=lambda err:issues.append(str(err));page.on('pageerror',listener)
    page.goto(base+'/lessons/'+path.name);page.locator('img').evaluate_all('es=>es.forEach(e=>e.loading="eager")');page.wait_for_function('Array.from(document.images).every(i=>i.complete)')
    broken=page.locator('img').evaluate_all('es=>es.filter(i=>!i.naturalWidth).map(i=>i.src)')
    if broken:issues.append({'broken':broken})
    if page.evaluate('document.documentElement.scrollWidth>innerWidth+2'):issues.append('document overflow')
    for a in page.locator('.responsive-map a').all():
     if page.locator('[id="'+a.get_attribute('href')[1:]+'"]').count()!=1:issues.append('missing route target '+a.get_attribute('href'))
    cv=page.locator('.course-visual');maps=page.locator('.responsive-map');states=0
    if cv.count():
     geometry=cv.locator('svg text').evaluate_all('''es=>es.filter(e=>{let b=e.getBBox();return b.x<0||b.y<0||b.x+b.width>300||b.y+b.height>270}).map(e=>e.textContent)''')
     if geometry:issues.append({'geometry':geometry})
     summary=cv.locator('summary');summary.focus();page.keyboard.press('Enter');assert cv.locator('details').evaluate('e=>e.open');page.keyboard.press('Enter');states+=2
     summary.evaluate('e=>e.blur()')
     cv.screenshot(path=str(out/f'{key}-{width}.png'))
     if args.export and width==1200:
      dest=ROOT/'assets/course-visuals';dest.mkdir(exist_ok=True);cv.screenshot(path=str(dest/f'{key}.png'))
    if width==375:
     for i,m in enumerate(maps.all()):
      if not m.is_visible():issues.append('mobile route hidden')
      else:m.screenshot(path=str(out/f'{key}-route-{i}.png'),animations='disabled',timeout=60000)
    # Every available visual control is exercised, excluding quiz/learning-state controls.
    if not args.quick:
     controls=page.locator('[class*="-viz"] button, [class*="-widget"] button, [class*="-widget"] select, [class*="-viz"] select, [class*="-viz"] input[type=range], [class*="-widget"] input[type=range]')
     for control in controls.all():
      if not control.is_visible() or not control.is_enabled():continue
      if control.evaluate('e=>!!e.closest(".quiz,.retrieval,.vs-figure-tools")'):continue
      tag=control.evaluate('e=>e.tagName')
      try:
       if tag=='SELECT':
        original=control.input_value()
        for opt in control.locator('option').all():
         if opt.is_enabled():control.select_option(opt.get_attribute('value') or opt.text_content());states+=1
        control.select_option(original)
       elif tag=='INPUT':
        original=control.input_value()
        for val in [control.get_attribute('min') or '0',control.get_attribute('max') or '100']:
         control.evaluate('(e,v)=>e.value=v',val);control.dispatch_event('input');control.dispatch_event('change');states+=1
        control.evaluate('(e,v)=>e.value=v',original);control.dispatch_event('input');control.dispatch_event('change')
       elif not any(w in control.inner_text().lower() for w in ['play','animate','download','export','run all']):
        control.focus();page.keyboard.press('Enter');states+=1
      except Exception as ex:issues.append('control: '+str(ex).splitlines()[0])
    tools=page.locator('.vs-figure-tools button:visible')
    if tools.count():
     tools.first.focus();page.keyboard.press('Enter');page.wait_for_function('document.querySelector("dialog[open] img")?.complete')
     if not page.locator('dialog img').evaluate('i=>i.naturalWidth>0'):issues.append('viewer image broken')
     fit=page.locator('dialog button[aria-pressed]');fit.click();assert fit.get_attribute('aria-pressed')=='true'
     if page.locator('dialog img').evaluate('e=>e.clientWidth>e.parentElement.clientWidth+1'):issues.append('fit width')
     page.keyboard.press('Escape');assert tools.first.evaluate('e=>e===document.activeElement');states+=3
    rows.append(dict(lesson=path.name,width=width,visuals=page.locator('svg,img,canvas').count(),responsive_maps=maps.count(),trace=bool(cv.count()),interaction_states=states,issues=issues))
    errors.extend(dict(lesson=path.name,width=width,issue=i) for i in issues);page.remove_listener('pageerror',listener)
    if len(rows)%20==0:
     print(len(rows),path.name,'errors',len(errors),flush=True);(out/f'progress-{args.start}-{args.stop}.json').write_text(json.dumps(dict(rows=rows,errors=errors),indent=2))
   page.close()
  browser.close()
 if server:server.shutdown()
 (out/f'browser-{args.start}-{args.stop}.json').write_text(json.dumps(dict(lessons=len(selected()),pages=rows,errors=errors),indent=2)+'\n')
 print('DONE',len(rows),'visits;',len(errors),'errors')
 if errors:raise SystemExit(1)
if __name__=='__main__':main()
