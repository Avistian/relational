"""Export the portable computation panels after regenerating lesson HTML."""
import functools,hashlib,json,threading
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright
from refresh_lesson_visuals import ROOT,selected,key_for
from course_visual_specs import SPECS
SOURCES=['scripts/course_visual_specs.py','scripts/course_visuals.py','scripts/visual_detail_layouts.py','assets/course-visuals.css','assets/lesson.css']
def main():
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*a):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(ROOT)));threading.Thread(target=server.serve_forever,daemon=True).start()
 with sync_playwright() as p:
  browser=p.chromium.launch();page=browser.new_page(viewport={'width':1200,'height':1000},reduced_motion='reduce')
  for path in selected():
   key=key_for(path)
   if key not in SPECS:continue
   page.goto(f'http://127.0.0.1:{server.server_port}/lessons/{path.name}')
   page.locator('.course-visual').screenshot(path=str(ROOT/f'assets/course-visuals/{key}.png'),animations='disabled')
  browser.close()
 server.shutdown()
 paths=SOURCES+[f'assets/course-visuals/{key}.png' for key in SPECS]
 manifest={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths}
 (ROOT/'assets/course-visuals/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Exported',len(SPECS),'portable diagrams')
if __name__=='__main__':main()
