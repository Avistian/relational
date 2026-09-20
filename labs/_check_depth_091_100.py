"""Editorial delivery audit; model experiments and scientific verdicts are unchanged."""
from pathlib import Path
from urllib.parse import urlsplit,unquote
import argparse,base64,hashlib,json,re,subprocess,os,tempfile,functools,threading
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import nbformat
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1]
BASE='657c4433aa608cacbeb1f002e5a3d670568c9d3b'
def links(page, root):
 soup=BeautifulSoup(page.read_text(),'html.parser');count=0
 for el in soup.select('[href],img[src],script[src]'):
  url=el.get('href',el.get('src',''));u=urlsplit(url)
  if u.scheme or u.netloc:continue
  target=(page.parent/unquote(u.path)).resolve() if u.path else page
  assert target.exists(),(str(page.relative_to(root)),url)
  assert not target.is_symlink(),target
  if u.fragment and target.suffix=='.html':
   doc=soup if target==page else BeautifulSoup(target.read_text(),'html.parser')
   assert doc.find(id=u.fragment) or doc.find(id=unquote(u.fragment)) or doc.find('a',attrs={'name':unquote(u.fragment)}),(str(page),url)
  count+=1
 return count

def packages(root):
 reports=[]
 for n in range(91,101):
  page=next((root/'lessons').glob(f'{n:04}-*.html'));soup=BeautifulSoup(page.read_text(),'html.parser');maps=soup.select('.model-map img')
  assert soup.select_one('.learning-route') and maps,n
  assert soup.select_one('link[href="../assets/learning-walkthrough.css"]'),n
  assert not re.search(r'\[\[[A-Z_]+',soup.get_text()),(n,'unexpanded marker')
  notebooks=[]
  for folder in ['labs','labs/solutions']:
   p=root/folder/(page.stem+'.ipynb');nb=nbformat.read(p,4);nbformat.validate(nb)
   md='\n'.join(c.source for c in nb.cells if c.cell_type=='markdown')
   assert md.count('THE BIG PICTURE')==1,(n,folder,'duplicated walkthrough')
   for img in maps:
    data=base64.b64encode((root/'assets/architectures'/Path(img['src']).with_suffix('.png').name).read_bytes()).decode()
    assert 'data:image/png;base64,'+data in md,(n,folder,'missing portable image')
   assert not re.search(r'src="[^"\n]*assets/architectures/[^"\n]*\.svg"',md),(n,'external notebook image')
   for href in re.findall(r'href="([^"\n]+)"',md):
    assert not re.match(r'(?:\.\./|\d{4}-)',href),(n,'nonportable link',href)
   previous=nbformat.reads(subprocess.check_output(['git','show',f'{BASE}:{folder}/{p.name}'],cwd=R,text=True),4)
   code=lambda b:[(c.source,c.execution_count,c.outputs) for c in b.cells if c.cell_type=='code']
   assert code(nb)==code(previous),(n,folder,'code or outputs changed')
   notebooks.append({'path':str(p.relative_to(root)),'code_cells':len(code(nb)),'code_outputs_counts':'UNCHANGED'})
  reports.append({'lesson':n,'diagrams':len(maps),'links':links(page,root),'notebooks':notebooks})
 links(root/'reference/0091-0100-model-map.html',root)
 return reports

def arithmetic():
 import math
 import numpy as np
 x=np.array([1.,2.]);a=np.diag([2.,1.]);b=np.array([[0.,1.],[1.,0.]])
 assert np.allclose(x@a@b,[2,2]) and np.allclose(x@b@a,[4,1])
 z=.75*np.array([2.,0.])+.25*np.array([0.,4.]);assert np.allclose(z,[1.5,1])
 assert np.allclose(z@np.array([[1.,-1.],[-1.,1.]]),[.5,-.5])
 B=np.array([[1.,1.,0.],[0.,1.,1.]]);p=B/B.sum(1,keepdims=True);q=B.T/B.T.sum(1,keepdims=True)
 assert np.allclose((p@q@p)[0],[3/8,1/2,1/8])
 assert abs(math.log1p(math.exp(-1))-.3132616875)<1e-10
 assert abs((2+24+2+24-8)/10-4.4)<1e-12
 theta=0.;theta-=.1*(theta-1);theta-=.1*(theta+1);assert abs(theta+.01)<1e-12
 assert abs(.9*(.5/.9)*.2+.1*(.5/.1)*.8-.5)<1e-12
 return 'PASS: relation order, semantic fusion, walk mass, BPR, duplicate relation, SGD trajectory, importance weights'

class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass

def browser(root):
 libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
 if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(root)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 base=f'http://127.0.0.1:{server.server_port}/';errors=[];screens=[]
 try:
  with sync_playwright() as pw:
   b=pw.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote']);p=b.new_page();p.on('pageerror',lambda e:errors.append(str(e)))
   for width in [1440,390]:
    p.set_viewport_size({'width':width,'height':1000})
    for n in range(91,101):
     file=next((root/'lessons').glob(f'{n:04}-*.html'));p.goto(base+'lessons/'+file.name,wait_until='networkidle')
     assert p.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),(n,width,'page overflow')
     for img in p.locator('img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0'),(n,width,'image')
     assert p.locator('.learning-route').inner_text().count('THE BIG PICTURE')==1
     for i,fig in enumerate(p.locator('.model-map').all()):
      path=f'/tmp/l{n:03}-depth-{width}-{i}.png';fig.screenshot(path=path);screens.append(path)
      scroller=fig.locator('.model-map-scroll');scroller.focus();assert scroller.evaluate('(e)=>e===document.activeElement')
      if width==390:
       assert scroller.evaluate('(e)=>e.scrollWidth>e.clientWidth'),(n,'no mobile scroll')
       p.keyboard.press('ArrowRight');p.wait_for_timeout(120);assert scroller.evaluate('(e)=>e.scrollLeft>0'),(n,'keyboard scroll')
     assert p.locator('details').count()>0
    p.goto(base+'reference/0091-0100-model-map.html');assert p.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),('reference',width,'overflow')
   # Bounds per SVG text node; check each card's text against its own rectangle too.
   for file in (root/'assets/architectures').glob('*.svg'):
    if not 91<=int(file.name[:3])<=100:continue
    p.goto(base+'assets/architectures/'+file.name)
    bounds=p.locator('svg').evaluate('''s=>{let v=s.viewBox.baseVal;let failures=[];for(let t of s.querySelectorAll('text')){let b=t.getBBox();if(b.x<0||b.y<0||b.x+b.width>v.width||b.y+b.height>v.height)failures.push(t.textContent);let prev=t.previousElementSibling;while(prev&&prev.tagName==='text')prev=prev.previousElementSibling;if(prev&&+prev.getAttribute('width')===4)prev=prev.previousElementSibling;if(prev&&+prev.getAttribute('width')===230){let x=+prev.getAttribute('x'),y=+prev.getAttribute('y');if(b.x<x||b.x+b.width>x+230||b.y<y||b.y+b.height>y+98)failures.push('card: '+t.textContent)}}return failures}''')
    assert not bounds,(file.name,bounds)
   for n in range(91,101):
    file=next((root/'labs/html').glob(f'{n:04}-*.html'));p.goto(base+'labs/html/'+file.name)
    for img in p.locator('img').all():assert img.evaluate('(e)=>e.complete&&e.naturalWidth>0'),(n,'preview image')
    assert p.locator('.model-map').count()>=1
   p.goto(base+'lessons/0100-heterogeneous-gnn-checkpoint.html');widget=p.locator('#batch-weighting');select=widget.locator('select')
   for size in range(1,6):
    select.select_option(str(size));assert 'Weighted mean: 0.600' in widget.locator('output').inner_text()
   p.emulate_media(media='print');assert p.locator('.model-map img').evaluate('(e)=>getComputedStyle(e).minWidth')=='0px'
   p.emulate_media(media='screen');p.goto(base+'index.html');p.wait_for_selector('a[href="lessons/0100-heterogeneous-gnn-checkpoint.html"]')
   b.close()
 finally:server.shutdown();server.server_close();thread.join()
 assert not errors,errors
 return {'widths':[1440,390],'lesson_views':20,'svg_bounds':'PASS','portable_previews':10,'keyboard_scrolling':'PASS','checkpoint_widget':'PASS','print':'CHECKED','manifest_navigation':'PASS','javascript_errors':errors,'screenshots':screens}

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--browser',action='store_true');ap.add_argument('--stage',action='store_true');args=ap.parse_args()
 report={'scope':'Editorial delivery only; no new training or paper-result claims','baseline':BASE,'packages':packages(R),'arithmetic':arithmetic(),'live_colab':'NOT_CHECKED'}
 if args.stage:
  with tempfile.TemporaryDirectory(prefix='depth-091-100-') as tmp:
   stage=Path(tmp)/'public';workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
   lines=[x[10:] for x in block.splitlines() if x.startswith('          ')];lines=[x for x in lines if not x.startswith(('VER=','sed -i'))]
   script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),x) for x in lines)
   subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
   packages(stage);report['copied_pages']='PASS'
   if args.browser:report['browser']=browser(stage)
 elif args.browser:report['browser']=browser(R)
 (R/'reviews/lessons-091-100-depth-check.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ['packages','browser']},indent=2))
 print('PASS',len(report['packages']),'packages;',sum(x['links'] for x in report['packages']),'lesson links')
