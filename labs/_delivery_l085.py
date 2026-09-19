"""Browser interactions and copied Pages asset/link checks for L085."""
import json,os,re,shutil,tempfile,hashlib
from pathlib import Path
from urllib.parse import urlsplit,unquote
from html.parser import HTMLParser
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-zygote'])
    page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((ROOT/'lessons/0085-over-smoothing.html').as_uri())
    assert page.locator('#smoothing .smooth-nodes>div').count()==3
    for width in [1200,375]:
        page.set_viewport_size({'width':width,'height':900})
        for mode in ['symmetric','mean']:
            page.locator('#smoothing select').select_option(mode)
            for cut in [False,True]:
                page.locator('#smoothing input[type=checkbox]').set_checked(cut)
                for depth in [0,1,100]:
                    page.locator('#smoothing input[type=range]').fill(str(depth));page.locator('#smoothing input[type=range]').dispatch_event('input')
                    assert page.locator('#smoothing .smooth-output').inner_text().startswith('Depth '+str(depth))
                    result=page.evaluate('([k,m,c])=>OversmoothingViz.compute(k,m,c)',[depth,mode,cut])
                    if depth==100 and cut:assert result['h'][2]==8
                    if depth==0:assert result['h']==[2,4,8]
                    if depth==1 and mode=='symmetric' and not cut:assert abs(result['h'][1]-(10/6**.5+4/3))<1e-12
        page.locator('#smoothing button').click();page.locator('#smoothing').scroll_into_view_if_needed();page.screenshot(path=f'/tmp/l085-{width}.png')
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1'), 'page overflow'
    for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete && e.naturalWidth>0')
    page.goto((LAB/'html/0085-over-smoothing.html').as_uri())
    assert page.locator('img[src^="data:image/png;base64,"]').count()>=5
    for img in page.locator('img').all():assert img.evaluate('(e)=>e.complete && e.naturalWidth>0')
    assert not errors,errors
    browser.close()

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        self.links.extend(v for k,v in attrs if k in ('href','src'))

with tempfile.TemporaryDirectory(prefix='l085-pages-') as tmp:
    stage=Path(tmp)
    # Same copied asset trees as pages.yml, restricted to L085 evidence/notebooks.
    for folder in ['assets','lessons','reference']:shutil.copytree(ROOT/folder,stage/folder)
    (stage/'labs/solutions').mkdir(parents=True);(stage/'labs/html').mkdir();(stage/'labs/figures').mkdir()
    shutil.copytree(LAB/'figures/l085',stage/'labs/figures/l085')
    for f in ['index.html','notebooks.html']:shutil.copy2(ROOT/f,stage/f)
    for source in list(LAB.glob('*l085*'))+[LAB/'0085-over-smoothing.ipynb']:
        if source.is_file():shutil.copy2(source,stage/'labs'/source.name)
    for folder in ['solutions','html']:shutil.copy2(LAB/folder/('0085-over-smoothing.'+('ipynb' if folder=='solutions' else 'html')),stage/'labs'/folder)
    checks=0
    for path in [stage/'lessons/0085-over-smoothing.html',stage/'reference/over-smoothing.html']:
        parser=Links();parser.feed(path.read_text())
        for url in parser.links:
            part=urlsplit(url)
            if part.scheme or not part.path:continue
            target=(path.parent/unquote(part.path)).resolve()
            assert target.exists(),(path,url)
            checks+=1
result={'status':'PASS','browser_widths':[1200,375],'widget_states':24,'page_errors':errors,'portable_notebook_images':5,'copied_pages_local_links':checks,'pages_scope':'L085 page/reference and copied shared asset trees; no symlinks','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(LAB/'_delivery_l085_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
