"""Geometry of the hand-laid pipeline diagram and its actual arithmetic."""
import json,os,xml.etree.ElementTree as ET
from pathlib import Path
from playwright.sync_api import sync_playwright
P=Path(__file__).resolve().parent;F=P/'figures/l155';root=ET.parse(F/'pipelines.svg').getroot();ns={'s':'http://www.w3.org/2000/svg'}
rects=[]
for el in root.findall('.//s:rect',ns):
 x=float(el.get('x',0));y=float(el.get('y',0));w=float(el.get('width'));h=float(el.get('height'))
 assert 0<=x and 0<=y and x+w<=940 and y+h<=700
 if w==940:continue
 for xx,yy,ww,hh in rects:assert not (x<xx+ww and xx<x+w and y<yy+hh and yy<y+h)
 rects.append((x,y,w,h))
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
with sync_playwright() as pw:
 b=pw.chromium.launch(headless=True,args=['--no-sandbox']);p=b.new_page();p.goto((F/'pipelines.svg').as_uri())
 boxes=p.locator('text').evaluate_all('(xs)=>xs.map(x=>{let b=x.getBBox();return {text:x.textContent,x:b.x,y:b.y,w:b.width,h:b.height}})')
 for v in boxes:assert v['x']>=0 and v['y']>=0 and v['x']+v['w']<=940 and v['y']+v['h']<=700,v
 b.close()
t=json.loads((P/'evidence/l155/fe/prediction-trace.json').read_text());assert abs(t['first_ten_sum']+t['remaining_sum']-t['prediction'])<1e-10
r=dict(status='PASS',nonoverlapping_panels=len(rects),bounded_labels=len(boxes),actual_tree_sum='EXACT');(P/'_viz_check_l155_results.json').write_text(json.dumps(r,indent=2));print(r)
