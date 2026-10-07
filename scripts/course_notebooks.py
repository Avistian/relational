"""Carry the reading route into portable notebooks without executing any cells.

Called by refresh_lesson_visuals after any lesson/notebook builder. The original
architecture export is retained; this adds its readable operation companion.
"""
from pathlib import Path
import base64,hashlib,json,re,subprocess
from html import escape
from course_visual_specs import SPECS
ROOT=Path(__file__).resolve().parents[1]
START='<!-- course-portable:start -->';END='<!-- course-portable:end -->'
def outputs():
 tracked=set(subprocess.check_output(["git","ls-files"],cwd=ROOT,text=True).splitlines())
 for key,s in SPECS.items():
  stem=key.zfill(4) if key.isdigit() else key
  notebooks=sorted((ROOT/'labs').glob(stem+'-*.ipynb'))+sorted((ROOT/'labs/solutions').glob(stem+'-*.ipynb'))
  notebooks=[p for p in notebooks if str(p.relative_to(ROOT)) in tracked]
  if not notebooks:continue
  png=ROOT/f'assets/course-visuals/{key}.png'
  if not png.exists():raise ValueError(f'Export the reviewed route first: {png}')
  data=base64.b64encode(png.read_bytes()).decode();image=f'<img alt="{escape(s["title"],quote=True)}" src="data:image/png;base64,{data}" style="width:100%;max-width:820px;height:auto"/>'
  body=START+'\n### Follow the computation\n\n'+image+'\n\n'+s['scope']+'\n\n**Trace question:** '+s['question']+'\n\n**Trace answer:** '+s['answer']+'\n'+END
  for p in notebooks:
   text=p.read_text();nb=json.loads(text)
   cells=[c for c in nb['cells'] if START not in ''.join(c.get('source',[]))]
   i=next((i for i,c in enumerate(cells) if c['cell_type']=='markdown'),-1)
   cell=dict(cell_type='markdown',metadata={},source=body.splitlines(keepends=True))
   if nb.get('nbformat_minor',0)>=5:cell['id']='course-visual-'+key
   cells.insert(i+1,cell);nb['cells']=cells
   indent=1 if re.search(r'^ \"',text,re.M) else 2
   value=json.dumps(nb,ensure_ascii=False,indent=indent)+'\n'
   yield p,value
  for p in sorted((ROOT/'labs/html').glob(stem+'-*.html')):
   if str(p.relative_to(ROOT)) not in tracked:continue
   text=re.sub(re.escape(START)+'.*?'+re.escape(END),'',p.read_text(),flags=re.S)
   match=re.search(r'</h1>',text)
   if not match:raise ValueError(f'No notebook heading: {p}')
   block=START+'<section class="course-portable" style="max-width:820px;margin:24px auto"><h2>Follow the computation</h2>'+image+'<p>'+escape(s['scope'])+'</p><p><strong>Trace question:</strong> '+escape(s['question'])+'</p><p><strong>Trace answer:</strong> '+escape(s['answer'])+'</p></section>'+END
   yield p,text[:match.end()]+block+text[match.end():]

def verify_exports():
 manifest=json.loads((ROOT/'assets/course-visuals/manifest.json').read_text())
 for path,digest in manifest.items():
  assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest, f'Re-export portable diagrams after changing {path}'
 assert all(f'assets/course-visuals/{key}.png' in manifest for key in SPECS)
