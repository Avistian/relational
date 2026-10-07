"""Render existing graph nodes/edges as native narrow-screen architecture maps."""
from pathlib import Path
from html import escape as e
import json,re,hashlib
ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'assets/architecture-routes.json').read_text())
START='<!-- responsive-map:start -->';END='<!-- responsive-map:end -->'
def verify():
 for path,digest in DATA['sources'].items():
  assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest, f'Re-export architecture routes after changing {path}'
def strip(text):return re.sub(re.escape(START)+'.*?'+re.escape(END),'',text,flags=re.S)
def render(key,s):
 prefix='rm-'+key;ids={n['id']:i for i,n in enumerate(s['nodes'],1)};titles={n['id']:n['title'] for n in s['nodes']};nodes=[]
 for n in s['nodes']:
  outputs=[]
  for edge in s['edges']:
   if edge['a']!=n['id']:continue
   b=edge['b'];label=(' · '+edge['label']) if edge['label'] else ''
   outputs.append(f'<a href="#{prefix}-{e(b)}">→ {ids[b]} · {e(titles[b])}</a>'+e(label)+(' <span>(dashed path)</span>' if edge['dashed'] else ''))
  role={'learn':'Training','loss':'Training objective','query':'Input / readout','input':'Data / fitted state','model':'Computation','compute':'Computation','output':'Output'}.get(n['role'],n['role'])
  nodes.append(f'<li id="{prefix}-{e(n["id"])}" data-role="{e(n["role"])}"><header><span class="rm-number">{ids[n["id"]]}</span><div><small>{e(role)}</small><h4>{e(n["title"])}</h4></div></header><p>{e(n["body"])}</p>'+('<ul>'+''.join('<li>'+o+'</li>' for o in outputs)+'</ul>' if outputs else '<p class="rm-terminal">Endpoint of this pictured path.</p>')+'</li>')
 return START+f'<div class="responsive-map" aria-label="{e(s["title"],quote=True)}"><p class="rm-guide">Follow the arrows by block number. Branches are separate paths; numbers identify blocks, not a mandatory execution order.</p><ol class="rm-nodes">'+''.join(nodes)+f'</ol><p class="rm-scope">{e(s["scope"])}</p></div>'+END

def inject(text):
 text=strip(text)
 def replace(m):
  figure=m[0];key=None
  sol=re.search(r'data-solution-map="([^"]+)"',figure)
  if sol:key='solution-'+sol[1]
  else:
   src=re.search(r'src="../assets/architectures/([^".]+)\.(?:png|svg)"',figure)
   if src:key=src[1]
  if key not in DATA['routes']:return figure
  figure=re.sub(r'\sdata-responsive-map="[^"]*"','',figure)
  figure=figure.replace('<figure','<figure data-responsive-map="'+key+'"',1)
  # The original image remains the printable/exportable desktop architecture.
  pos=figure.index('>')+1
  return figure[:pos]+render(key,DATA['routes'][key])+figure[pos:]
 return re.sub(r'<figure\b[^>]*>.*?</figure>',replace,text,flags=re.S)
