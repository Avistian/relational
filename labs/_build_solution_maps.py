"""Idempotent editorial finishing pass; original builders call revise(n).

Run .venv/bin/python labs/_build_solution_maps.py [49 ... 70] after a legacy
builder, or without arguments to render the whole sequence. No code cell,
execution count, output, source audit or measured result is regenerated.
"""
import argparse,base64,hashlib,html,json,re
from pathlib import Path
from urllib.parse import urlsplit,unquote
import nbformat
from bs4 import BeautifulSoup
from _solution_map_content import MAPS,STORIES
ROOT=Path(__file__).resolve().parents[1]
E=html.escape
COLORS={'compute':('#e8f4f0','#32877e'),'query':('#fff1de','#b27428'),'learn':('#eeebf8','#77629d')}
def lesson(n):return next((ROOT/'lessons').glob(f'{n:04}-*.html'))
def rich(text,notebook=False):
    def link(m):
        n=int(m[1]);path=lesson(n).name
        return f'<a href="'+('../lessons/' if notebook else '')+path+f'">{E(m[2])}</a>'
    return re.sub(r'\[\[(\d+)\|([^]]+)\]\]',link,E(text))
def plain(text):return re.sub(r'\[\[(\d+)\|([^]]+)\]\]',r'\2',text)
def normalize(s):return s.casefold().replace('’', "'").replace('‘', "'")
def titletext(h):return h.get_text(' ',strip=True).replace('¶','').strip()
def match_heading(soup,anchor):
    matches=[h for h in soup.find_all('h2') if normalize(anchor) in normalize(titletext(h)) and not h.find_parent(class_='solution-story')]
    assert len(matches)==1,(anchor,[titletext(x) for x in matches])
    h=matches[0]
    if not h.get('id'):h['id']='read-'+re.sub(r'[^a-z0-9]+','-',titletext(h).lower()).strip('-')
    return h

def port(node,p):
    x,y,w,h=[node[k] for k in ['x','y','w','h']]
    return {'t':(x+w/2,y),'b':(x+w/2,y+h),'l':(x,y+h/2),'r':(x+w,y+h/2)}[p]
def edge_points(m,e):
    nodes={n['key']:n for n in m.nodes};a=port(nodes[e['a']],e['ports'][0]);b=port(nodes[e['b']],e['ports'][1])
    via=e['via']
    if via is None:
        if a[0]==b[0] or a[1]==b[1]:via=[]
        elif e['ports'][0] in 'bt':via=[(a[0],(a[1]+b[1])/2),(b[0],(a[1]+b[1])/2)]
        else:via=[((a[0]+b[0])/2,a[1]),((a[0]+b[0])/2,b[1])]
    return [a,*via,b]

def svg(m,n):
    uid=f'sm-{n}-{m.key}'
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {m.height}" role="img" aria-labelledby="{uid}-title {uid}-desc">',f'<title id="{uid}-title">{E(m.title)}</title>',f'<desc id="{uid}-desc">{E(m.subtitle+" "+m.answer)}</desc>',
       '<defs><marker id="'+uid+'-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#426975"/></marker><marker id="'+uid+'-control" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#80669d"/></marker></defs>',
       f'<rect width="1000" height="{m.height}" fill="#fcfdfd"/>',
       '<rect width="1000" height="116" fill="#173b4a"/>',
       f'<text x="35" y="36" font-family="sans-serif" font-size="13" letter-spacing="2" fill="#95d5cb">LESSON {n:03} · SOLUTION MAP</text>',
       f'<text x="35" y="66" font-family="sans-serif" font-size="20" font-weight="bold" fill="white">{E(m.subtitle)}</text>',
       '<text x="35" y="94" font-family="sans-serif" font-size="13" fill="#d6e6e9">Solid arrows: information flow   ·   Dashed arrows: fitting / selection control</text>']
    for g in m.groups:
        fill,stroke=COLORS[g['role']]
        s.append(f'<rect x="{g["x"]}" y="{g["y"]}" width="{g["w"]}" height="{g["h"]}" rx="12" fill="{fill}" fill-opacity=".35" stroke="{stroke}" stroke-opacity=".35"/>')
        # Group labels fit their lane; long captions may wrap onto a second line.
        label=g['label'];font=min(12, (g['w']-26)/(max(len(label),1)*.58))
    group_labels=[]
    for g in m.groups:
        label=g['label'];font=min(12,(g['w']-26)/(max(len(label),1)*.58))
        group_labels.append(f'<rect x="{g["x"]+10}" y="{g["y"]+7}" width="{min(g["w"]-20,len(label)*font*.61+10)}" height="20" fill="#f7faf9"/><text x="{g["x"]+14}" y="{g["y"]+22}" font-family="sans-serif" font-size="{font:.2f}" font-weight="bold" fill="#496573">{E(label)}</text>')
    for i,e in enumerate(m.edges):
        points=edge_points(m,e);path=' '.join(f'{x},{y}' for x,y in points);stroke='#80669d' if e['control'] else '#426975';marker='control' if e['control'] else 'arrow'
        s.append(f'<polyline data-edge="{e["a"]}:{e["b"]}" points="{path}" fill="none" stroke="{stroke}" stroke-width="2" stroke-linejoin="round" '+('stroke-dasharray="6 4" ' if e['control'] else '')+f'marker-end="url(#{uid}-{marker})"/>')
        if e['label']:
            x,y=points[len(points)//2];s.append(f'<text x="{x+6}" y="{y-9}" font-family="sans-serif" font-size="13" fill="{stroke}">{E(e["label"])}</text>')
    s.extend(group_labels)
    for node in m.nodes:
        x,y,w,h=[node[k] for k in ['x','y','w','h']];fill,stroke=COLORS[node['role']]
        s.append(f'<g data-node="{node["key"]}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1.25"/>')
        for i,line in enumerate([node['title'],*node['lines']]):
            size=17 if i==0 else 14
            s.append(f'<text x="{x+14}" y="{y+26+i*22}" font-family="DejaVu Sans,sans-serif" font-size="{size}" '+('font-weight="bold" ' if i==0 else '')+f'fill="#183c4b">{E(line)}</text>')
        s.append('</g>')
    for x,y,text in m.notes:s.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="15" fill="#486777">{E(text)}</text>')
    return ''.join(s)+'</svg>'

def transcript(m):
    nodes={v['key']:v for v in m.nodes}
    out='<ol>'
    for node in m.nodes:
        outgoing=[nodes[e['b']]['title']+(' (fitting/selection control)' if e['control'] else '') for e in m.edges if e['a']==node['key']]
        out+=f'<li><strong>{E(node["title"])}</strong>: {E("; ".join(node["lines"]))}.'
        if outgoing:out+=' Connects to '+E('; '.join(outgoing))+'.'
        out+='</li>'
    return out+'</ol>'

def figure(m,n):
    return f'''<figure class="solution-map" id="solution-{m.key}" data-solution-map="{m.key}">
<header><p class="sm-eyebrow">See the whole solution</p><h3>{E(m.title)}</h3><p><strong>Trace before reading:</strong> {E(m.question)}</p></header>
<details><summary>Read every component and connection as text</summary>{transcript(m)}</details>
<p class="sm-scroll-hint">↔ On narrow screens, scroll horizontally to see all branches. The text route above reflows to your screen.</p>
<div class="sm-scroll" role="region" tabindex="0" aria-label="Scrollable {E(m.title)} diagram">{svg(m,n)}</div>
<figcaption><strong>How the parts work together.</strong> {E(m.answer)}<p class="sm-scope"><strong>Pictured scope.</strong> {E(m.scope)}</p><p class="sm-scope"><a href="../assets/solution-maps/{n:04}-{m.key}.svg">Open full-size diagram</a> · <a href="../reference/solution-map-atlas.html#lesson-{n}">Place this solution in the course</a></p></figcaption>
</figure>'''

def export(m,n):
    import cairosvg
    directory=ROOT/'assets/solution-maps';directory.mkdir(exist_ok=True)
    path=directory/f'{n:04}-{m.key}.svg';content=svg(m,n)
    if not path.exists() or path.read_text()!=content:path.write_text(content)
    png=directory/f'{n:04}-{m.key}.png'
    if not png.exists() or png.stat().st_mtime<path.stat().st_mtime:cairosvg.svg2png(bytestring=content.encode(),write_to=str(png),output_width=1600)


def revise_html(n):
    path=lesson(n);s=BeautifulSoup(path.read_text(),'html.parser');story=STORIES[n]
    for old in s.select('[data-solution-story], [data-solution-map]'):old.decompose()
    # Older worked arithmetic is useful as detail, but no longer the system overview.
    for old in s.select('.arch-atlas'):
        if old.find_parent(class_='solution-operation'):continue
        details=s.new_tag('details',attrs={'class':'solution-operation'})
        summary=s.new_tag('summary');summary.string='Work through the key operation: '+old.select_one('h3').get_text(' ',strip=True)
        old.wrap(details);details.insert(0,summary)
    # Supersede only full architecture images, not mechanism / result figures.
    for img in list(s.select('figure img')):
        if 'architecture' in img.get('src',''):
            img.find_parent('figure').decompose()
    route=[]
    for i,(anchor,text) in enumerate(story['seams']):
        heading=match_heading(s,anchor)
        route.append(f'<a href="#{E(heading["id"])}">{E(titletext(heading))}</a>')
        heading.insert_after(BeautifulSoup(f'<p class="solution-seam" data-solution-story="seam-{i}">{rich(text)}</p>','html.parser'))
    first=next(h for h in s.find_all('h2') if not any(t in titletext(h).lower() for t in ['recall','retrieve','warm up']) and not h.find_parent(class_='solution-operation'))
    intro=f'<section class="solution-story" data-solution-story="opening"><p class="sm-eyebrow">The argument of this lesson</p><h2>{E(story["title"])}</h2><p>{rich(story["opening"])}</p><nav class="solution-route" aria-label="Follow the lesson argument">'+''.join(route)+'</nav></section>'
    parent=first.parent if first.parent.name=='section' else first
    parent.insert_before(BeautifulSoup(intro,'html.parser'))
    for m in MAPS[n]:
        heading=match_heading(s,m.anchor)
        target=heading
        nxt=heading.find_next_sibling()
        if nxt and 'solution-seam' in nxt.get('class',[]):target=nxt
        target.insert_after(BeautifulSoup(figure(m,n),'html.parser'))
    end=s.find('article') or s.find('main') or s.body
    end.append(BeautifulSoup(f'<aside class="solution-handoff" data-solution-story="handoff"><p><strong>Carry this forward.</strong> {rich(story["handoff"])}</p></aside>','html.parser'))
    if not s.select_one('link[href="../assets/solution-maps.css"]'):s.head.append(s.new_tag('link',rel='stylesheet',href='../assets/solution-maps.css'))
    path.write_text(str(s))
    # Export the author-controlled story independently from generated HTML.
    if story.get('export_markdown',True):
        (ROOT/'lessons/story'/f'{n:04}.md').write_text('# '+story['title']+'\n\n'+story['opening']+'\n\n'+ '\n\n'.join('## Before: '+a+'\n\n'+t for a,t in story['seams'])+'\n\n## Carry this forward\n\n'+story['handoff']+'\n')
    return s

def notebook_figure(m,n):
    data=base64.b64encode((ROOT/'assets/solution-maps'/f'{n:04}-{m.key}.png').read_bytes()).decode()
    return f'''### {m.title}

**Trace before reading:** {m.question}

<div style="max-width:100%;overflow-x:auto"><img src="data:image/png;base64,{data}" alt="{E(m.title+'. '+m.answer)}" style="width:100%;min-width:900px;max-width:1000px;height:auto"></div>

**How the parts work together:** {m.answer}

**Pictured scope:** {m.scope}

[Full-size vector diagram](../assets/solution-maps/{n:04}-{m.key}.svg). Scroll on narrow screens to keep labels legible.

<details><summary>Text route: components and connections</summary>{transcript(m)}</details>'''

def revise_notebook(n,path):
    from _check_solution_maps import digest
    nb=nbformat.read(path,4);before=digest(nb);story=STORIES[n]
    image_hashes={hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'labs/figures'/f'l{n:03}').glob('*architecture*.png')}
    image_hashes|={hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'labs/figures/architecture-revision').glob(f'{n:04}-*.png')}
    kept=[]
    for c in nb.cells:
        if c.metadata.get('solution_story') or c.metadata.get('solution_map') or c.metadata.get('architecture_revision'):continue
        if c.cell_type=='markdown' and 'data:image/png;base64,' in c.source:
            payloads=re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',c.source)
            if any(hashlib.sha256(base64.b64decode(v)).hexdigest() in image_hashes for v in payloads):
                # Existing architecture image cells are dedicated figures; if mixed,
                # remove only the image and retain the surrounding explanation.
                c.source=re.sub(r'!\[[^\]]*\]\(data:image/png;base64,[A-Za-z0-9+/=]+\)','',c.source)
                c.source=re.sub(r'<img\b[^>]*src=[\"\']data:image/png;base64,[A-Za-z0-9+/=]+[\"\'][^>]*>','',c.source)
                if not c.source.strip():continue
        if c.cell_type=='markdown':
            # Legacy builders copy lesson paragraphs without their DOM metadata.
            # Remove our exact authored transitions before inserting canonical cells.
            for _,text in story['seams']:
                c.source=c.source.replace(text,'')
        kept.append(c)
    nb.cells=kept
    def cell(text,kind,key):
        c=nbformat.v4.new_markdown_cell(text,metadata={kind:key});c.id=f'solution-{n}-{key}';return c
    opening='## '+story['title']+'\n\n'+rich(story['opening'],True)
    # Leave the notebook's title / bootstrap intact. Place overview before its first
    # substantive text section, with component maps alongside their existing topic.
    first=next((i for i,c in enumerate(nb.cells) if i>0 and c.cell_type=='markdown' and re.search(r'^##? ',c.source,re.M)),1)
    nb.cells.insert(first,cell(opening,'solution_story','opening'))
    placements=[]
    def find(anchor):
        aliases={48:{'Derive the cross layer':'Task 1 — the dense cross update','Why the degree grows':'Task 2 — a low-rank cross','Low rank:':'Task 2 — a low-rank cross','Where does the deep':'Task 3 — route the deep branch','Extension: a mixture':'Task 4 — mix nonlinear expert updates','The local evidence':'Train the comparison','Reproduction:':'The full-data paper-results attempt'},57:{'Different mistakes':'Concept recap','Measure the gain':'Interpretation checkpoint'},63:{'Sparsity is a task':'Sparsity changes the world'},62:{'Combining several views':'Ensembling means aligning','What the experiment measured':'What was measured here'}}
        anchor=aliases.get(n,{}).get(anchor,anchor)
        for i,c in enumerate(nb.cells):
            if c.cell_type!='markdown' or c.metadata.get('solution_map') or c.metadata.get('solution_story'):continue
            headings=re.findall(r'^#{1,3}\s+(.+)',c.source,re.M)
            if any(normalize(anchor) in normalize(BeautifulSoup(h,'html.parser').get_text()) for h in headings):return i
        return None
    for i,(anchor,text) in enumerate(story['seams']):
        idx=find(anchor);assert idx is not None,(n,'notebook section missing',anchor)
        placements.append(dict(kind='seam',anchor=anchor,matched=idx is not None))
        value=cell('**Connecting the ideas.** '+rich(text,True),'solution_story',f'seam-{i}')
        # Some older notebooks bundle a section into one cell. The transition goes
        # before that section; a missing heading falls back to the opening route.
        nb.cells.insert(idx if idx is not None else first+1+i,value)
    for i,m in enumerate(MAPS[n]):
        idx=find(m.anchor);assert idx is not None,(n,'notebook map section missing',m.anchor)
        placements.append(dict(kind='map',anchor=m.anchor,matched=idx is not None))
        nb.cells.insert(idx if idx is not None else first+1,cell(notebook_figure(m,n),'solution_map',m.key))
    nb.cells.append(cell('## Carry this forward\n\n'+rich(story['handoff'],True),'solution_story','handoff'))
    assert digest(nb)==before,(n,path,'executable content changed')
    nbformat.write(nb,path)
    return placements


def prepared(n):
    """Patch the prepared notebook's markdown delivery without executing cells.

Original launcher and specialized styles are retained, while the notebook is
re-rendered to ensure that every new cell is actually visible in the HTML.
"""
    from nbconvert import HTMLExporter
    path=ROOT/'labs'/lesson(n).with_suffix('.ipynb').name
    output=ROOT/'labs/html'/path.with_suffix('.html').name
    old=BeautifulSoup(output.read_text(),'html.parser') if output.exists() else None
    nb=nbformat.read(path,4);page,_=HTMLExporter(template_name='lab').from_notebook_node(nb);s=BeautifulSoup(page,'html.parser')
    for tag in s.select('[id]'):tag['id']=unquote(tag['id'])
    for tag in s.select('[href],[src]'):
        for key in ['href','src']:
            if not tag.has_attr(key):continue
            u=urlsplit(tag[key])
            if not u.scheme and not u.netloc and u.path:tag[key]='../'+tag[key]
            elif key=='href':tag[key]=unquote(tag[key])
    if old:
        for tag in old.head.select('link[rel=stylesheet]'):
            if tag.get('href','').startswith('../../assets/') and not s.select_one(f'link[href="{tag["href"]}"]'):s.head.append(tag)
        # Keep any dedicated launcher emitted before the notebook container.
        for tag in old.body.find_all(recursive=False):
            if tag.name in ('nav','aside','header') or 'lab-launcher' in tag.get('class',[]) or 'lab-access' in tag.get('class',[]):s.body.insert(0,tag)
    for table in s.find_all('table'):table.wrap(s.new_tag('div',attrs={'style':'max-width:100%;overflow-x:auto','role':'region','tabindex':'0','aria-label':'Scrollable table'}))
    style=s.new_tag('style');style.string='.jp-RenderedHTMLCommon img{height:auto}';s.head.append(style)
    todo=next((h for h in s.find_all('h2') if h.get_text().startswith('TODO 1')),None)
    if todo and not s.find(id='lab-exercises'):todo.insert_before(s.new_tag('span',id='lab-exercises'))
    output.write_text(str(s))


def atlas():
    groups=[(48,54,'Build the predictor','Routing, retrieval, recipes and shared ensembles.'),(55,60,'Make the comparison meaningful','Availability, aggregation, selection and matched evidence.'),(61,66,'Learn inference across tasks','Objective → prior → row / feature representations → labels.'),(67,70,'Adapt, stress and decide','Local contexts, changing worlds, broken contracts and a research handoff.')]
    body=''
    for lo,hi,title,desc in groups:
        body+=f'<h2>{title}</h2><p>{desc}</p><div class="atlas-grid">'
        for n in range(lo,hi+1):
            st=STORIES[n];body+=f'<article id="lesson-{n}"><p class="eyebrow">LESSON {n:03}</p><h3><a href="../lessons/{lesson(n).name}">{E(st["title"])}</a></h3><p>{rich(st["opening"],True)}</p>'
            for m in MAPS[n]:body+=f'<a class="map-link" href="../lessons/{lesson(n).name}#solution-{m.key}">{E(m.title)} →</a>'
            body+='</article>'
        body+='</div>'
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>How the solutions connect · Lessons 048–070</title><style>body{background:#f3f7f7;color:#193b4a;font:16px/1.7 system-ui,sans-serif;margin:0}main{max-width:1100px;margin:50px auto;padding:0 24px}h1{font:48px/1.12 Georgia,serif;max-width:800px}h2{font:32px/1.2 Georgia,serif;margin:50px 0 12px}h3{font:25px/1.25 Georgia,serif}a{color:#08737b;text-underline-offset:3px}.atlas-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr));gap:22px}article{padding:26px;background:white;border:1px solid #d0e0e2;border-radius:12px}.eyebrow{font-size:11px;letter-spacing:.15em;font-weight:700}.map-link{display:block;margin-top:14px}a:focus-visible{outline:3px solid #ae601e;outline-offset:4px}@media(max-width:600px){h1{font-size:36px}main{padding:0 18px}article{padding:20px}}</style></head><body><main><nav><a href="../index.html">← Course</a></nav><h1>Different solutions.<br>A connected argument.</h1><p>Follow what each paper changes, why that change matters, and which question remains. Each lesson pairs a complete solution map with worked mechanisms and an explicit evidence boundary.</p>'''+body+'</main></body></html>'
    (ROOT/'reference/solution-map-atlas.html').write_text(page)

def revise(n):
    if n not in STORIES:return
    for m in MAPS[n]:export(m,n)
    revise_html(n)
    name=lesson(n).with_suffix('.ipynb').name
    placements={}
    for directory in [ROOT/'labs',ROOT/'labs/solutions']:
        p=directory/name
        if p.exists():placements[str(p.relative_to(ROOT))]=revise_notebook(n,p)
    prepared(n);atlas()
    print(json.dumps(dict(lesson=n,maps=len(MAPS[n]),unmatched_notebook_anchors=[v for rows in placements.values() for v in rows if not v['matched']])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('lessons',nargs='*',type=int);args=p.parse_args()
    for n in args.lessons or range(48,71):revise(n)
