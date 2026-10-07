"""Idempotent visual finishing pass. Run after an individual lesson builder.

Standard library only. SVG stories are editable assets; inline copies work with
JavaScript disabled. --check rejects stale generated publication files.
"""
from pathlib import Path
from html import escape, unescape
import argparse
import json
import re
import sys
from visual_story_specs import STORIES, EXTRA_STORIES
from visual_story_drawings import svg_scene
import visual_details
ROOT=Path(__file__).resolve().parents[1]
START='<!-- lesson-visuals:start -->'; END='<!-- lesson-visuals:end -->'
TOOLS_START='<!-- visual-tools:start -->'; TOOLS_END='<!-- visual-tools:end -->'

def clean_text(s):
    return ' '.join(unescape(re.sub('<[^>]+>', ' ',s)).split())

def key_for(path):
    key=path.stem.split('-')[0]
    return str(int(key)) if key.isdigit() else key

def selected():
    return [p for p in sorted((ROOT/'lessons').glob('*.html')) if re.match(r'b\d|\d{4}',p.name) and (p.name.startswith('b') or int(p.name[:4])>=50)]

def story_html(key,s):
    prefix=f'vs-{key}'
    parts=[f'<section class="visual-story" id="{prefix}" aria-labelledby="{prefix}-heading"><header><p class="vs-kicker">See the idea · then trace the details</p><h2 id="{prefix}-heading">{escape(s["title"])}</h2><p>{escape(s["why"])}</p></header><ol class="vs-stages">']
    scenes=[]
    for i,stage in enumerate(s['stages'],1):
        scene=svg_scene(stage['drawing'],f'{prefix}-{i}',stage['title'],stage['explanation'])
        scenes.append(scene)
        parts.append(f'<li class="vs-stage" data-visual-step="{i}"><h3><span class="vs-number">{i}</span>{escape(stage["title"])}</h3>{scene}<p class="vs-stage-label">{escape(stage["label"])}</p><p class="vs-explanation">{escape(stage["explanation"])}</p></li>')
    parts.append(f'</ol><details class="vs-predict"><summary>Pause and predict: {escape(s["question"])}</summary><p>{escape(s["answer"])}</p></details><p class="vs-print-answer"><strong>Answer:</strong> {escape(s["answer"])}</p><footer><p>{escape(s["scope"])} Small graphs, cells and weights are schematic, not measured results.</p><p><a href="{escape(s["source"],quote=True)}">Primary source</a> · <a href="../assets/visual-stories/{key}.svg">Open the complete drawing</a> · <a href="../reference/visual-reading-guide.html">How to read these diagrams</a></p></footer></section>')
    # Standalone export carries titles, stage labels, scope and source as text.
    import textwrap
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="630" viewBox="0 0 1080 630" role="img" aria-labelledby="title desc"><title id="title">{escape(s["title"])}</title><desc id="desc">{escape(s["why"]+" "+s["scope"])}</desc><rect width="1080" height="630" fill="#fff"/><style>text{{font-family:system-ui,sans-serif;fill:#233c50}}</style>']
    svg.append(f'<text x="28" y="37" font-size="{20 if len(s['title'])>60 else 24}" font-weight="650">{escape(s["title"])}</text><text x="28" y="67" font-size="15">{escape(s["why"])}</text>')
    for i,(stage,scene) in enumerate(zip(s['stages'],scenes)):
        x=20+i*355
        svg.append(f'<rect x="{x}" y="92" width="335" height="420" rx="14" fill="#f6fafb" stroke="#d1dfe4"/>')
        svg.append(f'<text x="{x+16}" y="121" font-size="{13 if len(stage['title'])>32 else 16}" font-weight="650">{i+1}. {escape(stage["title"])}</text>')
        inner=re.sub(r'^<svg[^>]+>|</svg>$','',scene)
        svg.append(f'<g transform="translate({x+16} 133)">{inner}</g>')
        for j,line in enumerate(textwrap.wrap(stage['label'],38)):
            svg.append(f'<text x="{x+16}" y="{374+j*20}" font-size="14" font-weight="650">{escape(line)}</text>')
        for j,line in enumerate(textwrap.wrap(stage['explanation'],46)):
            svg.append(f'<text x="{x+16}" y="{426+j*17}" font-size="12">{escape(line)}</text>')
    for j,line in enumerate(textwrap.wrap(s['scope'],130)):
        svg.append(f'<text x="28" y="{548+j*18}" font-size="13">{escape(line)}</text>')
    svg.append(f'<a href="{escape(s["source"],quote=True)}"><text x="28" y="604" font-size="13">Source: {escape(s["source"])}</text></a></svg>')
    return ''.join(parts), ''.join(svg)

def update(path):
    text=visual_details.strip(path.read_text())
    # Marked regions are the only generated lesson content removed on refresh.
    text=re.sub(re.escape(START)+r'.*?'+re.escape(END)+'\n?', '',text,flags=re.S)
    text=re.sub(re.escape(TOOLS_START)+r'.*?'+re.escape(TOOLS_END), '',text,flags=re.S)
    text=re.sub(r'\s*<link\b[^>]*href=["\x27]../assets/visual-stories.css["\x27][^>]*>', '',text)
    text=re.sub(r'\s*<script\b[^>]*src=["\x27]../assets/visual-stories.js["\x27][^>]*>\s*</script>', '',text)
    key=key_for(path); links=[]; index=0
    text=visual_details.inject(text,key)
    def figure(m):
        nonlocal index
        full=m[0]
        if not re.search(r'<(?:img|svg)\b',full):return full
        index+=1
        opening=re.match(r'<figure\b[^>]*>',full)[0]
        existing=re.search(r'\bid=["\x27]([^"\x27]+)',opening)
        anchor=existing[1] if existing else f'visual-figure-{index}'
        if not existing:full=full.replace(opening,opening[:-1]+f' id="{anchor}">',1)
        caption=re.search(r'<figcaption\b[^>]*>(.*?)</figcaption>',full,re.S)
        title=clean_text(caption[1]) if caption else ''
        img=re.search(r'<img\b[^>]*>',full)
        if not title and img:
            alt=re.search(r'\balt=["\x27]([^"\x27]*)',img[0]);title=unescape(alt[1]) if alt else ''
        if not title:
            t=re.search(r'<title\b[^>]*>(.*?)</title>',full,re.S);title=clean_text(t[1]) if t else f'Diagram {index}'
        title=title.split('. ')[0]
        if len(title)>100:title=title[:97].rsplit(' ',1)[0]+'…'
        links.append((anchor,title))
        src=re.search(r'\bsrc=["\x27]([^"\x27]+)',img[0]) if img else None
        if src:
            tools=f'<div class="vs-figure-tools"><a href="{escape(src[1],quote=True)}" data-enlarge="">Open original figure</a></div>'
        elif '<svg' in full:
            # A static jump keeps no-JS usable; the browser adds the enlarged view.
            tools=f'<div class="vs-figure-tools"><a href="#{anchor}" data-enlarge="" data-figure-svg="">Figure {index}</a></div>'
        else:tools=''
        return full.replace('</figure>',TOOLS_START+tools+TOOLS_END+'</figure>')
    text=re.sub(r'<figure\b[^>]*>.*?</figure>',figure,text,flags=re.S)
    story='';asset=None
    if key in STORIES:
        story,asset=story_html(key,STORIES[key]);links.insert(0,(f'vs-{key}',STORIES[key]['title']))
        for extra in EXTRA_STORIES.get(key,[]):
            story+=story_html(extra,STORIES[extra])[0]
            links.append((f'vs-{extra}',STORIES[extra]['title']))
    nav='<nav class="visual-reading" data-visual-reading="" aria-label="Visual reading route"><details><summary>Visual reading route'+(f' · {len(links)} figures' if links else '')+'</summary>'
    if links:
        nav+='<ol>'+''.join(f'<li><a href="#{escape(a,quote=True)}">{escape(t)}</a></li>' for a,t in links)+'</ol><p>Start with the overall idea, then inspect the mechanism and evidence. Use “Enlarge figure” for dense diagrams; press Escape to return.</p>'
    else:
        # Writing/community lessons should not acquire invented model diagrams.
        nav+='<p>This lesson develops a written argument or practical workflow. Follow its worked examples and linked evidence; it does not introduce a new model architecture.</p>'
    nav+='</details></nav>'
    block=START+'\n'+nav+story+'\n'+END+'\n'
    h1=re.search(r'</h1>',text)
    if not h1:raise ValueError(f'No lesson heading: {path}')
    # Insert at an existing section boundary after the title. No prose is replaced.
    boundary=re.search(r'<(?:h2|section)\b',text[h1.end():])
    pos=h1.end()+boundary.start() if boundary else h1.end()
    text=text[:pos]+block+text[pos:]
    text=text.replace('</head>','<link rel="stylesheet" href="../assets/visual-stories.css"/></head>',1)
    text=text.replace('</body>','<script src="../assets/visual-stories.js" defer></script></body>',1)
    return text,asset,dict(lesson=path.name,story=key in STORIES,figures=len(links),source=STORIES.get(key,{}).get('source'))

def reference():
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Read an architecture diagram</title><link rel="stylesheet" href="../assets/lesson.css"></head><body><article><nav><a href="../index.html">Course</a></nav><h1>Read an architecture diagram</h1><p>A useful diagram lets you trace a prediction, identify the new operation, and name the evidence that can affect the answer.</p><h2>Trace three things</h2><ol><li><strong>Inputs:</strong> distinguish rows, cells, labels, edges and time. Mark what the query is allowed to observe.</li><li><strong>Computation:</strong> follow the arrows. A branch means parallel routes; a repeated block reuses a computation pattern; shared weights are different from shared activations.</li><li><strong>Output:</strong> identify the head and its task. Separate the forward prediction from the loss that updates weights during training.</li></ol><h2>Read the pictures</h2><p>Small grids depict tables or token arrays; strips depict vector coordinates; circles joined by lines depict graphs. Amber often marks a query or distinguished role; teal and violet distinguish computational routes. Read labels as well as colors. Cell counts, graph sizes, bar heights and edge widths in the visual stories are schematic, not experimental measurements.</p><h2>Ask what the paper added</h2><p>Cover the caption and explain the mechanism aloud. Which representation changes? Which nodes or rows can communicate? What state is retained? What is shared? Then reveal the question’s answer and check your explanation against the detailed architecture and the cited primary source.</p><h2>Keep the evidence separate</h2><p>An explanatory diagram is not proof that a result was reproduced. Use each lesson’s protocol and results to distinguish a teaching example, saved-evidence replay, checkpoint inference and fresh training.</p><h2>When the lesson introduces no model</h2><p>Read its protocol, timeline, comparison or evidence plot instead. Ask what is held fixed, what changes, and which observation would support the conclusion.</p><p>Ask the teaching agent to walk one concrete row, token or message through any diagram that is still unclear.</p><p><a href="../lessons/0043-tabnet.html">Earlier example: TabNet</a> · <a href="../lessons/0052-tabr-retrieval.html">Retrieval: TabR</a> · <a href="../lessons/0164-griffin-graph-centric-rdb-fm.html">Relational cells: Griffin</a></p></article></body></html>'''

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    changes=[];coverage=[]
    def emit(path,value):
        if not path.exists() or path.read_text()!=value:
            changes.append(str(path.relative_to(ROOT)))
            if not args.check:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
    for path in selected():
        html,svg,record=update(path);emit(path,html);coverage.append(record)
        if svg:emit(ROOT/'assets/visual-stories'/f'{key_for(path)}.svg',svg)
    for key in visual_details.DETAILS:
        emit(ROOT/'assets/visual-details'/f'{key}.svg',visual_details.render(key))
    for keys in EXTRA_STORIES.values():
        for key in keys:emit(ROOT/'assets/visual-stories'/f'{key}.svg',story_html(key,STORIES[key])[1])
    emit(ROOT/'reference/visual-reading-guide.html',reference())
    emit(ROOT/'reviews/lesson-visuals-2026-10-07/coverage.json',json.dumps(coverage,indent=2)+'\n')
    print(json.dumps(dict(lessons=len(coverage),stories=sum(r['story'] for r in coverage),changed=len(changes))))
    if args.check and changes:
        print('Stale generated files: '+', '.join(changes[:8]),file=sys.stderr);return 1
    return 0
if __name__=='__main__':raise SystemExit(main())
