"""Synchronize editorial walkthroughs into the canonical lesson manuscripts."""
from pathlib import Path
from html import escape
import re
from nbconvert.filters.markdown import markdown2html_mistune as render
import _architecture_091_100 as maps
R=Path(__file__).resolve().parents[1]
START='<!-- depth-walkthrough:start -->';END='<!-- depth-walkthrough:end -->'
# Keep cold retrieval ahead of the new synthesis, and the detailed derivation after it.
BEFORE={91:'## 1 ·',92:'## 2 ·',93:'## 1 ·',94:'## 1 ·',95:'## 2 ·',96:'## 2 ·',97:'## 2 ·',98:'## 2 ·',99:'## 2 ·',100:'## 2 ·'}
def figure(key):
 title,sub,*_,footer=maps.art.SPECS[key]
 caption=sub+' '+footer
 return f'<figure class="model-map"><div class="model-map-scroll" tabindex="0" role="region" aria-label="Scrollable architecture: {escape(title,quote=True)}"><img src="../assets/architectures/{key}.svg" alt="{escape(caption,quote=True)}"></div><figcaption>{escape(caption)}</figcaption></figure>'
entries=[]
for n in range(91,101):
 source=next((R/'lessons/content').glob(f'{n:04}-*.md'))
 depth=(R/'lessons/depth'/source.name).read_text()
 depth=re.sub(r'\[\[DEPTH_MAP:([^]]+)\]\]',lambda m:figure(m[1]),depth)
 manuscript=source.read_text()
 manuscript=re.sub(re.escape(START)+r'.*?'+re.escape(END)+r'\n*','',manuscript,flags=re.S)
 at=manuscript.index(BEFORE[n]);manuscript=manuscript[:at]+START+'\n'+depth+'\n'+END+'\n\n'+manuscript[at:]
 source.write_text(manuscript)
 builder=R/f'labs/_build_l{n:03}.py';s=builder.read_text()
 if 'from _walkthrough_delivery import snapshot, finalize' not in s:
  s=s.replace('from pathlib import Path','from pathlib import Path\nfrom _walkthrough_delivery import snapshot, finalize\nsnapshot('+str(n)+')',1)
  s+='\nfinalize('+str(n)+')\n';builder.write_text(s)
 heading=depth.splitlines()[0].removeprefix('## ')
 route=['Preserve relation roles','Separate route and neighbor weights','Trace typed score and value branches','Classify assumptions, not scores','Return from users to ranked items','Preserve rows, roles and multiplicity','Define the pairwise objective','Recover query and batch identities','Control the model comparison','Defend the full prediction procedure'][n-91]
 entries.append(f'<section><p class="route-kicker">LESSON {n:03}</p><h2><a href="../lessons/{source.stem}.html">{route}</a></h2><p>{escape(heading.removeprefix("The big picture · "))}</p></section>')
reference='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Lessons 91–100 · Typed graph learning map</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/learning-walkthrough.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="0071-0090-model-map.html">Previous model map</a></nav><header><p>Relational learning · 091–100</p><h1>From typed relationships to defensible predictions</h1><p>Use this map after retrieval. First explain what each method preserves, then inspect its computation and evidence boundary.</p></header><div class="model-index">'''+''.join(entries)+'''</div><h2>Keep the axes separate</h2><table><thead><tr><th>Choice</th><th>Question to recover</th><th>Where to practice</th></tr></thead><tbody><tr><td>Representation</td><td>Which identities, roles, counts and timestamps survive?</td><td>091, 092, 095, 096</td></tr><tr><td>Learned operation</td><td>Which parameters share, and over what set do weights normalize?</td><td>091–093, 097</td></tr><tr><td>Computation boundary</td><td>Which context is legal, and which outputs enter the loss?</td><td>096, 098, 100</td></tr><tr><td>Evidence</td><td>Which procedure was selected, executed and actually compared?</td><td>094, 099, 100</td></tr></tbody></table><h2>Paper reading route</h2><p><a href="https://arxiv.org/abs/1703.06103">R-GCN §2–3</a> → <a href="https://arxiv.org/abs/1903.07293">HAN §3–4</a> → <a href="https://arxiv.org/abs/2003.01332">HGT §3–4</a>. Compare the operators before their datasets or scores. <a href="https://arxiv.org/abs/1205.2618">BPR §4–5</a> supplies the ranking objective; <a href="https://arxiv.org/abs/2312.04615">Relational Deep Learning</a> motivates the database handoff.</p><h2>Three checks to do from memory</h2><ol><li>Explain why a mean per relation differs from a mean over all incoming edges.</li><li>Trace a database key through a typed graph index into a batch-local row.</li><li>Explain why fixed-weight gradient parity does not imply the same optimizer trajectory.</li></ol><p>Open the linked lessons for worked checks and visible labs. Prepared material is not learner mastery; submit a written defense and ask the teacher to challenge it.</p></article></body></html>'''
(R/'reference/0091-0100-model-map.html').write_text(reference+'\n')
print('Synchronized ten walkthroughs and reference map')
