"""Register prepared L087 without recording learner mastery or publishing."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0087-link-prediction';TITLE='Link prediction: hide the edge, define the candidates'
p=ROOT/'lessons/manifest.json';data=json.loads(p.read_text())
if not any(x['id']==87 for x in data['lessons']):
 data['lessons'].append({'id':87,'slug':SLUG,'year':3,'quarter':1,'checkpoint':False,'labPath':f'labs/{SLUG}.ipynb','title':TITLE,'published':True});data['version']+=1;p.write_text(json.dumps(data,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
 p=ROOT/name;s=p.read_text();s=re.sub(r'(name="rdl-manifest-version" content=")\d+',r'\g<1>'+str(data['version']),s)
 if name=='notebooks.html' and 'id="lab-87"' not in s:s=s.replace('<ul class="nb-gallery">',f'<ul class="nb-gallery"><li id="lab-87"><div class="nb-head"><span class="num">Lesson 0087</span><span class="title">{TITLE}</span></div><div class="nb-links"><a href="labs/html/{SLUG}.html">Read lab</a><a href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{SLUG}.ipynb">Run in Colab</a><a href="labs/{SLUG}.ipynb">Download notebook</a><a href="lessons/{SLUG}.html">Lesson</a></div></li>',1)
 p.write_text(s)
p=ROOT/'CURRICULUM.md';p.write_text(p.read_text().replace('| 087 | Link prediction | Zhang & Chen 2018 (skim) | Edge split metrics |',f'| 087 | [Link prediction](lessons/{SLUG}.html) | Zhang & Chen 2018 (skim) | [Edge splits, decoder, ranking; full baseline reconstruction](labs/{SLUG}.ipynb) |'))
p=ROOT/'plan/year-3.md';s=p.read_text().replace('### 087 · Link prediction — *Zhang & Chen 2018 (skim)*',f'### 087 · [Link prediction](../lessons/{SLUG}.html) — *Zhang & Chen 2018 (skim)*\n- **Prepared package** — [lab](../labs/{SLUG}.ipynb), [reproduction guide](../labs/l087-reproduction.md): all8 datasets ×10 splits ×3 Table1 heuristics executed; separate live GCN decoder/ranking lab. Historical parity INCOMPARABLE; full SEAL classifier NOT_RUN. No learner completion inferred.');p.write_text(s)
p=ROOT/'.github/workflows/pages.yml';s=p.read_text()
if '# L087 link' not in s:
 s=s.replace('          # L086 PyG','''          # L087 link prediction, full baseline reconstruction and explicit evidence.
          cp labs/l087-reproduction.md labs/requirements-l087-runtime.txt labs/requirements-l087-lock.txt labs/_sources_l087.json labs/_targets_l087.json labs/_paper_l087_results.json labs/_teaching_l087_results.json labs/_verify_l087_results.json labs/_audit_l087_results.json labs/_execution_l087_results.json labs/_delivery_l087_results.json labs/_clean_environment_l087_results.json public/labs/
          cp labs/_run_l087.py labs/_verify_l087.py labs/_audit_l087.py labs/_figures_l087.py labs/_build_l087.py labs/_execute_l087.py labs/_delivery_l087.py labs/_original_l087.m labs/_clean_environment_l087.py public/labs/
          mkdir -p public/labs/solutions public/labs/results
          cp labs/solutions/0087-link-prediction.ipynb public/labs/solutions/
          cp -r labs/results/l087 public/labs/results/
          # L086 PyG''',1)
 p.write_text(s)
for name,marker,text in [
 ('NOTES.md','Lesson087 created','\n## Lesson087 created · Link prediction · 2026-09-19\n\nFull reproducibility requested. Prepared split/decoder/ranking lesson with three live TODOs; full Table1 CN/AA/RA reconstruction on all eight released graphs and ten splits (240 evaluations), plus separate three-seed USAir GCN teaching runs. Hash-pinned data/source, raw split/score artifacts, independent audits and native MATLAB replay provided. Historical parity INCOMPARABLE (permutation/source-era gap); MATLAB replay and full SEAL classifier NOT_RUN. No learner completion inferred.\n'),
 ('RESOURCES.md','L087 · Link prediction','\n## L087 · Link prediction\n\n- [Zhang & Chen2018](https://arxiv.org/abs/1802.09691): primary reading, enclosing subgraphs/DRNL; Table1 baseline target.\n- [Pinned author source](https://github.com/muhanzhang/SEAL/tree/ca1f019a15fb0c21796042165b4e6bee73981dd3): data, split/sampling, heuristic scores, metric and labeling implementation.\n- [OGB task contracts](https://ogb.stanford.edu/docs/linkprop/) and [evaluator](https://github.com/snap-stanford/ogb/blob/master/ogb/linkproppred/evaluate.py): candidate policies and average tied ranks.\n'),
 ('labs/README.md','L087:',f'\n- L087: [Link prediction]({SLUG}.ipynb) — [full baseline reproduction guide](l087-reproduction.md), visible decoder, leakage checks and candidate-defined MRR/Hits.\n'),
 ('thesis-dossier.md','| L087 |','\n| L087 | BAR | Full8-graph,10-split CN/AA/RA baseline reconstruction and separate GCN decoder/ranking lab expose how graph visibility and candidate selection define the task. Static edge recovery is not temporal recommendation evidence; historical numerical parity INCOMPARABLE, full SEAL training NOT_RUN. |\n')]:
 p=ROOT/name;s=p.read_text()
 if marker not in s:p.write_text(s+text)
p=ROOT/'reference/glossary.html';s=p.read_text()
if 'id="l087-terms"' not in s:s=s.replace('</article>','''<section id="l087-terms"><h2>Link prediction · L087</h2><dl><dt>Context graph</dt><dd>Edges supplied to the encoder, distinct from pairs carrying supervised labels.</dd><dt>Filtered candidate set</dt><dd>Alternatives with other known positives excluded under a declared observation rule.</dd><dt>Mean reciprocal rank (MRR)</dt><dd>Mean of one divided by each query's positive rank; requires candidate and tie conventions.</dd><dt>Hits@k</dt><dd>Fraction of queries whose positive rank is at most k under the declared ranking convention.</dd><dt>Enclosing subgraph</dt><dd>Induced graph on nodes within h hops of either candidate endpoint; SEAL removes the target edge before classification.</dd><dt>DRNL</dt><dd>Double-radius node labeling: roots receive1; other labels encode distances computed with the opposite root removed, with unreachable nodes assigned0.</dd></dl><a href="../lessons/0087-link-prediction.html">Lesson87</a></section></article>''')
p.write_text(s)
p=ROOT/'assets/retrieval-pool.js';s=p.read_text()
if 'l087-reverse-edge' not in s:
 item={'id':'l087-reverse-edge','lesson':87,'quarter':'Q1','concept':'edge-leakage','question':'For an undirected held-out link, what must be removed from the encoder context?','options':[{'label':'Both directed edge copies','value':'both'},{'label':'Only forward edge copies','value':'forward'},{'label':'Only reverse edge copies','value':'reverse'}],'correct':'both','explain':'Canonicalize unordered pairs before splitting; add both message directions only for context pairs.'}
 s=s.replace('\n];\n})(window);',',\n'+json.dumps(item,indent=2)+'\n];\n})(window);');p.write_text(s)
print('Registered L087 and updated course/reference links')
