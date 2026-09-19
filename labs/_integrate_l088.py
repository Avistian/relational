"""Register L088 materials without recording learner completion or publishing."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1];SLUG='0088-graph-classification';TITLE='Graph classification: what can GIN distinguish?'
p=ROOT/'lessons/manifest.json';d=json.loads(p.read_text())
if not any(x['id']==88 for x in d['lessons']):
 d['lessons'].append({'id':88,'slug':SLUG,'year':3,'quarter':1,'checkpoint':False,'labPath':f'labs/{SLUG}.ipynb','title':TITLE,'published':True});d['version']+=1;p.write_text(json.dumps(d,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
 p=ROOT/name;s=p.read_text();s=re.sub(r'(name="rdl-manifest-version" content=")\d+',r'\g<1>'+str(d['version']),s)
 if name=='notebooks.html' and 'id="lab-88"' not in s:
  s=s.replace('<ul class="nb-gallery">',f'<ul class="nb-gallery"><li id="lab-88"><div class="nb-head"><span class="num">Lesson 0088</span><span class="title">{TITLE}</span></div><div class="nb-links"><a href="labs/html/{SLUG}.html">Read lab</a><a href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{SLUG}.ipynb">Run in Colab</a><a href="labs/{SLUG}.ipynb">Download notebook</a><a href="lessons/{SLUG}.html">Lesson</a></div></li>',1)
 p.write_text(s)
p=ROOT/'CURRICULUM.md';p.write_text(p.read_text().replace('| 088 | Graph classification | Xu et al. 2019 (GIN) | Whole-graph readout |',f'| 088 | [Graph classification](lessons/{SLUG}.html) | Xu et al. 2019 (GIN) | [GIN, WL and whole-graph readout; MUTAG reproduction track](labs/{SLUG}.ipynb) |'))
p=ROOT/'plan/year-3.md';p.write_text(p.read_text().replace('### 088 · Graph classification — *Xu 2019 (GIN), `1810.00826`*',f'### 088 · [Graph classification](../lessons/{SLUG}.html) — *Xu 2019 (GIN), `1810.00826`*\n- **Prepared package** — [lab](../labs/{SLUG}.ipynb), [full reproduction contract](../labs/l088-reproduction.md). Complete inline GIN and MUTAG grid, separate readout intervention, WL refinement visual. Consult execution JSON for actual grid coverage; historical parity INCOMPARABLE. No learner completion inferred.'))
p=ROOT/'.github/workflows/pages.yml';s=p.read_text()
if '# L088 graph' not in s:
 s=s.replace('          # L087 link','''          # L088 graph classification: full visible model, source/data and run evidence.
          cp labs/l088-reproduction.md labs/requirements-l088-runtime.txt labs/requirements-l088-lock.txt labs/_sources_l088.json labs/_paper_l088_results.json labs/_teaching_l088_results.json labs/_verify_l088_results.json labs/_audit_l088_results.json labs/_execution_l088_results.json labs/_delivery_l088_results.json labs/_clean_environment_l088_results.json public/labs/
          cp labs/_run_l088.py labs/_teaching_l088.py labs/_verify_l088.py labs/_audit_l088.py labs/_figures_l088.py labs/_build_l088.py labs/_execute_l088.py labs/_delivery_l088.py public/labs/
          mkdir -p public/labs/solutions public/labs/data public/labs/sources public/labs/results
          cp labs/solutions/0088-graph-classification.ipynb public/labs/solutions/
          cp -r labs/data/l088 public/labs/data/
          cp -r labs/sources/l088 public/labs/sources/
          cp -r labs/results/l088 public/labs/results/
          # L087 link''',1)
 p.write_text(s)
for name,marker,entry in [
 ('NOTES.md','Lesson088 created','\n## Lesson088 created · Graph classification / GIN · 2026-09-19\n\nUser requested full reproducibility. Complete visible GIN-0, hash-pinned MUTAG loader, release-aligned training/grid and common-epoch CV selection. Separate readout intervention, three live TODOs, WL counterexample widget and portable architecture figures. Actual execution coverage is in labs/_paper_l088_results.json; runnable search is not a completion claim. Historical parity INCOMPARABLE; no learner completion inferred.\n'),
 ('RESOURCES.md','L088 · Graph classification','\n## L088 · Graph classification\n\n- [Xu et al., ICLR2019](https://arxiv.org/abs/1810.00826v3): §4 GIN and graph readout, §7 Table1 MUTAG target, expressiveness bound and scope.\n- [Pinned powerful-gnns release](https://github.com/weihua916/powerful-gnns/tree/9a2ce8ac3e99278307093a464a95caf0fb04b602): model, data, training recipe and clarified common-epoch cross-validation.\n- [PyTorch1.0 scheduler](https://github.com/pytorch/pytorch/blob/v1.0.0/torch/optim/lr_scheduler.py): historical StepLR initialization for faithful decay timing.\n'),
 ('labs/README.md','L088:',f'\n- L088: [Graph classification / GIN]({SLUG}.ipynb) — [MUTAG reproduction contract](l088-reproduction.md), full inline implementation, WL limit and readout intervention.\n'),
 ('thesis-dossier.md','| L088 |','\n| L088 | BAR | GIN makes multiset counting and the 1-WL ceiling explicit. A six-cycle and two triangles remain indistinguishable under identical initial labels; empirical graph classification cannot establish universal relational expressiveness. MUTAG execution coverage and historical protocol gaps are reported separately. |\n')]:
 p=ROOT/name;s=p.read_text()
 if marker not in s:p.write_text(s+entry)
p=ROOT/'reference/glossary.html';s=p.read_text()
if 'id="l088-terms"' not in s:
 s=s.replace('</article>','''<section id="l088-terms"><h2>Graph classification · L088</h2><dl><dt>Graph readout</dt><dd>Permutation-invariant reduction from the nodes of one graph to a fixed-size representation of that graph.</dd><dt>Multiset</dt><dd>A collection in which repetitions and their counts are retained.</dd><dt>Injective aggregation</dt><dd>An aggregation mapping distinct inputs to distinct outputs on its stated domain.</dd><dt>1-WL refinement</dt><dd>Simultaneous relabeling by each node's previous label and multiset of neighbor labels, using shared label identities across graphs.</dd><dt>GIN</dt><dd>Graph Isomorphism Network: sum neighborhood aggregation followed by an MLP, with graph readouts across depths.</dd><dt>Common-epoch cross-validation</dt><dd>Choosing one epoch by maximizing the mean validation curve across folds, then reporting fold dispersion at that epoch.</dd></dl><a href="../lessons/0088-graph-classification.html">Lesson88</a></section></article>''');p.write_text(s)
p=ROOT/'assets/retrieval-pool.js';s=p.read_text()
if 'l088-wl-limit' not in s:
 item={'id':'l088-wl-limit','lesson':88,'quarter':'Q1','concept':'gin-expressiveness','question':'With identical initial node labels, which pair remains indistinguishable to 1-WL?','options':[{'label':'Six-cycle versus two triangles','value':'cycle'},{'label':'Four-path versus four-star graphs','value':'path'},{'label':'Four-cycle versus four-clique graphs','value':'clique'}],'correct':'cycle','explain':'Each graph has six nodes with two neighbors per node. Joint WL refinement preserves identical color histograms forever.'}
 s=s.replace('\n];\n})(window);',',\n'+json.dumps(item,indent=2)+'\n];\n})(window);');p.write_text(s)
p=ROOT/'.gitignore';s=p.read_text()
if '!labs/solutions/0088-' not in s:p.write_text(s+'\n# L088 prepared solution; learner output stays local.\n!labs/solutions/0088-graph-classification.ipynb\nlabs/l088-exit.json\n')
print('Registered L088')
