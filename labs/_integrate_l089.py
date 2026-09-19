"""Register L089 package; no learner completion and no deployment implied."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1];SLUG='0089-sampling-at-scale';TITLE='Sampling at scale: Cluster-GCN'
p=ROOT/'lessons/manifest.json';d=json.loads(p.read_text())
if not any(x['id']==89 for x in d['lessons']):
 d['lessons'].append({'id':89,'slug':SLUG,'year':3,'quarter':1,'checkpoint':False,'labPath':f'labs/{SLUG}.ipynb','title':TITLE,'published':True});d['version']+=1;p.write_text(json.dumps(d,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
 p=ROOT/name;s=p.read_text();s=re.sub(r'(name="rdl-manifest-version" content=")\d+',r'\g<1>'+str(d['version']),s)
 if name=='notebooks.html' and 'id="lab-89"' not in s:s=s.replace('<ul class="nb-gallery">',f'<ul class="nb-gallery"><li id="lab-89"><div class="nb-head"><span class="num">Lesson 0089</span><span class="title">{TITLE}</span></div><div class="nb-links"><a href="labs/html/{SLUG}.html">Read lab</a><a href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{SLUG}.ipynb">Run in Colab</a><a href="labs/{SLUG}.ipynb">Download notebook</a><a href="lessons/{SLUG}.html">Lesson</a></div></li>',1)
 p.write_text(s)
p=ROOT/'CURRICULUM.md';p.write_text(p.read_text().replace('| 089 | Sampling at scale | Chiang et al. 2019 (Cluster-GCN) | Large-graph training |',f'| 089 | [Sampling at scale](lessons/{SLUG}.html) | Chiang et al. 2019 (Cluster-GCN) | [Cluster batches, scale/F1 comparison and full PPI recipe](labs/{SLUG}.ipynb) |'))
p=ROOT/'plan/year-3.md';s=p.read_text().replace('### 089 · Sampling at scale — *Chiang 2019 (Cluster-GCN), ◆ `1905.07953`*',f'### 089 · [Sampling at scale](../lessons/{SLUG}.html) — *Chiang 2019 (Cluster-GCN), ◆ `1905.07953`*\n- **Prepared package** — [lab](../labs/{SLUG}.ipynb), [full reproduction contract](../labs/l089-reproduction.md). Complete inline PPI release reconstruction;12 full-data teaching runs and full-width update executed. Full400-epoch target NOT_RUN: Modal GPU blocked by account payment requirement. Historical parity INCOMPARABLE; no learner completion inferred.')
s=s.replace('- **Viz** — reuse `group-viz.js` (partitioned clusters).','- **Viz** — `cluster-sampling-viz.js`: induced graph union restores cut edges and changes normalized messages; `group-viz.js` concerns leakage folds and is not a graph-partition visual.')
p.write_text(s)
for name,marker,entry in [
 ('NOTES.md','Lesson089 created','\n## Lesson089 created · Cluster-GCN · 2026-09-19\n\nUser requested full reproducibility. Full visible PPI release model/trainer, source/data hashes, original split and a runnable five-layer width2048/400-epoch recipe. Twelve separate teaching runs cover full/random/cluster1/cluster5 across3 seeds. One real full-width update passed; full target NOT_RUN because Modal rejected GPU execution pending a payment method. Modern PyTorch/PyMetis historical parity INCOMPARABLE. No learner completion inferred.\n'),
 ('RESOURCES.md','L089 · Cluster-GCN','\n## L089 · Cluster-GCN\n\n- [Chiang et al., KDD2019](https://arxiv.org/abs/1905.07953v2): §§3.1–3.3,Algorithm1 and Table10; batching, diagonal enhancement and PPI result target.\n- [2019 release](https://github.com/google-research/google-research/tree/89c16e403d42015c3133634788ed0b7965f56395/cluster_gcn): PPI shell recipe, precomputed first layer, loss, partition and inference details.\n- [Stanford GraphSAGE datasets](https://snap.stanford.edu/graphsage/): original PPI archive, fixed split, features and multi-label targets.\n'),
 ('labs/README.md','L089:',f'\n- L089: [Sampling at scale / Cluster-GCN]({SLUG}.ipynb) — [reproduction contract](l089-reproduction.md), full inline model/trainer and12-run PPI batching intervention. Full paper execution remains NOT_RUN.\n'),
 ('thesis-dossier.md','| L089 |','\n| L089 | BAR | Training feasibility depends on which relational messages fit inside each batch. PPI teaching runs quantify retained edges, hidden-state proxies and F1, but equal passes give different update counts. Full published-scale recipe is supplied; its result is not reproduced. |\n')]:
 p=ROOT/name;s=p.read_text()
 if marker not in s:p.write_text(s+entry)
p=ROOT/'reference/glossary.html';s=p.read_text()
if 'id="l089-terms"' not in s:s=s.replace('</article>','''<section id="l089-terms"><h2>Graph sampling · L089</h2><dl><dt>Cluster batch</dt><dd>A union of graph partitions used for one optimizer update, with induced edges restored.</dd><dt>Cut edge</dt><dd>An edge whose endpoints belong to different partitions.</dd><dt>Induced subgraph</dt><dd>Selected nodes together with every original edge between them.</dd><dt>Diagonal enhancement</dt><dd>Adding a scaled copy of a normalized adjacency’s diagonal to strengthen self contributions.</dd><dt>Hidden-state memory proxy</dt><dd>An explicit array-size estimate that excludes other allocations and is not measured peak memory.</dd></dl><a href="../lessons/0089-sampling-at-scale.html">Lesson89</a></section></article>''');p.write_text(s)
p=ROOT/'assets/retrieval-pool.js';s=p.read_text()
if 'l089-induced-union' not in s:
 item={'id':'l089-induced-union','lesson':89,'quarter':'Q1','concept':'cluster-sampling','question':'When two graph partitions enter one Cluster-GCN batch, which edges belong in its adjacency?','options':[{'label':'All original edges within union','value':'union'},{'label':'Only original edges within partitions','value':'blocks'},{'label':'All original edges touching union','value':'touch'}],'correct':'union','explain':'The induced subgraph restores crossing edges with both endpoints selected; it excludes edges to unselected nodes.'}
 s=s.replace('\n];\n})(window);',',\n'+json.dumps(item,indent=2)+'\n];\n})(window);');p.write_text(s)
p=ROOT/'.gitignore';s=p.read_text()
if '!labs/solutions/0089-' not in s:p.write_text(s+'\n# L089 prepared solution; downloadable data and learner output stay local.\n!labs/solutions/0089-sampling-at-scale.ipynb\nlabs/data/l089/ppi.zip\nlabs/l089-exit.json\nlabs/results/l089/teaching/*/model.pt\n')
p=ROOT/'requirements-labs.txt';s=p.read_text()
if 'pymetis' not in s:p.write_text(s+'\n# L089: graph partitioning (exact lesson pins in labs/requirements-l089-runtime.txt).\npymetis==2025.2.2\n')
p=ROOT/'.github/workflows/pages.yml';s=p.read_text()
if '# L089 Cluster' not in s:
 s=s.replace('          # L088 graph','''          # L089 Cluster-GCN: inline model, audited recipe, evidence and sources.
          cp labs/l089-reproduction.md labs/requirements-l089-*.txt labs/_sources_l089.json labs/_paper_l089_results.json labs/_teaching_l089_results.json labs/_verify_l089_results.json labs/_scale_l089_results.json labs/_audit_l089_results.json labs/_execution_l089_results.json labs/_delivery_l089_results.json labs/_clean_environment_l089_results.json public/labs/
          cp labs/_run_l089.py labs/_teaching_l089.py labs/_verify_l089.py labs/_audit_l089.py labs/_scale_check_l089.py labs/_figures_l089.py labs/_build_l089.py labs/_execute_l089.py labs/_delivery_l089.py public/labs/
          mkdir -p public/labs/solutions public/labs/sources public/labs/results
          cp labs/solutions/0089-sampling-at-scale.ipynb public/labs/solutions/
          cp -r labs/sources/l089 public/labs/sources/
          cp -r labs/results/l089 public/labs/results/
          # L088 graph''',1)
 p.write_text(s)
print('Registered L089')
