"""Register the prepared lesson without asserting learner completion or remote publication."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1];SLUG='0083-graphsage';TITLE='GraphSAGE: sample, aggregate, generalize'
p=ROOT/'lessons/manifest.json';m=json.loads(p.read_text())
if not any(x['id']==83 for x in m['lessons']):
 m['lessons'].append({'id':83,'slug':SLUG,'year':3,'quarter':1,'checkpoint':False,'labPath':f'labs/{SLUG}.ipynb','title':TITLE,'published':True});m['version']+=1;p.write_text(json.dumps(m,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
 p=ROOT/name;s=p.read_text()
 s=re.sub(r'<meta\b[^>]*name="rdl-manifest-version"[^>]*>',f'<meta name="rdl-manifest-version" content="{m["version"]}">',s)
 if name=='notebooks.html' and 'id="lab-83"' not in s:
  s=s.replace('<ul class="nb-gallery">',f'<ul class="nb-gallery"><li id="lab-83"><div class="nb-head"><span class="num">Lesson 0083</span><span class="title">{TITLE}</span></div><div class="nb-links"><a href="labs/html/{SLUG}.html">Read lab</a><a href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{SLUG}.ipynb">Run in Colab</a><a href="labs/{SLUG}.ipynb" download>Download notebook</a><a href="lessons/{SLUG}.html">Lesson</a></div></li>',1)
 p.write_text(s)
p=ROOT/'CURRICULUM.md';p.write_text(p.read_text().replace('| 083 | GraphSAGE | Hamilton 2017 | Inductive mini-batch training |',f'| 083 | [GraphSAGE](lessons/{SLUG}.html) | Hamilton 2017 | [Inductive sampling + full PPI experiment port](labs/{SLUG}.ipynb) |'))
p=ROOT/'.github/workflows/pages.yml';s=p.read_text()
if '# L083 GraphSAGE' not in s:
 s=s.replace('          # L082 GCN', '''          # L083 GraphSAGE, full PPI port, reproducibility evidence and inline solution.
          cp labs/l083-reproduction.md labs/requirements-l083-observed.txt labs/_sources_l083.json labs/_paper_l083_results.json labs/_verify_l083_results.json labs/_execution_l083_results.json labs/_browser_l083_results.json public/labs/
          cp labs/_run_l083.py labs/_verify_l083.py public/labs/
          mkdir -p public/labs/solutions
          cp labs/solutions/0083-graphsage.ipynb public/labs/solutions/
          # L082 GCN''',1);p.write_text(s)
for name,marker,extra in [
 ('NOTES.md','Lesson 083 created','\n## Lesson 083 created · GraphSAGE · 2026-09-19\n\nFull supervised PPI mean port, three seeds × three learning rates × ten epochs; mean micro-F1 .59093, sample SD .00606 versus paper .598. Full inline sampler, aggregator, loader and trainer. Released-code/pseudocode differences, historical selection/seed gaps and PyTorch differences explicit; historical parity INCOMPARABLE. Held-out-feature/label training intervention passes. No learner completion inferred.\n'),
 ('RESOURCES.md','L083 · GraphSAGE','\n## L083 · GraphSAGE\n\n- [Hamilton, Ying & Leskovec 2017](https://arxiv.org/html/1706.02216v4): Algorithms1–2, Table1 PPI supervised mean .598, AppendixC hyperparameters.\n- [Pinned official implementation](https://github.com/williamleif/GraphSAGE/tree/a0fdef95dca7b456dab01cb35034717c8b6dd017): inspect mean concat, reversed sampling, final normalization, fixed adjacency and training access.\n- [Full PPI data](https://snap.stanford.edu/graphsage/ppi.zip): archive SHA-256 in labs/_sources_l083.json; not the tiny example_data subset.\n'),
 ('labs/README.md','L083:',f'\n- L083: [GraphSAGE]({SLUG}.ipynb) — three live TODOs, inductive audit and [full PPI reproduction contract](l083-reproduction.md).\n'),
 ('labs/data/README.md','L083 PPI','\n## L083 PPI\n\nTier B, original complete Stanford GraphSAGE PPI archive. Download on demand into data/l083/ppi.zip; SHA-256 checked before reading.56,944 nodes,50 features,121 labels; split44,906/6,514/5,524. Standardization fits training nodes only. Original graph split, full ten-epoch schedule; see l083-reproduction.md for historical gaps.\n'),
 ('plan/year-3.md','L083 delivery evidence','\n## L083 delivery evidence\n\nPrepared [lesson](../lessons/0083-graphsage.html), [lab](../labs/0083-graphsage.ipynb) and [contract](../labs/l083-reproduction.md). Native PyTorch mini-batches deliberately expose the released padded-table sampler instead of substituting modern NeighborLoader semantics. Full PPI release port executed; paper-result parity remains INCOMPARABLE. Creation does not assert mastery.\n')]:
 p=ROOT/name;s=p.read_text()
 if marker not in s:p.write_text(s+extra)
p=ROOT/'reference/glossary.html';s=p.read_text()
if 'id="graphsage-l083"' not in s:
 s=s.replace('</article>','<h2 id="graphsage-l083">L083 · GraphSAGE</h2><dl><dt>Inductive evaluation</dt><dd>Evaluate shared learned functions on nodes or graphs absent from the training computation.</dd><dt>Fanout</dt><dd>The number of sampled neighbor occurrences per receiver at one outward expansion step.</dd><dt>Support nodes</dt><dd>Nodes whose features or intermediate states are needed to compute predictions for the supervised roots.</dd></dl></article>');p.write_text(s)
print('Integrated L083')
