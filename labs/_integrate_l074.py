"""Idempotent course navigation and source ledger integration."""
import json,hashlib,platform,importlib.metadata
from pathlib import Path
from bs4 import BeautifulSoup
R=Path(__file__).resolve().parents[1];L=R/'labs';slug='0074-carte-cross-table-transfer';title='CARTE: transfer across table schemas'
p=R/'lessons/manifest.json';m=json.loads(p.read_text())
if not any(z['id']==74 for z in m['lessons']):
 m['lessons'].append({'id':74,'slug':slug,'year':2,'quarter':4,'checkpoint':False,'labPath':'labs/'+slug+'.ipynb','title':title,'published':True});m['version']+=1;p.write_text(json.dumps(m,indent=2)+'\n')
for file in ['index.html','notebooks.html']:
 p=R/file;s=p.read_text()
 if f'lessons/{slug}.html' not in s:
  if file=='index.html':s=s.replace('<p><a href="lessons/0073-',f'<p><a href="lessons/{slug}.html">Lesson 074 · {title}</a> · <a href="labs/html/{slug}.html">Lab</a></p>\n  <p><a href="lessons/0073-',1)
  else:s=s.replace('<ul class="nb-gallery">',f'<ul class="nb-gallery"><li id="lab-74"><div class="nb-head"><span class="num">Lesson 0074</span><span class="title">{title}</span></div><div class="nb-links"><a href="labs/html/{slug}.html">Read lab</a><a href="labs/{slug}.ipynb" download>Download notebook</a><a href="lessons/{slug}.html">Lesson</a></div></li>',1)
  p.write_text(s)
p=R/'.github/workflows/pages.yml';s=p.read_text()
if '# L074 CARTE' not in s:s=s.replace('          # L073 label-budget', '          # L074 CARTE source-checked transfer, compact real vectors and checkpoint.\n          cp labs/_verify_l074_results.json labs/_sources_l074.json labs/l074-reproduction.md labs/_check_l074_results.json labs/_execution_l074_results.json labs/_browser_l074_results.json public/labs/\n          cp labs/_verify_l074.py labs/_run_l074.py labs/_check_l074.py labs/_prepare_l074.py public/labs/\n          mkdir -p public/labs/data\n          cp -r labs/data/l074 public/labs/data/\n          # L073 label-budget');p.write_text(s)
p=R/'assets/retrieval-pool.js';s=p.read_text()
if 'l074-schema-transfer' not in s:
 item={'id':'l074-schema-transfer','lesson':74,'quarter':'Q4','concept':'schema-transfer','question':'Why can CARTE accept previously unseen column sets?','options':[{'label':'It shares learned operations across value-relation pairs','value':'shared'},{'label':'It learns separate positions for every schema','value':'positions'},{'label':'It aligns all source columns before inference','value':'align'}],'correct':'shared','explain':'Shared message operations act on cell and column embeddings. Input compatibility still requires empirical transfer evaluation.'}
 s=s.replace('global.RETRIEVAL_POOL = [','global.RETRIEVAL_POOL = [\n'+json.dumps(item,indent=2)+',',1);p.write_text(s)
for file,marker,extra in [
 ('labs/README.md','L074:','\n- L074: [CARTE cross-table transfer](0074-carte-cross-table-transfer.ipynb) — real FastText/YAGO embeddings, graph attention, matched scratch controls.\n'),
 ('labs/data/README.md','L074 CARTE','\n## L074 CARTE\n\nTier A: three 384-row subsets of released wine tables. Fixed real FastText sentence-vector cache and YAGO checkpoint accompany source IDs and SHA256 digests in `l074/manifest.json`. No full string-model download is needed for the lab; see `../l074-reproduction.md`.\n'),
 ('NOTES.md','Lesson 074 created','\n## Lesson 074 created · CARTE cross-table transfer\n\nSource-visible single-readout CARTE, real FastText vectors and selected YAGO checkpoint transfer to three wine schemas. Paper/release center, sender-index and initial_x loading discrepancies are explicit. Authorship does not advance learner completion. Full benchmark INCOMPARABLE; joint source-table learning and new pretraining NOT_RUN. See per-lesson evidence and contract.\n'),
 ('RESOURCES.md','L074 CARTE','\n## L074 CARTE\n\n- [CARTE v2](https://arxiv.org/html/2402.16785v2), Figures 1–3 and Sections 3.1–3.3: graph representation and cross-schema transfer.\n- [Pinned release](https://github.com/soda-inria/carte/tree/f54690da4cddbedd1e1a9113a312f85783d2c125): implementation/checkpoint/example-table provenance.\n- [FastText English crawl vectors](https://fasttext.cc/docs/en/crawl-vectors.html): real language inputs and licensing.\n'),
 ('thesis-dossier.md','L074 |','\n| L074 | BAR | CARTE provides schema-variable within-row encodings; local YAGO transfer is measured on three related wine tables with scratch controls. This does not establish foreign-key reasoning or temporal validity. Full benchmark INCOMPARABLE. See labs/_verify_l074_results.json. |\n')]:
 p=R/file;s=p.read_text()
 if marker not in s:p.write_text(s+extra)
p=R/'reference/glossary.html';s=p.read_text()
if 'id="carte-schema-transfer"' not in s:s=s.replace('</article>','<section id="carte-schema-transfer"><h2>CARTE: schema transfer</h2><p><strong>Schema matching:</strong> explicit correspondences between columns across tables. <strong>Entity matching:</strong> explicit identification of records referring to the same real-world entity. <strong>Row graph:</strong> a center representing one row connected to its observed cell nodes, with column vectors on edges. <strong>Frozen probe:</strong> a target predictor fitted on fixed encoder outputs.</p></section></article>');p.write_text(s)
files=['sources/carte-l074/carte_model.py','sources/carte-l074/carte_table_to_graph.py','sources/carte-l074/LICENSE.txt','relkit/carte_l074.py','data/l074/manifest.json']
manifest={'paper':'https://arxiv.org/html/2402.16785v2','release':'f54690da4cddbedd1e1a9113a312f85783d2c125','files':{f:hashlib.sha256((L/f).read_bytes()).hexdigest() for f in files},'versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','torch','scikit-learn','catboost','fasttext-wheel','torch-geometric']},'python':platform.python_version(),'deviations':['released edge-conditioned center','sender-indexed messages','one-readout architecture with no ordinary blocks','strict initial_x transfer differs from estimator','ridge or linear target head','zero dropout','constant-column train-only standard-scaling fallback','capped target tables and fixed local budgets'],'original_benchmark':'INCOMPARABLE'}
(L/'_sources_l074.json').write_text(json.dumps(manifest,indent=2))
print('Integrated L074 navigation and provenance')
