"""Idempotent navigation and evidence integration; no learner mastery claim."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0077-single-table-ceiling';TITLE='Single-table ceiling: an information audit'
p=ROOT/'lessons/manifest.json';m=json.loads(p.read_text())
if not any(x['id']==77 for x in m['lessons']):
    m['lessons'].append({'id':77,'slug':SLUG,'year':2,'quarter':4,'checkpoint':False,'labPath':'labs/'+SLUG+'.ipynb','title':TITLE,'published':True});m['version']+=1;p.write_text(json.dumps(m,indent=2)+'\n')
p=ROOT/'notebooks.html';s=p.read_text()
if f'lessons/{SLUG}.html' not in s:
    s=s.replace('<ul class="nb-gallery">',f'<ul class="nb-gallery"><li id="lab-77"><div class="nb-head"><span class="num">Lesson 0077</span><span class="title">{TITLE}</span></div><div class="nb-links"><a href="labs/html/{SLUG}.html">Read lab</a><a href="labs/{SLUG}.ipynb" download>Download notebook</a><a href="lessons/{SLUG}.html">Lesson</a></div></li>',1);p.write_text(s)
p=ROOT/'.github/workflows/pages.yml';s=p.read_text()
if '# L077 ceiling' not in s:
    s=s.replace('          # L076 stack','          # L077 ceiling: full synthetic reproduction and delivery evidence.\n          cp labs/_verify_l077_results.json labs/_execution_l077_results.json labs/_browser_l077_results.json labs/_sources_l077.json labs/l077-reproduction.md public/labs/\n          cp labs/_verify_l077.py public/labs/\n          # L076 stack',1);p.write_text(s)
p=ROOT/'assets/retrieval-pool.js';s=p.read_text()
if 'l077-ceiling' not in s:
    item={'id':'l077-ceiling','lesson':77,'quarter':'Q4','concept':'representation-ceiling','question':'Opposite labels share identical flat vectors in balanced pairs. What repairs this information gap?','options':[{'label':'Supply an informative history feature','value':'feature'},{'label':'Increase the existing model depth','value':'depth'},{'label':'Repeat the existing training loop','value':'repeat'}],'correct':'feature','explain':'A deterministic function gives identical vectors the same prediction. A permissible feature that distinguishes the histories can break the collision.'}
    s=s.replace('global.RETRIEVAL_POOL = [','global.RETRIEVAL_POOL = [\n'+json.dumps(item,indent=2)+',',1);p.write_text(s)
for name,marker,extra in [
('NOTES.md','Lesson 077 created','\n## Lesson 077 created · single-table ceiling · 2026-09-19\n\nUser reiterated full reproduction. L077 is the planned synthesis construction, not a new paper architecture. Full 1000-pair × five-seed experiment reproduced locally: deterministic flat stump accuracy/ceiling .5; same learner with eligible last-minus-first accuracy/ceiling1. Complete visible generator/split/aggregation/trainer and three live TODOs, SQL and library-tree controls, interventions, four portable figures and browser-checked controls. This proves a specified information ceiling, not real-data RDL superiority. L076 historical benchmark remains separate and NOT_RUN. No learner completion inferred.\n'),
('RESOURCES.md','L077 single-table ceiling','\n## L077 single-table ceiling\n\n- [Fey et al., ICML2024 position paper](https://proceedings.mlr.press/v235/fey24a.html): primary motivation for graph learning over databases.\n- [Zaheer et al., Deep Sets](https://arxiv.org/abs/1703.06114): permutation-invariant set functions; supporting reading, no paper experiment claimed.\n- [RelBench v1](https://arxiv.org/html/2407.20060v1): real-data evaluation and feature-engineering comparison, distinct from the authored collision construction.\n'),
('labs/README.md','L077:',f'\n- L077: [Single-table ceiling]({SLUG}.ipynb) — complete five-seed synthetic reproduction, exact information bound and tabular feature-repair control.\n'),
('labs/data/README.md','L077 paired histories','\n## L077 paired histories\n\nTier C, generated entirely in `relkit/ceiling_l077.py`. Full experiment uses 1000 pairs/seed and seeds0–4; 2000 customers and10000 event rows per seed. Pair-group 60/20/20 splits, cutoff10 with event/availability masks. Target is a known eligible historical pattern. Synthetic data is intentional for an exact impossibility construction; no external dataset substitute is presented as a paper reproduction. Source and per-seed dataset hashes are saved.\n'),
('thesis-dossier.md','L077 |','\n| L077 | FOR / BAR | Full paired-history construction reproduced on five seeds: specified flat input has exact .5 accuracy ceiling; adding eligible order delta permits1 with the same tabular stump. Supports an information-loss mechanism. Limits the broader claim: adequate manually derived relational features solve this task; neither GNN superiority nor real-world prevalence follows. |\n')]:
    p=ROOT/name;s=p.read_text()
    if marker not in s:p.write_text(s+extra)
p=ROOT/'reference/glossary.html';s=p.read_text()
if 'id="representation-ceiling"' not in s:
    s=s.replace('</article>','<section id="representation-ceiling"><h2>Representation ceiling</h2><p><strong>Collision class:</strong> examples with exactly the same predictor input. <strong>Empirical accuracy ceiling:</strong> sum of the largest label count in each collision class divided by the total sample size. It audits a specified information interface and is not a generalization estimate. <strong>Sufficient feature for a deterministic target:</strong> a feature representation from which that target can be recovered without ambiguity. <strong>Cardinality:</strong> number of records in a collection.</p></section></article>');p.write_text(s)
provenance={'kind':'original synthesis experiment; no published-table match asserted','primary_sources':[
{'url':'https://proceedings.mlr.press/v235/fey24a.html','role':'relational-learning motivation','accessed':'2026-09-19'},
{'url':'https://arxiv.org/abs/1703.06114','role':'set-function background','accessed':'2026-09-19'},
{'url':'https://arxiv.org/html/2407.20060v1','role':'separate real-data empirical question','accessed':'2026-09-19'}],
'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [LAB/'relkit/ceiling_l077.py',LAB/'_verify_l077.py',ROOT/'lessons/content'/f'{SLUG}.md',LAB/'l077-reproduction.md']}}
(LAB/'_sources_l077.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('Integrated L077; no learner mastery inferred')
