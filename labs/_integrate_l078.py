"""Idempotent L078 course integration; no learner mastery claim."""
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SLUG='0078-message-passing-preview';TITLE='Message passing: from a join to a GCN'
p=ROOT/'lessons/manifest.json';m=json.loads(p.read_text())
if not any(x['id']==78 for x in m['lessons']):
    m['lessons'].append({'id':78,'slug':SLUG,'year':2,'quarter':4,'checkpoint':False,'labPath':'labs/'+SLUG+'.ipynb','title':TITLE,'published':True});m['version']+=1;p.write_text(json.dumps(m,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
    p=ROOT/name;s=p.read_text();s=re.sub(r'(name="rdl-manifest-version" content=")\d+',r'\g<1>'+str(m['version']),s)
    if name=='notebooks.html' and f'lessons/{SLUG}.html' not in s:s=s.replace('<ul class="nb-gallery">',f'<ul class="nb-gallery"><li id="lab-78"><div class="nb-head"><span class="num">Lesson 0078</span><span class="title">{TITLE}</span></div><div class="nb-links"><a href="labs/html/{SLUG}.html">Read lab</a><a href="labs/{SLUG}.ipynb" download>Download notebook</a><a href="lessons/{SLUG}.html">Lesson</a></div></li>',1)
    p.write_text(s)
p=ROOT/'.github/workflows/pages.yml';s=p.read_text()
if '# L078 message' not in s:s=s.replace('          # L077 ceiling','          # L078 message passing, complete Cora protocol and measured evidence.\n          cp labs/_verify_l078_results.json labs/_execution_l078_results.json labs/_browser_l078_results.json labs/_sources_l078.json labs/_paper_l078_results.json labs/l078-reproduction.md labs/requirements-l078-observed.txt public/labs/\n          cp labs/_run_l078.py labs/_verify_l078.py public/labs/\n          mkdir -p public/labs/data public/modal\n          cp -r labs/data/l078 public/labs/data/\n          cp modal/l078_paper_repro.py public/modal/\n          # L077 ceiling',1);p.write_text(s)
p=ROOT/'assets/retrieval-pool.js';s=p.read_text()
if 'l078-synchronous' not in s:
    item={'id':'l078-synchronous','lesson':78,'quarter':'Q4','concept':'synchronous-message-passing','question':'On A—B—C, change only C’s initial feature. Under the lesson’s local synchronous update, when can A first change?','options':[{'label':'After two message rounds','value':'two'},{'label':'After one message round','value':'one'},{'label':'Before any message rounds','value':'zero'}],'correct':'two','explain':'The first round changes B; the second can carry that changed state to A. All messages within one round read old states.'}
    s=s.replace('global.RETRIEVAL_POOL = [','global.RETRIEVAL_POOL = [\n'+json.dumps(item,indent=2)+',',1);p.write_text(s)
for name,marker,extra in [
('NOTES.md','Lesson 078 created','\n## Lesson 078 created · message passing · 2026-09-19\n\nFull reproducibility requested. Added manual mean-message bridge and complete inline Cora GCN port of the pinned release. Executed100 initializations on the full fixed split:81.401% mean,0.658 percentage-point sample SD. Paper target81.5%; modern-framework/original-seed differences remain explicit. Gilmer QM9 and original TensorFlow parity NOT_RUN. Student/solution notebooks, portable architecture and arithmetic figures, reusable Y3 widget, source/data hashes and full commands included. No learner completion inferred.\n'),
('RESOURCES.md','L078 message passing','\n## L078 message passing\n\n- [Gilmer et al., ICML2017, §2](https://proceedings.mlr.press/v70/gilmer17a.html): message, sum, update and graph readout.\n- [Kipf & Welling, ICLR2017](https://arxiv.org/html/1609.02907v4): Eq2/9 and Table2 Cora fixed-split experiment.\n- [Pinned GCN release](https://github.com/tkipf/gcn/tree/39a4089fe72ad9f055ed6fdb9746abdcfebc4d81): preprocessing, architecture, loss, initialization and exact stopping code; paper/code stopping mismatch documented.\n'),
('labs/README.md','L078:',f'\n- L078: [Message passing]({SLUG}.ipynb) — hand aggregation plus the full100-seed Cora GCN experiment with visible code and pinned bytes.\n'),
('labs/data/README.md','L078 Cora','\n## L078 Cora\n\nFull raw released Cora files from tkipf/gcn revision39a4089fe72ad9f055ed6fdb9746abdcfebc4d81; `_sources_l078.json` pins every file hash. Load only after verification. Fixed140/500/1000 train/validation/test label split, all2708 nodes visible under the transductive protocol. Separate four-node synthetic fixture exists only for manual arithmetic.\n'),
('thesis-dossier.md','L078 |','\n| L078 | BAR | Full Cora GCN release-protocol port,100 initializations:81.401% mean versus published81.5%; implementation makes learned neighbor aggregation concrete. Single static transductive graph, modern-framework deviations and no tabular baseline comparison: neither temporal safety nor relational-database superiority follows. |\n')]:
    p=ROOT/name;s=p.read_text()
    if marker not in s:p.write_text(s+extra)
p=ROOT/'reference/glossary.html';s=p.read_text()
if 'id="message-passing-l078"' not in s:
    s=s.replace('</article>','<section id="message-passing-l078"><h2>Message passing</h2><p><strong>Message:</strong> vector sent along an eligible edge. <strong>Reduction:</strong> combine incoming vectors, e.g. sum or mean. <strong>Synchronous update:</strong> all new node states read the same old snapshot. <strong>Permutation equivariance:</strong> consistent node relabeling reorders node outputs in the same way. <strong>Receptive field:</strong> input nodes that can influence a given output. <strong>GCN support:</strong> adjacency with added identity, normalized by both endpoint augmented degrees. <strong>Transductive protocol:</strong> all node features and graph structure are available while only specified labels train the model.</p></section></article>');p.write_text(s)
print('Integrated L078')
