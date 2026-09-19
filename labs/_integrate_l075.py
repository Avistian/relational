"""Idempotent course integration; authored content does not mark learner mastery."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[1];SLUG='0075-pytorch-frame-row-encoder';TITLE='PyTorch Frame: the row encoder'
p=R/'lessons/manifest.json';m=json.loads(p.read_text())
if not any(z['id']==75 for z in m['lessons']):
 m['lessons'].append({'id':75,'slug':SLUG,'year':2,'quarter':4,'checkpoint':False,'labPath':'labs/'+SLUG+'.ipynb','title':TITLE,'published':True});m['version']+=1;p.write_text(json.dumps(m,indent=2)+'\n')
p=R/'notebooks.html';s=p.read_text()
if f'lessons/{SLUG}.html' not in s:
 s=s.replace('<ul class="nb-gallery">',f'<ul class="nb-gallery"><li id="lab-75"><div class="nb-head"><span class="num">Lesson 0075</span><span class="title">{TITLE}</span></div><div class="nb-links"><a href="labs/html/{SLUG}.html">Read lab</a><a href="labs/{SLUG}.ipynb" download>Download notebook</a><a href="lessons/{SLUG}.html">Lesson</a></div></li>',1);p.write_text(s)
p=R/'.github/workflows/pages.yml';s=p.read_text()
if '# L075 PyTorch Frame' not in s:
 s=s.replace('          # L074 CARTE','          # L075 PyTorch Frame API evidence and commands.\n          cp labs/_verify_l075_results.json labs/_sources_l075.json labs/l075-reproduction.md labs/_execution_l075_results.json labs/_browser_l075_results.json public/labs/\n          cp labs/_verify_l075.py labs/_check_l075.py labs/_prepare_l075.py labs/requirements-l075-audit.txt public/labs/\n          # L074 CARTE',1);p.write_text(s)
p=R/'assets/retrieval-pool.js';s=p.read_text()
if 'l075-materialization' not in s:
 item={'id':'l075-materialization','lesson':75,'quarter':'Q4','concept':'fit-scope','question':'In PyTorch Frame 0.3.0, how do you keep materialization statistics training-only?','options':[{'label':'Fit training rows; reuse their converter','value':'reuse'},{'label':'Fit all rows; assign split metadata','value':'all'},{'label':'Fit query rows; rebuild their vocabulary','value':'query'}],'correct':'reuse','explain':'The release computes statistics on the supplied DataFrame. Split metadata does not restrict fitting. Reuse training-fitted state for query conversion.'}
 s=s.replace('global.RETRIEVAL_POOL = [','global.RETRIEVAL_POOL = [\n'+json.dumps(item,indent=2)+',',1);p.write_text(s)
p=R/'assets/paper-deck.js';s=p.read_text()
if 'hu2024-frame' not in s:
 item={'id':'hu2024-frame','paper':'Hu et al. — PyTorch Frame','year':2024,'lesson':75,'front':'What separates a TensorFrame from a row embedding in PyTorch Frame?','back':'Materialization organizes typed tensor data. Type-specific encoders create column vectors; column interaction and decoding produce a row vector. Relationships between database records need a separate graph and temporal protocol.'}
 s=s.replace('global.PAPER_DECK = [','global.PAPER_DECK = [\n'+json.dumps(item,indent=2)+',',1);p.write_text(s)
for name,marker,extra in [
 ('labs/README.md','L075:',f'\n- L075: [PyTorch Frame row encoder]({SLUG}.ipynb) — five-stype API trace, train-only materialization, real credit_g row vectors; no accuracy claim.\n'),
 ('labs/data/README.md','L075 PyTorch Frame','\n## L075 PyTorch Frame\n\nTier C: four training/two query rows generated visibly in `relkit/frame_l075.py`, isolating five semantic types. Tier A: existing OpenML31 credit_g cache, actual numeric/categorical columns, seed75 fixed 128/32 rows, labels unused. Cache hash and IDs are recorded in `_verify_l075_results.json`; no new download required when cached.\n'),
 ('NOTES.md','Lesson 075 created','\n## Lesson 075 created · PyTorch Frame row encoder\n\nAPI/tool lesson per curriculum; not a downscaled paper benchmark. Pinned0.3.0 source, training-only materializer, four parent encoders/five input columns, visible MLP readout, numeric reconstruction and real credit_g encoding. Split-column leakage demonstrated (mean20 vs265). Authorship does not mark learner completion. Paper benchmarks NOT_RUN; no predictive comparison. See per-lesson execution/browser/delivery reports.\n'),
 ('RESOURCES.md','L075 PyTorch Frame','\n## L075 PyTorch Frame\n\n- [Hu et al., v2, Figure1 and §3](https://arxiv.org/html/2404.00776v2): materialization, encoding, column interactions and decoding.\n- [Pinned release0.3.0](https://github.com/pyg-team/pytorch-frame/tree/d998aae368db6a4e36139ccc56bd54579a70874b): executable enum, converter, statistics and encoders; local files byte-matched.\n- [Official heterogeneous-type tutorial](https://pytorch-frame.readthedocs.io/en/latest/handling_advanced_stypes/handle_heterogeneous_stypes.html): conceptual configuration guide; latest docs can differ from the pin.\n'),
 ('thesis-dossier.md','L075 |','\n| L075 | BAR | PyTorch Frame provides the typed row-encoding interface needed before relational message passing. Local five-type and real credit_g checks establish shape, fit scope and gradient behavior, not relational advantage or predictive quality. No benchmark run; random row vectors remain interface evidence. |\n')]:
 p=R/name;s=p.read_text()
 if marker not in s:p.write_text(s+extra)
p=R/'reference/glossary.html';s=p.read_text()
if 'id="frame-semantic-type"' not in s:
 s=s.replace('</article>','<section id="frame-semantic-type"><h2>Semantic type and row encoding</h2><p><strong>Semantic type (stype):</strong> a modeling interpretation of a column, distinct from its storage dtype. <strong>Materialization:</strong> fitted conversion of raw values to typed tensor blocks. <strong>TensorFrame:</strong> named feature blocks grouped by parent type. <strong>Feature encoder:</strong> maps typed values to column tokens [B,C,d]. <strong>Row encoder:</strong> combines column representations into one vector [B,D] per row; does not itself supply relational edges.</p></section></article>');p.write_text(s)
print('Integrated lesson75; learner mastery state unchanged')
