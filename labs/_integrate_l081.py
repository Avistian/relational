"""Course registration, provenance and evidence; does not advance learner progress."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0081-mpnn-framework';TITLE='MPNNs: message, aggregate, update, readout'
p=ROOT/'lessons/manifest.json';m=json.loads(p.read_text())
if not any(x['id']==81 for x in m['lessons']):
    m['lessons'].append({'id':81,'slug':SLUG,'year':3,'quarter':1,'checkpoint':False,'labPath':f'labs/{SLUG}.ipynb','title':TITLE,'published':True});m['version']+=1;p.write_text(json.dumps(m,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
    p=ROOT/name;s=p.read_text();s=re.sub(r'(name="rdl-manifest-version" content=")\d+',r'\g<1>'+str(m['version']),s)
    if name=='notebooks.html' and 'id="lab-81"' not in s:
        s=s.replace('<ul class="nb-gallery">',f'<ul class="nb-gallery"><li id="lab-81"><div class="nb-head"><span class="num">Lesson 0081</span><span class="title">{TITLE}</span></div><div class="nb-links"><a href="labs/html/{SLUG}.html">Read lab</a><a href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{SLUG}.ipynb">Run in Colab</a><a href="labs/{SLUG}.ipynb" download>Download notebook</a><a href="lessons/{SLUG}.html">Lesson</a></div></li>',1)
    p.write_text(s)
p=ROOT/'CURRICULUM.md';p.write_text(p.read_text().replace('| 081 | MPNN framework | Gilmer 2017 | Implement generic MPNN |',f'| 081 | [MPNN framework](lessons/{SLUG}.html) | Gilmer 2017 | [Generic MPNN + auditable QM9 reconstruction](labs/{SLUG}.ipynb) |'))
p=ROOT/'.github/workflows/pages.yml';s=p.read_text()
if '# L081 MPNN' not in s:
    s=s.replace('          # L080 exit', '''          # L081 MPNN lesson, source provenance and complete reconstruction evidence.
          cp labs/l081-reproduction.md labs/requirements-l081-observed.txt labs/_sources_l081.json labs/_verify_l081_results.json labs/_execution_l081_results.json labs/_browser_l081_results.json labs/_check_l081_evidence_results.json public/labs/
          cp labs/_run_l081.py labs/_verify_l081.py public/labs/
          mkdir -p public/labs/sources public/labs/evidence
          cp -r labs/sources/l081 public/labs/sources/
          cp -r labs/evidence/l081 public/labs/evidence/
          # L080 exit''',1);p.write_text(s)
for name,marker,extra in [
('NOTES.md','Lesson 081 created','\n## Lesson 081 created · MPNN framework · 2026-09-19\n\nUser explicitly requested full reproducibility. Generic routing lab plus full sparse GG-NN QM9 model, loader, trainer and budgeted search are visible. Named target: Gilmer supplementary Table3 GG-NN mu (.394 Debye). Official source lacks reader/trainer/search configurations; fresh IDs and modern features are documented gaps. Three fresh 512-molecule/40-update smoke runs execute; full historical result NOT_RUN, reconstruction scores INCOMPARABLE. No learner completion inferred.\n'),
('RESOURCES.md','L081 · Gilmer','\n## L081 · Gilmer MPNN framework\n\n- [Primary paper](https://proceedings.mlr.press/v70/gilmer17a.html): framework §2, features §6, training §7.\n- [Supplement](https://proceedings.mlr.press/v70/gilmer17a/gilmer17a-supp.pdf): GCN mapping §1.1; sparse GG-NN Table3 and chemical accuracy Table1.\n- [Official source](https://github.com/brain-research/mpnn/tree/4a1f0ddea3cd7de5eebc96e509da2161624aaacd): model-only release with missing reader/trainer; pin and hashes in labs/_sources_l081.json.\n'),
('labs/README.md','L081:',f'\n- L081: [MPNN framework]({SLUG}.ipynb) — routing, permutation tests, full sparse molecular GG-NN and [reconstruction protocol](l081-reproduction.md).\n'),
('labs/data/README.md','L081 QM9','\n## L081 QM9\n\nTier C toy graph for routing; Tier B real molecular graphs for named-target reconstruction. Hash-pinned QM9 SDF/CSV and public uncharacterized list download into ignored data/cache/l081. RDKit sanitization failures and molecule IDs are retained. Smoke takes first512 valid molecules, then seeded splits; the file-order cap is not representative. Full historical population/IDs unavailable.\n'),
('thesis-dossier.md','| L081 |','\n| L081 | BAR | Executable learned graph aggregation and invariant molecular readout, with honest QM9 reconstruction boundaries. Mechanism tests and smoke training do not establish relational advantage, historical paper parity, or learner mastery. |\n')]:
    p=ROOT/name;s=p.read_text()
    if marker not in s:p.write_text(s+extra)
p=ROOT/'requirements-labs.txt';s=p.read_text()
if 'rdkit==' not in s:p.write_text(s+'\nrdkit==2025.9.6  # L081 — explicit QM9 chemical features\n')
p=ROOT/'assets/retrieval-pool.js';s=p.read_text()
if 'l081-equivariance' not in s:
    item={'id':'l081-equivariance','lesson':81,'quarter':'Q1','concept':'graph-permutation','question':'After consistently relabeling a graph, what happens to node and graph predictions?','options':[{'label':'Node outputs permute; graph output remains','value':'correct'},{'label':'Node outputs remain; graph output permutes','value':'wrong1'},{'label':'Node outputs change; graph output changes','value':'wrong2'}],'correct':'correct','explain':'Equivariance transports node outputs with their nodes; invariant readout preserves the whole-graph prediction.'}
    s=s.replace('\n];',',\n'+json.dumps(item,indent=2)+'\n];');p.write_text(s)
p=ROOT/'reference/glossary.html';s=p.read_text()
if 'id="mpnn-l081"' not in s:
    section='<h2 id="mpnn-l081">L081 · Message passing</h2><dl><dt>Message</dt><dd>Information computed for a directed source-to-destination edge from states and edge features.</dd><dt>Permutation equivariance</dt><dd>Relabeling inputs relabels node outputs in the same way.</dd><dt>Permutation invariance</dt><dd>Relabeling nodes leaves a graph-level output unchanged.</dd><dt>Readout</dt><dd>A map from node states to the required graph output; it must respect arbitrary node order.</dd></dl>'
    s=s.replace('</article>',section+'</article>');p.write_text(s)
import torch,numpy,rdkit,torch_geometric
(LAB/'requirements-l081-observed.txt').write_text(f'# Observed author runtime: Python 3.12.3; CPU. Not a claim of public-wheel availability.\ntorch=={torch.__version__}\nnumpy=={numpy.__version__}\nrdkit=={rdkit.__version__}\ntorch-geometric=={torch_geometric.__version__}\n')
files=['sources/l081/mpnn.py','sources/l081/README','sources/l081/LICENSE','relkit/mpnn_l081.py','relkit/qm9_l081.py','_run_l081.py']
provenance={'paper':'https://proceedings.mlr.press/v70/gilmer17a.html','supplement':'https://proceedings.mlr.press/v70/gilmer17a/gilmer17a-supp.pdf','source_commit':'4a1f0ddea3cd7de5eebc96e509da2161624aaacd','source_repo':'https://github.com/brain-research/mpnn','sha256':{n:hashlib.sha256((LAB/n).read_bytes()).hexdigest() for n in files},'target':{'table':'Supplement Table 3','model':'GG-NN','property':'mu','error_ratio':3.94,'chemical_accuracy_debye':.1,'mae_debye':.394},'source_runtime_parity':'NOT_CHECKED','full_reproduction':'NOT_RUN'}
(LAB/'_sources_l081.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('Integrated L081')
