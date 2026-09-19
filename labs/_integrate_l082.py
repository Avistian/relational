"""Register L082 without marking learner completion or publishing remotely."""
from pathlib import Path
import json,re,hashlib
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0082-gcn';TITLE='GCN: normalize, propagate, reproduce'
p=ROOT/'lessons/manifest.json';m=json.loads(p.read_text())
if not any(x['id']==82 for x in m['lessons']):
    m['lessons'].append({'id':82,'slug':SLUG,'year':3,'quarter':1,'checkpoint':False,'labPath':f'labs/{SLUG}.ipynb','title':TITLE,'published':True});m['version']+=1;p.write_text(json.dumps(m,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
    p=ROOT/name;s=p.read_text();s=re.sub(r'(name="rdl-manifest-version" content=")\d+',r'\g<1>'+str(m['version']),s)
    if name=='notebooks.html' and 'id="lab-82"' not in s:
        s=s.replace('<ul class="nb-gallery">',f'<ul class="nb-gallery"><li id="lab-82"><div class="nb-head"><span class="num">Lesson 0082</span><span class="title">{TITLE}</span></div><div class="nb-links"><a href="labs/html/{SLUG}.html">Read lab</a><a href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{SLUG}.ipynb">Run in Colab</a><a href="labs/{SLUG}.ipynb" download>Download notebook</a><a href="lessons/{SLUG}.html">Lesson</a></div></li>',1)
    p.write_text(s)
p=ROOT/'CURRICULUM.md';p.write_text(p.read_text().replace('| 082 | GCN | Kipf 2017 | Cora node classification |',f'| 082 | [GCN](lessons/{SLUG}.html) | Kipf 2017 | [Full Cora fixed-split reproduction port](labs/{SLUG}.ipynb) |'))
p=ROOT/'.github/workflows/pages.yml';s=p.read_text()
if '# L082 GCN' not in s:
    s=s.replace('          # L081 MPNN', '''          # L082 GCN, full Cora experiment and source-visible solution.
          cp labs/l082-reproduction.md labs/requirements-l082-observed.txt labs/_sources_l082.json labs/_paper_l082_results.json labs/_verify_l082_results.json labs/_execution_l082_results.json labs/_browser_l082_results.json public/labs/
          cp labs/_run_l082.py labs/_verify_l082.py labs/_isolation_l082.py labs/_isolation_l082_results.json public/labs/
          mkdir -p public/labs/solutions
          cp labs/solutions/0082-gcn.ipynb public/labs/solutions/
          # L081 MPNN''',1);p.write_text(s)
for name,marker,extra in [
('NOTES.md','Lesson 082 created','\n## Lesson 082 created · GCN · 2026-09-19\n\nFull Cora Table2 experiment port: 100 fresh runs, mean81.401%, sample SD0.658pp; target81.5%. Inline normalization, propagation and masked-loss TODOs feed the complete trainer. Full-size release protocol, source/data hashes, four computation figures, hidden-state diagnostic and exact commands. PyTorch versus TensorFlow1 and unavailable historical seeds remain explicit. Original-framework parity NOT_RUN. Authorship does not advance learner completion.\n'),
('RESOURCES.md','L082 · GCN','\n## L082 · GCN\n\n- [Kipf & Welling 2017](https://arxiv.org/html/1609.02907v4): spectral motivation, Eq2 propagation, Table2 Cora target.\n- [Pinned TensorFlow release](https://github.com/tkipf/gcn/tree/39a4089fe72ad9f055ed6fdb9746abdcfebc4d81): actual stopping rule, regularization, dropout and Planetoid data. Source/data provenance shared with L078 and verified again for L082.\n'),
('labs/README.md','L082:',f'\n- L082: [GCN]({SLUG}.ipynb) — exact normalization, sparse propagation, masked gradients and [full Cora experiment](l082-reproduction.md).\n'),
('thesis-dossier.md','| L082 |','\n| L082 | BAR | Full Cora fixed-split GCN port,100 initializations; measured81.401% versus81.5% target. Supports reproducible static graph aggregation, not temporal RDL advantage or cross-framework identity. |\n')]:
    p=ROOT/name;s=p.read_text()
    if marker not in s:p.write_text(s+extra)
p=ROOT/'reference/glossary.html';s=p.read_text()
if 'id="gcn-l082"' not in s:s=s.replace('</article>','<h2 id="gcn-l082">L082 · GCN</h2><dl><dt>Symmetric normalization</dt><dd>Each edge message is weighted by the reciprocal square root of sender and receiver augmented degrees.</dd><dt>Transductive node classification</dt><dd>Unlabeled evaluation nodes and their features/edges participate in the observed graph; their labels remain excluded from training supervision.</dd></dl></article>');p.write_text(s)
import importlib.metadata as metadata,sys
names=['numpy','scipy','torch','nbformat','nbclient','nbconvert','beautifulsoup4','matplotlib','ipykernel','networkx']
(LAB/'requirements-l082-observed.txt').write_text('# Observed Python '+sys.version.split()[0]+' CPU runtime; public-wheel availability not asserted.\n'+'\n'.join(n+'=='+metadata.version(n) for n in names)+'\n')
source=json.loads((LAB/'_sources_l078.json').read_text())
source.update({'lesson':82,'shared_data_manifest':'_sources_l078.json','paper':'https://arxiv.org/html/1609.02907v4','named_target':'Table2 / GCN / Cora /81.5%','local_sha256':{n:hashlib.sha256((LAB/n).read_bytes()).hexdigest() for n in ['relkit/gcn_l082.py','_run_l082.py']}})
(LAB/'_sources_l082.json').write_text(json.dumps(source,indent=2)+'\n')

p=ROOT/'assets/retrieval-pool.js';s=p.read_text()
if 'l082-endpoint-degrees' not in s:
    item={'id':'l082-endpoint-degrees','lesson':82,'quarter':'Q1','concept':'gcn-normalization','question':'What determines an edge coefficient in symmetric GCN normalization?','options':[{'label':'Both augmented endpoint degrees determine weights','value':'correct'},{'label':'Only augmented receiver degrees determine weights','value':'receiver'},{'label':'Only augmented sender degrees determine weights','value':'sender'}],'correct':'correct','explain':'Add self-loops first, then use 1/sqrt(d_source*d_destination). This is not ordinary row averaging.'}
    s=s.replace('\n];',',\n'+json.dumps(item,indent=2)+'\n];');p.write_text(s)

print('Registered L082')
