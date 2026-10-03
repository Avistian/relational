from pathlib import Path
import json,re
R=Path(__file__).resolve().parents[1]
p=R/'lessons/manifest.json';j=json.loads(p.read_text());j['lessons']=[x for x in j['lessons'] if x['id']!='B09'];j['lessons'].append(dict(id='B09',slug='b09-cost-frontier',year=5,quarter=5,sortOrder=200.09,checkpoint=False,labPath='labs/b09-cost-frontier.ipynb',title='Current cost frontier: TabFM, EXAONE and Nori',published=True));j['version']=max(j.get('version',1),8);p.write_text(json.dumps(j,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
 p=R/name;p.write_text(re.sub(r'(name="rdl-manifest-version" content=")[0-9]+',lambda m:m[1]+str(j['version']),p.read_text()))
p=R/'CURRICULUM.md';s=p.read_text().replace('B01–B08, B04a and B07a are prepared','B01–B09, B04a and B07a are prepared').replace('The other 19 units remain planned.','B09 adds a matched CPU cost protocol and source-gated EXAONE Figure 3 frontier. The other 18 units remain planned.').replace('[Current cost frontier: TabFM, EXAONE and Nori](./plan/year-5-6-bridge.md#b09) | Matched latency, memory and quality protocol.','[Current cost frontier: TabFM, EXAONE and Nori](lessons/b09-cost-frontier.html) | [Lab](labs/b09-cost-frontier.ipynb): matched rows, measured cost, Figure 3 source gate.');p.write_text(s)
p=R/'reference/curriculum.html';s=p.read_text().replace('B01–B08, B04a and B07a prepared; 19 planned','B01–B09, B04a and B07a prepared; 18 planned').replace('../plan/year-5-6-bridge.md#b09','../lessons/b09-cost-frontier.html');p.write_text(s)
adds={'labs/README.md':'\n\n### B09 · Current cost frontier\n\n[Student](b09-cost-frontier.ipynb) · [Executed solution](html/b09-cost-frontier.html) · [Protocol](b09-reproduction.md). Identity-safe metrics, Pareto dominance and support visibility; separate current-release CPU inference, saved-evidence replay and historical Figure3 source gate.\n','.gitignore':'\n# B09 portable solution; checkpoints remain external.\n!labs/solutions/b09-cost-frontier.ipynb\n','RESOURCES.md':'\n\n## B09 · Cost frontier primary sources\n\n[EXAONE §2 and Figure3](https://arxiv.org/html/2608.25774v1), [TabFM §§3–5](https://arxiv.org/html/2609.37959v1), [Nori model card](https://huggingface.co/Synthefy/Nori/blob/main/README.md). Sources, current code commits and weight revisions pinned in `labs/sources/b09/`. Published GPU/TabArena claims remain distinct from current-release CPU diabetes inference. Seldon and NEXUS are provider-only access inventory entries.\n','thesis-dossier.md':'\n\n- **B09 · BAR:** A credible relational advantage requires matched end-to-end baseline costs. Current release identities, ensembles and transductive preprocessing complicate comparisons. The course CPU matrix cannot establish the published regression Elo frontier or a general model ranking; see `labs/b09-reproduction.md`.\n'}
for file,addition in adds.items():
 p=R/file
 if addition.strip() not in p.read_text():p.write_text(p.read_text()+addition)
p=R/'assets/retrieval-pool.js';s=p.read_text()
if 'b09-operating-point' not in s:
 pos=s.rfind('];');item="\n, {id:'b09-operating-point',lesson:200.09,quarter:'Q4',concept:'prediction-cost',question:'A model wins on warm latency. What else is needed for a one-request cost claim?',options:[{label:'Include loading and support preparation',value:'complete'},{label:'Include only stored parameter counts',value:'weights'}],correct:'complete',explain:'One-request latency pays cold load, preprocessing and first-call work. Warm timing alone describes a different workload.'}\n"
 s=s[:pos]+item+s[pos:];p.write_text(s)
p=R/'reference/glossary.html';s=p.read_text()
if 'id="b09-operating-point"' not in s:s=s.replace('</body>','<section id="b09-operating-point"><h2>Operating point · B09</h2><p>The complete prediction configuration: checkpoint, input/support rows, preprocessing, ensemble, cache, precision, hardware and workload. A Pareto point is not dominated on every declared cost axis; at least one strict improvement is needed for dominance.</p></section></body>');p.write_text(s)
p=R/'.github/workflows/pages.yml';s=p.read_text()
if '# B09:' not in s:
 block='''          # B09: portable evidence and pinned source archives; no external weights.
          mkdir -p public/labs/evidence/b09 public/labs/sources/b09 public/labs/data/b09 public/labs/figures/b09
          cp labs/b09-reproduction.md labs/_*b09*.py labs/_*b09*.json public/labs/
          cp labs/relkit/cost_b09.py public/labs/relkit/
          cp -r labs/evidence/b09/. public/labs/evidence/b09/
          cp -r labs/sources/b09/. public/labs/sources/b09/
          cp labs/data/b09/*.npz public/labs/data/b09/
          cp labs/figures/b09/* public/labs/figures/b09/
          cp labs/solutions/b09-cost-frontier.ipynb public/labs/solutions/

'''
 s=s.replace('      - name: Setup Pages',block+'      - name: Setup Pages');p.write_text(s)
