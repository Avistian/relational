"""Register the prepared lesson; never mark learner completion or deployment."""
import json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];S='0095-bipartite-graphs';T='Bipartite graphs: from interactions to recommendations'
p=R/'lessons/manifest.json';m=json.loads(p.read_text())
if not any(x['id']==95 for x in m['lessons']):
    m['lessons'].append({'id':95,'slug':S,'year':3,'quarter':2,'checkpoint':False,'labPath':f'labs/{S}.ipynb','title':T,'published':True});m['version']+=1;p.write_text(json.dumps(m,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
    p=R/name;s=p.read_text();s=re.sub(r'(name="rdl-manifest-version" content=")\d+',r'\g<1>'+str(m['version']),s)
    if name=='notebooks.html' and 'id="lab-95"' not in s:
        s=s.replace('<ul class="nb-gallery">',f'<ul class="nb-gallery"><li id="lab-95"><div class="nb-head"><span class="num">Lesson 0095</span><span class="title">{T}</span></div><div class="nb-links"><a href="labs/html/{S}.html">Read lab</a><a class="lab-access-primary" href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{S}.ipynb">Run in Colab</a><a download="" href="labs/{S}.ipynb">Download notebook</a><a href="lessons/{S}.html">Lesson</a></div></li>',1)
    p.write_text(s)
p=R/'CURRICULUM.md';s=p.read_text().replace('| 095 | Bipartite graphs | — | User-item as hetero graph |',f'| 095 | [Bipartite graphs](lessons/{S}.html) | GroupLens / PyG | [Typed graph, full five-fold recommendation](labs/{S}.ipynb) |');p.write_text(s)
p=R/'plan/year-3.md';s=p.read_text().replace('### 095 · Bipartite graphs — *— (user-item as hetero graph)*',f'### 095 · [Bipartite graphs](../lessons/{S}.html) — *GroupLens / PyG (user-item as hetero graph)*\n- **Prepared package** — [lab](../labs/{S}.ipynb), [reproduction contract](../labs/l095-reproduction.md). Full ML-100K release audit MATCH; all five course ranking folds MEASURED. No historical model-score reproduction claimed. Learner defense pending.');p.write_text(s)
entries={
'NOTES.md':'\n## Lesson095 created · Bipartite graphs · 2026-09-20\n\nFull reproduction requested. Full ML-100K release counts and five official partitions audited MATCH; complete five-fold course ranking experiment MEASURED, distinct from historical model-paper scores. Typed IDs, reverse-edge boundary, projections, three-hop walk and full-catalog evaluation; three live tasks and complete inline implementation. Mean NDCG@10 .327886 walk versus .204553 popularity. Fold SD is descriptive, not dataset-level uncertainty. No learner mastery or publication inferred.\n',
'RESOURCES.md':'\n## L095 · Bipartite recommendation contract\n\n- [GroupLens ML-100K release](https://grouplens.org/datasets/movielens/100k/) and [README](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt): complete counts and official five partitions. Data acknowledgment: Harper & Konstan (2015), DOI 10.1145/2827872.\n- [PyG heterogeneous tutorial](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html): typed stores and bipartite identity.\n- [PyG 2.8.0 RandomLinkSplit](https://pytorch-geometric.readthedocs.io/en/2.8.0/generated/torch_geometric.transforms.RandomLinkSplit.html): paired reverse relations; is_undirected alone does not cover bipartite types.\n- [Local full-run protocol](labs/l095-reproduction.md): complete release audit plus explicitly course-defined walk and ranking experiment.\n',
'labs/README.md':'\n- **L095 Bipartite graphs:** `0095-bipartite-graphs.ipynb`; full ML-100K release audit and five-fold graph-walk recommendation, typed IDs, reverse-edge leakage checks and full-catalog ranking. See `l095-reproduction.md`.\n',
'labs/data/README.md':'\n## L095 · Full MovieLens 100K\n\nTier B relational interaction data: 100,000 ratings, 943 users and 1,682 items. Downloader verifies the GroupLens archive and all official split members. Data are downloaded directly, not redistributed. Tier C two-user synthetic graph is used only for mechanism checks. Full five-fold ranking is a course protocol, not a claimed published-model result.\n',
'thesis-dossier.md':'\n- **L095 · BAR · 2026-09-20:** Complete ML-100K five-fold course experiment gives walk NDCG@10 .327886 versus popularity .204553. This shows useful relational structure under one declared offline protocol, not GNN superiority, temporal forecasting success or general database superiority. Published release counts and partitions match; historical model-score parity is not claimed.\n'}
for name,entry in entries.items():
    p=R/name;s=p.read_text()
    if entry.strip().splitlines()[0] not in s:p.write_text(s+entry)
p=R/'reference/glossary.html';s=p.read_text()
if 'id="bipartite-l095"' not in s:s=s.replace('</body>','<section id="bipartite-l095"><h2>Bipartite recommendation · L095</h2><dl><dt>Bipartite graph</dt><dd>Two disjoint node sets, with every edge joining one set to the other.</dd><dt>Projection</dt><dd>A derived same-type relation, such as shared-item counts BBᵀ, which loses the intermediate item identity.</dd><dt>Candidate mask</dt><dd>The explicit eligibility rule applied before ranking; L095 excludes every fitting-rated item.</dd></dl><p><a href="bipartite-contract.html">Reference contract</a></p></section></body>')
p.write_text(s)
p=R/'assets/retrieval-pool.js';s=p.read_text()
if 'l095-reverse-boundary' not in s:
    item={'id':'l095-reverse-boundary','lesson':95,'quarter':'Q2','concept':'bipartite-split','question':'A held-out user–item target is absent forward but present in reverse. Which claim holds?','options':[{'label':'The target remains visible','value':'leak'},{'label':'The target remains hidden','value':'hidden'},{'label':'The target becomes negative','value':'negative'}],'correct':'leak','explain':'Reverse connectivity identifies the same interaction. Both directions must derive from the fitting split.'}
    s=s.replace('];\n})(window);',', '+json.dumps(item)+'\n];\n})(window);');p.write_text(s)
p=R/'.gitignore';s=p.read_text()
for line in ['!labs/solutions/0095-bipartite-graphs.ipynb','labs/data/l095/','labs/results/l095/*-inner.npz']:
    if line not in s:s+='\n'+line+'\n'
p.write_text(s)
p=R/'.github/workflows/pages.yml';s=p.read_text()
if '# L095 bipartite' not in s:s=s.replace('          # L094 HIN survey:', '          # L095 bipartite: complete course run and explicit release target.\n          cp labs/l095-reproduction.md labs/requirements-l095-*.txt labs/_*l095*.json labs/_*l095.py public/labs/\n          mkdir -p public/labs/results/l095 public/labs/solutions\n          cp labs/results/l095/*.json public/labs/results/l095/\n          cp labs/solutions/0095-bipartite-graphs.ipynb public/labs/solutions/\n          # L094 HIN survey:',1)
p.write_text(s)
print('Registered lesson 95')
