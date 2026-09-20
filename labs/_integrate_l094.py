"""Register L094 without claiming publication or learner mastery."""
import json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];S='0094-hin-survey';T='HIN survey: map the mechanism, audit the evidence'
p=R/'lessons/manifest.json';m=json.loads(p.read_text())
if not any(x['id']==94 for x in m['lessons']):
 m['lessons'].append({'id':94,'slug':S,'year':3,'quarter':2,'checkpoint':False,'labPath':f'labs/{S}.ipynb','title':T,'published':True});m['version']+=1;p.write_text(json.dumps(m,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
 p=R/name;s=p.read_text();s=re.sub(r'(name="rdl-manifest-version" content=")\d+',r'\g<1>'+str(m['version']),s)
 if name=='notebooks.html' and 'id="lab-94"' not in s:
  s=s.replace('<ul class="nb-gallery">',f'<ul class="nb-gallery"><li id="lab-94"><div class="nb-head"><span class="num">Lesson 0094</span><span class="title">{T}</span></div><div class="nb-links"><a href="labs/html/{S}.html">Read lab</a><a class="lab-access-primary" href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{S}.ipynb">Run in Colab</a><a download="" href="labs/{S}.ipynb">Download notebook</a><a href="lessons/{S}.html">Lesson</a></div></li>',1)
 p.write_text(s)
p=R/'CURRICULUM.md';s=p.read_text().replace('Sun 2020 (HIN survey)','Dong et al. 2020 (HIN survey)').replace('| 094 | HIN survey | Sun & Han 2020 | Taxonomy of heterogeneity |',f'| 094 | [HIN survey](lessons/{S}.html) | Dong et al. 2020 | [Taxonomy and complete NN statistics audit](labs/{S}.ipynb) |');p.write_text(s)
p=R/'plan/year-3.md';s=p.read_text().replace('### 094 · HIN survey — *Sun & Han 2020*',f'### 094 · [HIN survey](../lessons/{S}.html) — *Dong et al. 2020*\n- **Prepared package** — [lab](../labs/{S}.ipynb), [reproduction contract](../labs/l094-reproduction.md). Full released NN statistics audit and all-row printed arithmetic; edge mismatches preserved. CS/OAG graph audit NOT_RUN; full-paper parity NOT_ESTABLISHED. Learner defense pending.');p.write_text(s)
entries={
'NOTES.md':'\n## Lesson094 created · HIN survey · 2026-09-20\n\nFull reproduction requested. Corrected the ambiguous Sun & Han 2020 placeholder to Dong et al. 2020. Two-axis taxonomy, route/meta-graph trace, three live tasks and full released NN statistics audit. All node counts match; edge discrepancies remain. All three printed rows receive arithmetic checks, not all three graph reconstructions. CS/OAG graph audits NOT_RUN; historical snapshot identity NOT_ESTABLISHED. Prior L091–L093 evidence is inspected without relabeling it as fresh training. WSL interrupted initial notebook execution; bounded rerun coverage is recorded in _execution_l094_results.json. No learner mastery inferred.\n',
'RESOURCES.md':'\n## L094 · HIN taxonomy and evidence accounting\n\n- [Dong, Hu, Wang, Sun and Tang (2020), Heterogeneous Network Representation Learning](https://web.cs.ucla.edu/~yzsun/papers/2020_IJCAI_HIN_Survey.pdf): §1 schema, §3 representations, §4 and Table1. Corrects the roadmap citation placeholder.\n- [metapath2vec author release](https://ericdongyx.github.io/metapath2vec/m2v.html): typed walks and lookup embeddings as a contrast to GNN encoders.\n- [Pinned OAG source](https://github.com/acbull/pyHGT/tree/85eaccd482bc1d1af56c2de297b6e3a88b96d5cd/OAG): relation storage and released graph provenance; historical snapshot equality remains unestablished.\n',
'labs/README.md':'\n- **L094 HIN survey:** `0094-hin-survey.ipynb`; taxonomy, route composition, protocol eligibility and complete NN statistics audit. No new model training. See `l094-reproduction.md`.\n',
'labs/data/README.md':'\n## L094 · Reuse complete NN, synthetic routes separately\n\nTier B: the entire hash-pinned L093 NN archive for Table1 statistics. Tier C: four-author matrices for route interventions only. No sampled graph is labeled as the complete NN target.\n',
 'thesis-dossier.md':'\n- **L094 · BAR · 2026-09-20:** Taxonomy does not rank model quality. The complete NN release matches all five node counts yet differs from printed edge statistics. Data identity and matched task protocols must precede claims that relational encoders outperform alternatives. No database-superiority or full-paper reproduction claim follows.\n'}
for name,entry in entries.items():
 p=R/name;s=p.read_text()
 if entry.strip().splitlines()[0] not in s:p.write_text(s+entry)
p=R/'reference/glossary.html';s=p.read_text()
if 'id="hin-l094"' not in s:s=s.replace('</body>','<section id="hin-l094"><h2>HIN taxonomy · L094</h2><dl><dt>Meta-graph</dt><dd>A typed structural pattern that may branch and combine relation constraints.</dd><dt>Lookup embedding</dt><dd>A learned vector indexed by a known entity ID; unseen IDs require an additional policy.</dd><dt>Protocol eligibility</dt><dd>Recorded agreement on task, information access and evaluation needed before comparing outcomes; not proof of reproduction.</dd></dl></section></body>')
p.write_text(s)
p=R/'assets/retrieval-pool.js';s=p.read_text()
if 'l094-taxonomy-axes' not in s:s=s.replace('];\n})(window);',', '+json.dumps({'id':'l094-taxonomy-axes','lesson':94,'quarter':'Q2','concept':'hin-taxonomy','question':'HAN uses explicit meta-paths and learns attention updates. Which taxonomy statement is valid?','options':[{'label':'Both axes describe HAN','value':'both'},{'label':'Only routes describe HAN','value':'route'},{'label':'Only encoders describe HAN','value':'encoder'}],'correct':'both','explain':'Route construction and encoder family are independent axes; HAN belongs to both categories.'})+'\n];\n})(window);')
p.write_text(s)
p=R/'.gitignore';s=p.read_text()
if '!labs/solutions/0094-hin-survey.ipynb' not in s:p.write_text(s+'\n!labs/solutions/0094-hin-survey.ipynb\n')
p=R/'.github/workflows/pages.yml';s=p.read_text()
if '# L094 HIN survey' not in s:s=s.replace('          # L093 HGT:', '          # L094 HIN survey: portable audit and explicit evidence boundaries.\n          cp labs/l094-reproduction.md labs/requirements-l094-*.txt labs/_*l094*.json labs/_*l094.py public/labs/\n          mkdir -p public/labs/solutions\n          cp labs/solutions/0094-hin-survey.ipynb public/labs/solutions/\n          # L093 HGT:',1)
p.write_text(s)
print('Registered lesson 94')
