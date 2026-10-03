"""B07a-only course integration; preserve existing uncommitted bridge work."""
import json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];S='b07a-hypernetworks'
p=R/'lessons/manifest.json';m=json.loads(p.read_text())
if not any(x['id']=='B07a' for x in m['lessons']):
 m['lessons'].append(dict(id='B07a',slug=S,year=5,quarter=5,sortOrder=200.071,checkpoint=False,labPath='labs/'+S+'.ipynb',title='Hypernetworks: generate a predictor from a table',published=True));m['version']+=1
p.write_text(json.dumps(m,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
 p=R/name;text=p.read_text();text=re.sub(r'(name="rdl-manifest-version" content=")[0-9]+',lambda x:x[1]+str(m['version']),text);p.write_text(text)
p=R/'plan/year-5-6-bridge.md';text=p.read_text().replace('B01–B07 and B04a prepared; 21 planned','B01–B07, B04a and B07a prepared; 20 planned');marker='### B07a ◆ · Hypernetworks: generate a predictor from a table\n'
if '**Prepared 2026-10-03:** [lesson](../lessons/'+S not in text:text=text.replace(marker,marker+'\n**Prepared 2026-10-03:** [lesson](../lessons/'+S+'.html) · [lab](../labs/'+S+'.ipynb). Full released HyperFast course inference:9 predictors/18 paired retrieval arms plus one refresh. Original Table7 banknote INCOMPLETE_SOURCE_PROTOCOL. MotherNet/iLTM architecture comparisons only; learner defense pending.\n\n')
p.write_text(text)
p=R/'CURRICULUM.md';text=p.read_text().replace('B01–B07 and B04a are prepared','B01–B07, B04a and B07a are prepared').replace('The other 21 units remain planned.','B07a adds full-dimensional HyperFast course inference and a source-gated Table 7 target. The other 20 units remain planned.').replace('[Hypernetworks: generate a predictor from a table](./plan/year-5-6-bridge.md#b07a) | Support-to-weights trace and query-volume cost comparison.','[Hypernetworks: generate a predictor from a table](lessons/'+S+'.html) | [Lab](labs/'+S+'.ipynb): 18 paired arms; query costs; Table 7 source gate.');p.write_text(text)
p=R/'reference/curriculum.html';text=p.read_text().replace('B01–B07 and B04a prepared; 21 planned','B01–B07, B04a and B07a prepared; 20 planned').replace('../plan/year-5-6-bridge.md#b07a"','../lessons/'+S+'.html"').replace('Support-to-weights trace and query-volume cost comparison</td>','18 paired HyperFast arms; query cost and Table 7 source gate</td>');p.write_text(text)
sections={
'labs/README.md':'\n### B07a · Hypernetworks\n\n[Student]('+S+'.ipynb) · [Executed solution](html/'+S+'.html) · [Protocol](b07a-reproduction.md). Three live functions, full visible HyperFast generation/inference and optional downstream optimizer. Nine full-dimensional predictors,18paired arms and one refresh; original Table7 source-gated. Fresh lane downloads5.09GB checkpoint; offline evidence lane needs no checkpoint.\n',
'RESOURCES.md':'\n## B07a · Hypernetworks (2026-10-03)\n\n[MotherNet v2 §§3.1–3.2](https://arxiv.org/html/2312.08598v2#S3), [HyperFast v1 model/Appendix A/Table7](https://arxiv.org/html/2402.14335v1), [iLTM v1 §3](https://arxiv.org/html/2511.15941v1#S3). Full HyperFast source current d1f1c3b0c45572dc09173733b194c0e7384cd696 and publication-era9a25ed34edf1d9896efde32500707fb8689e1005 archived. Checkpoint43484094 authenticated by SHA256 in labs/sources/b07a/checkpoint.json. Model generation does not remove optional retrieval/context costs.\n',
'NOTES.md':'\n## B07a · approved HyperFast inference (2026-10-03)\n\nFull released dimensions/checkpoint retained. Three numeric tables×three fresh split/model seeds×retrieval off/on;18paired prediction arms share9generated predictors. One extra banknote support refresh is timing-only. Original Table7 banknote requires10mini-test repetitions/300s each; original split/subsample/search identities remain missing:INCOMPLETE_SOURCE_PROTOCOL. Code/current checkpoint parity does not establish historical paper reproduction. Whole-paper/meta-training/MotherNet/iLTM executions NOT_RUN. No learner mastery inferred.\n',
'thesis-dossier.md':'\n- **B07a · BAR (2026-10-03):** generated predictor weights do not alone specify serving cost; RF/PCA, retrieval and support refreshes matter. Paired HyperFast query paths on three numeric tables do not establish a relational advantage or a model-family ranking. Original Table7 target source-gated. [Audit](labs/evidence/b07a/course-audit.json) · [Protocol](labs/b07a-reproduction.md).\n'}
for name,section in sections.items():
 p=R/name;t=p.read_text()
 if section.strip().split('\n')[0] not in t:p.write_text(t+section)
p=R/'reference/glossary.html';t=p.read_text()
if 'id="hypernetwork-b07a"' not in t:t=t.replace('</body>','<section id="hypernetwork-b07a"><h2>Hypernetworks · B07a</h2><dl><dt>Hypernetwork</dt><dd>A network generating parameters for another network from task information.</dd><dt>Generated predictor</dt><dd>The task model produced by the hypernetwork; its complete serving state also includes preprocessing and any retrieval context.</dd><dt>Amortized construction</dt><dd>Spreading predictor construction cost across repeated predictions, accounting for later rebuilds.</dd><dt>Operating point</dt><dd>A concrete model/configuration including adaptation, preprocessing, retrieval, cache, hardware and batch policy.</dd></dl><a href="b07a-hypernetworks.html">Field guide</a></section></body>');p.write_text(t)
p=R/'assets/retrieval-pool.js';t=p.read_text()
if 'b07a-retained-support' not in t:t=t.replace('global.RETRIEVAL_POOL = [','global.RETRIEVAL_POOL = [\n'+json.dumps(dict(id='b07a-retained-support',lesson=200.071,quarter='Q5',concept='hypernetworks',question='B07a: what determines whether generated predictors can discard support?',options=[dict(label='The complete query path',value='path'),dict(label='The hypernetwork family name',value='name')],correct='path',explain='A pure child can discard support, but retrieval still consumes context. Include fitted preprocessing and update policy.'))+',');p.write_text(t)
p=R/'assets/paper-deck.js';t=p.read_text()
if 'hypernetwork-b07a-cost' not in t:t=t.replace('global.PAPER_DECK = [','global.PAPER_DECK = [\n'+json.dumps(dict(id='hypernetwork-b07a-cost',paper='MotherNet, HyperFast and iLTM',year=2025,lesson=200.071,front='Does generating predictor weights eliminate support dependence?',back='Only a query path that no longer consults support can discard it. HyperFast nearest-neighbor bias and iLTM retrieval retain context; count preprocessing, optional tuning and every rebuild.',source='https://arxiv.org/html/2402.14335v1'))+',');p.write_text(t)
p=R/'.github/workflows/pages.yml';t=p.read_text()
if '# B07a:' not in t:
 block='''          # B07a: portable hypernetwork course artifacts; exclude external5GB checkpoint.
          mkdir -p public/labs/evidence/b07a public/labs/sources/b07a public/labs/figures/b07a public/labs/data/b07a
          cp labs/b07a-reproduction.md labs/_*b07a*.py labs/_*b07a*.json public/labs/
          cp labs/relkit/*b07a.py public/labs/relkit/
          cp -r labs/evidence/b07a/. public/labs/evidence/b07a/
          cp -r labs/sources/b07a/. public/labs/sources/b07a/
          cp labs/data/b07a/*.npz public/labs/data/b07a/
          cp labs/figures/b07a/* public/labs/figures/b07a/
          cp labs/solutions/b07a-hypernetworks.ipynb public/labs/solutions/

'''
 t=t.replace('      - name: Setup Pages',block+'      - name: Setup Pages');p.write_text(t)
p=R/'.gitignore';t=p.read_text()
if '# B07a external checkpoint' not in t:p.write_text(t+'\n# B07a external checkpoint and portable executed solution.\nlabs/data/b07a/*.ckpt\nlabs/data/b07a/*.partial\n!labs/solutions/'+S+'.ipynb\n')
print('Integrated B07a manifest',m['version'])
