"""Integrate B13 through existing manifest-driven navigation and evidence ledgers."""
import json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];S='b13-synthetic-relational-data'
p=R/'lessons/manifest.json';j=json.loads(p.read_text());j['lessons']=[x for x in j['lessons'] if x['id']!='B13'];j['lessons'].append(dict(id='B13',slug=S,year=5,quarter=5,sortOrder=200.13,checkpoint=False,labPath=f'labs/{S}.ipynb',title='RDB-PFN and PluRel: two roles for relational synthetic data',published=True));j['version']=max(j.get('version',1),12);p.write_text(json.dumps(j,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
    p=R/name;p.write_text(re.sub(r'(name="rdl-manifest-version" content=")[0-9]+',lambda m:m[1]+str(j['version']),p.read_text()))
p=R/'CURRICULUM.md';s=p.read_text().replace('B01–B12, B04a and B07a are prepared','B01–B13, B04a and B07a are prepared').replace('The other 15 units remain planned.','B13 adds a complete 12-fit held-out-schema course experiment, full 30-evaluation RDB-PFN replay and a source-audited PluRel Table 1 contract; paper pretraining remains NOT_RUN. The other 14 units remain planned.').replace('[RDB-PFN and PluRel: two roles for relational synthetic data](./plan/year-5-6-bridge.md#b13) | Generator-to-prediction pipelines and held-out-schema test.','[RDB-PFN and PluRel: two roles for relational synthetic data](lessons/'+S+'.html) | [Lab](labs/'+S+'.ipynb): 12-fit schema holdout, complete saved RDB-PFN replay and PluRel full-target audit.');p.write_text(s)
p=R/'reference/curriculum.html';s=p.read_text().replace('../plan/year-5-6-bridge.md#b13','../lessons/'+S+'.html').replace('B01–B12, B04a and B07a prepared; 15 planned','B01–B13, B04a and B07a prepared; 14 planned');p.write_text(s)
p=R/'plan/year-5-6-bridge.md';s=p.read_text();header='### B13 ★ · RDB-PFN and PluRel: two roles for relational synthetic data';addition='\n\n**Prepared 2026-10-03:** [lesson](../lessons/'+S+'.html) · [lab](../labs/'+S+'.ipynb) · [contract](../labs/b13-reproduction.md). Complete 12-fit course schema holdout and 30-evaluation/21,060-prediction saved RDB-PFN replay. PluRel paper code and checkpoints located; complete three-seed Table 1 training remains source/protocol/budget gated. USD0 paid execution; learner PENDING_WRITTEN_DEFENSE.\n'
if addition.strip() not in s:s=s.replace(header,header+addition)
p.write_text(s)
adds={'labs/README.md':'\n\n### B13 · Synthetic relational data\n\n[Student]('+S+'.ipynb) · [Executed solution](html/'+S+'.html) · [Contract](b13-reproduction.md). Three live mechanisms; 12 course fits, independent complete L200 replay, two paper architecture paths and pinned model/trainer source.\n',
'.gitignore':'\n# B13 portable solution\n!labs/solutions/'+S+'.ipynb\n',
'RESOURCES.md':'\n\n## B13 · Synthetic relational data\n\n[RDB-PFN v1](https://arxiv.org/html/2603.03805v1) for curriculum lineage; [v5 Table 9](https://arxiv.org/html/2603.03805v5) for the inherited selected numerical target. [PluRel v1 §§2–3/Table 1](https://arxiv.org/html/2602.04029v1), [official generator](https://github.com/stanford-star/plurel), [paper tag](https://github.com/stanford-star/plurel/tree/2a273cfd21933ee4893dfcb862a3edaed45ac665), [Hub paper card](https://huggingface.co/stanford-star/rt-plurel/blob/0cac262c0fc95353372b2cf1d5c1b0a1e649a449/paper/README.md). Later leaderboard checkpoints use regression NMAE selection and do not supply the original three-seed Table 1 evidence. Sources/identity inventory: `labs/sources/b13/`; full protocol: `labs/b13-reproduction.md`.\n',
'NOTES.md':'\n\n## B13 · Approved synthetic relational data (2026-10-03)\n\nUSD0 paid scope; 3600 aggregate local execution seconds. Complete 12-fit equal-feature-cell course schema holdout, 3072 intact + 3072 shuffled-FK predictions; a sufficient-statistic readout transfers, and schema diversity does not consistently improve the relational arm. Independent full L200 replay: 30 evaluations/21060 predictions, no fresh B13 checkpoint inference. PluRel source and paper tag are available; later leaderboard checkpoints differ from original three-seed R²-selected Table1. Full Table1 training/source identity remains gated; no claim of missing all code, paper pretraining, broad transfer or learner mastery.\n',
'thesis-dossier.md':'\n\n- **B13 · BAR:** An engineered parent-mean feature solves a held-out synthetic topology without learned graph reasoning. Equal generated-feature-cell budget, 12 fits, mixed tiny diversity changes; shuffled FKs worsen relational errors. Complete RDB-PFN replay adds audit confidence, not independent training evidence. PluRel Table1 full target remains source/protocol/budget gated despite available paper code and synthetic weights.\n'}
for name,addition in adds.items():
    p=R/name
    if addition.strip() not in p.read_text():p.write_text(p.read_text()+addition)
p=R/'assets/retrieval-pool.js';s=p.read_text();item=dict(id='b13-generator-representation',lesson=200.13,quarter='Q5',concept='synthetic-relational-priors',question='Does a relational synthetic generator require graph-native prediction?',options=[dict(label='No, representations can linearize',value='no'),dict(label='Yes, generators determine inference',value='yes')],correct='no',explain='Generation, representation and learner are separate. RDB-PFN uses relationally derived feature vectors; PluRel can supply cell contexts to RT.')
if item['id'] not in s:s=s.replace('global.RETRIEVAL_POOL = [','global.RETRIEVAL_POOL = [\n'+json.dumps(item)+',')
p.write_text(s)
p=R/'assets/paper-deck.js';s=p.read_text()
cards=[dict(id='plurel-b13-diversity',paper='Kothapalli et al. — PluRel',year=2026,lesson=200.13,front='Why separate database diversity from pretraining-token count?',back='More generated databases can broaden schema and parameter draws; more tokens can supply more observations from existing databases. A controlled comparison must identify which axis changed. PluRel generates data; RT is the learner used in the paper.',source='https://arxiv.org/html/2602.04029v1'),dict(id='rdbpfn-b13-linearization',paper='Wang et al. — RDB-PFN',year=2026,lesson=200.13,front='Where does relational structure enter RDB-PFN prediction?',back='The synthetic prior creates relational tasks. Linearization constructs fixed feature vectors from neighborhoods; the tabular in-context transformer reads those vectors and support labels. A relational generator does not make inference graph-native.',source='https://arxiv.org/html/2603.03805v5')]
for item in cards:
    if item['id'] not in s:s=s.replace('];\n})(window);',', '+json.dumps(item)+'\n];\n})(window);')
p.write_text(s)
p=R/'reference/glossary.html';s=p.read_text();entry='<section id="b13-schema-holdout"><h2>Schema identity and synthetic relational prior</h2><p>A schema holdout excludes the declared structural family from training, including graph-isomorphic renamings when structural identity defines the split. A synthetic relational prior is a chosen distribution of practice databases and tasks, not proof about arbitrary real schemas. Generation, representation and learner are distinct. <a href="../lessons/'+S+'.html">B13 worked pipelines</a>.</p></section>'
if 'id="b13-schema-holdout"' not in s:s=s.replace('</body>',entry+'\n</body>')
p.write_text(s)
p=R/'.github/workflows/pages.yml';s=p.read_text()
if '# B13:' not in s:
    block='''          # B13: complete course results, replay packet and paper-source audit.
          mkdir -p public/labs/evidence/b13 public/labs/sources/b13 public/labs/figures/b13
          cp labs/b13-reproduction.md labs/_*b13*.py labs/_*b13*.json public/labs/
          cp labs/relkit/synthetic_b13.py public/labs/relkit/
          cp -r labs/evidence/b13/. public/labs/evidence/b13/
          cp -r labs/sources/b13/. public/labs/sources/b13/
          cp labs/figures/b13/* public/labs/figures/b13/
          cp labs/solutions/b13-synthetic-relational-data.ipynb public/labs/solutions/

'''
    s=s.replace('      - name: Setup Pages',block+'      - name: Setup Pages')
p.write_text(s)
(R/'reviews/lesson-b13').mkdir(parents=True,exist_ok=True)
(R/'learning-records/0169-synthetic-relational-data-prepared.md').write_text('''# B13 prepared; learner defense pending

Author package: two model-specific generation-to-prediction paths, schema-isomorphism and FK-read exercises, 12 course ridge fits, and complete saved L200 replay. The new course experiment controls numeric-feature cells and exposes a sufficient-statistic shortcut; it does not establish learned schema reasoning or reproduce PluRel scaling laws.

Learner must implement the three live functions, explain source/version boundaries and defend the held-out-schema interpretation. PluRel Table1 needs 36 real-data training runs and 108 task evaluations plus synthetic-base reconstruction if required; current paper code is available but complete historical seed mapping and paid execution remain gated.

Revisit after 1/7/30 days. Status: PENDING_WRITTEN_DEFENSE. No learner mastery or Year5 exit status change.
''')
print('Integrated B13 manifest, curriculum, reference and evidence ledgers')
