"""Register B14 and its evidence boundaries, preserving unrelated lesson work."""
import json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];S='b14-flattening-challenge';E=R/'labs/evidence/b14';status=json.loads((E/'paper-status.json').read_text())
p=R/'lessons/manifest.json';j=json.loads(p.read_text());j['lessons']=[x for x in j['lessons'] if x['id']!='B14'];j['lessons'].append(dict(id='B14',slug=S,year=5,quarter=5,sortOrder=200.14,checkpoint=False,labPath=f'labs/{S}.ipynb',title='RDBLearn and TabPFN-Rel: the flattening challenge',published=True));j['version']=max(j.get('version',1),13);p.write_text(json.dumps(j,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
 p=R/name;p.write_text(re.sub(r'(name="rdl-manifest-version" content=")[0-9]+',lambda m:m[1]+str(j['version']),p.read_text()))
p=R/'CURRICULUM.md';s=p.read_text().replace('B01–B13, B04a and B07a are prepared','B01–B14, B04a and B07a are prepared').replace('The other 14 units remain planned.','B14 adds a complete 12-fit controlled flattening diagnostic, original preprocessing check and selected TabPFN-Rel F1 reproduction evidence; see its exact status ledger. The other 13 units remain planned.').replace('[RDBLearn and TabPFN-Rel: the flattening challenge](./plan/year-5-6-bridge.md#b14) | Time-safe flattening pipeline and backbone-swap ablation.','[RDBLearn and TabPFN-Rel: the flattening challenge](lessons/'+S+'.html) | [Lab](labs/'+S+'.ipynb): time-safe features, paired predictor swaps and selected TabPFN-Rel reproduction.');p.write_text(s)
p=R/'reference/curriculum.html';s=p.read_text().replace('../plan/year-5-6-bridge.md#b14','../lessons/'+S+'.html').replace('B01–B13, B04a and B07a prepared; 14 planned','B01–B14, B04a and B07a prepared; 13 planned');p.write_text(s)
p=R/'plan/year-5-6-bridge.md';s=p.read_text();header='### B14 ★ · RDBLearn and TabPFN-Rel: the flattening challenge';addition='\n\n**Prepared2026-10-03:** [lesson](../lessons/'+S+'.html) · [lab](../labs/'+S+'.ipynb) · [reproduction contract](../labs/b14-reproduction.md). Complete12-fit course ablation; selected paper status: '+status['status']+'. Learner PENDING_WRITTEN_DEFENSE.\n'
if addition.strip() not in s:s=s.replace(header,header+addition)
p.write_text(s)
adds={'labs/README.md':'\n\n### B14 · Flattening challenge\n\n[Student]('+S+'.ipynb) · [Executed solution](html/'+S+'.html) · [Contract](b14-reproduction.md). Three live mechanisms; two architecture paths; complete12-fit course ablation; exact source-backed selected TabPFN-Rel paper lane.\n',
'.gitignore':'\n# B14 portable solution\n!labs/solutions/'+S+'.ipynb\n',
'RESOURCES.md':'\n\n## B14 · The flattening challenge\n\n[RDBLearn toolkitv1](https://arxiv.org/html/2602.18495v1), [distinct encoder-analysisv2](https://arxiv.org/html/2602.13697v2), [RelArena/TabPFN-Relv2](https://arxiv.org/html/2608.16319v2), [releasee890022](https://github.com/PriorLabs/relarena/tree/e89002200e18be6d8d7a55f8a5ab50c993ce4d5d). Primary source bytes, original source and checkpoint metadata are under`labs/sources/b14/`. Model/system comparisons, API/OSS variants, single-seed evidence and omitted LimiX remain distinct.\n',
'NOTES.md':'\n\n## B14 · Approved flattening challenge (2026-10-03)\n\nComplete12-fit course ablation;3072fixed+3072rolling predictions; time/arrival and hidden-label interventions. Relational features help both course predictors; nonlinear predictor improves on the same table. Full original RDBLearn preprocessing diagnostic repeats category-code inconsistency; not automatically inherited by RelArena. Selected TabPFN-Rel target0.7145: '+status['status']+'. '+status['summary']+' Learner defense pending; no deployment.\n',
'thesis-dossier.md':'\n\n- **B14 · BAR:** Strong constructed relational features and tabular predictors are a necessary comparison for a learned relational encoder. Controlled synthetic experiment separates representation and predictor effects; it cannot establish real-database superiority. Selected TabPFN-Rel F1 evidence: '+status['status']+'. '+status['summary']+'\n'}
for name,addition in adds.items():
 p=R/name
 if addition.strip() not in p.read_text():p.write_text(p.read_text()+addition)
p=R/'assets/retrieval-pool.js';s=p.read_text();item=dict(id='b14-fixed-table-swap',lesson=200.14,quarter='Q5',concept='controlled-backbone-swap',question='Which comparison isolates the predictive backbone?',options=[dict(label='Same table, changed predictor',value='same'),dict(label='Changed table, changed predictor',value='both')],correct='same',explain='Freeze feature bytes, query/support keys, preprocessing, snapshots and label visibility. A complete pipeline comparison may change several factors.')
if item['id'] not in s:s=s.replace('global.RETRIEVAL_POOL = [','global.RETRIEVAL_POOL = [\n'+json.dumps(item)+',')
p.write_text(s)
p=R/'assets/paper-deck.js';s=p.read_text();cards=[dict(id='rdblearn-b14-recipe',paper='Zhang et al. — RDBLearn',year=2026,lesson=200.14,front='Where does relational information enter RDBLearn?',back='Deterministic key-path aggregations construct feature columns. A pretrained tabular ICL backbone predicts from the constructed table. Toolkit v1 selects depth and backend; it is distinct from the companion encoder analysis.',source='https://arxiv.org/html/2602.18495v1'),dict(id='tabpfnrel-b14-confounds',paper='Hayler et al. — RelArena and TabPFN-Rel',year=2026,lesson=200.14,front='Why is a TabPFN-Rel pipeline gain not necessarily a backbone gain?',back='The harness also changes temporal tuning, context selection/budget and refitting; the API adds text. Isolate the backbone with identical features and information access. The release uses one seed and distinguishes models from systems.',source='https://arxiv.org/html/2608.16319v2')]
for item in cards:
 if item['id'] not in s:s=s.replace('];\n})(window);',', '+json.dumps(item)+'\n];\n})(window);')
p.write_text(s)
p=R/'reference/glossary.html';s=p.read_text();entry='<section id="b14-backbone-ablation"><h2>Backbone ablation and phase snapshot</h2><p>A backbone ablation changes the predictor while preserving the feature table, support/query identities, preprocessing and information access. A phase snapshot fixes the latest database state for inner validation or outer evaluation; it does not establish when each historical field became available. <a href="../lessons/'+S+'.html">B14 worked trace</a>.</p></section>'
if 'id="b14-backbone-ablation"' not in s:s=s.replace('</body>',entry+'\n</body>')
p.write_text(s)
p=R/'.github/workflows/pages.yml';s=p.read_text()
if '# B14:' not in s:
 block='''          # B14: controlled flattening and selected released-model evidence.
          mkdir -p public/labs/evidence/b14 public/labs/sources/b14 public/labs/figures/b14
          cp labs/b14-reproduction.md labs/_*b14*.py labs/_*b14*.json public/labs/
          cp labs/relkit/flatten_b14.py public/labs/relkit/
          cp -r labs/evidence/b14/. public/labs/evidence/b14/
          cp -r labs/sources/b14/. public/labs/sources/b14/
          cp labs/figures/b14/* public/labs/figures/b14/
          cp labs/solutions/b14-flattening-challenge.ipynb public/labs/solutions/
          mkdir -p public/modal
          cp modal/b14_tabpfn_rel.py public/modal/

'''
 s=s.replace('      - name: Setup Pages',block+'      - name: Setup Pages')
p.write_text(s)
print('B14 integrated')
