"""Add B12 to the established manifest and course ledgers without changing staging."""
import json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];S='b12-adaptation-mechanisms'
p=R/'lessons/manifest.json';j=json.loads(p.read_text());j['lessons']=[x for x in j['lessons'] if x['id']!='B12'];j['lessons'].append(dict(id='B12',slug=S,year=5,quarter=5,sortOrder=200.12,checkpoint=False,labPath=f'labs/{S}.ipynb',title='Griffin, OpenRFM and KumoRFM: adaptation mechanisms',published=True));j['version']=max(j.get('version',1),11);p.write_text(json.dumps(j,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
 p=R/name;p.write_text(re.sub(r'(name="rdl-manifest-version" content=")[0-9]+',lambda m:m[1]+str(j['version']),p.read_text()))
p=R/'CURRICULUM.md';s=p.read_text().replace('B01–B11, B04a and B07a are prepared','B01–B12, B04a and B07a are prepared').replace('The other 16 units remain planned.','B12 adds a complete36-condition support-access diagnostic and source-gated adaptation comparison; full paper runs remain NOT_RUN. The other 15 units remain planned.').replace('[Griffin, OpenRFM and KumoRFM: adaptation mechanisms](./plan/year-5-6-bridge.md#b12) | Pretraining/adaptation matrix and support reachability test.','[Griffin, OpenRFM and KumoRFM: adaptation mechanisms](lessons/'+S+'.html) | [Lab](labs/'+S+'.ipynb):36-condition support reachability diagnostic and full-target source audit.');p.write_text(s)
p=R/'reference/curriculum.html';s=p.read_text().replace('../plan/year-5-6-bridge.md#b12','../lessons/'+S+'.html').replace('B01–B11, B04a and B07a prepared; 16 planned','B01–B12, B04a and B07a prepared; 15 planned');p.write_text(s)
p=R/'plan/year-5-6-bridge.md';s=p.read_text();header='### B12 ★ · Griffin, OpenRFM and KumoRFM: adaptation mechanisms';addition='\n\n**Prepared 2026-10-03:** [lesson](../lessons/'+S+'.html) · [lab](../labs/'+S+'.ipynb) · [contract](../labs/b12-reproduction.md). Complete36-condition fixed-kernel diagnostic,432keyed predictions, three architecture paths. Griffin20fit protocol preserved; full model runs NOT_RUN behind budget/source/identity gates. USD0 paid execution; learner PENDING_WRITTEN_DEFENSE.\n'
if addition.strip() not in s:s=s.replace(header,header+addition)
p.write_text(s)
adds={'labs/README.md':'\n\n### B12 · Adaptation mechanisms\n\n[Student]('+S+'.ipynb) · [Executed solution](html/'+S+'.html) · [Protocol](b12-reproduction.md). Three live operations, full36-condition/432-prediction course diagnostic, complete inherited Griffin source and explicit OpenRFM/Kumo gates.\n','.gitignore':'\n# B12 portable solution\n!labs/solutions/'+S+'.ipynb\n','RESOURCES.md':'\n\n## B12 · Adaptation mechanisms\n\n[Griffin ICML2025](https://proceedings.mlr.press/v267/wang25da.html), [OpenRFM v1 §§3–5](https://arxiv.org/html/2606.04320v1), [KumoRFM-2 v1 §3](https://arxiv.org/html/2604.12596v1). Frozen source ledger: `labs/sources/b12/source-ledger.json`. OpenRFM author release identity unresolved; T-Lab/OpenRFM is an independent Kumo reproduction. Kumo main ICL evaluation and optional fine-tuning are separate modes.\n','NOTES.md':'\n\n## B12 · Approved adaptation mechanisms (2026-10-03)\n\nUSD0 paid scope. Complete36-condition fixed-kernel diagnostic (seeds0/1/2, high/lowreachability, intact/shuffled/hiddenlabels, relational/dualchannels),432keyed predictions. Not an OpenRFM model reproduction. Full Griffin20fit source/contract retained with inherited budget STOP; OpenRFM author artifacts and Kumo historical model identity unresolved. Frozen weights do not mean fixed predictions; sensitivity does not establish accuracy or feature learning. Learner PENDING_WRITTEN_DEFENSE.\n','thesis-dossier.md':'\n\n- **B12 · BAR:** Controlled support access shows how fixed-weight predictions can depend on labels outside a relational walk. An untrained kernel can also be sensitive; this does not prove useful feature learning or an RFM advantage. Complete36-condition course diagnostic; Griffin/OpenRFM/Kumo full targets remain budget/source/identity gated.\n'}
for name,addition in adds.items():
 p=R/name
 if addition.strip() not in p.read_text():p.write_text(p.read_text()+addition)
p=R/'assets/retrieval-pool.js';s=p.read_text();item=dict(id='b12-adaptation-path',lesson=200.12,quarter='Q5',concept='adaptation-mechanisms',question='Frozen weights and changed support labels: what can change?',options=[dict(label='Activations and query predictions',value='predictions'),dict(label='Parameters and optimizer moments',value='parameters')],correct='predictions',explain='Context inputs change activations and predictions without updating stored parameters. Label sensitivity alone does not establish useful learned adaptation.')
if item['id'] not in s:s=s.replace('global.RETRIEVAL_POOL = [','global.RETRIEVAL_POOL = [\n'+json.dumps(item)+',')
p.write_text(s)
p=R/'assets/paper-deck.js';s=p.read_text();item=dict(id='openrfm-b12-support-path',paper='Chen et al. — OpenRFM',year=2026,lesson=200.12,front='Why distinguish relational support reachability from cross-example ICL?',back='A sampled relational walk may contain few labeled task rows. A cross-example stage can access eligible support outside that walk. Compare available and used support budgets; a path permits label influence but does not prove accuracy or learned feature adaptation.',source='https://arxiv.org/html/2606.04320v1')
if item['id'] not in s:s=s.replace('];\n})(window);',', '+json.dumps(item)+'\n];\n})(window);')
p.write_text(s)
p=R/'reference/glossary.html';s=p.read_text();entry='<section id="b12-support-reachability"><h2>Support reachability and contextual adaptation</h2><p>Support reachability describes whether eligible labeled examples enter a query’s sampled context. Contextual adaptation changes input-dependent activations and predictions; fine-tuning changes parameters. Available support budget and labels actually read are distinct. <a href="../lessons/'+S+'.html">B12 worked paths</a>.</p></section>'
if 'id="b12-support-reachability"' not in s:s=s.replace('</body>',entry+'\n</body>')
p.write_text(s)
p=R/'.github/workflows/pages.yml';s=p.read_text()
if '# B12:' not in s:
 block='''          # B12: adaptation mechanisms, portable source packet and all predictions.
          mkdir -p public/labs/evidence/b12 public/labs/sources/b12 public/labs/figures/b12
          cp labs/b12-reproduction.md labs/_*b12*.py labs/_*b12*.json public/labs/
          cp labs/relkit/adaptation_b12.py public/labs/relkit/
          cp -r labs/evidence/b12/. public/labs/evidence/b12/
          cp -r labs/sources/b12/. public/labs/sources/b12/
          cp labs/figures/b12/* public/labs/figures/b12/
          cp labs/solutions/b12-adaptation-mechanisms.ipynb public/labs/solutions/

'''
 s=s.replace('      - name: Setup Pages',block+'      - name: Setup Pages')
p.write_text(s)
(R/'learning-records/0168-adaptation-mechanisms-prepared.md').write_text('''# B12 prepared; learner defense pending

B12 separates fine-tuning from contextual adaptation with three architecture paths and a complete36-condition course experiment. Three live TODOs enforce temporal label eligibility, masked label attention and controlled label intervention. All432predictions are author evidence, not learner mastery or OpenRFM checkpoint results.

Full Griffin20fit source/contract remains budget-gated; OpenRFM author artifacts and Kumo historical identity are unresolved. The learner must implement the functions and defend the adaptation matrix, available-versus-used support budget, and sensitivity-versus-accuracy distinction. Revisit after1/7/30days; do not alter the personal mastery glossary or Year5 exit status.

Status: PENDING_WRITTEN_DEFENSE.
''')
print('Integrated B12 manifest, curriculum, references and source delivery')
