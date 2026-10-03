"""Narrow B11 additions; leave other lessons and user staging untouched."""
import json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];S='b11-supervised-relational-baselines'
p=R/'lessons/manifest.json';j=json.loads(p.read_text());j['lessons']=[x for x in j['lessons'] if x['id']!='B11'];j['lessons'].append(dict(id='B11',slug=S,year=5,quarter=5,sortOrder=200.11,checkpoint=False,labPath=f'labs/{S}.ipynb',title='RelGNN versus RelGT: supervised relational baselines',published=True));j['version']=max(j.get('version',1),10);p.write_text(json.dumps(j,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
    p=R/name;p.write_text(re.sub(r'(name="rdl-manifest-version" content=")[0-9]+',lambda m:m[1]+str(j['version']),p.read_text()))
p=R/'CURRICULUM.md';s=p.read_text().replace('B01–B10, B04a and B07a are prepared','B01–B11, B04a and B07a are prepared').replace('The other 17 units remain planned.','B11 adds three-seed RelGNN/RelGT block checks and complete eleven-run saved-evidence replay; a fresh matched benchmark remains NOT_RUN. The other 16 units remain planned.').replace('[RelGNN versus RelGT: supervised relational baselines](./plan/year-5-6-bridge.md#b11) | Matched RelGNN/RelGT computation and training comparison.','[RelGNN versus RelGT: supervised relational baselines](lessons/'+S+'.html) | [Lab](labs/'+S+'.ipynb): source block checks, saved-evidence audit and matched-study specification.');p.write_text(s)
p=R/'reference/curriculum.html';s=p.read_text().replace('../plan/year-5-6-bridge.md#b11','../lessons/'+S+'.html').replace('B01–B10, B04a and B07a prepared; 17 planned','B01–B11, B04a and B07a prepared; 16 planned');p.write_text(s)
p=R/'plan/year-5-6-bridge.md';s=p.read_text();header='### B11 ★ · RelGNN versus RelGT: supervised relational baselines';addition='\n\n**Prepared 2026-10-03:** [lesson](../lessons/'+S+'.html) · [lab](../labs/'+S+'.ipynb) · [protocol](../labs/b11-reproduction.md). Three-seed reduced source-block checks and complete11-run/13,849-prediction replay. Fresh matched benchmark NOT_RUN; full RelGT search INCOMPLETE_TEMPORAL_AND_BUDGET_GATE. Learner PENDING_WRITTEN_DEFENSE.\n'
if addition.strip() not in s:s=s.replace(header,header+addition)
p.write_text(s)
adds={'labs/README.md':'\n\n### B11 · Supervised relational baselines\n\n[Student]('+S+'.ipynb) · [Executed solution](html/'+S+'.html) · [Protocol](b11-reproduction.md). Visible source blocks, three live functions and complete11-run saved-prediction audit. Fresh matched RelGNN/RelGT benchmark NOT_RUN.\n','.gitignore':'\n# B11 portable solution\n!labs/solutions/'+S+'.ipynb\n','RESOURCES.md':'\n\n## B11 · Supervised relational baseline comparison\n\n[RelGNN v2 §3 and Table2](https://arxiv.org/html/2502.06784v2), [RelGT v1 §3 and Tables1/6](https://arxiv.org/html/2505.10960v1). Sources pinned to cffdb8b54627e92c7dd112c1243dde739c90d35b and19e423ca3e7cac761130aba790857f2dc3a46ef7 respectively. Full source, licenses, frozen protocols and saved evidence authenticated by `labs/sources/b11/source-ledger.json`; no current-SOTA claim.\n','NOTES.md':'\n\n## B11 · approved supervised baseline comparison (2026-10-03)\n\nApproved USD0 scope: three-seed source block/gradient checks, complete saved11-run/13,849-prediction audit, visible model code and student/solution notebooks. L146 typed-mean GNN is not RelGNN; no cross-protocol ranking. RelGT full nine-config search remains INCOMPLETE_TEMPORAL_AND_BUDGET_GATE; inherited projection aboutUSD80.41 is not new pricing. Fresh B11 benchmark fits NOT_RUN, whole-paper NOT_RUN, learner PENDING_WRITTEN_DEFENSE. See reviews/lesson-b11/review.md.\n','thesis-dossier.md':'\n\n- **B11 · BAR:** Matching local context alone does not align global training state. Reduced RelGNN/RelGT source blocks agree numerically; the11-run replay does not establish a matched architecture advantage. Full RelGT temporal/budget blockers persist. `labs/b11-reproduction.md` separates executable evidence from the unrun matched study.\n'}
for name,addition in adds.items():
    p=R/name
    if addition.strip() not in p.read_text():p.write_text(p.read_text()+addition)
p=R/'.github/workflows/pages.yml';s=p.read_text()
if '# B11:' not in s:
    block='''          # B11: supervised baseline mechanisms and complete saved-evidence packet.
          mkdir -p public/labs/evidence/b11 public/labs/sources/b11 public/labs/figures/b11
          cp labs/b11-reproduction.md labs/_*b11*.py labs/_*b11*.json public/labs/
          cp labs/relkit/baselines_b11.py public/labs/relkit/
          cp -r labs/evidence/b11/. public/labs/evidence/b11/
          cp -r labs/sources/b11/. public/labs/sources/b11/
          cp labs/figures/b11/* public/labs/figures/b11/
          cp labs/solutions/b11-supervised-relational-baselines.ipynb public/labs/solutions/

'''
    s=s.replace('      - name: Setup Pages',block+'      - name: Setup Pages')
p.write_text(s)
print('Integrated B11 navigation and protocol links')
