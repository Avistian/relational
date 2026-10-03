"""Integrate only B07, preserving existing bridge lessons and shared metadata."""
import json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];S='b07-semantic-transfer'
p=R/'lessons/manifest.json';m=json.loads(p.read_text())
if not any(x['id']=='B07' for x in m['lessons']):
 m['lessons'].append(dict(id='B07',slug=S,year=5,quarter=5,sortOrder=200.07,checkpoint=False,labPath='labs/'+S+'.ipynb',title='Semantic transfer: CARTE, ConTextTab and TabSTAR',published=True));m['version']+=1
p.write_text(json.dumps(m,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
 p=R/name;text=p.read_text();text=re.sub(r'(name="rdl-manifest-version" content=")[0-9]+',lambda x:x[1]+str(m['version']),text);p.write_text(text)
p=R/'plan/year-5-6-bridge.md';text=p.read_text().replace('B01–B06 and B04a prepared; 22 planned','B01–B07 and B04a prepared; 21 planned');marker='### B07 ★ · Semantic transfer: CARTE, ConTextTab and TabSTAR\n'
if '**Prepared 2026-10-03:** [lesson](../lessons/'+S not in text:text=text.replace(marker,marker+'\n**Prepared 2026-10-03:** [lesson](../lessons/'+S+'.html) · [lab](../labs/'+S+'.ipynb). Complete27-fit CARTE semantic ablation/6912predictions; meaningful names help one of three table means. Original ConTextTab Table2 INCOMPLETE_SOURCE_PROTOCOL; no three-model ranking or new pretraining. Learner PENDING_WRITTEN_DEFENSE.\n\n')
p.write_text(text)
p=R/'CURRICULUM.md';text=p.read_text().replace('B01–B06 and B04a are prepared','B01–B07 and B04a are prepared').replace('The other 22 units remain planned.','B07 adds a complete27-fit semantic ablation with mixed effects and source-gated ConTextTab Table2. The other21units remain planned.').replace('[Semantic transfer: CARTE, ConTextTab and TabSTAR](./plan/year-5-6-bridge.md#b07) | Semantic-name ablation and adaptation comparison.','[Semantic transfer: CARTE, ConTextTab and TabSTAR](lessons/'+S+'.html) | [Lab](labs/'+S+'.ipynb):27fits; mixed semantic effects; Table2 source gate; defense pending.');p.write_text(text)
p=R/'reference/curriculum.html';text=p.read_text().replace('B01–B06 and B04a prepared; 22 planned','B01–B07 and B04a prepared; 21 planned').replace('../plan/year-5-6-bridge.md#b07"','../lessons/'+S+'.html"').replace('Semantic-name ablation and adaptation comparison</td>','27fit semantic ablation; mixed effects; original Table2 source gate</td>');p.write_text(text)
sections={
'labs/README.md':'\n### B07 · Semantic transfer\n\n[Student]('+S+'.ipynb) · [Executed solution](html/'+S+'.html) · [Protocol](b07-reproduction.md). Complete27-fit course CARTE ablation,6912predictions; meaningful headers help only one of three table means. ConTextTab Table2 source-gated; TabSTAR architecture explained, no fresh fit. Learner defense pending.\n',
'RESOURCES.md':'\n## B07 · Semantic transfer (2026-10-03)\n\n[CARTE v2 §§3.1–3.3](https://arxiv.org/html/2402.16785v2#S3), [ConTextTab v1 §§3–5/Table2](https://arxiv.org/html/2506.10707v1#S5.T2), [TabSTAR v2 §3/Appendices A.2,B.2](https://arxiv.org/html/2505.18125v2#S3), [SAP-RPT-1-OSS model card](https://huggingface.co/SAP/sap-rpt-1-oss). Original June ConTextTab source f1e4560e28c59dd632a41affb0a6ed4f30af9ebe archived alongside current source and TabSTAR files in labs/sources/b07. Current wrapper bagging differs from original; binning checkpoint and exact original evaluator/splits remain unauthenticated.\n',
'NOTES.md':'\n## B07 · approved semantic ablation (2026-10-03)\n\n27fresh frozen-encoder CARTE/ridge fits;6912test predictions;3related wine tables×3paired split seeds×3arms. Meaningful headers improve mean R² on wine_pl but hurt on wine.com/Vivino. Numeric-only also has mixed effects. Independent dual ridge reconstruction and complete key/label/selection audit PASS. Preserve negative results and split disagreement. Header-digit cache and all-missing numeric-row failures were corrected before complete reports/scoring; failed attempts retained, all27fits restarted. Original ConTextTab binning Table2 remains INCOMPLETE_SOURCE_PROTOCOL. Current/June wrapper replacement behavior differs. No fresh ConTextTab/TabSTAR inference or new pretraining; USD0cloud/API. Learner PENDING_WRITTEN_DEFENSE.\n',
 'thesis-dossier.md':'\n- **B07 · BAR (2026-10-03):** complete27-fit semantic-name ablation has mixed effects on three related wine tables. A matched information contract is required before assigning gains to relational structure. The frozen CARTE probe does not establish semantic or relational superiority. Original ConTextTab Table2 source-gated. [Evidence](labs/evidence/b07/course-audit.json) · [Protocol](labs/b07-reproduction.md).\n'}
for name,section in sections.items():
 p=R/name;text=p.read_text()
 if section.strip().split('\n')[0] not in text:p.write_text(text+section)
p=R/'reference/glossary.html';text=p.read_text()
if 'id="semantic-transfer-b07"' not in text:text=text.replace('</body>','<section id="semantic-transfer-b07"><h2>Semantic transfer · B07</h2><dl><dt>Semantic-name ablation</dt><dd>Replacing natural column names while preserving values and evaluation identities.</dd><dt>Text-removal intervention</dt><dd>Removing text-valued predictors; changes information availability as well as representation.</dd><dt>Target-candidate token</dt><dd>A representation of each allowed target class, supplied independently of a query\'s unknown answer.</dd><dt>Adaptation regime</dt><dd>How task data changes a predictor: supplied context, fitted heads or parameter updates.</dd></dl><p><a href="b07-semantic-transfer.html">Field guide</a></p></section></body>');p.write_text(text)
p=R/'assets/retrieval-pool.js';text=p.read_text()
if 'b07-target-candidates' not in text:text=text.replace('global.RETRIEVAL_POOL = [','global.RETRIEVAL_POOL = [\n'+json.dumps(dict(id='b07-target-candidates',lesson=200.07,quarter='Q5',concept='target-candidates',question='B07: which target-token input preserves the unknown query answer?',options=[dict(label='Include every class candidate',value='all'),dict(label='Include only correct class',value='answer')],correct='all',explain='Every candidate is present regardless of the true class. Selecting only the correct class reveals the query label.'))+',');p.write_text(text)
p=R/'assets/paper-deck.js';text=p.read_text()
if 'semantic-b07-adaptation' not in text:text=text.replace('global.PAPER_DECK = [','global.PAPER_DECK = [\n'+json.dumps(dict(id='semantic-b07-adaptation',paper='ConTextTab and TabSTAR',year=2025,lesson=200.07,front='How do ConTextTab and TabSTAR adapt to a new table?',back='ConTextTab ordinarily changes labeled context with frozen weights. TabSTAR learns downstream LoRA updates, including eligible text-encoder blocks. Both can use semantic inputs; they are different adaptation regimes.',source='https://arxiv.org/html/2505.18125v2#S3'))+',');p.write_text(text)
p=R/'.github/workflows/pages.yml';text=p.read_text()
if '# B07:' not in text:
 block='''          # B07: paired semantic ablation and original ConTextTab source gate.
          mkdir -p public/labs/evidence/b07 public/labs/sources/b07 public/labs/figures/b07 public/labs/data/b07
          cp labs/b07-reproduction.md labs/_*b07*.py labs/_*b07*.json public/labs/
          cp labs/relkit/*b07.py public/labs/relkit/
          cp -r labs/evidence/b07/. public/labs/evidence/b07/
          cp -r labs/sources/b07/. public/labs/sources/b07/
          cp -r labs/data/b07/. public/labs/data/b07/
          cp labs/figures/b07/* public/labs/figures/b07/
          cp labs/solutions/b07-semantic-transfer.ipynb public/labs/solutions/

'''
 text=text.replace('      - name: Setup Pages',block+'      - name: Setup Pages');p.write_text(text)
p=R/'.gitignore';text=p.read_text()
if '!labs/solutions/'+S+'.ipynb' not in text:p.write_text(text+'\n# B07 portable executed solution.\n!labs/solutions/'+S+'.ipynb\n')
print('Integrated B07, manifest',m['version'])
