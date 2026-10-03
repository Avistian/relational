"""Integrate B06 only; preserve prior prepared lessons and manifest metadata."""
import json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];S='b06-mitra-prior-mixtures'
p=R/'lessons/manifest.json';m=json.loads(p.read_text())
if not any(x['id']=='B06' for x in m['lessons']):
 m['lessons'].append(dict(id='B06',slug=S,year=5,quarter=5,sortOrder=200.06,checkpoint=False,labPath='labs/'+S+'.ipynb',title='Mitra: the prior is part of the model',published=True));m['version']+=1
p.write_text(json.dumps(m,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
 p=R/name;text=p.read_text();text=re.sub(r'(name="rdl-manifest-version" content=")[0-9]+',lambda x:x[1]+str(m['version']),text);p.write_text(text)
p=R/'plan/year-5-6-bridge.md';text=p.read_text().replace('B01–B05 and B04a prepared; 23 planned','B01–B06 and B04a prepared; 22 planned');marker='### B06 ★ · Mitra: the prior is part of the model\n'
if '**Prepared 2026-10-03:** [lesson](../lessons/'+S not in text:text=text.replace(marker,marker+'\n**Prepared 2026-10-03:** [lesson](../lessons/'+S+'.html) · [lab](../labs/'+S+'.ipynb). Nine fresh paired course fits / 4,320 predictions; near chance and worse mean cross-entropy than uniform. Original Table 12 INCOMPLETE_SOURCE_PROTOCOL; full pretraining NOT_RUN; learner PENDING_WRITTEN_DEFENSE.\n\n')
p.write_text(text)
p=R/'CURRICULUM.md';text=p.read_text().replace('B01–B05 and B04a are prepared','B01–B06 and B04a are prepared').replace('The other 23 units remain planned.','B06 adds nine controlled prior fits with a negative result and source-gated Mitra Table 12. The other 22 units remain planned.').replace('[Mitra: the prior is part of the model](./plan/year-5-6-bridge.md#b06) | Prior-mixture ablation with fixed learner and budget.','[Mitra: the prior is part of the model](lessons/'+S+'.html) | [Lab](labs/'+S+'.ipynb): nine fresh prior fits; near-chance result; Table 12 source gate; defense pending.');p.write_text(text)
p=R/'reference/curriculum.html';text=p.read_text().replace('B01–B05 and B04a prepared; 23 planned','B01–B06 and B04a prepared; 22 planned').replace('../plan/year-5-6-bridge.md#b06','../lessons/'+S+'.html').replace('Prior-mixture ablation with fixed learner and budget</td>','Nine matched prior fits; negative course result; Table 12 source gate</td>');p.write_text(text)
append={
'labs/README.md':'\n### B06 · Mitra: the prior is part of the model\n\n[Student notebook]('+S+'.ipynb) · [Executed solution](html/'+S+'.html) · [Protocol](b06-reproduction.md). Nine fresh course fits and 4,320 predictions; near chance, no prior benefit established. Original Table 12 INCOMPLETE_SOURCE_PROTOCOL. Learner defense pending.\n',
'RESOURCES.md':'\n## B06 · Mitra prior design (2026-10-03)\n\n[Mitra v1 §§3–4, Appendix B.3 and Table 12](https://arxiv.org/html/2510.21204v1), [Mitra-v2 §2](https://arxiv.org/html/2609.04540v1), [classifier release](https://huggingface.co/autogluon/mitra-classifier/tree/c425e9fa0910a6be1c494321792e7ba2a1367b1a), [v2 fine-tuning release](https://huggingface.co/autogluon/mitra-finetune/tree/b4701e8148dc33b00ed15d7086ff59816957cde4). Archived source files and release inventories under labs/sources/b06. Original six-arm Table 12 protocol not authenticated. Course proxy experiment and version boundaries are explicit.\n',
'NOTES.md':'\n## B06 · approved controlled prior experiment (2026-10-03)\n\nNine fresh paired course fits, 1,440 updates, 5,760 training tasks and 4,320 held-out predictions. Every family-mean cross-entropy exceeds uniform ln2; accuracy near chance. Preserve the negative result; no prior benefit or real-data generalization established. All nine checkpoint outputs independently replayed exactly; scalar scorer, complete identities, initial pairing and corruption checks pass. Original Table 12 source gate remains INCOMPLETE_SOURCE_PROTOCOL: original six checkpoints/pretraining histories, fold identities and evaluator unavailable in authenticated releases. Current fine-tuning release is v2. Full paper training/full benchmark NOT_RUN. USD0 paid spend; aggregate numerical ledger includes failures. See reviews/lesson-b06 for delivery checks. Learner PENDING_WRITTEN_DEFENSE; no learner completion inferred.\n',
 'thesis-dossier.md':'\n- **B06 · BAR (2026-10-03):** complete nine-fit synthetic prior comparison is near chance and does not establish mixed-prior benefit. Generator choice needs controlled, capable-learner tests before extension to relational priors. Original Mitra Table 12 remains source-gated; these measurements provide no direct evidence of relational superiority. [Protocol](labs/b06-reproduction.md).\n'}
for name,section in append.items():
 p=R/name;text=p.read_text()
 if section.strip().split('\n')[0] not in text:p.write_text(text+section)
p=R/'reference/glossary.html';text=p.read_text()
if 'id="prior-mixtures-b06"' not in text:text=text.replace('</body>','<section id="prior-mixtures-b06"><h2>Prior mixtures · B06</h2><dl><dt>Task prior</dt><dd>A probability distribution over tasks used to train a learner.</dd><dt>Outer prior mixture</dt><dd>A distribution that selects one generator for each complete task.</dd><dt>Within-task hybrid mechanism</dt><dd>Different mechanism families combined within the generation of a single task.</dd><dt>Prior-selection exposure</dt><dd>Using benchmark performance to choose a generator or mixture; the benchmark then supplies development evidence.</dd></dl><p><a href="b06-prior-mixtures.html">Prior-mixture field guide</a></p></section></body>');p.write_text(text)
p=R/'assets/retrieval-pool.js';text=p.read_text()
if 'b06-prior-selection' not in text:text=text.replace('global.RETRIEVAL_POOL = [','global.RETRIEVAL_POOL = [\n'+json.dumps(dict(id='b06-prior-selection',lesson=200.06,quarter='Q5',concept='prior-selection',question='B06: a benchmark chooses the prior mixture. What kind of evidence does it now supply?',options=[dict(label='Development evidence for selection',value='dev'),dict(label='Untouched evidence for generality',value='test')],correct='dev',explain='Choosing a prior from benchmark scores makes that benchmark development evidence, even without gradient training on its rows.'))+',');p.write_text(text)
p=R/'assets/paper-deck.js';text=p.read_text()
if 'mitra-b06-prior' not in text:text=text.replace('global.PAPER_DECK = [','global.PAPER_DECK = [\n'+json.dumps(dict(id='mitra-b06-prior',paper='Mitra: Mixed Synthetic Priors',year=2025,lesson=200.06,front='How can prior design change a fixed architecture, and what must its ablation hold constant?',back='Pretraining-task distributions change the learning experience. Match architecture, initialization, task/update budget and evaluation identities; change the generator. Keep prior-selection datasets separate from final tests. The B06 proxy experiment is near chance and does not reproduce Mitra.',source='https://arxiv.org/html/2510.21204v1'))+',');p.write_text(text)
p=R/'.github/workflows/pages.yml';text=p.read_text()
if '# B06:' not in text:
 block='''          # B06: complete controlled course fits and original Mitra Table12 source gate.
          mkdir -p public/labs/evidence/b06 public/labs/sources/b06 public/labs/figures/b06
          cp labs/b06-reproduction.md labs/_*b06*.py labs/_*b06*.json public/labs/
          cp labs/relkit/prior_b06.py public/labs/relkit/
          cp -r labs/evidence/b06/. public/labs/evidence/b06/
          cp -r labs/sources/b06/. public/labs/sources/b06/
          cp labs/figures/b06/* public/labs/figures/b06/
          cp labs/solutions/b06-mitra-prior-mixtures.ipynb public/labs/solutions/

'''
 text=text.replace('      - name: Setup Pages',block+'      - name: Setup Pages');p.write_text(text)
p=R/'.gitignore';text=p.read_text()
if '!labs/solutions/'+S+'.ipynb' not in text:p.write_text(text+'\n# B06 portable executed solution.\n!labs/solutions/'+S+'.ipynb\n')
print('Integrated B06; manifest version',m['version'])
