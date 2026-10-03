"""Integrate B08 into manifest-driven navigation and teaching records."""
import json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];S='b08-structured-objectives'
p=R/'lessons/manifest.json';m=json.loads(p.read_text())
if not any(x['id']=='B08' for x in m['lessons']):
 m['lessons'].append(dict(id='B08',slug=S,year=5,quarter=5,sortOrder=200.08,checkpoint=False,labPath='labs/'+S+'.ipynb',title='LimiX: alternative structured-data objectives',published=True));m['version']+=1
p.write_text(json.dumps(m,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
 p=R/name;t=p.read_text();t=re.sub(r'(name="rdl-manifest-version" content=")[0-9]+',lambda x:x[1]+str(m['version']),t);p.write_text(t)
p=R/'CURRICULUM.md';t=p.read_text().replace('B01–B07, B04a and B07a are prepared','B01–B08, B04a and B07a are prepared').replace('The other 20 units remain planned.','B08 adds nine paired objective fits with mixed target effects and a source-gated Table 23 imputation target. The other 19 units remain planned.').replace('[LimiX: alternative structured-data objectives](./plan/year-5-6-bridge.md#b08) | Feature/target attention and objective ablation.','[LimiX: alternative structured-data objectives](lessons/'+S+'.html) | [Lab](labs/'+S+'.ipynb): nine objective fits; paired masks; Table 23 source gate.');p.write_text(t)
p=R/'plan/year-5-6-bridge.md';t=p.read_text().replace('B01–B07, B04a and B07a prepared; 20 planned','B01–B08, B04a and B07a prepared; 19 planned');marker='### B08 ◆ · LimiX: alternative structured-data objectives\n'
if '**Prepared 2026-10-03:** [lesson](../lessons/'+S not in t:t=t.replace(marker,marker+'\n**Prepared 2026-10-03:** [lesson](../lessons/'+S+'.html) · [lab](../labs/'+S+'.ipynb). Complete nine-fit objective ablation: combined target error improves in one seed and worsens in two. Original Table23 Analcatdata imputation remains INCOMPLETE_SOURCE_PROTOCOL; full pretraining NOT_RUN; learner defense pending.\n\n')
p.write_text(t)
p=R/'reference/curriculum.html';t=p.read_text().replace('B01–B07, B04a and B07a prepared; 20 planned','B01–B08, B04a and B07a prepared; 19 planned').replace('../plan/year-5-6-bridge.md#b08"','../lessons/'+S+'.html"').replace('Feature/target attention and objective ablation</td>','Nine paired objective fits; Table 23 source gate</td>');p.write_text(t)
sections={
'labs/README.md':'\n### B08 · LimiX objectives\n\n[Student]('+S+'.ipynb) · [Executed solution](html/'+S+'.html) · [Contract](b08-reproduction.md). Three live functions, visible dual-axis course model/trainer, nine fresh objective fits and full original release source appendix. Table23 Analcatdata source-gated. Portable packet includes frozen inputs and source pins; no paid service required.\n',
'RESOURCES.md':'\n## B08 · LimiX objectives (2026-10-03)\n\n[LimiX v2 §§2–3 and Table23](https://arxiv.org/html/2509.03505v2), [LimiX-2M v2](https://arxiv.org/abs/2606.04485v2), [LimiX-2 v1 §§2–3](https://arxiv.org/html/2609.17488v1#S2), [official release](https://github.com/limix-ldm-ai/LimiX). Snapshots and source/checkpoint identities under labs/sources/b08. LimiX-2M and LimiX-2 are distinct releases. Current checkpoint license archived separately from code; the older abstract is not current licensing evidence.\n',
'NOTES.md':'\n## B08 · approved objective ablation (2026-10-03)\n\nB08-OBJECTIVE-ABLATION: three objectives×three seeds, same inputs/masks/initialization/120 steps within seed. Complete nine fits; target MSE0.6704 target-only versus0.7536 combined. Paired target differences+0.00384,−0.04761,+0.29332; mixed effects. Feature-only leaves the final target head untrained. Two-block width16 scalar-head mechanism is explicitly separate from full released LimiX architectures. Selected Table23 Analcatdata RMSE0.194 remains INCOMPLETE_SOURCE_PROTOCOL: original data/split/mask/scaler/repetitions/historical weights unestablished. No paid dispatch, full pretraining, deployment or learner mastery claim.\n',
'thesis-dossier.md':'\n- **B08 · BAR (2026-10-03):** more reconstruction supervision does not guarantee better target prediction. Nine paired synthetic course fits gave combined target MSE0.7536 versus0.6704, with one improving seed and two worsening seeds. No real-dataset or relational advantage inferred. Table23 selected historical target source-gated. [Audit](labs/evidence/b08/course-audit.json) · [Contract](labs/b08-reproduction.md).\n'}
for name,section in sections.items():
 p=R/name;t=p.read_text()
 if section.strip().split('\n')[0] not in t:p.write_text(t+section)
p=R/'reference/glossary.html';t=p.read_text()
if 'id="structured-objectives-b08"' not in t:
 t=t.replace('</body>','<section id="structured-objectives-b08"><h2>Structured objectives · B08</h2><dl><dt>Context-conditional masked modeling</dt><dd>Predicting hidden query entries from observed query entries and support context across episodes.</dd><dt>Task slot</dt><dd>One internal vector contributing to a target representation; not an extra observed label.</dd><dt>Attention visibility</dt><dd>The allowed reader-to-source edges, separate from input hiding and loss scoring.</dd><dt>Scored-cell mask</dt><dd>The set of originally known, intentionally hidden cells whose errors enter a loss or metric.</dd></dl><a href="b08-structured-objectives.html">Field guide</a></section></body>');p.write_text(t)
p=R/'assets/retrieval-pool.js';t=p.read_text()
if 'b08-three-masks' not in t:t=t.replace('global.RETRIEVAL_POOL = [','global.RETRIEVAL_POOL = [\n'+json.dumps(dict(id='b08-three-masks',lesson=200.08,quarter='Q5',concept='structured-objectives',question='B08: does hiding query labels alone prevent cross-query information flow?',options=[dict(label='Yes, all routes disappear',value='yes'),dict(label='No, support can relay',value='no')],correct='no',explain='If support can read queries, a later query can receive another query through support. Restrict both support and query readers to support keys.'))+',');p.write_text(t)
p=R/'assets/paper-deck.js';t=p.read_text()
if 'limix-b08-objective' not in t:t=t.replace('global.PAPER_DECK = [','global.PAPER_DECK = [\n'+json.dumps(dict(id='limix-b08-objective',paper='LimiX and LimiX-2',year=2026,lesson=200.08,front='Does feature reconstruction guarantee improved target prediction or causal identification?',back='No. It changes supervision and shared gradients. A paired objective ablation measures the effect in a declared setting; observational reconstruction and attention do not by themselves identify causal effects.',source='https://arxiv.org/html/2609.17488v1#S3'))+',');p.write_text(t)
p=R/'.github/workflows/pages.yml';t=p.read_text()
if '# B08:' not in t:
 block='''          # B08: complete course packet and source-gated published imputation target.
          mkdir -p public/labs/evidence/b08 public/labs/sources/b08 public/labs/data/b08 public/labs/solutions
          cp labs/b08-reproduction.md labs/_*b08*.py labs/_*b08*.json public/labs/
          cp labs/relkit/*b08.py public/labs/relkit/
          cp -r labs/evidence/b08/. public/labs/evidence/b08/
          cp -r labs/sources/b08/. public/labs/sources/b08/
          cp labs/data/b08/*.npz public/labs/data/b08/
          cp labs/solutions/b08-structured-objectives.ipynb public/labs/solutions/

'''
 t=t.replace('      - name: Setup Pages',block+'      - name: Setup Pages');p.write_text(t)
p=R/'.gitignore';t=p.read_text()
if '# B08 portable solution' not in t:p.write_text(t+'\n# B08 portable solution and optional external checkpoint.\n!labs/solutions/'+S+'.ipynb\nlabs/data/b08/*.ckpt\nlabs/sources/b08/release.tar.gz\n')
p=R/'learning-records/0165-structured-objectives-prepared.md'
if not p.exists():p.write_text('''# Structured-objective lesson prepared; learner defense pending

B08 teaches separate input hiding, attention visibility and scored-cell masks. LimiX-16M is the selected published target; LimiX-2 provides the asymmetric feature/task architecture discussion. LimiX-2M is a different smaller release.

Author evidence: nine complete course fits with paired initialization, inputs and schedules. Adding reconstruction improves target MSE in one seed and worsens it in two. This is a small synthetic mechanism experiment, not released-checkpoint performance or causal identification. Original Table23 Analcatdata RMSE0.194 remains INCOMPLETE_SOURCE_PROTOCOL; historical input/weight identities are missing.

The student must implement three live functions and defend both attention axes and evidence boundaries. Learner status PENDING_WRITTEN_DEFENSE. Schedule1/7/30day retrieval after learner completion, not author preparation. No personal mastery entry is justified.
''')
print('Integrated B08 manifest version',m['version'])
