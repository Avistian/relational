"""Register B15, retaining all existing bridge packages and evidence statuses."""
import json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];S='b15-parameter-free-encoders'
p=R/'lessons/manifest.json';j=json.loads(p.read_text());j['lessons']=[x for x in j['lessons'] if x['id']!='B15'];j['lessons'].append(dict(id='B15',slug=S,year=5,quarter=5,sortOrder=200.15,checkpoint=False,labPath=f'labs/{S}.ipynb',title='Parameter-free encoders: limits and assumptions',published=True));j['version']=max(j['version'],14);p.write_text(json.dumps(j,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
 p=R/name;p.write_text(re.sub(r'(name="rdl-manifest-version" content=")[0-9]+',lambda m:m[1]+str(j['version']),p.read_text()))
p=R/'CURRICULUM.md';s=p.read_text().replace('B01–B14, B04a and B07a are prepared','B01–B15, B04a and B07a are prepared').replace('The other 13 units remain planned.','B15 adds exhaustive label-visibility and feature-identifiability diagnostics; the RDBLearn v1.1 Table 5 target remains source-gated. The other 12 units remain planned.').replace('[Parameter-free encoders: limits and assumptions](./plan/year-5-6-bridge.md#b15) | Encoder assumptions and label-visibility counterexample.','[Parameter-free encoders: limits and assumptions](lessons/'+S+'.html) | [Lab](labs/'+S+'.ipynb): exhaustive information-boundary counterexamples and source-gated Table 5 target.');p.write_text(s)
p=R/'reference/curriculum.html';p.write_text(p.read_text().replace('../plan/year-5-6-bridge.md#b15','../lessons/'+S+'.html').replace('B01–B14, B04a and B07a prepared; 13 planned','B01–B15, B04a and B07a prepared; 12 planned'))
p=R/'plan/year-5-6-bridge.md';s=p.read_text();header='### B15 ◆ · Parameter-free encoders: limits and assumptions';addition='\n\n**Prepared 2026-10-04:** [lesson](../lessons/'+S+'.html) · [lab](../labs/'+S+'.ipynb) · [contract](../labs/b15-reproduction.md). Complete finite 4-rule/8-column worlds and 16 interventions; Table 5 RDBLearn trial target INCOMPLETE_SOURCE_PROTOCOL_GATE, inference NOT_RUN. Learner PENDING_WRITTEN_DEFENSE. Dedicated label-visibility widgets replace the generic label-count visual because access, not just quantity, is the mechanism.\n'
if '**Prepared 2026-10-04:** [lesson](../lessons/'+S not in s:s=s.replace(header,header+addition)
p.write_text(s)
adds={'.gitignore':'\n# B15 portable solution\n!labs/solutions/'+S+'.ipynb\n','labs/README.md':'\n\n### B15 · Parameter-free encoders\n\n[Student]('+S+'.ipynb) · [Executed solution](html/'+S+'.html) · [Contract](b15-reproduction.md). Three live mechanisms; complete finite information experiment; pinned source/default audit and explicitly blocked Table 5 inference.\n',
'RESOURCES.md':'\n\n## B15 · Parameter-free encoders: limits and assumptions\n\n[Paper v2](https://arxiv.org/html/2607.05476v2), §§3–4, Appendices A/D, Table 5; [RDBLearn v1.1 source](https://github.com/HKUSHXLab/rdblearn/tree/78561f0a9c1dd231d44659e761d5d85e18c82f6e). Actual default target-history flag is False; README says True. Code default agrees with paper, but exact run configuration is not established. Complete source packet and missing protocol ledger under labs/sources/b15 and labs/evidence/b15.\n',
'NOTES.md':'\n\n## B15 · Approved information-boundary lesson (2026-10-04)\n\nComplete finite 4-rule/8-column worlds and 16 hidden/future-label interventions. Legal external support resolves designed ambiguity; this changes available information, not encoder training. Three live student functions and portable executed solution. RDBLearn v1.1 actual history default OFF resolves README mismatch; exact Table 5 trial candidates/seeds/checkpoints/refit unknown, so selected inference INCOMPLETE_SOURCE_PROTOCOL_GATE/NOT_RUN. Cloud USD0; full suite/pretraining NOT_RUN. Learner PENDING_WRITTEN_DEFENSE. User authorized push and Pages deployment.\n',
'thesis-dossier.md':'\n\n- **B15 · BAR:** Learned parameters cannot universally recover task information absent from the input. Exact finite counterexamples show ambiguity and a legal expanded-support rescue, not universal uselessness of learned encoders. RDBLearn v1.1 Table 5 trial target0.7271 remains source-gated; no fresh paper score or real-data superiority inferred.\n'}
for name,addition in adds.items():
 p=R/name
 if addition.strip() not in p.read_text():p.write_text(p.read_text()+addition)
p=R/'assets/retrieval-pool.js';s=p.read_text();item=dict(id='b15-existence-quantifier',lesson=200.15,quarter='Q5',concept='encoder-information-boundary',question='What does an existence counterexample establish about learned encoders?',options=[dict(label='Some distributions defeat guarantees',value='some'),dict(label='All distributions defeat learning',value='all')],correct='some',explain='An existence result defeats a universal guarantee. It does not say every task is hard or trained encoders cannot help on particular distributions.')
if item['id'] not in s:s=s.replace('global.RETRIEVAL_POOL = [','global.RETRIEVAL_POOL = [\n'+json.dumps(item)+',')
p.write_text(s)
p=R/'reference/glossary.html';s=p.read_text();entry='<section id="b15-identifiability"><h2>Identifiability and information boundary</h2><p>Identifiability asks whether competing rules can be distinguished from the available observations. An information boundary specifies which rows, labels and times the encoder and prediction head may read. A parameter-free encoder has no learned encoder weights; its head may still be pretrained. <a href="../lessons/'+S+'.html">B15 worked counterexamples</a>.</p></section>'
if 'id="b15-identifiability"' not in s:s=s.replace('</body>',entry+'\n</body>')
p.write_text(s)
p=R/'.github/workflows/pages.yml';s=p.read_text()
if '# B15:' not in s:
 block='''          # B15: finite information diagnostics and source-gated paper contract.
          mkdir -p public/labs/evidence/b15 public/labs/sources/b15 public/labs/relkit public/labs/solutions
          cp labs/b15-reproduction.md labs/_*b15*.py labs/_*b15*.json public/labs/
          cp labs/relkit/labels_b15.py public/labs/relkit/
          cp -r labs/evidence/b15/. public/labs/evidence/b15/
          cp -r labs/sources/b15/. public/labs/sources/b15/
          cp labs/solutions/b15-parameter-free-encoders.ipynb public/labs/solutions/

'''
 s=s.replace('      - name: Setup Pages',block+'      - name: Setup Pages')
p.write_text(s)
p=R/'learning-records/0171-parameter-free-encoders-prepared.md';p.write_text('''# B15 prepared; learner defense pending

Author evidence: exhaustive four rule worlds, eight column worlds, sixteen forbidden-label interventions; independent scoring. Local column accuracy75% coexists with zero bits about column identity. Expanded support changes information, not trainable weights. Complete source archive: v1.1 history OFF in actual config, contrary to README. Table5 trial reproduction is source-gated, inference NOT_RUN. Full suite and pretraining NOT_RUN; USD0cloud.

Learner must implement visibility, compatible rules and compatible columns, then defend the existence quantifier and label access. Revisit after1/7/30days. Status:PENDING_WRITTEN_DEFENSE. Author tests and deployment do not establish learner mastery.
''');print('B15 integrated')
