"""Add only B10 navigation and evidence entries; preserve existing working changes."""
from pathlib import Path
import json,re
R=Path(__file__).resolve().parents[1];S='b10-relational-transformer'
p=R/'lessons/manifest.json';j=json.loads(p.read_text());j['lessons']=[x for x in j['lessons'] if x['id']!='B10'];j['lessons'].append(dict(id='B10',slug=S,year=5,quarter=5,sortOrder=200.10,checkpoint=False,labPath=f'labs/{S}.ipynb',title='Relational Transformer: cells, tasks and relational attention',published=True));j['version']=max(j.get('version',1),9);p.write_text(json.dumps(j,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
 p=R/name;p.write_text(re.sub(r'(name="rdl-manifest-version" content=")[0-9]+',lambda m:m[1]+str(j['version']),p.read_text()))
p=R/'CURRICULUM.md';s=p.read_text().replace('B01–B09, B04a and B07a are prepared','B01–B10, B04a and B07a are prepared').replace('The other 18 units remain planned.','B10 adds complete RT-v1 CPU mechanism checks and saved-context replay; its Table 1 inference remains blocked by the temporal gate. The other 17 units remain planned.').replace('[Relational Transformer: cells, tasks and relational attention](./plan/year-5-6-bridge.md#b10) | Cell-level relational attention and visibility trace.','[Relational Transformer: cells, tasks and relational attention](lessons/b10-relational-transformer.html) | [Lab](labs/b10-relational-transformer.ipynb): directed masks, source parity and full saved-context audit.');p.write_text(s)
p=R/'reference/curriculum.html';s=p.read_text().replace('B01–B09, B04a and B07a prepared; 18 planned','B01–B10, B04a and B07a prepared; 17 planned').replace('../plan/year-5-6-bridge.md#b10','../lessons/b10-relational-transformer.html');p.write_text(s)
p=R/'plan/year-5-6-bridge.md';s=p.read_text();header='### B10 ★ · Relational Transformer: cells, tasks and relational attention';addition='\n\n**Prepared 2026-10-03:** [lesson](../lessons/b10-relational-transformer.html) · [lab](../labs/b10-relational-transformer.ipynb) · [protocol](../labs/b10-reproduction.md). Complete three-seed CPU source-shaped mechanism checks and 2106-context saved replay; Table1 inference INCOMPLETE_TEMPORAL_GATE / NOT_RUN. Learner PENDING_WRITTEN_DEFENSE.\n'
if addition.strip() not in s:s=s.replace(header,header+addition)
p.write_text(s)
adds={'labs/README.md':'\n\n### B10 · Relational Transformer\n\n[Student](b10-relational-transformer.ipynb) · [Executed solution](html/b10-relational-transformer.html) · [Protocol](b10-reproduction.md). Visible typed model, directed cell attention, CPU source output/gradient checks and complete saved-context replay. Paper inference is blocked by the temporal gate.\n','.gitignore':'\n# B10 portable solution and archived RT-v1 source.\n!labs/solutions/b10-relational-transformer.ipynb\n','RESOURCES.md':'\n\n## B10 · Relational Transformer primary sources\n\n[RT-v1 §§3–5, Table1, AppendixH](https://arxiv.org/html/2510.06377v1), [original implementation](https://github.com/stanford-star/relational-transformer/tree/8d83590b5ae7fba9e40e8df463ed2dd9066ce5fb), [RT-J author update](https://star-project.stanford.edu/rt-j/). Original source/preprocessing/checkpoint revisions and inherited input hashes are in `labs/sources/b10/source-ledger.json`; CPU adapter and fixture deviations are in `labs/b10-reproduction.md`. Current RT-J is not a substitute for the original Table1 target.\n','thesis-dossier.md':'\n\n- **B10 · BAR:** Cell-level relational structure can be implemented and independently checked without establishing a performance advantage. Three-seed CPU RT-v1 output/gradient checks pass, while complete saved-context replay preserves the temporal blocker (385 future-dated cells in77contexts). Paper82.0%AUROC is not reproduced; historical availability and architecture benefit remain unestablished. See `labs/b10-reproduction.md`.\n','NOTES.md':'\n\n## B10 · approved RT-v1 mechanism and reproduction gate (2026-10-03)\n\nUser approved a full lesson/lab plus selected Table1 driver-dnf protocol. Three-seed CPU dense source-shaped mirror passes outputs/gradients/permutation/hidden-target checks; complete saved-context replay reconstructs2106labels and2,156,544slots.385future-dated cells in77contexts preserve INCOMPLETE_TEMPORAL_GATE. Benchmark inference, fresh pretraining and whole-paper reproduction NOT_RUN. Original checkpoint bytes unauthenticated. Local3600second aggregate budget, USD0paid; no silent RT-J substitution. Learner PENDING_WRITTEN_DEFENSE; no deployment or live Colab claim.\n'}
for name,addition in adds.items():
 p=R/name
 if addition.strip() not in p.read_text():p.write_text(p.read_text()+addition)
p=R/'assets/retrieval-pool.js';s=p.read_text()
if 'b10-directed-cell-attention' not in s:
 pos=s.rfind('];');s=s[:pos]+"\n, {id:'b10-directed-cell-attention',lesson:200.10,quarter:'Q4',concept:'relational-attention',question:'A task row references a customer. Which attention lets the task cell read customer cells?',options:[{label:'Feature attention reads parent cells',value:'feature'},{label:'Neighbor attention reads parent cells',value:'neighbor'}],correct:'feature',explain:'Feature attention reads the same row and foreign-to-primary parents. Neighbor attention reads child rows in the reverse direction.'}\n"+s[pos:];p.write_text(s)
p=R/'assets/paper-deck.js';s=p.read_text()
if 'rt-b10-masks' not in s:
 pos=s.rfind('];');s=s[:pos]+"\n, {id:'rt-b10-masks',paper:'Ranjan et al. — Relational Transformer',year:2025,lesson:200.10,front:'Why does correct relational attention not prove temporal validity?',back:'Attention mixes the cells admitted by context construction. Full attention can propagate forbidden evidence even if column and foreign-key masks are correct. Audit event time, availability and completed label windows before encoding.',source:'https://arxiv.org/html/2510.06377v1'}\n"+s[pos:];p.write_text(s)
p=R/'reference/glossary.html';s=p.read_text()
if 'id="b10-cell-attention"' not in s:s=s.replace('</body>','<section id="b10-cell-attention"><h2>Relational cell attention · B10</h2><p>Column attention joins cells with the same table and column identity. Feature attention reads the same row and referenced parents. Neighbor attention reads referencing child rows. Full attention reads the admitted non-padding context. Permissions are directed from reader to source; sampling visibility is a separate constraint.</p></section></body>');p.write_text(s)
p=R/'.github/workflows/pages.yml';s=p.read_text()
if '# B10:' not in s:
 block='''          # B10: source-visible cell model, audit receipts and portable notebook.
          mkdir -p public/labs/evidence/b10 public/labs/sources/b10 public/labs/figures/b10
          cp labs/b10-reproduction.md labs/_*b10*.py labs/_*b10*.json public/labs/
          cp labs/relkit/rt_b10.py public/labs/relkit/
          cp -r labs/evidence/b10/. public/labs/evidence/b10/
          cp -r labs/sources/b10/. public/labs/sources/b10/
          cp labs/figures/b10/* public/labs/figures/b10/
          cp labs/solutions/b10-relational-transformer.ipynb public/labs/solutions/

'''
 s=s.replace('      - name: Setup Pages',block+'      - name: Setup Pages');p.write_text(s)
(R/'learning-records/0167-relational-cell-attention-prepared.md').write_text('''# B10 prepared; learner defense pending

B10 builds task-table integration, typed cell tokens and column/feature/neighbor/full attention. The three live tasks implement directed permissions, safe masked attention and a strict temporal evidence policy. CPU source-shaped model output/gradient parity and intervention checks are author verification, not learner mastery or pretrained model performance.

The full saved-context replay covers2106queries and2,156,544slots;385future-dated schedule cells preserve INCOMPLETE_TEMPORAL_GATE. No checkpoint inference or pretraining ran. Historical availability remains unestablished; a future event timestamp does not prove late arrival. RT-J is separate from RT-v1.

Learner status PENDING_WRITTEN_DEFENSE. Assess the functions and written explanation, then schedule1/7/30day retrieval. No change to the Year5 exit gate or personal mastery log.
''')
print('Integrated B10 without changing previous lesson artifacts')
