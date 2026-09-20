"""Register prepared L099 artifacts, preserving learner progress."""
from pathlib import Path
import json,re
R=Path(__file__).resolve().parents[1];S='0099-rgcn-vs-hgt';T='R-GCN vs HGT: a controlled comparison'
p=R/'lessons/manifest.json';d=json.loads(p.read_text())
if not any(x['id']==99 for x in d['lessons']):
 d['lessons'].append({'id':99,'slug':S,'year':3,'quarter':2,'checkpoint':False,'labPath':f'labs/{S}.ipynb','title':T,'published':True});d['version']+=1;p.write_text(json.dumps(d,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
 p=R/name;s=p.read_text();s=re.sub(r'(name="rdl-manifest-version" content=")\d+',r'\g<1>'+str(d['version']),s)
 if name=='notebooks.html' and 'id="lab-99"' not in s:s=s.replace('<ul class="nb-gallery">',f'<ul class="nb-gallery"><li id="lab-99"><div class="nb-head"><span class="num">Lesson0099</span><span class="title">{T}</span></div><div class="nb-links"><a href="labs/html/{S}.html">Read lab</a><a class="lab-access-primary" href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{S}.ipynb">Run in Colab</a><a download="" href="labs/{S}.ipynb">Download notebook</a><a href="lessons/{S}.html">Lesson</a></div></li>',1)
 p.write_text(s)
p=R/'CURRICULUM.md';p.write_text(p.read_text().replace('| 099 | Compare R-GCN vs HGT | — | Same graph, two architectures |',f'| 099 | [Compare R-GCN vs HGT](lessons/{S}.html) | Controlled analysis · Schlichtkrull + Hu | [Full ACM comparison, 24 fits + attention ablation](labs/{S}.ipynb) |'))
p=R/'plan/year-3.md';s=p.read_text();s=s.replace('### 099 · Compare R-GCN vs HGT — *— (analysis unit)*',f'### 099 · [Compare R-GCN vs HGT](../lessons/{S}.html) — *controlled analysis unit*\n- **Prepared package** — [lab](../labs/{S}.ipynb), [full protocol](../labs/l099-reproduction.md): all24 ACM course fits and a fresh ten-run AIFB port. Uniform HGT slightly exceeds learned attention here; no universal winner inferred. HGT CS NOT_RUN; full-paper parity NOT_ESTABLISHED. Learner defense pending.');p.write_text(s)
entries={
'NOTES.md':'\n## Lesson099 created · R-GCN vs HGT · 2026-09-20\n\nFull reproduction requested; combined scope approved. Entire24-fit ACM course comparison executed: same graph/split/features/search budget, paired common HGT/ablation initialization, full traces/checkpoints/predictions. Uniform HGT slightly exceeds HGT; architecture gap cannot be credited to learned attention. Fresh complete ten-run AIFB port executed separately (95.8333% mean). HGT CS remains NOT_RUN with prior resource findings explicitly attributed to L093. Full-paper parity NOT_ESTABLISHED; no learner mastery or deployment inferred.\n',
'RESOURCES.md':'\n## Lesson 099 · Controlled architecture comparison\n\n- [R-GCN, Schlichtkrull et al.](https://arxiv.org/abs/1703.06103v4): Eq2–3, Table2 AIFB; named ten-run port replay.\n- [HGT, Hu et al.](https://arxiv.org/abs/2003.01332v1): §3 operators, Table2 CS; full-setting track remains unrun.\n- [Pinned ACM raw loader](https://github.com/dmlc/dgl/blob/3d16000b4170fa741ed9e9667f22ba84d3493026/examples/pytorch/han/utils.py): conference mapping and typed incidence; course split/trainer explicitly differ.\n- [L099 protocol and evidence](labs/l099-reproduction.md):24 complete course fits, paired attention intervention, independent checkpoint/metric audit and separate published targets.\n',
'labs/README.md':f'\n### Lesson099 · R-GCN versus HGT\n\n[Student]({S}.ipynb) · [Solution](solutions/{S}.ipynb) · [Read lab](html/{S}.html) · [Protocol](l099-reproduction.md). Full24-fit ACM comparison, attention ablation, fresh ten-run AIFB port; HGT CS NOT_RUN. Three live tasks and written attribution defense.\n'}
for name,entry in entries.items():
 p=R/name;s=p.read_text()
 if entry.strip().splitlines()[0] not in s:p.write_text(s+entry)
p=R/'.github/workflows/pages.yml';s=p.read_text();marker='          # L098 native heterogeneous sampling:'
if '# L099 controlled' not in s:s=s.replace(marker,'          # L099 controlled ACM comparison and separate published-experiment evidence.\n          cp labs/l099-reproduction.md labs/requirements-l099-*.txt labs/_*l099*.json labs/_*l099.py public/labs/\n          mkdir -p public/labs/solutions public/plan\n          cp labs/solutions/0099-rgcn-vs-hgt.ipynb public/labs/solutions/\n          cp -r labs/_experiment_l099_results-checkpoints public/labs/\n          cp plan/year-3.md public/plan/\n'+marker)
p.write_text(s)
print('Registered lesson 99; no learning record or mastery status changed')
