"""Register prepared lesson without claiming learner completion or remote deployment."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1];SLUG='0084-gat';TITLE='GAT: learn which neighbors to weight'
p=ROOT/'lessons/manifest.json';m=json.loads(p.read_text())
if not any(x['id']==84 for x in m['lessons']):
 m['lessons'].append({'id':84,'slug':SLUG,'year':3,'quarter':1,'checkpoint':False,'labPath':f'labs/{SLUG}.ipynb','title':TITLE,'published':True});m['version']+=1;p.write_text(json.dumps(m,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
 p=ROOT/name;s=p.read_text();s=re.sub(r'<meta\b[^>]*name="rdl-manifest-version"[^>]*>',f'<meta name="rdl-manifest-version" content="{m["version"]}">',s)
 if name=='notebooks.html' and 'id="lab-84"' not in s:
  s=s.replace('<ul class="nb-gallery">',f'<ul class="nb-gallery"><li id="lab-84"><div class="nb-head"><span class="num">Lesson 0084</span><span class="title">{TITLE}</span></div><div class="nb-links"><a href="labs/html/{SLUG}.html">Read lab</a><a href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{SLUG}.ipynb">Run in Colab</a><a href="labs/{SLUG}.ipynb" download>Download notebook</a><a href="lessons/{SLUG}.html">Lesson</a></div></li>',1)
 p.write_text(s)
p=ROOT/'CURRICULUM.md';p.write_text(p.read_text().replace('| 084 | GAT | Veličković 2018 | Attention aggregation |',f'| 084 | [GAT](lessons/{SLUG}.html) | Veličković 2018 | [Attention aggregation + full Cora port](labs/{SLUG}.ipynb) |'))
p=ROOT/'.github/workflows/pages.yml';s=p.read_text()
if '# L084 GAT' not in s:
 s=s.replace('          # L083 GraphSAGE', '''          # L084 GAT, full Cora port and reproducibility evidence.
          cp labs/l084-reproduction.md labs/requirements-l084-observed.txt labs/requirements-l084-runtime.txt labs/_clean_environment_l084.py labs/_clean_environment_l084_results.json labs/_sources_l084.json labs/_paper_l084_results.json labs/_attention_l084.json labs/_verify_l084_results.json labs/_execution_l084_results.json labs/_browser_l084_results.json public/labs/
          cp labs/_run_l084.py labs/_verify_l084.py labs/_sources_l084.py labs/_data_identity_l084_results.json labs/_check_fresh_cli_l084.py labs/_fresh_cli_l084_results.json public/labs/
          mkdir -p public/labs/solutions
          cp labs/solutions/0084-gat.ipynb public/labs/solutions/
          # L083 GraphSAGE''',1);p.write_text(s)
for name,marker,extra in [
 ('NOTES.md','Lesson 084 created','\n## Lesson 084 created · GAT · 2026-09-19\n\nUser requested full reproducibility. Complete visible Cora GAT release port, three live TODOs, full100-seed command, learned attention visualization and source/data hashes. Release biases, three dropout sites and OR-reset/AND-save checkpoint rule preserved. Modern-framework and historical-seed deviations explicit. See _paper_l084_results.json for actual execution and l084-reproduction.md for the evidence boundary. No learner completion inferred.\n'),
 ('RESOURCES.md','L084 · GAT','\n## L084 · GAT\n\n- [Veličković et al.2018](https://arxiv.org/html/1710.10903v3): attention equations, Table2 Cora target and100-run evaluation.\n- [Pinned official implementation](https://github.com/PetarV-/GAT/tree/5af87e7fce2b90ae1cbd621cd58059036a3c7436): inspect head biases/dropout, parameter regularization and checkpoint decisions.\n- [Lesson reproduction contract](labs/l084-reproduction.md): full protocol audit, local run evidence and historical gaps.\n'),
 ('labs/README.md','L084:',f'\n- L084: [GAT]({SLUG}.ipynb) — visible multi-head attention, three live TODOs and [full Cora reproduction contract](l084-reproduction.md).\n'),
 ('plan/year-3.md','L084 delivery evidence','\n## L084 delivery evidence\n\nPrepared [lesson](../lessons/0084-gat.html), [lab](../labs/0084-gat.ipynb) and [contract](../labs/l084-reproduction.md). Native visible attention exposes receiver normalization and original release semantics; full100-run Cora operator, fixed source/data identities, independent dense oracle and learned attention visualization. Historical parity remains separate. Creation does not assert mastery.\n')]:
 p=ROOT/name;s=p.read_text()
 if marker not in s:p.write_text(s+extra)
p=ROOT/'reference/glossary.html';s=p.read_text()
if 'id="gat-l084"' not in s:
 s=s.replace('</article>','<h2 id="gat-l084">L084 · Graph attention</h2><dl><dt>Neighbor softmax</dt><dd>Normalize allowed sender scores separately for each receiver and attention head.</dd><dt>Attention head</dt><dd>One independently parameterized projection and scoring function shared across nodes and edges.</dd><dt>Coefficient dropout</dt><dd>Drop normalized attention coefficients with inverted scaling during training; realized rows need not sum to one.</dd></dl></article>');p.write_text(s)
print('Integrated L084')
if __name__=='__main__':pass
