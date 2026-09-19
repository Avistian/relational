"""Register prepared L086 materials; learner completion is unchanged."""
from pathlib import Path
import json,re,importlib.metadata as m,sys
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0086-pyg-fundamentals';TITLE='PyG fundamentals: preserve the graph computation'
p=ROOT/'lessons/manifest.json';data=json.loads(p.read_text())
if not any(r['id']==86 for r in data['lessons']):
 data['lessons'].append({'id':86,'slug':SLUG,'year':3,'quarter':1,'checkpoint':False,'labPath':f'labs/{SLUG}.ipynb','title':TITLE,'published':True});data['version']+=1;p.write_text(json.dumps(data,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
 p=ROOT/name;s=p.read_text();s=re.sub(r'(name="rdl-manifest-version" content=")\d+',r'\g<1>'+str(data['version']),s)
 if name=='notebooks.html' and 'id="lab-86"' not in s:s=s.replace('<ul class="nb-gallery">',f'<ul class="nb-gallery"><li id="lab-86"><div class="nb-head"><span class="num">Lesson 0086</span><span class="title">{TITLE}</span></div><div class="nb-links"><a href="labs/html/{SLUG}.html">Read lab</a><a href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{SLUG}.ipynb">Run in Colab</a><a href="labs/{SLUG}.ipynb">Download notebook</a><a href="lessons/{SLUG}.html">Lesson</a></div></li>',1)
 p.write_text(s)
p=ROOT/'CURRICULUM.md';p.write_text(p.read_text().replace('| 086 | PyG fundamentals | Fey & Lenssen 2019 | Data, NeighborLoader |',f'| 086 | [PyG fundamentals](lessons/{SLUG}.html) | Fey & Lenssen 2019 | [MessagePassing parity, NeighborLoader, full100-seed Cora reconstruction](labs/{SLUG}.ipynb) |'))
p=ROOT/'plan/year-3.md';s=p.read_text();s=s.replace('### 086 · PyG fundamentals — *Fey & Lenssen 2019*',f'### 086 · [PyG fundamentals](../lessons/{SLUG}.html) — *Fey & Lenssen 2019*\n- **Prepared package** — [lab](../labs/{SLUG}.ipynb), [protocol](../labs/l086-reproduction.md): real NeighborLoader checks and full100-seed fixed-Cora PyG release reconstruction. Historical parity INCOMPARABLE; no learner completion inferred.');p.write_text(s)
p=ROOT/'.github/workflows/pages.yml';s=p.read_text()
if '# L086 PyG' not in s:s=s.replace('          # L085 over-smoothing', '''          # L086 PyG containers, routing, sampling and full Cora reconstruction.
          cp labs/l086-reproduction.md labs/requirements-l086-runtime.txt labs/requirements-l086-observed.txt labs/_sources_l086.json labs/_paper_l086_results.json labs/_verify_l086_results.json labs/_execution_l086_results.json labs/_delivery_l086_results.json labs/_clean_environment_l086_results.json public/labs/
          cp labs/_run_l086.py labs/_verify_l086.py labs/_build_l086.py labs/_execute_l086.py labs/_delivery_l086.py public/labs/
          mkdir -p public/labs/solutions
          cp labs/solutions/0086-pyg-fundamentals.ipynb public/labs/solutions/
          # L085 over-smoothing''',1);p.write_text(s)
for name,marker,addition in [
 ('NOTES.md','Lesson086 created','\n## Lesson086 created · PyG fundamentals · 2026-09-19\n\nFull reproducibility requested. Full100-seed PyG1.2.0 fixed-Cora protocol reconstruction:81.305% mean,0.714pp sampleSD. Exact paper-run revision and historical seeds unavailable; historical parity INCOMPARABLE. Real compiled NeighborLoader, output/gradient oracles, typed IDs, three live TODOs and complete inline model/loader/trainer. Source build of pyg-lib pinned. No learner completion inferred.\n'),
 ('RESOURCES.md','L086 · PyG','\n## L086 · PyG fundamentals\n\n- [Fey & Lenssen2019](https://arxiv.org/abs/1903.02428): primary framework paper; Table1 fixed-Cora target and100-run protocol.\n- [Archived PyG1.2.0](https://github.com/pyg-team/pytorch_geometric/tree/d5aff37604c8e3247f5e807f2ba0ec6eeb4c661b/benchmark/citation): audited release proxy, model and selection details differ from original Kipf trainer.\n- [MessagePassing](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.MessagePassing.html) and [NeighborLoader](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/neighbor_loader.html): source/destination lifting, seed ordering and ID contracts. Executed against PyG2.8.0.post1.\n'),
 ('labs/README.md','L086:',f'\n- L086: [PyG fundamentals]({SLUG}.ipynb) — [full reproduction guide](l086-reproduction.md), real sampling and operator/gradient parity.\n')]:
 p=ROOT/name;s=p.read_text()
 if marker not in s:p.write_text(s+addition)
names=['torch','torch-geometric','pyg-lib','numpy','scipy','matplotlib','nbformat','nbclient','nbconvert','ipykernel','cmake','ninja']
(LAB/'requirements-l086-observed.txt').write_text('# Observed Python '+sys.version.split()[0]+' Linux aarch64; pyg-lib source pin in protocol.\n'+'\n'.join(n+'=='+m.version(n) for n in names)+'\n')
print('Registered L086')
