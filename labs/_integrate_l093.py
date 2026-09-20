"""Register the approved L093 package without marking learner mastery."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1];SLUG='0093-hgt';TITLE='HGT: typed attention through time'
p=ROOT/'lessons/manifest.json';d=json.loads(p.read_text())
if not any(x['id']==93 for x in d['lessons']):
 d['lessons'].append({'id':93,'slug':SLUG,'year':3,'quarter':2,'checkpoint':False,'labPath':f'labs/{SLUG}.ipynb','title':TITLE,'published':True});d['version']+=1;p.write_text(json.dumps(d,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
 p=ROOT/name;s=p.read_text();s=re.sub(r'(name="rdl-manifest-version" content=")\d+',r'\g<1>'+str(d['version']),s)
 if name=='notebooks.html' and 'id="lab-93"' not in s:s=s.replace('<ul class="nb-gallery">',f'<ul class="nb-gallery"><li id="lab-93"><div class="nb-head"><span class="num">Lesson0093</span><span class="title">{TITLE}</span></div><div class="nb-links"><a href="labs/html/{SLUG}.html">Read lab</a><a class="lab-access-primary" href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{SLUG}.ipynb">Run in Colab</a><a download="" href="labs/{SLUG}.ipynb">Download notebook</a><a href="lessons/{SLUG}.html">Lesson</a></div></li>',1)
 p.write_text(s)
p=ROOT/'CURRICULUM.md';p.write_text(p.read_text().replace('| 093 | HGT | Hu 2020 | Transformer on heterogeneous graphs |','| 093 | [HGT](lessons/0093-hgt.html) | Hu2020 | [Typed attention, time, matched OAG comparison and CS reproduction track](labs/0093-hgt.ipynb) |'))
p=ROOT/'plan/year-3.md';s=p.read_text();s=s.replace('### 093 · HGT — *Hu 2020, ★ `2003.01332`*','### 093 · [HGT](../lessons/0093-hgt.html) — *Hu2020, ★ `2003.01332`*\n- **Prepared package** — [lab](../labs/0093-hgt.ipynb), [protocol/evidence](../labs/l093-reproduction.md). Nine real OAG NN fits, source operator/sampler checks and a frequency baseline. CS bytes acquired; full CS target NOT_RUN due local memory/GPU account limits. Historical parity INCOMPARABLE; no learner completion inferred.');p.write_text(s)
entries={
'NOTES.md':'\n## Lesson093 created · HGT · 2026-09-20\n\nFull reproduction requested and combined scope approved. Visible typed attention/RTE/HGSampling/trainer; nine real NN fits and training-frequency baseline. Perfect NN MRR is trivial because every paper has the Artificial neural network field. Modern operator outputs/gradients and sample content match pinned source; publication-era source differences are archived. Both NN and CS bytes acquired/hash-pinned. Full-width NN update passed; CS loading exceeded10GiB address-space guard and Modal GPU requires a payment method. Full CS target NOT_RUN; full-paper parity NOT_ESTABLISHED. No learner completion inferred.\n',
'RESOURCES.md':'\n## L093 · Heterogeneous Graph Transformer\n\n- [Hu et al., WWW2020](https://arxiv.org/html/2003.01332v1): Figure2,§§3–4 and Table2. Typed attention, relative time, sampling and named CS Paper-Field L2 target.\n- [Audited modern OAG release](https://github.com/acbull/pyHGT/tree/85eaccd482bc1d1af56c2de297b6e3a88b96d5cd/OAG): executable source operator and sampler oracle.\n- [Publication-era source](https://github.com/acbull/pyHGT/tree/fd4a244db8efc72410537f3effec3b0c432892f7): historical comparison; differs from modern architecture.\n- [Authors\' OAG files](https://drive.google.com/drive/folders/1a85skqsMBwnJ151QpurLFSa9o2ymc_rq): NN and CS snapshots acquired and hashed; paper-era identity unestablished.\n',
'labs/README.md':'\n- **L093 HGT:** `0093-hgt.ipynb`; three live tasks, visible source-conditioned model and full OAG trainer. `_run_l093.py` provides smoke/teaching/paper/release tracks. Read `l093-reproduction.md` before interpreting scores.\n',
'labs/data/README.md':'\n## L093 · OAG NN and CS\n\nTier B, real heterogeneous academic graphs from the authors. Both archives acquired locally; excluded from git/Pages. `_fetch_l093.py` verifies pinned bytes. NN is the teaching subset, not a paper Table2 dataset. CS full-target loading/training has explicit resource blockers. See `../l093-reproduction.md`.\n',
'GLOSSARY.md':'\n## HGT additions · Lesson093\n\n- **Meta-relation:** one source-node-type, edge-relation-type, target-node-type triple.\n- **Relative temporal encoding:** a representation of the receiver/source time gap added to the source before key/value projection. It does not impose a data cutoff.\n- **HGSampling budget:** accumulated normalized neighbor scores, maintained separately per node type; squared scores determine sampling probabilities.\n'
}
for name,entry in entries.items():
 p=ROOT/name;s=p.read_text()
 if entry.splitlines()[1] not in s:p.write_text(s+entry)
p=ROOT/'.gitignore';s=p.read_text()
if 'labs/data/l093/' not in s:p.write_text(s+'\n# L093: downloaded OAG graphs stay local; prepared solution is distributable.\nlabs/data/l093/\n!labs/solutions/0093-hgt.ipynb\n')
p=ROOT/'.github/workflows/pages.yml';s=p.read_text()
if '# L093 HGT' not in s:
 s=s.replace('          # L092 HAN:', '          # L093 HGT: portable teaching/reproduction code and measured boundaries.\n          cp labs/l093-reproduction.md labs/requirements-l093-*.txt labs/_*l093*.json labs/_*l093.py public/labs/\n          mkdir -p public/labs/solutions public/modal\n          cp labs/solutions/0093-hgt.ipynb public/labs/solutions/\n          cp modal/l093_paper_repro.py public/modal/\n          # L092 HAN:',1);p.write_text(s)
print('Registered L093')
