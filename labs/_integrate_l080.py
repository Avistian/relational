"""Integrate L080 without changing learner progress or adding home announcements."""
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SLUG='0080-year-2-exit-exam';TITLE='Year 2 exit exam: defend a reproducible comparison'
p=ROOT/'lessons/manifest.json';m=json.loads(p.read_text())
if not any(x['id']==80 for x in m['lessons']):
    m['lessons'].append({'id':80,'slug':SLUG,'year':2,'quarter':4,'checkpoint':True,'labPath':'labs/'+SLUG+'.ipynb','title':TITLE,'published':True});m['version']+=1;p.write_text(json.dumps(m,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
    p=ROOT/name;s=p.read_text();s=re.sub(r'(name="rdl-manifest-version" content=")\d+',r'\g<1>'+str(m['version']),s)
    if name=='notebooks.html' and 'id="lab-80"' not in s:
        s=s.replace('<ul class="nb-gallery">',f'<ul class="nb-gallery"><li id="lab-80"><div class="nb-head"><span class="num">Lesson 0080</span><span class="title">{TITLE}</span></div><div class="nb-links"><a href="labs/html/{SLUG}.html">Read lab</a><a href="labs/{SLUG}.ipynb" download>Download notebook</a><a href="lessons/{SLUG}.html">Lesson</a></div></li>',1)
    p.write_text(s)
p=ROOT/'CURRICULUM.md';s=p.read_text().replace('| 080 | **Year 2 exit exam** | All Y2 papers | Teach-back: 3 biases + TabM + TabPFN v2 + TabICL |',f'| 080 | [**Year 2 exit exam**](lessons/{SLUG}.html) | All Y2 papers | [Four-model random/temporal comparison + teach-back](labs/{SLUG}.ipynb) |');p.write_text(s)
p=ROOT/'.github/workflows/pages.yml';s=p.read_text()
if '# L080 exit' not in s:
    s=s.replace('          # L079 decision', '''          # L080 exit exam, complete local evidence and replay inputs.
          cp labs/_sources_l080.json labs/_verify_l080_results.json labs/_check_l080_results.json labs/_execution_l080_results.json labs/_browser_l080_results.json labs/l080-reproduction.md labs/l080-submission.md labs/requirements-l080-observed.txt public/labs/
          cp labs/_verify_l080.py labs/_check_l080.py labs/_prepare_l080.py labs/_fetch_l055.py public/labs/
          mkdir -p public/labs/data
          cp -r labs/data/l080 public/labs/data/
          # L079 decision''',1);p.write_text(s)
for name,marker,extra in [
('NOTES.md','Lesson 080 created','\n## Lesson 080 created · Year 2 exit exam · 2026-09-19\n\nFull reproducibility requested. Prepared cross-paper exam with four required arms, two real TabReD classification tasks, both split regimes and three seeds. Full local 48-record comparison and complete visible notebook implementations; historical v2 weights with disclosed simple wrapper. Small caps are local evidence, not original paper benchmark parity. Includes cold teach-back, fail-closed audits, rubric and submission template. Authorship does not advance learner completion.\n'),
('RESOURCES.md','L080 exit exam','\n## L080 exit exam\n\nHistorical synthesis of Grinsztajn three biases, FT-Transformer, TabM, Nature TabPFN v2, TabICL 2025 and TabReD. Pinned sources: `labs/_sources_l080.json`. FT uses the released architecture (including ReGLU and first-layer normalization exception). Local cross-paper comparison has no matching published target; original benchmarks remain NOT_RUN.\n'),
('labs/README.md','L080:',f'\n- L080: [Year 2 exit exam]({SLUG}.ipynb) — fresh four-family comparison, random + temporal, cold teach-back and rubric. [Reproduce](l080-reproduction.md).\n'),
('labs/data/README.md','L080 raw subsets','\n## L080 raw subsets\n\n`l080/{smoke,exam}.npz` stores complete float32 numeric/binary raw subsets and int64 targets from hash-pinned TabReD Ecom Offers/Homesite release archives. Matching JSON contains exact original IDs, full split boundaries, omitted categorical counts and archive/NPZ hashes. No preprocessing is fitted before packaging. `_prepare_l080.py` rebuilds the extraction.\n'),
('thesis-dossier.md','L080 |','\n| L080 | BAR | Prepared reproducible four-family random/temporal comparison and an assessed information-ceiling argument. Two capped datasets cannot establish population superiority or relational benefit; no learner pass inferred. |\n')]:
    p=ROOT/name;s=p.read_text()
    if marker not in s:p.write_text(s+extra)
print('Integrated L080')
