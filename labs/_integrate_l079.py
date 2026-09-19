"""Idempotent course integration for the writing capstone, without mastery claims."""
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SLUG='0079-neural-tabular-decision-guide';TITLE='Year 2 essay: choose a model you can defend'
p=ROOT/'lessons/manifest.json';m=json.loads(p.read_text())
if not any(x['id']==79 for x in m['lessons']):
    m['lessons'].append({'id':79,'slug':SLUG,'year':2,'quarter':4,'checkpoint':False,'labPath':'labs/'+SLUG+'.ipynb','title':TITLE,'published':True});m['version']+=1;p.write_text(json.dumps(m,indent=2)+'\n')
for name in ['index.html','notebooks.html']:
    p=ROOT/name;s=p.read_text();s=re.sub(r'(name="rdl-manifest-version" content=")\d+',r'\g<1>'+str(m['version']),s)
    if name=='notebooks.html' and f'lessons/{SLUG}.html' not in s:s=s.replace('<ul class="nb-gallery">',f'<ul class="nb-gallery"><li id="lab-79"><div class="nb-head"><span class="num">Lesson 0079</span><span class="title">{TITLE}</span></div><div class="nb-links"><a href="labs/html/{SLUG}.html">Read lab</a><a href="labs/{SLUG}.ipynb" download>Download notebook</a><a href="lessons/{SLUG}.html">Lesson</a></div></li>',1)
    p.write_text(s)
p=ROOT/'.github/workflows/pages.yml';s=p.read_text()
if '# L079 decision' not in s:
    s=s.replace('          # L078 message','          # L079 decision guide, complete frozen evidence and replay contract.\n          cp labs/_sources_l079.json labs/_verify_l079_results.json labs/_execution_l079_results.json labs/_browser_l079_results.json labs/l079-reproduction.md labs/l079-decision-template.md labs/requirements-l079-observed.txt public/labs/\n          cp labs/_verify_l079.py labs/_check_l079.py public/labs/\n          mkdir -p public/labs/data public/solutions\n          cp -r labs/data/l079 public/labs/data/\n          cp solutions/l079-example-guide.md public/solutions/\n          # L078 message',1);p.write_text(s)
for name,marker,extra in [
('NOTES.md','Lesson 079 created','\n## Lesson 079 created · Year 2 decision guide · 2026-09-19\n\nWriting capstone with a one-page template/rubric, source-grounded historical family shortlist, matched-cohort rank analysis and synthetic latency constraint. Full corrected L060 v2 prediction audit (210 records) is reproducible offline with embedded/pinned bytes and complete inline analysis. It is not fresh training or paper reproduction. No learner completion inferred.\n'),
('RESOURCES.md','L079 decision guide','\n## L079 decision guide\n\nPrimary historical sources for mechanism-based shortlists: [trees](https://arxiv.org/abs/2207.08815), [FT-Transformer](https://arxiv.org/abs/2106.11959), [RealMLP](https://arxiv.org/abs/2407.04491), [TabM v3](https://arxiv.org/abs/2410.24210v3), [TabPFN v2](https://www.nature.com/articles/s41586-024-08328-6), [TabICL 2025 v2](https://arxiv.org/abs/2502.05564v2). This is a course synthesis, not a current leaderboard. Local quantitative evidence uses corrected L060 v2 predictions only; source/hash inventory: labs/_sources_l079.json.\n'),
('labs/README.md','L079:',f'\n- L079: [Decision guide]({SLUG}.ipynb) — one-page writing deliverable plus complete frozen-prediction audit. [Reproduction](l079-reproduction.md).\n'),
('labs/data/README.md','L079 decision evidence','\n## L079 decision evidence\n\n`l079/l060-v2.json.gz` is a deterministic gzip copy of the complete corrected `_verify_l060_v2_results.json`, including all 210 prediction records. Uncompressed hash and source-course revision are pinned in `_sources_l079.json`; no sampling or rounding was introduced. Reanalysis only, not new training.\n')]:
    p=ROOT/name;s=p.read_text()
    if marker not in s:p.write_text(s+extra)
print('Integrated L079')
