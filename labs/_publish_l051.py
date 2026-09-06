"""Publish tables from measured evidence; never hand-enter scores."""
import json,re,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent;COURSE=ROOT.parent
r=json.loads((ROOT/'_verify_l051_results.json').read_text());models=['MLP','FT-T','XGBoost']
rows=[]
for name,row in r['datasets'].items():
 for condition,runs in row['runs'].items():
  values=[]
  for m in models:
   a=np.array([v['accuracy'] for v in runs[m]]);values.append(f'{a.mean():.4f} ± {a.std(ddof=1):.4f}')
  rows.append('<tr><td>'+name+'</td><td>'+condition+'</td>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>')
table='<div class="evidence-table"><table><thead><tr><th>Dataset</th><th>Condition</th>'+''.join('<th>'+m+'</th>' for m in models)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'
means={c:{m:float(np.mean([v['effects'][c][m]['mean'] for v in r['datasets'].values()])) for m in models} for c in ['rotated','noise','smoothed']}
summary='<p><strong>Author measurement:</strong> '+str(round(r['elapsed_s']))+' seconds for the local CPU experiment. Accuracy ± sample SD across three seeds; the table contains local evidence, not paper targets.</p>'+table
summary+='<p>Rotation effects averaged equally over the three tasks: '+', '.join(f'{m} {100*means["rotated"][m]:+.2f} percentage points' for m in models)+'. These are task averages, not nine independent task observations.</p>'
summary+='<p>For added noise, electricity XGBoost rises while its other two task scores fall. Smoothing changes '+', '.join(f'{n}: {v["changed_training_labels"]}/{len(v["split_rows"][0])}' for n,v in r['datasets'].items())+' training labels. FT-T improves slightly on smoothed bank marketing while the other smoothing effects are negative. These exceptions belong in the report.</p>'
summary+='<figure><img class="lesson-figure" src="../labs/figures/l051/effects.png" alt="Paired intervention effects with seed points and conditional intervals"><figcaption>Local evidence. Effects use original baseline for rotation/noise and top-five baseline for smoothing.</figcaption></figure>'
summary+='<p>Original-condition Friedman p='+f'{r["statistics"]["original"]["friedman_p"]:.3f}'+'. The noise condition has unadjusted asymptotic p≈0.0498, but it is one of five exploratory tests on only three tasks; it does not survive a Bonferroni threshold of 0.01. Do not use this isolated threshold crossing as the conclusion.</p>'
lesson=COURSE/'lessons/0051-why-trees-still-win.html';s=lesson.read_text();s=re.sub(r'<!-- RESULTS_START -->.*?<!-- RESULTS_END -->','<!-- RESULTS_START -->'+summary+'<!-- RESULTS_END -->',s,flags=re.S)
closer=ROOT/'data/cache/l051-closer/summary.json'
if closer.exists():
 c=json.loads(closer.read_text());shutil.copyfile(closer,ROOT/'_paper_repro_l051_closer_summary.json')
 cr=[]
 for n,row in c['datasets'].items():
  cr.append(n+': '+', '.join(m+' '+f'{np.mean([a["accuracy"] for a in row["runs"]["original"][m]]):.4f} ± {np.std([a["accuracy"] for a in row["runs"]["original"][m]],ddof=1):.4f}' for m in models))
 scale='<p><strong>Larger attempt completed:</strong> '+str(round(c['elapsed_s']))+' seconds for the CPU experiment over all three tasks and five conditions. Original-condition accuracy means ± sample SD over three seeds — '+'; '.join(cr)+'. <strong>INCOMPARABLE</strong> to the paper; the full resource preset and original search curves remain NOT_RUN. <a href="../labs/_paper_repro_l051_closer_summary.json">Full scale-up evidence, paired effects and ledger</a>. The timing includes heavy workspace CPU contention and is not a model-speed comparison.</p>'
 s=re.sub(r'<!-- SCALE_START -->.*?<!-- SCALE_END -->','<!-- SCALE_START -->'+scale+'<!-- SCALE_END -->',s,flags=re.S)
lesson.write_text(s)
(COURSE/'assets/intervention-results.js').write_text('window.InterventionResults = '+json.dumps({'statistics':r['statistics'],'effects':means},indent=2)+';\n')
print('Published measured lesson evidence; closer included:',closer.exists())
