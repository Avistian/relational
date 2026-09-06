"""Refresh measured prose from canonical evidence; never hand-copy result numbers."""
import json,re
import numpy as np
from relkit.tabr_experiment import seed_interval
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def update():
 r=json.loads((HERE/'_verify_l052_results.json').read_text());p=ROOT/'lessons/0052-tabr-retrieval.html';s=p.read_text()
 table='<table><thead><tr><th>Dataset / metric</th><th>MLP</th><th>TabR-S</th><th>XGBoost</th></tr></thead><tbody>'
 for name,t in r['results'].items():
  table+=f'<tr><td>{name} / {t["metric"]}</td>'
  for arm,v in t['summary'].items():
   digits=0 if name=='house' else 4
   table+=f'<td>{v["mean"]:.{digits}f} ± {v["sd"]:.{digits}f}<br><small>95% CI [{v["ci95"][0]:.{digits}f}, {v["ci95"][1]:.{digits}f}]</small></td>'
  table+='</tr>'
 table+='</tbody></table>'
 st=r['stats'];table+=f'<p>Mean ± sample SD across three seeds; t intervals (2 degrees of freedom) condition on the fixed split and sampled rows. Mean ranks: MLP {st["mean_ranks"][0]:.3f}, TabR-S {st["mean_ranks"][1]:.3f}, XGBoost {st["mean_ranks"][2]:.3f}. Friedman p={st["friedman_p"]:.4f}; Nemenyi CD={st["cd"]:.4f}. No pair exceeds the critical difference. The observed tree lead is descriptive; this small rank test does not establish a general winner. Training elapsed {r["elapsed_seconds"]:.1f} seconds on the author CPU; this is not a model-speed benchmark.</p>'
 table+='<details><summary>Measured intervention: permute memory labels at inference</summary><table><thead><tr><th>Task</th><th>Clean mean</th><th>Permuted mean</th><th>Paired change ± SD</th></tr></thead><tbody>'
 for name,task in r['results'].items():
  clean=np.array([a['score'] for a in task['runs']['TabR-S']]);changed=np.array([a['shuffled_label_score'] for a in task['runs']['TabR-S']]);delta=seed_interval(changed-clean);digits=0 if name=='house' else 4
  table+=f'<tr><td>{name}</td><td>{clean.mean():.{digits}f}</td><td>{changed.mean():.{digits}f}</td><td>{delta["mean"]:+.{digits}f} ± {delta["sd"]:.{digits}f}</td></tr>'
 table+='</tbody></table><p>Change = permuted minus clean on paired model seeds. Positive RMSE change and negative accuracy change are worse. This one fixed label permutation demonstrates dependence on correctly aligned stored labels; it does not measure performance of a retrained label-free model or variation across permutations.</p></details>'
 s=re.sub(r'<!-- RESULTS_START -->.*?<!-- RESULTS_END -->','<!-- RESULTS_START -->'+table+'<!-- RESULTS_END -->',s,flags=re.S)
 scale=HERE/'_paper_repro_l052_closer_summary.json'
 if scale.exists():
  a=json.loads(scale.read_text());v=a['summary'];text=f'<p><strong>Larger run completed:</strong> California RMSE {v["mean"]:.4f} ± {v["sd"]:.4f} (sample SD, {v["n"]} seeds); conditional 95% seed interval [{v["ci95"][0]:.4f}, {v["ci95"][1]:.4f}]. Elapsed {a["elapsed_seconds"]:.0f} CPU seconds. <strong>INCOMPARABLE</strong> to the paper target .403 because training size, width, optimization and preprocessing differ. <a href="../labs/_paper_repro_l052_closer_summary.json">Measured larger-run ledger</a>. Full selected-configuration replay remains NOT_RUN.</p>'
 else:text='<p><strong>Larger run:</strong> NOT_RUN in this artifact snapshot. The operators below are provided; the paper result remains cited, not reproduced.</p>'
 s=re.sub(r'<!-- SCALE_START -->.*?<!-- SCALE_END -->','<!-- SCALE_START -->'+text+'<!-- SCALE_END -->',s,flags=re.S);p.write_text(s)
if __name__=='__main__':update()
