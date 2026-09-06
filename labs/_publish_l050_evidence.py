"""Regenerate lesson evidence from recorded measurements, never hand-copy scores."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
r=json.loads((HERE/'_verify_l050_results.json').read_text())
path=ROOT/'lessons/0050-q1-checkpoint.html';html=path.read_text()
models=r['models'];rows=[]
for name,row in r['datasets'].items():
    values=''.join(f'<td>{row["summary"][m]["mean"]:.4f} ± {row["summary"][m]["sd"]:.4f}</td>' for m in models)
    rows.append(f'<tr><td>{name}</td>'+values+'</tr>')
body='<div class="table-scroll"><table><caption>Test AUROC · mean ± sample SD across 3 seeds</caption><thead><tr><th>Task</th>'+''.join(f'<th>{m}</th>' for m in models)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'
s=r['statistics']
body+=f'<p>FT-T leads on {s["ft_wins"]}/3 tasks. Mean ranks: '+', '.join(f'{m} {v:.3f}' for m,v in r['mean_ranks'].items())+f'. Friedman χ²={s["friedman_chi2"]:.3f}, p={s["friedman_p"]:.3f}; Nemenyi CD={s["nemenyi_cd"]:.3f}. Exact two-sided sign-test p={s["exact_sign_p"]:.3f}. The rank test detects no overall difference; equivalence is not established.</p>'
body+='<p>FT-T minus XGBoost mean AUROC and conditional 95% t intervals: '+ '; '.join(f'{n}: {v["ft_minus_xgb"]["mean"]:+.4f} [{v["ft_minus_xgb"]["ci95"][0]:+.4f}, {v["ft_minus_xgb"]["ci95"][1]:+.4f}]' for n,v in r['datasets'].items())+'. On phoneme the seed interval lies below zero; this remains conditional on the chosen split and search. Other intervals include zero.</p>'
body+=f'<p>The author run took {r["elapsed_s"]:.1f} seconds on CPU with numerical thread limits. Training time excludes lesson work. The copied-weight logit and input-gradient maximum errors were both {r["reference_parity"]["max_logit_error"]:.1e} in the specified reference check.</p>'
a,b=html.split('<!-- RESULTS_START -->');_,c=b.split('<!-- RESULTS_END -->');html=a+'<!-- RESULTS_START -->\n'+body+'\n<!-- RESULTS_END -->'+c
scale=HERE/'data/cache/l050-closer/summary.json'
if scale.exists():
    closer=json.loads(scale.read_text());(HERE/'_paper_repro_l050_closer_summary.json').write_text(json.dumps(closer,indent=2))
    body='<p><strong>Measured closer attempt:</strong> Higgs Small accuracy (mean ± sample SD): '+ '; '.join(f'{m} {s["mean"]:.4f} ± {s["sd"]:.4f}' for m,s in closer['accuracy'].items())+f'. CPU fitting took {closer["elapsed_s"]:.0f} seconds. <strong>INCOMPARABLE</strong> to Table 2 because the split, preprocessing, budget and selection differ. The full resource preset and Table 4 ensemble suite remain NOT_RUN. <a href="../labs/_paper_repro_l050_closer_summary.json">Scale-up ledger</a>.</p>'
else:
    body='<p><strong>Scale-up status: NOT_RUN.</strong> Use the supplied operators to produce the required ledger; no paper result is claimed.</p>'
a,b=html.split('<!-- SCALE_START -->');_,c=b.split('<!-- SCALE_END -->');html=a+'<!-- SCALE_START -->\n'+body+'\n<!-- SCALE_END -->'+c
path.write_text(html)
(ROOT/'assets/checkpoint-results.js').write_text('window.CHECKPOINT_RESULTS = '+json.dumps({k:r[k] for k in ['mean_ranks','statistics']})+';\n')
print('Published measured checkpoint evidence.')
