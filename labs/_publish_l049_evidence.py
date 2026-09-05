"""Refresh measured lesson panels from artifacts; never invent unrun scores."""
import json,re
from pathlib import Path
HERE=Path(__file__).parent
lesson=HERE.parent/'lessons/0049-excelformer-trompt.html'
r=json.loads((HERE/'_verify_l049_results.json').read_text());s=lesson.read_text()

def table(summary,caption):
 h='<div class="table-scroll"><table><caption>'+caption+'</caption><thead><tr><th>Task</th><th>Excel · no DA</th><th>Excel · Feat-Mix</th><th>CatBoost</th></tr></thead><tbody>'
 for ds,models in summary['per_dataset'].items():
  h+='<tr><td>'+ds+'</td>'+''.join(f"<td>{v['mean']:.4f} ± {v['sample_std']:.4f}</td>" for v in models.values())+'</tr>'
 return h+'</tbody></table></div>'

def setblock(name,content):
 global s
 s=re.sub(r'<!-- '+name+r'_START -->.*?<!-- '+name+r'_END -->','<!-- '+name+'_START -->\n'+content+'\n<!-- '+name+'_END -->',s,flags=re.S)

summary=r['summary'];ranks=summary['mean_ranks'];cd=summary['nemenyi_cd'];p=summary['friedman']['p']
html=table(summary,'Measured test AUROC · mean ± sample SD · seeds 0, 1, 2')
html+='<figure class="claim-figure"><img src="../labs/figures/l049/scores.png" alt="Measured per-task test AUROC with three seed points and sample standard deviation"><figcaption>Author-reference detail views. Each task uses its own labeled AUROC range; compare models within a panel. The table gives exact rounded scores.</figcaption></figure>'
html+=f'<p>Mean ranks: Excel no DA {ranks["excel_no_da"]:.3f}; Feat-Mix {ranks["excel_feat_mix"]:.3f}; CatBoost {ranks["catboost"]:.3f}. Friedman p={p:.3f}; Nemenyi critical difference={cd:.3f}. This three-task experiment detects no overall difference. Banknote has AUROC 1.0 for all arms on this small test split; that is saturation, not proof of perfect future predictions. The breast task also has very little headroom.</p>'
html+='<div id="cd"></div><script>document.addEventListener("DOMContentLoaded",function(){CdDiagramViz.mount(document.getElementById("cd"),'+json.dumps({'models':[{'name':{'excel_no_da':'Excel no DA','excel_feat_mix':'Excel Feat-Mix','catboost':'CatBoost'}[m],'rank':v} for m,v in ranks.items()],'cd':cd,'N':3,'friedmanP':f'{p:.3f}'})+');});</script>'
setblock('RESULTS',html)
html=table(r['transfer_summary'],'Separate MovieLens transfer · test AUROC · mean ± sample SD')
html+='<figure class="claim-figure"><img src="../labs/figures/l049/transfer.png" alt="Random split favors CatBoost locally; temporal split favors the two neural variants"><figcaption>Different test populations. Error bars show training-seed variability, not uncertainty over future periods.</figcaption></figure>'
html+='<p>Here CatBoost leads under random splitting, while the neural arms lead on the later interval. This is a measured ranking reversal for our fixed pipeline, not a universal temporal advantage. Excel no DA barely changes in mean AUROC; a temporal split does not mechanically lower every model’s score.</p>'
setblock('TRANSFER',html)
pth=HERE/'data/cache/l049-closer/summary.json'
if pth.exists():
 q=json.loads(pth.read_text());html=table(q['summary'],'Measured larger Pima attempt · mean ± sample SD over three seeds')
 html+='<p><strong>Larger-run verdict: INCOMPARABLE.</strong> Width, depth and attention heads align more closely with the paper, but a single released split, chosen epoch cap and batch size, augmentation sampling details and absent hyperparameter search prevent an exact table reproduction. Selected weights, predictions and histories are saved under <code>labs/data/cache/l049-closer/</code>; regenerate them with the runner.</p>'
 (HERE/'_paper_repro_l049_closer_summary.json').write_text(json.dumps(q,indent=2))
 setblock('SCALE',html)
lesson.write_text(s)
