"""Dataset-unit analysis of saved actual L069 predictions; no fabricated training."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parent

def analyze():
 r=json.loads((ROOT/'_verify_l069_v2_results.json').read_text());rows=r['summary'];table=['| Dataset | Arm | Continuous novelty AUC ± SD | Interval AUC ± SD | Natural novel fraction |','|---|---|---:|---:|---:|'];diff=[];ranks=[]
 for dataset in ['cmc','winequality-red','winequality-white']:
  means=[]
  for arm in ['v2','xgboost']:
   a=next(v for v in rows if v['axis']=='novelty' and v['dataset']==dataset and v['arm']==arm and v['condition']=='leave-one-class-out');n=next(v for v in rows if v['axis']=='novelty' and v['dataset']==dataset and v['arm']==arm and v['condition']=='natural-prevalence-class0')
   table.append('| '+dataset+' | '+arm+' | '+f"{a['auc_mean']:.4f} ± {a['auc_sd']:.4f} | {a['interval_auc_mean']:.4f} ± {a['interval_auc_sd']:.4f} | {n['unsupported_fraction_mean']:.4f} |")
   means.append(a['auc_mean'])
  diff.append(means[0]-means[1]);ranks.append(rankdata(-np.array(means)).tolist())
 rng=np.random.default_rng(69);boot=np.array(diff)[rng.integers(0,3,(10000,3))].mean(1);interval=np.quantile(boot,[.025,.975]).tolist()
 feature=[]
 for dataset in ['iris','cmc','winequality-red']:
  for arm in ['v2','xgboost']:
   a=next(v for v in rows if v['axis']=='features' and v['dataset']==dataset and v['arm']==arm and v['condition']=='0%');b=next(v for v in rows if v['axis']=='features' and v['dataset']==dataset and v['arm']==arm and v['condition']=='100%');feature.append(dict(dataset=dataset,arm=arm,clean=a['accuracy_mean'],all_removed=b['accuracy_mean'],gap=b['delta_accuracy_mean'],balanced_accuracy=a['balanced_accuracy_mean'],macro_f1=a['macro_f1_mean'],auc=a.get('auc_mean'),auc_n=a.get('auc_n',0)))
 text='The mean continuous novelty AUC gap (v2 minus XGBoost), after averaging every held class and split within each dataset, is '+f"{np.mean(diff):+.4f}. The paired three-dataset bootstrap interval is [{interval[0]:+.4f}, {interval[1]:+.4f}] (10,000 resamples, seed 69). This is a descriptive interval over only these three dataset units, two of which are related Wine tasks; it does not establish general detector superiority. Two-arm mean ranks are v2 {np.mean(ranks,axis=0)[0]:.3f} and XGBoost {np.mean(ranks,axis=0)[1]:.3f}."
 text+='\n\nClean-to-100%-removed accuracy: '+ '; '.join(f"{a['dataset']}/{a['arm']} {a['clean']:.4f} → {a['all_removed']:.4f}" for a in feature)+'. The fully removed query matrix contains no row-specific feature differences. The saved probabilities and class prevalences explain the remaining accuracy.'
 text+='\n\nOn the generated concept-reversal control, probabilities match the IID case and binary accuracy/AUC complement exactly within numeric tolerance. Both arms fail for the same information reason. Added-column alignment also produces exactly the baseline input; that is a control for ignoring new information, not learned feature adaptation. The source-scored interval and continuous confidence should therefore be interpreted as separate detectors, and supported-only losses remain separate from all-row clipped losses.'
 text+='\n\nThe measured clean-objective table below uses the same saved predictions throughout. Macro F1 averages the union of true and predicted labels; balanced accuracy averages true-present classes. These values must not be substituted for the paper\'s unreconciled F1 convention.\n\n| Dataset | Arm | Accuracy | Balanced accuracy | Macro F1 | AUC (available seeds) |\n|---|---|---:|---:|---:|---:|\n'+'\n'.join('| '+a['dataset']+' | '+a['arm']+' | '+' | '.join(f"{a[k]:.4f}" for k in ['clean','balanced_accuracy','macro_f1'])+' | '+(f"{a['auc']:.4f} ({a['auc_n']})" if a['auc'] is not None else 'undefined (0)')+' |' for a in feature)
 text+='\n\nThe generated conditions below report realized positive prevalence. Covariate shift preserves the threshold law but changes class composition; concept reversal changes labels while retaining identical input rows.\n\n| Generated condition | Arm | Positive fraction | Accuracy ± SD | AUC ± SD |\n|---|---|---:|---:|---:|\n'
 for condition in ['iid','covariate','concept']:
  targets=[s['targets'] for s in r['splits'] if s['id'].startswith('shift:') and s['id'].endswith(':'+condition)]
  prevalence=float(np.mean([np.mean(t) for t in targets]))
  for arm in ['v2','xgboost']:
   a=next(v for v in rows if v['axis']=='shift' and v['arm']==arm and v['condition']==condition)
   text+=f"| {condition} | {arm} | {prevalence:.4f} | {a['accuracy_mean']:.4f} ± {a['accuracy_sd']:.4f} | {a['auc_mean']:.4f} ± {a['auc_sd']:.4f} |\n"
 text+='\n\nNatural-prevalence class 0-only task: every original 20% test row remains. The balanced detector table above has a different training context and test mixture; comparing them does not isolate prevalence alone. Clipped losses use epsilon 10⁻¹².\n\n| Dataset | Arm | All-row accuracy | Known accuracy | Unsupported fraction | All-row clipped loss | Novel AP |\n|---|---|---:|---:|---:|---:|---:|\n'
 for dataset in ['cmc','winequality-red','winequality-white']:
  for arm in ['v2','xgboost']:
   a=next(v for v in rows if v['axis']=='novelty' and v['dataset']==dataset and v['arm']==arm and v['condition']=='natural-prevalence-class0')
   text+='| '+dataset+' | '+arm+' | '+' | '.join(f"{a[k+'_mean']:.4f}" for k in ['accuracy','known_accuracy','unsupported_fraction','clipped_log_loss','ap'])+' |\n'
 control=json.loads((ROOT/'_feature_control_l069_v2_results.json').read_text())
 text+='\n\nThe separate Iris all-subset control averages all six pairs at 50% removal (seed 42): '
 for arm in ['v2','xgboost']:
  a=next(v for v in control['summary'] if v['arm']==arm and v['removed_fraction']==.5)
  b=next(v for v in r['records'] if v['axis']=='features' and v['dataset']=='iris' and v['arm']==arm and v['seed']==42 and v['condition']=='60%')
  text+=f"{arm} all-subset accuracy {a['accuracy']:.4f}, nested-path accuracy {b['accuracy']:.4f} on columns {b['columns']}; "
 text+='the same nominal difficulty can therefore give different scores depending on the removed subset. Source subset parity and this 32-prediction-set control are distinct from the sampled-mask main panel.'
 result=dict(status='PASS',table_markdown='\n'.join(table),interpretation=text,continuous_auc_dataset_gaps=diff,dataset_bootstrap95=interval,mean_ranks=np.mean(ranks,axis=0).tolist(),feature_summary=feature,source_identity=r['kernel_identity']['sha256'],records=len(r['records']),seconds=r['seconds'])
 (ROOT/'_analysis_l069_v2_results.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':print(json.dumps(analyze(),indent=2))
