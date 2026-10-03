"""Build lesson, reference and portable notebooks from shared prose and implementation."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0185-causal-relational-data'
report=json.loads((P/'evidence/l185/report.json').read_text());summary=report['summary']
source=(P/'relkit/causal_l185.py').read_text()
functions={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
captions={'mechanism':'Declared causal graph. Company U and B are shared by twenty customers; A and Y belong to individual customers. Arrows are causal assumptions, not foreign keys. Cutting the badge assignment leaves no causal path from B to Y.',
'adjustment':'Exact population illustration, not measured sample means. Standardization holds the demand mix at 50/50: .5 × (.10 − .05) + .5 × (.95 − .90) = .05.',
'results':'Measured synthetic evidence. Each dot is one of seeds 0–4. Ranking and action effects have different units; a strong AUROC is not an intervention effect. The dotted line marks the known 5-point action effect.'}
rows=[('Badge-only test AUROC','test_badge_auroc',1),('Action-only test AUROC','test_action_auroc',1),('Demand + action test AUROC','test_demand_action_auroc',1),('Observed action gap (points)','naive_action_difference',100),('Adjusted action effect (points)','adjusted_action_effect',100),('Paired action effect (points)','paired_action_effect',100),('Paired badge effect (points)','paired_badge_effect',100)]
result_table='| Quantity | Mean ± seed SD |\n|---|---:|\n'+'\n'.join(f'| {label} | {summary[key]["mean"]*scale:.4f} ± {summary[key]["sd_across_seeds"]*scale:.4f} |' for label,key,scale in rows)
seed_table='| Seed | Badge AUROC | Observed action gap (points) | Adjusted effect (points) | Paired effect (points) |\n|---|---:|---:|---:|---:|\n'+'\n'.join(f'| {r["seed"]} | {r["test_badge_auroc"]:.4f} | {100*r["naive_action_difference"]:.3f} | {100*r["adjusted_action_effect"]:.3f} | {100*r["paired_action_effect"]:.3f} |' for r in report['per_seed'])

def prose(portable=False):
 s=(R/'lessons/content'/(S+'.md')).read_text().replace('[[BADGE_AUC]]',f'{summary["test_badge_auroc"]["mean"]:.3f}').replace('[[POLICY_GAIN]]',f'{100*summary["action1_policy_gain"]["mean"]:.2f}').replace('[[RESULT_TABLE]]',result_table).replace('[[SEED_TABLE]]',seed_table)
 for name,caption in captions.items():
  if portable:
   payload=base64.b64encode((P/'figures/l185'/(name+'.png')).read_bytes()).decode()
   img=f'![{caption}](data:image/png;base64,{payload})\n\n*{caption}*'
  else:img=f'<figure><div class="figure-scroll" tabindex="0" aria-label="Scrollable {name} diagram"><img src="../labs/figures/l185/{name}.svg" alt="{caption}"></div><figcaption>{caption} On narrow screens, scroll the figure horizontally.</figcaption></figure>'
  s=s.replace('[[FIG:'+name+']]',img)
 if portable:
  s=re.sub(r'<div id="(?:warmup|predict|intervention|teachback)"></div>','',s)
  s=s.replace('<noscript>','').replace('</noscript>','')
  s=re.sub(r'(?<=["(])\.\./', 'https://avistian.github.io/relational/',s)
  s=s.replace('](0181-relbench-v2-autocomplete.html)','](https://avistian.github.io/relational/lessons/0181-relbench-v2-autocomplete.html)')
 return s

def doc(title,body,interactive=False):
 html=render(body).replace('<table>','<div class="table-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','causal-intervention','l185-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/causal-intervention.css"></head><body class="causal-page"><article><nav><a href="../index.html">Course</a> · <a href="../notebooks.html">Notebooks</a></nav><p class="eyebrow">Year 5 · Lesson 185 · From prediction to action</p><h1>'+title+'</h1>'+html+'</article>'+''.join('<script src="../assets/'+s+'.js"></script>' for s in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('A good prediction can suggest a useless action',prose(),True))
reference='''**Decision test:** name the outcome, the action and the population. Draw the assumed causal paths. Separate ranking evidence from intervention evidence. Check timing, joins, confounding, overlap and possible spillovers. Then estimate the effect under a defensible design.

| Term | Operational meaning in L185 |
|---|---|
| Conditioning | Select customers already having a value; demand composition may change. |
| Intervention | Replace an assignment rule; retain other causal rules. |
| Confounding | Demand causes both action and purchase. |
| Standardization | Compare action groups within demand strata; average with common population weights. |
| Overlap | Both action states occur in each demand group. |
| Potential outcome | Purchase under a specified action and the same customer's underlying state. |
| ATE | Mean purchase under action1 minus mean purchase under action0. |
| Policy gain | Change from observed assignments to a specified new assignment policy. |
| Interference | One customer's outcome depends on another customer's action; excluded in this simulator. |

**Worked arithmetic:** low demand purchases .05 without action and .10 with it; high demand .90 and .95. Both within-group effects are .05. Observed groups have different demand mixtures, producing .865 − .135 = .73. Standardizing to 50/50 produces .525 − .475 = .05. Badge observations give .9005 versus .0995, but either badge intervention leaves .5.

**Relational guard:** customer → company is many-to-one; customer → action/outcome is one-to-one. Reject duplicate/missing identities. Split at company level. Shared companies imply dependent customers; do not treat all rows as independent uncertainty units.

**Assumption guard:** a foreign key does not establish causal direction. A time-safe feature may still be a noncausal proxy. Adjustment works under the declared causal graph; it is not evidence that every real confounder was measured.

'''+result_table+'''

**Evidence:** complete original five-seed synthetic experiment; mean ± sample SD. No real-world causal or paper-reproduction claim. Learner PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/0185-causal-relational-data.html) · [Student notebook](../labs/0185-causal-relational-data.ipynb) · [Reproduction](../labs/l185-reproduction.md) · [Pearl 2009 §§2–3](https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf).
'''
(R/'reference/causal-relational-data.html').write_text(doc('Causal relational data — quick reference',reference))

def make(solution):
 cells=[]
 def md(s):cells.append(nb.v4.new_markdown_cell(s))
 def code(s,tags=None):cells.append(nb.v4.new_code_cell(s,metadata={'tags':tags or []}))
 md('# Lesson 185 · Causal & relational data\n\n**Tier C: original synthetic mechanism experiment.** Full five-seed run; no paper benchmark. Read → predict → implement → compare → defend. Author-reference tables below are saved evidence, not your kernel output. No network, cloud or dataset download is required.\n\nFirst recall: why does a company-level split matter? Why does a legal feature not necessarily make a useful action?\n\nThis self-contained notebook embeds all source and figures. CPU only; Python 3 with NumPy, pandas and scikit-learn. The exact author environment is listed below. Colab is unverified; portable images are embedded as PNG data URLs.')
 env=json.loads((P/'evidence/l185/environment.json').read_text())
 md('Author versions: Python '+env['python']+'; '+', '.join(f'{k} {v}' for k,v in env['packages'].items())+'. For exact parity use the pinned requirements linked in the protocol. No installation cell runs automatically.')
 code('# @colab-bootstrap — standalone, no clone or installation needed if dependencies are present.\nimport json\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\nfrom sklearn.metrics import roc_auc_score\nSEEDS = tuple(range(5))\nprint("Original synthetic experiment; 5 seeds; cloud/API $0")',['provided'])
 md(prose(True))
 tasks=[('assemble','Task 1 · Preserve the relational query','Implement the complete-key and time-safe assembly contract. Reject duplicated primary keys, missing/extra action or outcome rows, unknown companies, invalid availability, and companies split across groups. Return rows sorted by customer_id and cutoff. The input is a dictionary of the four DataFrames described above.','test_duplicate_and_missing_relationships_rejected'),('intervene','Task 2 · Change a mechanism','Return a NumPy integer outcome array in input order. Accept None, 0 or 1 for badge/action. Replace only requested assignments, retain underlying U and E, and leave input rows unchanged. Explain why badge is not a parent of Y before coding.','test_do_replaces_action_and_reuses_noise'),('adjusted_effect','Task 3 · Compare like with like','Return the demand-standardized action risk difference as a float, using this frame’s demand proportions. Reject missing action overlap in any observed demand stratum. Inputs contain U, A and Y.','test_adjustment_removes_composition_effect')]
 tests=(P/'_test_l185.py').read_text();cls=next(n for n in ast.parse(tests).body if isinstance(n,ast.ClassDef))
 code('import unittest\n'+ast.get_source_segment(tests,cls),['provided','checks'])
 for name,title,goal,check in tasks:
  md('## TODO · '+title+'\n\n'+goal+'\n\n**Why this matters:** this function is called by the complete experiment below. The CHECK uses a small hand-computable counterexample, not your final score.')
  signature=functions[name].split('\n',1)[0]
  code(functions[name] if solution else signature+'\n    # TODO: implement the contract above.\n    raise NotImplementedError("'+name+'")',['solution' if solution else 'todo'])
  code('CausalContracts().'+check+'()\nprint("CHECK passed: '+name+'")',['check'])
 md('## PROVIDED · Generate the normalized database\n\nFour independent random streams separate company traits, assignments, customer thresholds and group splitting. This code calls your intervention to generate observed Y. The company ID carries the shared U/B values into the customer rows; it is not used as a predictor.')
 code(functions['generate'],['provided'])
 md('## PROVIDED · Fit transparent probability tables\n\nThis is the entire predictive model: count outcomes within each permitted feature combination, using training data only. There is no optimizer or checkpoint. Predicting an outcome from A is distinct from choosing A to improve that outcome.')
 code(functions['predict_frequency'],['provided'])
 md('## PROVIDED · Score one seed\n\nFollow validation/test query keys into each predictor. Then follow the test customers into four paired interventions. The adjusted estimate is calculated from held-out observations; E is used only for simulator truth. **Predict first:** which is closer to 5 percentage points, the observed action gap or the adjusted effect?')
 code(functions['run_seed'],['provided'])
 md('## PROVIDED · Aggregate the full experiment\n\nAll five seeds are mandatory. Standard deviation describes variation across complete simulated datasets. It does not turn shared company rows into independent observations.')
 code(functions['full_experiment'],['provided'])
 md('## RUN · Full approved experiment\n\nNo smaller preset is substituted. Save the report generated by your own functions and compare it with the author-reference evidence.')
 code('outputs, report = full_experiment()\nPath("l185-report.json").write_text(json.dumps(report, indent=2)+"\\n")\ncolumns = ["seed", "test_badge_auroc", "naive_action_difference", "adjusted_action_effect", "paired_action_effect", "paired_badge_effect"]\ndisplay(pd.DataFrame(report["per_seed"])[columns])',['run'])
 code('assert report["seeds"] == [0,1,2,3,4]\nfor row in report["per_seed"]:\n    assert row["paired_badge_effect"] == 0\n    assert abs(row["paired_action_effect"]-.05) < .025\n    assert abs(row["adjusted_action_effect"]-.05) < .025\nprint("CHECK: all five seeds and causal bounds passed")',['check'])
 md('## EXIT TICKET · Defend the strategy\n\nExplain, without quoting the lesson: (1) why high badge AUROC and zero badge effect can coexist; (2) why the observed action gap is confounded; (3) why all-treated policy gain is smaller than ATE; (4) how spillovers would change the experiment; (5) what real-world evidence is missing. Paste your explanation and table into the teaching chat.\n\nThese assertions cannot grade your explanation. Learner status stays PENDING_WRITTEN_DEFENSE. There is no unrun paper-results lane hidden behind this experiment: the approved unit is synthetic. A real causal benchmark would require a separately specified task and protocol.')
 code('submission = {"experiment": report["experiment"], "execution": report["status"], "learner": "PENDING_WRITTEN_DEFENSE", "paper_reproduction": "NOT_ESTABLISHED"}\nPath("l185-submission.json").write_text(json.dumps(submission, indent=2)+"\\n")\nprint(submission)',['exit'])
 for i,c in enumerate(cells):c['id']=f'l185-{i:03d}'
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':env['python']}})
 return book
nb.write(make(False),P/(S+'.ipynb'));nb.write(make(True),P/'solutions'/(S+'.ipynb'))
(P/'evidence/l185/report.md').write_text('# L185 observed evidence\n\n'+result_table+'\n\n'+seed_table+'\n\nComplete original synthetic experiment. Paper/real-world causal claims NOT_ESTABLISHED.\n')
env=json.loads((P/'evidence/l185/environment.json').read_text())
(P/'sources/l185/requirements.txt').write_text('\n'.join(k+'=='+v for k,v in env['packages'].items())+'\n')
print('Built lesson, reference, student and solution notebooks')
