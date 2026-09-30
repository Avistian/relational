"""Build L147 lesson, reference and portable audit notebooks from canonical sources."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l147';S='0147-next-generation-architectures';TITLE='Next-generation architectures: turn gaps into experiments'
summary=json.loads((E/'summary.json').read_text());questions=json.loads((E/'questions.json').read_text())
results='| Seed | GNN validation | RelGT validation | GNN test | RelGT test |\n|---|---:|---:|---:|---:|\n'
for seed in [0,1,2]:
 a=next(r for r in summary['runs'] if r['arm']=='gnn' and r['seed']==seed);b=next(r for r in summary['runs'] if r['arm']=='relgt' and r['seed']==seed)
 results+=f"| {seed} | {a['val']:.6f} | {b['val']:.6f} | {a['test']:.6f} | {b['test']:.6f} |\n"
results+='\nFresh reanalysis of **7,554 reused L146 predictions**. Paired validation difference **+0.149340 ± 0.125959**; test **−0.323904 ± 0.253784** MAE (mean ± sample seed SD). No fresh training.\n'
captions={'map':'Four distinct evidence boundaries: legal context, representation, prediction and transfer. Conceptual course synthesis, not a measured ranking.','evidence':'Every point is one paired seed difference from saved L146 predictions, independently rescored in L147. SD describes three seeds, not uncertainty across databases.','ranking':'Declared planning assumptions produce different feasible sets. An ordering is conditional on those judgments; it is not a measured probability of research success.'}
fallbacks={'WARMUP':'Recall: query identity includes entity and cutoff; attention cannot read excluded attributes; seeds do not create new databases.','PREDICT':'Before reading: does reproducing six fits on F1 establish cross-database transfer? No; the database never changes.','MAP_WIDGET':'Baseline at four hours: selection → ownership → routes. At one hour only selection remains. At eight hours coverage enters. Transfer stays blocked because required inputs and protocol are missing.','TEACHBACK':'Write: choose one unresolved mechanism, a minimal falsifiable test and a claim your current evidence cannot support. Ask the teaching agent for feedback.'}
def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l147/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l147/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="research-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption} Scroll horizontally for detail on a narrow screen.</figcaption></figure>')
 for name,fallback in fallbacks.items():
  s=s.replace('[['+name+']]',fallback if portable else f'<div id="{name.lower()}"></div><noscript>{fallback}</noscript>')
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def document(title,body,interactive=False):
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','research-map','l147-lesson'] if interactive else []
 html=render(body).replace('<table>','<div class="research-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/research-map.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../notebooks.html">Labs</a></nav><header><p class="eyebrow">Year 4 · Quarter 3 · Lesson 147</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join(f'<script src="../assets/{s}.js"></script>' for s in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
ref='''## Turn an open problem into a falsifiable test

| Boundary | Ask | Minimum evidence |
|---|---|---|
| Access | Which rows can this query see? | Query-specific cutoff and coverage audit |
| Representation | Which distinctions survive? | A controlled counterexample tied to a target |
| Prediction | Which complete procedure performs better? | Fixed protocol, full keyed targets, paired seeds |
| Transfer | Do weights help on an unseen database? | Database-level holdout and matched adaptation baseline |

## Ranking contract
First reject unavailable inputs and over-limit effort/cost. Then sort by declared evidence level (descending), hours (ascending), stable ID. Evidence levels: 3 local empirical audit, 2 controlled example, 1 mechanism argument, 0 untested transfer proposal. These are judgments, not measured success probabilities. Each candidate is considered separately.

## Scoring contract
Require unique complete (entity, cutoff) keys on both sides. Check targets against the reference after key alignment. Reject nonfinite numbers. Compute query-level absolute losses, average within each seed, subtract paired model MAEs, then report the mean and sample SD over seeds. Never use row order or entity alone as identity.

## Verified reference evidence
'''+results+'''
## Evidence boundary
L147 reuses six L146 fitted models' predictions. It does not retrain, independently rebuild raw labels, isolate architecture effects or test database transfer. Full RelGT selected reproduction remains INCOMPLETE; whole paper NOT_RUN; historical identity NOT_ESTABLISHED. No paid compute.

[Lesson](../lessons/0147-next-generation-architectures.html) · [Protocol](../labs/l147-reproduction.md) · [Question ledger](../labs/evidence/l147/questions.json) · [Primary survey](https://arxiv.org/html/2506.16654v1). Ask the teaching agent to review your falsifier.
'''
(R/'reference/next-generation-architectures.html').write_text(document('Research-map reference',ref))
js='ResearchMap.mount(document.getElementById("map_widget"),'+json.dumps(questions)+');\n'
js+='''RetrievalBank.mount(document.getElementById('warmup'),{upTo:147,count:3});
Predict.mount(document.getElementById('predict'),{prompt:'Six saved fits on F1 can be reproduced exactly. Does that establish transfer to unseen databases?',options:[{label:'It establishes cross-database transfer',value:'transfer'},{label:'It establishes within-task repeatability',value:'repeat'}],correct:'repeat',reveal:'Repeating seeds leaves the database fixed. This audit establishes reproducible scoring of saved within-task evidence; it does not train or test transferred weights.'});
Teachback.mount(document.getElementById('teachback'),{prompt:'Choose one research question. State its mechanism, minimal test, falsifier and one unsupported claim. Explain how the test could change your next decision.',points:['Connect the hypothesis to a concrete information boundary.','Name the fixed quantities and a result that would refute the claim.','Separate reused measurements from new experiments.','Treat effort and evidence levels as revisable judgments.','A single database cannot establish cross-database transfer.'],model:'I would audit context coverage before changing attention depth. Hold queries, graph and cutoff rules fixed, then measure whether the allegedly missing rows already occur in the legal sampled contexts. If they do, missing access cannot explain those cases. This is a diagnostic motivated by existing F1 results, not evidence that a new architecture or foundation model will win.'});
'''
(R/'assets/l147-lesson.js').write_text(js)

def functions(path):
 s=path.read_text();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
fns=functions(P/'relkit/survey_l147.py');checks=functions(P/'_check_l147.py')
bootstrap='''# @colab-bootstrap: self-contained CPU audit; no repository or GPU required.
import sys,subprocess
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable,'-m','pip','install','-q','numpy==1.26.4'])
import base64,hashlib,io,json,math
from pathlib import Path
import numpy as np
from IPython.display import display,Markdown
'''
raw=(E/'audit-inputs.npz').read_bytes();payload="# PROVIDED: immutable author inputs, not newly trained predictions.\nraw=base64.b64decode("+repr(base64.b64encode(raw).decode())+")\nassert hashlib.sha256(raw).hexdigest()=="+repr(hashlib.sha256(raw).hexdigest())+"\nDATA=np.load(io.BytesIO(raw),allow_pickle=False)\nQUESTIONS=json.loads("+repr(json.dumps(questions))+")\nAUTHOR_SUMMARY=json.loads("+repr(json.dumps(summary))+")"
audit='''# CHECK: your live functions rescore every saved prediction after shuffling rows.
means={s:{a:{} for a in ['gnn','relgt']} for s in ['val','test']}
verified=0;rows=[]
for arm in ['gnn','relgt']:
    for seed in [0,1,2]:
        row={'arm':arm,'seed':seed}
        for split in ['val','test']:
            key=f'{arm}_{seed}_{split}_';ref=f'ref_{split}_'
            n=len(DATA[ref+'target']);order=np.random.default_rng(147+seed).permutation(n)
            losses=keyed_losses(list(zip(DATA[ref+'entity'],DATA[ref+'cutoff'])),DATA[ref+'target'],
                list(zip(DATA[key+'entity'][order],DATA[key+'cutoff'][order])),
                DATA[key+'target'][order],DATA[key+'pred'][order])
            means[split][arm][seed]=float(losses.mean());row[split]=float(losses.mean());verified+=n
        rows.append(row)
paired={s:paired_summary(means[s]['gnn'],means[s]['relgt']) for s in means}
assert verified==7554
for split in paired:
    np.testing.assert_allclose(paired[split]['differences'],AUTHOR_SUMMARY['paired'][split]['differences'],rtol=0,atol=1e-12)
table='| Arm | Seed | Validation MAE | Test MAE |\\n|---|---:|---:|---:|\\n'
for r in rows:table+=f"| {r['arm']} | {r['seed']} | {r['val']:.6f} | {r['test']:.6f} |\\n"
display(Markdown(table.replace('\\n','\n')))
for split,d in paired.items():print(split, 'GNN minus RelGT:',round(d['mean'],6),'+/-',round(d['sample_sd'],6),'sample seed SD')
print('Freshly rescored reused predictions:',verified)
'''
# Avoid escape ambiguity: the visible cell uses ordinary escaped newlines.
audit=audit.replace("replace('\\n','\n')","replace('\\\\n','\\n')")
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson 147 · '+TITLE+'\n\n'+('Reference solution' if solution else 'Student lab')+' · CPU only · PROVIDED / TODO / CHECK / EXIT. Complete visible evidence contracts, no new model. Tier B: saved real rel-f1 predictions, not synthetic benchmark data. Small synthetic fixtures test failure handling only. Three functions you write are called on all 7,554 held-out predictions or on the research map.'),nb.v4.new_code_cell(bootstrap)]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if section.strip():cells.append(nb.v4.new_markdown_cell(section))
 c=nb.v4.new_code_cell(payload);c.metadata['tags']=['data-payload'];cells.append(c)
 for name,check,explanation in [('keyed_losses','check_keyed_losses','Align full keys, reject duplicate/missing/extra identities, nonfinite values and target disagreements, then return absolute losses in reference order.'),('paired_summary','check_paired_summary','Require matching sets of at least two seeds. Return sorted seeds, left-minus-right differences, their mean and sample SD (ddof=1).'),('rank_questions','check_rank_questions','Validate planning inputs; filter readiness and limits; order by evidence descending, hours ascending, then ID. Return IDs. Reject duplicate IDs and invalid costs.')]:
  cells.append(nb.v4.new_markdown_cell('## TODO · `'+name+'`\n\n'+explanation+'\n\nPredict the first fixture result before running the CHECK. This is the live function used below.'))
  cells.append(nb.v4.new_code_cell(fns[name] if solution else fns[name].splitlines()[0]+'\n    raise NotImplementedError("TODO: '+name+'")'))
  cells.append(nb.v4.new_code_cell(checks[check]+'\n'+check+'('+name+')\nprint("PASS: '+name+'")'))
 cells.extend([nb.v4.new_markdown_cell('## CHECK · full evidence reanalysis\n\nTargets come from saved L146 preparation. Raw database labels are not regenerated here. Predictions are shuffled deliberately; scores must remain unchanged. Original input hashes and source protocol are linked in the lesson.'),nb.v4.new_code_cell(audit)])
 cells.append(nb.v4.new_markdown_cell('## CHECK · change an assumption, not a measured score\n\nPredict what happens if ownership has evidence level 1 instead of 3. The sensitivity case below is hypothetical; it does not rewrite the author ledger.'))
 cells.append(nb.v4.new_code_cell("baseline=rank_questions(QUESTIONS,4,10)\nassert baseline==['selection','ownership','routes']\nchanged=[dict(q,evidence=1) if q['id']=='ownership' else dict(q) for q in QUESTIONS]\nassert rank_questions(changed,4,10)==['selection','routes','ownership']\nassert 'transfer' not in rank_questions(QUESTIONS,40,10)\nprint('Baseline:',baseline)\nprint('Changed judgment:',rank_questions(changed,4,10))\nMY_QUESTIONS=[dict(q) for q in QUESTIONS]  # Revise assumptions and justify them in your EXIT.\nPath('l147-open-problems.json').write_text(json.dumps(MY_QUESTIONS,indent=2))"))
 cells.append(nb.v4.new_markdown_cell('## EXIT · your research defense\n\nWrite 150–250 words covering a hypothesis, minimal test, fixed quantities, falsifier, cost assumptions and unsupported claim. Modify `MY_QUESTIONS` and rerun its export when needed. Submit the map and defense to the teaching agent. Automated checks do not grade your scientific judgment.'))
 cells.append(nb.v4.new_code_cell("report=dict(status='PASS',verified_author_predictions=verified,paired=paired,ranking=baseline,cloud_spend_usd=0,fresh_training='NOT_RUN',full_selected_reproduction='INCOMPLETE',learner='PENDING_WRITTEN_DEFENSE')\nPath('l147-report.json').write_text(json.dumps(report,indent=2))\nprint('Saved evidence report and open-problems map; written defense remains pending.')"))
 cells.append(nb.v4.new_markdown_cell('## Paper-results lane · retained without launching compute\n\nThe survey has no new training experiment. Named retained lane: RelGT v1 Table 1, F1 driver-position. [Complete original model](https://avistian.github.io/relational/labs/relkit/relgt_l145.py), [complete trainer](https://avistian.github.io/relational/labs/_full_l145.py), [source/data/optimizer/selection/seed ledger](https://avistian.github.io/relational/labs/l145-reproduction.md), [safe preflight](https://avistian.github.io/relational/labs/_reproduce_l146.py). From the repository run `.venv/bin/python labs/_reproduce_l146.py --audit`. The full `.venv/bin/modal run modal/l145_repro.py::full` command is guarded and currently stops: source temporal FAIL and complete search exceeds the standing budget. No training call or forensic override is hidden in this notebook. Nine 100-epoch fits NOT_RUN; historical identity NOT_ESTABLISHED. L147 audit completion does not change these statuses.'))
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
 for i,c in enumerate(cells):c.id=hashlib.sha256((str(i)+c.source).encode()).hexdigest()[:12]
 path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
  old=nb.read(path,4)
  if [c.source for c in old.cells]==[c.source for c in cells]:notebook=old
 nb.write(notebook,path)
print('Built Lesson 147, reference and two portable notebooks')
