"""Build lesson, reference, portable audit ZIP and visible-code notebooks."""
import ast,base64,hashlib,html,io,json,re,zipfile,importlib.metadata
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l197';Q=E/'packet';S='0197-year-5-essay'
r=json.loads((E/'report.json').read_text());p=r['predictions'];g=p['regression']['test'];i=p['icl']['advantage'];ci=g['conditional_interval']
F=P/'figures/l197';F.mkdir(parents=True,exist_ok=True)
# Code-native SVG, vertically stacked routes retain legible text when scrolled on mobile.
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="790" viewBox="0 0 960 790" role="img" aria-labelledby="title desc"><title id="title">Three routes from relational information to a prediction</title><desc id="desc">Graph-native processing uses linked rows. Synthetic-prior RDB-PFN uses generated relational tasks and DFS features. RDBLearn reuses a pretrained tabular predictor on relational summaries. All routes obey query-time information limits.</desc><rect width="960" height="790" rx="22" fill="#f1f7f5"/><style>text{font-family:Arial,sans-serif;fill:#183e47}.title{font-size:27px;font-weight:700}.label{font-size:20px;font-weight:700}.small{font-size:17px}.box{fill:white;stroke:#91b8b2;stroke-width:1.5}</style><text x="35" y="47" class="title">ONE QUERY · THREE DESIGN STRATEGIES</text><text x="35" y="79" class="small">Ada at day 30: known support labels + eligible history → hidden query prediction</text>']
lanes=[(115,'01 · Graph-native learning','#d7ebe7',['Eligible linked rows','Row / cell encoder','Graph mixing','Task prediction'],'Griffin: pretrain → fine-tune. Relational ICL: support conditions frozen prediction.'),
       (310,'02 · Synthetic relational prior','#e4e5f3',['Generated DB tasks','DFS linearization','Pretrain predictor','New-task ICL'],'RDB-PFN: relational structure shapes synthetic experience; inference uses feature rows.'),
       (505,'03 · Reuse a tabular foundation model','#f1e6d8',['Eligible linked rows','Relational summaries','Pretrained backend','New-task ICL'],'RDBLearn: reuse earlier tabular pretraining; no new relational pretraining.')]
for y,title,color,steps,note in lanes:
 svg.append(f'<rect x="24" y="{y}" width="912" height="176" rx="14" fill="{color}"/><text x="44" y="{y+33}" class="label">{title}</text>')
 for j,step in enumerate(steps):
  x=44+j*220;svg.append(f'<rect x="{x}" y="{y+53}" width="205" height="57" rx="9" class="box"/><text x="{x+102}" y="{y+87}" text-anchor="middle" class="small">{step}</text>')
  if j<3:svg.append(f'<path d="M{x+207} {y+80}h10m-5 -5l5 5l-5 5" stroke="#35655e" fill="none" stroke-width="2"/>')
 svg.append(f'<text x="44" y="{y+146}" class="small">{html.escape(note)}</text>')
svg+=['<text x="35" y="725" class="label">Keep three questions separate</text><text x="35" y="757" class="small">Representation: what is seen? · Prior: what was learned? · Adaptation: what changes?</text></svg>']
svg=''.join(svg);(F/'landscape.svg').write_text(svg)
figure='<figure class="landscape-figure"><div class="figure-scroll" tabindex="0" role="region" aria-label="Scrollable architecture map"><img src="../labs/figures/l197/landscape.svg" alt="Three routes: graph-native encoding and mixing; synthetic tasks and DFS-trained predictor; relational summaries and a reused tabular model."></div><figcaption>Same query, different representation and prior. Strategies can overlap; arrows are conceptual stages, not measured latency.</figcaption></figure>'
status='<div class="landscape-status"><strong>Complete selected evidence audit.</strong> 491 published numeric cells; 33,650 saved predictions; all 21 fresh-result slots remain unrun. Full model reproduction: <code>INCOMPLETE_SOURCE_PREPROCESSING_GATE</code>. Your essay: <code>PENDING_WRITTEN_DEFENSE</code>.</div>'
gaps='| Table / task count | Metric units | Best foundation gap | Best combined-pool gap | Combined taskwise oracle gap |\n|---|---|---|---|---|\n'
for t in r['tables']['tables']:
 def fmt(pool):
  x=t['pools'][pool]
  return 'No eligible comparator' if x['status']=='NO_ELIGIBLE_COMPARATOR' else ', '.join(x['single_methods'])+f": {x['single_gap']:.4f}"
 units='AUROC percentage points' if t['metric']=='AUROC' else 'Normalized MAE ratio'
 gaps+=f"| {t['number']} / {len(t['tasks'])} | {units} | {fmt('foundation')} | {fmt('all')} | {t['pools']['all']['oracle_gap']:.4f} |\n"
replay=f'''| Saved comparison | Complete measurement | What it supports |
|---|---|---|
| L149, five seeds per pipeline | Test GNN MAE {g['gnn_mean']:.4f}; relational-FE MAE {g['fe_mean']:.4f}; GNN advantage {g['gnn_advantage']:+.4f} | FE has lower mean loss in this saved comparison. |
| L149 conditional uncertainty | 95% driver interval [{ci['low']:.4f}, {ci['high']:.4f}] | Neither superiority nor equivalence established. |
| L182, three arms × ten draws | RDB-PFN minus TabICL {i['mean']:+.6f} AUROC; positive on {i['positive']}/10 draws | Scoped saved-system comparison on one task. |
| L194, 21 declared tasks | 0 fresh tasks; published signs 17 / 3 / 1 | Complete inventory of missing fresh evidence. |
'''
rubric='''| Dimension | Ready for review when… | Revise when… |
|---|---|---|
| Mechanism | Same query traced; representation, prior and adaptation separated | Treating the strategies as disjoint or equating frozen weights with no label access |
| Comparison | Version, pool, task coverage and metric are explicit | Cherry-picking tasks or silently changing the eligible pool |
| Evidence | Published, replayed and fresh lanes are distinguished | Promoting missing scores, replay or a source diagnostic to new model results |
| Argument | Evidence supports the actual claim through an explicit warrant | Jumping from one pipeline comparison to all architectures or economic value |
| Counterargument | Strongest objection and uncertainty are addressed fairly | Treating a zero-crossing interval as proof of equivalence |
| Revision | A feasible comparison and result would change the conclusion | Proposing a study that cannot challenge the preferred answer |
'''
essay=(E/'worked-essay.md').read_text();essay_embed=essay.replace('](report.json)','](../labs/evidence/l197/report.json)').replace('](../l195/falsification-brief.md)','](../labs/evidence/l195/falsification-brief.md)')
prose=(R/'lessons/content'/(S+'.md')).read_text()
for key,value in dict(STATUS=status,FIGURE=figure,GAPS=gaps,REPLAY=replay,RUBRIC=rubric,ESSAY=essay_embed).items():prose=prose.replace('[['+key+']]',value)
def doc(title,body,scripts=False):
 body=body.replace('<table>','<div class="repro-table" tabindex="0" role="region" aria-label="Scrollable evidence table"><table>').replace('</table>','</table></div>')
 tags=''.join('<script src="../assets/'+x+'.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','teachback','arch-family-viz','landscape-essay']) if scripts else ''
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/landscape-essay.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html">Curriculum</a></nav>'+body+'</article>'+tags+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Lesson 197 — Year 5 landscape essay',render(prose),True))
reference='''# The landscape essay field guide

**Five argument parts:** claim → evidence → warrant (why it supports the claim) → limitation → revision condition.

**Three strategy questions:** where does relational information enter; where does prior experience come from; what changes on the new task? Graph-native processing, synthetic relational pretraining and reuse of a tabular predictor can overlap. Training-free deployment does not mean the backend was never trained.

**Five openness axes:** implementation, weights, pretraining data, license, runnable protocol. Inspect each separately. Code access is not historical reproduction.

'''+gaps+'\n'+replay+'''
**Do not cross these boundaries:** published tables → published comparison; saved predictions → scoped replay; source diagnostic → implementation observation. None automatically proves architectural cause, new model reproduction or economic undervaluation. No eligible comparator is not a win. Missing scores are not zero. A zero-crossing interval is not equivalence.

**A fair next study:** untouched tasks, fixed temporal availability, equally informed comparators, validation-only selection, declared budget and practically meaningful margin, appropriate uncertainty unit, and total human/compute/serving cost.

'''+rubric+'''
[Lesson](../lessons/0197-year-5-essay.html) · [Protocol](../labs/l197-reproduction.md) · [Worked essay](../labs/evidence/l197/worked-essay.md) · [Blank template](../labs/evidence/l197/essay-template.md) · [Complete report](../labs/evidence/l197/report.json).

Primary sources: [KumoRFM-2 v1](https://arxiv.org/html/2604.12596v1), [Griffin v1](https://arxiv.org/html/2505.05568v1), [RDB-PFN v5](https://arxiv.org/html/2603.03805v5), [RDBLearn v1](https://arxiv.org/html/2602.18495v1). Learner PENDING_WRITTEN_DEFENSE; whole-paper reproduction and new model runs not established by this audit.
'''
(R/'reference/year-5-essay.html').write_text(doc('Year 5 landscape essay field guide',render(reference)))
requirements='\n'.join(name+'=='+importlib.metadata.version(name) for name in ['numpy','scipy','beautifulsoup4'])+'\n'
(E/'requirements-audit.txt').write_text(requirements)
readme='''# L197 complete landscape evidence audit

Python 3.12 recommended. CPU; no model weights or external API calls. From this extracted directory:

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements-audit.txt
    .venv/bin/python labs/_budget_l197.py .venv/bin/python labs/_audit_l197.py
    .venv/bin/python labs/_budget_l197.py .venv/bin/python labs/_verify_l197.py

The budget wrapper caps all its runs together at 1800 seconds. Installation is environment preparation; the author's preparation is included in the repository ledger. Account your own preparation time separately. The audit reconstructs the full selected tables and rescored predictions. Fresh model reproduction remains incomplete. All inputs and original sources are hash checked. Independent verifiers run in temporary copies. Output: labs/evidence/l197/report.json and labs/_verify_l197_results.json. This is replay, not an independent replication.
'''
entries={'README.md':readme.encode(),'requirements-audit.txt':requirements.encode()}
for name in ['_audit_l197.py','_verify_l197.py','_test_l197.py','_budget_l197.py','relkit/landscape_l197.py']:
 entries['labs/'+name]=(P/name).read_bytes()
entries['labs/relkit/__init__.py']=b''
for path in sorted(Q.rglob('*')):
 if path.is_file() and '__pycache__' not in str(path):entries['labs/evidence/l197/packet/'+str(path.relative_to(Q))]=path.read_bytes()
for name in ['input-manifest.json','protocol.json','report.json']:
 entries['labs/evidence/l197/'+name]=(E/name).read_bytes()
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for name,data in sorted(entries.items()):
  info=zipfile.ZipInfo(name,(2026,10,2,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
payload=buf.getvalue();(E/'reproducer.zip').write_bytes(payload)
def code_source(path):
 source=path.read_text();nodes=ast.parse(source).body
 return '\n\n'.join(ast.get_source_segment(source,n) for n in nodes if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef)) and not (isinstance(n,ast.ImportFrom) and (n.module or '').startswith('relkit')))
learner=(P/'relkit/landscape_l197.py').read_text();funcs={n.name:ast.get_source_segment(learner,n) for n in ast.parse(learner).body if isinstance(n,ast.FunctionDef)}
contracts=[('coverage_report','expected, rows','Require exactly the unique declared task IDs. Each row has task, status, score. NOT_RUN must have score=None; MEASURED must have a finite non-boolean number. Reject missing/duplicate/extra IDs and unknown statuses. Return declared count, measured count and missing IDs in declared order.'),('admit_claim','lane, claim, authenticated, complete','Implement the explicit packet policy: published_table admits published_comparison; saved_predictions admits scoped_pipeline_comparison; source_diagnostic admits implementation_observation. Admission needs both flags exactly True. Other known claims (fresh_model_reproduction, architecture_cause, economic_undervaluation) return NOT_ESTABLISHED. Unknown lanes/claims or non-boolean flags raise ValueError. Return ADMISSIBLE_SCOPED only when allowed.'),('essay_readiness','sections','Require textual claim/evidence/warrant/limitation/revision fields. Return missing whitespace-only fields in that order, state DRAFT if any are missing else READY_FOR_REVIEW, and mastery PENDING_WRITTEN_DEFENSE in both cases. This tests structure only.')]
portable=re.sub(r'<div id="[^"]+"></div>','',prose)
portable=portable.replace(figure,'![Three strategy routes](data:image/svg+xml;base64,'+base64.b64encode(svg.encode()).decode()+')')
portable=portable.replace('](../','](https://avistian.github.io/relational/')
portable=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',portable)
# Keep worked essay for the notebook exit, after learner exercises.
portable=portable.split('<details><summary>Read the author')[0]
def notebook(solution):
 cells=[]
 def md(s):cells.append(nb.v4.new_markdown_cell(s))
 def code(s,tags=None):cells.append(nb.v4.new_code_cell(s,metadata={'tags':tags or []}))
 md('# Lesson 197 · The Year 5 landscape essay\n\nComplete saved-evidence audit, followed by your written defense. No training, checkpoint inference or API calls. Independent verification is included. The author solution is not your mastery record.')
 code('# @colab-bootstrap\nimport ast, base64, hashlib, io, json, math, os, subprocess, sys, tempfile, zipfile, importlib.util\nfrom pathlib import Path\nneeded = '+repr(requirements.splitlines())+'\nmodules = ["numpy", "scipy", "bs4"]\nif any(importlib.util.find_spec(m) is None for m in modules):\n    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *needed], check=True, timeout=300)\nimport numpy as np\nworkspace = Path(tempfile.mkdtemp(prefix="l197-landscape-"))')
 md(portable)
 code('payload = base64.b64decode('+repr(base64.b64encode(payload).decode())+')\nassert hashlib.sha256(payload).hexdigest() == '+repr(hashlib.sha256(payload).hexdigest())+'\nwith zipfile.ZipFile(io.BytesIO(payload)) as archive:\n    archive.extractall(workspace)\nevidence = workspace / "labs/evidence/l197"\nprint("Authenticated complete packet:", len(payload), "bytes")',['data-payload'])
 for name,args,contract in contracts:
  md('## TODO · '+name+'\n\n'+contract)
  code(funcs[name] if solution else 'def '+name+'('+args+'):\n    raise NotImplementedError("'+name+'")')
 md('## CHECK · Evidence and writing contracts\n\nThese reject incomplete task inventories, unsupported evidence promotion and false mastery states.')
 code(code_source(P/'_test_l197.py')+'\nprint(check(coverage_report, admit_claim, essay_readiness))')
 for name in ['relkit/tracking_l191.py','_replay_l191.py','relkit/stress_l195.py','_replay_l195.py']:
  md('## PROVIDED · '+name+'\n\nExact frozen source functions are visible below. Inspect the metric direction, complete keys, task coverage and validation selection checks. These execute here; no hidden course helper supplies their results.')
  code(code_source(Q/name))
 md('## PROVIDED · Assemble the complete audit using your functions')
 code(code_source(P/'_audit_l197.py')+'\nreport = audit197(evidence, replay191, replay195, paired_mae, interval_verdict, claim_scope, keyed_auc, cluster_interval, coverage_report, admit_claim)\nassert report == json.loads((evidence / "report.json").read_text())\nPath("l197-report.json").write_text(json.dumps(report, indent=2))\nprint(report["status"], report["tables"]["task_cells"], report["predictions"]["prediction_rows"], report["coverage"])')
 md('## CHECK · Independent arithmetic\n\nThe embedded independent verifiers use rational table arithmetic, SQL joins, pairwise AUROC and explicit driver bootstrap blocks. They execute in a disposable copy of the same frozen packet, not the course repository.')
 code('subprocess.run([sys.executable, str(workspace / "labs/_budget_l197.py"), sys.executable, str(workspace / "labs/_verify_l197.py")], check=True, timeout=300)\nverification = json.loads((workspace / "labs/_verify_l197_results.json").read_text())\nassert verification["status"] == "PASS"\nPath("l197-verification.json").write_text(json.dumps(verification, indent=2))\nprint(verification)')
 md('## EXIT · Write your defense before reading the author example\n\nDraft 800–1,200 words using the six-dimensional rubric. Fill the five argument fields below with one paragraph from your draft. The readiness function checks presence only. Paste the essay to the agent for human review; incomplete prior exit requirements stay incomplete.')
 code('sections = {"claim": "", "evidence": "", "warrant": "", "limitation": "", "revision": ""}\nsubmission = {"sections": sections, "review": essay_readiness(sections), "essay": ""}\nPath("l197-submission.json").write_text(json.dumps(submission, indent=2))\nprint(submission["review"])')
 md('## Author reference essay — compare only after your own draft\n\n'+essay.replace('](report.json)','](https://avistian.github.io/relational/labs/evidence/l197/report.json)').replace('](../l195/falsification-brief.md)','](https://avistian.github.io/relational/labs/evidence/l195/falsification-brief.md)'))
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
 for idx,cell in enumerate(book.cells):cell.id=('solution' if solution else 'student')+'-l197-'+str(idx)
 return book
nb.write(notebook(False),P/(S+'.ipynb'));nb.write(notebook(True),P/'solutions'/(S+'.ipynb'))
print('Built lesson, reference, SVG, ZIP, student and solution')
