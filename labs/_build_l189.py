"""Deterministic lesson/reference/portable lab builder. No numerical model runs."""
import ast,base64,hashlib,html,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l189';Q=E/'packet';S='0189-identify-open-problems'
r=json.loads((E/'report.json').read_text());cases=json.loads((Q/'cases.json').read_text());sources=json.loads((Q/'sources.json').read_text());manifest=json.loads((E/'input-manifest.json').read_text())
status='**Author evidence:** `COMPLETE_SELECTED_AUDIT` — 3 case files, 6 primary sources, 6 inherited receipts, all 27 weight settings. Prediction replay, fresh training and whole-paper reproduction: `NOT_RUN`. Novelty and complete literature coverage: `NOT_ESTABLISHED`. Learner: `PENDING_WRITTEN_DEFENSE`.'
results='| Rank | Question | Equal-weight priority | Full-run cost |\n|---:|---|---:|---|\n'
for row in r['ranking']:
 c=next(c for c in cases if c['id']==row['id']);results+=f"| {row['rank']} | {c['title']} | {row['score']:.2f} | {row['cost_gate']['status']} |\n"
results+='\nThese are authored priorities reproduced by code, not measured model performance. [Frozen audit report](../labs/evidence/l189/report.json) · [Input hashes](../labs/evidence/l189/input-manifest.json).'
def case_md(c):
 text='### '+c['title']+'\n\n'
 for field,label in [('question','Question'),('hypothesis','Hypothesis'),('known','Already known'),('gap','Candidate gap'),('approach','Approach'),('baseline','Matched control'),('minimum_design','Minimum full comparison — proposed, unrun'),('estimand','Quantity to estimate'),('failure','Falsifier and limitation')]:text+='**'+label+'.** '+c[field]+'\n\n'
 text+='**Related work.** '+', '.join('['+s['id']+']('+s['url']+')' for s in sources if s['id'] in c['related_work'])+'.\n\n'
 text+='**Inherited receipts.** '+', '.join('['+n+'](../labs/evidence/l189/packet/inherited/'+n+')' for n in c['evidence'])+'. Reports are authenticated; underlying predictions are not replayed here.\n\n'
 text+='**Authored feasibility rationale.** '+' '.join(c['feasibility_reasons'])+'\n\n'
 text+=f"**Human effort estimate:** {c['human_hours'][0]}–{c['human_hours'][1]} hours to prepare the protocol. **Full-run cost:** unverified; all six phases unknown. **Next decision:** {c['next_step']}\n\n"
 return text
captions={'trace':'Worked planning example grounded in temporal pretraining §4.4: a limitation becomes a held-out-database question, then a test and an explicit evidence boundary. No experiment is executed by this diagram.', 'ranking':'All 27 rubric-weight settings using the authored scores. Diamonds show equal weights. The values measure research priorities under assumptions, not model quality or probability of success.'}
def prose(portable=False):
 text=(R/'lessons/content'/(S+'.md')).read_text().replace('[[STATUS]]',status).replace('[[RESULTS]]',results)
 for key,caption in captions.items():
  url='data:image/png;base64,'+base64.b64encode((P/'figures/l189'/(key+'.png')).read_bytes()).decode() if portable else '../labs/figures/l189/'+key+'.svg'
  picture=f'<img src="{url}" alt="{caption}">'
  if key=='trace' and not portable:picture='<picture><source media="(max-width:500px)" srcset="../labs/figures/l189/trace-mobile.svg">'+picture+'</picture>'
  text=text.replace('[[FIG:'+key+']]',f'<figure class="route-figure" tabindex="0">{picture}<figcaption>{caption}</figcaption></figure>')
 widgets={'PREDICT':('Predict: does a 2025 survey establish that a 2026 proposal is new? Answer: no; it supplies leads, not complete current coverage.','<div id="predict"></div>'), 'PRIORITY':('Recalculate after changing temporal impact from 4 to 3: temporal 14, composite 15, transfer 10. Composite leads.','<div id="priority"></div><noscript>Equal weights: temporal 18.67, composite 15, transfer 10. At temporal impact 3: temporal 14, composite 15, transfer 10.</noscript>'), 'CHECKLIST':('Check: nearest work; narrow question; matched baseline; full cost; falsifier; next cheap decision.','<div id="checklist"></div><noscript>Check the related work, hypothesis, matched controls, full cost and evidence boundaries.</noscript>'), 'TEACHBACK':('Write a defense of the first research decision. State the nearest work, useful effect, information contract, full cost, falsifier and uncertainty.','<div id="teachback"></div>')}
 for key,(plain,widget) in widgets.items():text=text.replace('[['+key+']]',plain if portable else widget)
 cards='\n\n'.join(case_md(c) for c in cases) if portable else '\n\n'.join('<details><summary>'+html.escape(c['title'])+'</summary>'+render(case_md(c))+'</details>' for c in cases)
 text=text.replace('[[CASES]]',cards)
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/')
  text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def doc(title,body,interactive=False):
 markup=render(body).replace('<table>','<div class="route-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 scripts=['predict','checklist','teachback','research-priority','l189-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 189 — '+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+s+'.css">' for s in ['lesson','atomic-route','checkpoint','research-priority'])+'</head><body class="checkpoint rp-lesson"><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html">Curriculum</a></nav><header><p class="route-kicker">Year 5 · Quarter 3 · Lesson 189</p><h1>'+title+'</h1><p class="subtitle">From a promising topic to a defensible next experiment</p></header>'+markup+'</article>'+''.join('<script src="../assets/'+s+'.js"></script>' for s in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Identify three open problems',prose(),True))
reference='''## Research question contract

Specify the closest related work; what remains unknown; the population; the intervention; the matched baseline; the estimand; a useful effect; an outcome that refutes it; and an inexpensive next decision. A source limitation is a lead, not proof of novelty. A software fix is not automatically a research contribution.

## Priority and feasibility

Impact × weighted mean(data access, implementation, compute), each scored 1–5. Weights are positive integers. The default is (1,1,1); the complete sensitivity sweep is {1,2,3}³. Keep ties. These scores are authored judgments about the next useful decision, not probabilities or full-experiment affordability.

'''+results+'''

## Cost gate

Include preparation, training, selection, evaluation, retries and validation. Any unknown phase, unfinished protocol or unverified bound → NOT_ESTABLISHED. Failed source audit → BLOCKED. Total above USD10 → OVER_CAP. A complete valid bound within USD10 → WITHIN_CAP; that is arithmetic feasibility, not user authorization. Human effort is estimated separately.

## Evidence ladder

Source states a limitation → candidate question. Frozen report hashes → byte identity. Executed audit → verified contracts and arithmetic. Fresh matched experiment → evidence about its measured population. None of these alone establishes novelty or mastery.

## Worked decision

Temporal: 4×(5+4+5)/3 = 18.67. Composite: 5×(4+3+2)/3 = 15.00. Transfer: 5×(3+2+1)/3 = 10.00. Lower temporal impact to 3 and its score becomes 14. Ranking robustness to weights does not establish robust judgments.

## Status and reading

'''+status+'\n\n[Lesson](../lessons/'+S+'.html) · [Student notebook](../labs/'+S+'.ipynb) · [Protocol](../labs/l189-reproduction.md) · [Survey](https://arxiv.org/html/2506.16654v1) · [Temporal pretraining limitations](https://arxiv.org/html/2609.35219v1#S4.SS4).'
(R/'reference/identify-open-problems.html').write_text(doc('Research-question selection — reference',reference))
(E/'shortlist.md').write_text(('# L189 ranked research shortlist\n\n'+status+'\n\n'+results+'\n\n'+'\n\n'.join(case_md(c) for c in cases)).replace('(../labs/','(../../'))
(E/'report.md').write_text(('# L189 audit results\n\n'+status+'\n\n'+results+'\n\nTemporal leads in all 27 declared weight settings; all full-run costs remain unverified.\n').replace('(../labs/','(../../'))

def defs(path):return {n.name:ast.get_source_segment(path.read_text(),n) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
functions=defs(P/'relkit/gaps_l189.py');auditor=defs(P/'_audit_l189.py')['audit'];checks=defs(P/'_test_l189.py')['checks']
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for name in sorted(manifest['files']):
  info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,(Q/name).read_bytes())
raw=buf.getvalue();encoded=base64.b64encode(raw).decode();digest=hashlib.sha256(raw).hexdigest()
def make(solution):
 cells=[]
 def md(s):cells.append(nb.v4.new_markdown_cell(s))
 def code(s,tags=None):cells.append(nb.v4.new_code_cell(s,metadata={'tags':tags or []}))
 md('# Lesson 189 · Identify three open problems\n\nResearch-decision lab (the curriculum’s Tier C synthesis scope). Source documents and reports are real; rubric ratings are authored judgments. The inputs are six selected primary documents, six inherited receipts and three authored research cards—not training data. Python 3 standard library only; all bytes and figures are embedded. No install, network, cloud, model fit or repository checkout is required. Complete audit replay is the experiment. Live Colab UI remains NOT_CHECKED.\n\nThree live TODO functions drive the real audit. Author-reference tables are not your execution results.')
 code('# @colab-bootstrap\nfrom pathlib import Path\nfrom decimal import Decimal\nfrom copy import deepcopy\nimport base64,hashlib,io,itertools,json,math,zipfile\nprint("L189: full frozen audit; $0 cloud/API; no model execution")',['colab-bootstrap'])
 md(prose(True))
 md('## PROVIDED · Unpack and authenticate\nThe embedded archive has a fixed SHA256. Every extracted file is checked again against the independent input manifest in the visible audit below. Hash equality is not evidence that a scientific claim is correct.')
 code('PACKET='+repr(encoded)+'\nraw=base64.b64decode(PACKET)\nassert hashlib.sha256(raw).hexdigest()=='+repr(digest)+"\npacket=Path('l189-packet');packet.mkdir(exist_ok=True)\nwith zipfile.ZipFile(io.BytesIO(raw)) as z:\n    for name in z.namelist():\n        assert not Path(name).is_absolute() and '..' not in Path(name).parts\n    z.extractall(packet)\nmanifest="+repr(manifest),['data-payload'])
 tests={'admission':"p={'phases_usd':dict(preparation=1,training=4,selection=1,evaluation=1,retries=1,validation=1),'bound_verified':True,'protocol_frozen':True,'source_audit':'PASS'}\nassert admission(p)=={'status':'WITHIN_CAP','total_usd':9.0}\np['phases_usd']['retries']=None\nassert admission(p)['status']=='NOT_ESTABLISHED'\nprint('Cost contract PASS')", 'priority':"assert priority(4,[5,4,5],[1,1,1])==56/3\nassert priority(3,[5,4,5],[1,1,1])==14\nprint('Priority arithmetic PASS')", 'sensitivity':"tied=[dict(id='a',impact=4,feasibility=[3,3,3]),dict(id='b',impact=3,feasibility=[4,4,4])]\ngrid=sensitivity(tied)\nassert len(grid)==27 and all(row['leaders']==['a','b'] for row in grid)\nprint('Full sweep and ties PASS')"}
 instructions={
 'admission':'Return {status,total_usd}. Require exactly preparation/training/selection/evaluation/retries/validation. Reject negative/nonfinite/bool costs. Missing or unknown phases give NOT_ESTABLISHED and null total. Source FAIL gives BLOCKED. Compare decimal sums with USD10; above cap gives OVER_CAP. Otherwise require bound_verified, protocol_frozen and source PASS; WITHIN_CAP does not authorize a run.',
 'priority':'Accept integer impact and exactly three integer feasibility ratings in 1..5, plus three positive integer weights. Reject invalid values including booleans. Return impact times the weighted mean. Explain why changing impact is different from changing a weight.',
 'sensitivity':'Accept nonempty cases with unique IDs, impact and feasibility. Enumerate all 27 weight triples in lexicographic order from {1,2,3}. Call your live priority function. Return weights, scores by ID, and every maximum-scoring ID in sorted order. Preserve ties; do not break them by input order.'}
 for name in ['admission','priority','sensitivity']:
  md('## TODO · '+name+'\n\n'+instructions[name])
  node=ast.parse(functions[name]).body[0]
  code(functions[name] if solution else 'def '+name+'('+ast.unparse(node.args)+'):\n    # TODO: implement the contract above.\n    raise NotImplementedError('+repr(name)+')',['solution' if solution else 'todo'])
  md('### CHECK · immediate feedback');code(tests[name])
 md('## CHECK · reject invalid budgets and fabricated certainty\nThese behavioral cases test missing costs, source failures, malformed ratings, ties and a genuine rank reversal. All three live functions feed these checks and the full audit.')
 code(checks);code("print(checks(admission,priority,sensitivity))")
 md('## PROVIDED · Visible complete audit\nRead the byte inventory, case-schema checks, source links, inherited boundary checks and cache arithmetic. The calls at the end use your three functions. No predicted probabilities or model checkpoints are loaded.')
 code(auditor)
 code("report=audit(packet,manifest,admission,priority,sensitivity)\nPath('l189-report.json').write_text(json.dumps(report,indent=2)+'\\n')\nfor row in report['ranking']:\n    print(row['rank'],row['id'],round(row['score'],2),row['cost_gate']['status'])\nprint('Weight scenarios:',report['weight_scenarios'])\nprint('Evidence:',report['observations'])")
 md('## Intervene · challenge an assumption\nPredict the result before executing. A score change does not rewrite the frozen author evidence. It creates a separate what-if analysis.')
 code("changed=json.loads((packet/'cases.json').read_text())\nchanged[0]['impact']=3\nscenario=sensitivity(changed)[0]\nassert scenario['leaders']==['composite']\nprint('What-if scores:',scenario['scores'])")
 md('## EXIT · defend your shortlist\nWrite your own rationale; automatic success never fills it for you. Name the nearest work and why a local bug is not a general gap. State one useful effect, uncertainty procedure still to freeze, control and stopping condition. Include human effort and the complete compute envelope.')
 code("defense=''  # Write your own argument here; the author solution does not impersonate you.\nsubmission={'audit':report['status'],'ranked_ids':[r['id'] for r in report['ranking']],'defense':defense,'learner':'PENDING_WRITTEN_DEFENSE'}\nPath('l189-submission.json').write_text(json.dumps(submission,indent=2)+'\\n')\nprint(submission['learner'])")
 md('## Reproduction boundary\nFull selected source/report audit COMPLETE. Fresh training, underlying prediction replay and whole-paper reproduction NOT_RUN. Novelty and complete literature coverage NOT_ESTABLISHED. Model proposals are plans, not ready or budget-approved trainers. Tomorrow retrieve the cost gate; in one week revisit related work and revise a justified score. Ask the agent to review your defense.')
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(book.cells):c['id']=hashlib.sha256((str(i)+c.source).encode()).hexdigest()[:12]
 dest=P/('solutions' if solution else '')/(S+'.ipynb');dest.parent.mkdir(exist_ok=True)
 if solution and dest.exists():
  old=nb.read(dest,4)
  if [c.source for c in old.cells]==[c.source for c in book.cells]:return
 nb.write(book,dest)
make(False);make(True)
print('Built lesson, reference, shortlist and portable notebooks')
