"""Build complete proposal cards, reference, lesson, portable ZIP and visible-code notebooks."""
import ast,base64,hashlib,html,importlib.metadata,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l198';Q=E/'packet';S='0198-three-research-directions'
r=json.loads((E/'report.json').read_text());cases=json.loads((E/'proposals.json').read_text());F=P/'figures/l198'
source_urls={s['id']:s['url'] for s in json.loads((Q/'evidence/l189/packet/sources.json').read_text())}
cards=[]
for c,o in zip(cases,r['proposals']):
 card='### '+c['short_title']+'\n\n'
 for title,key in [('Question','question'),('What is already known','known'),('Candidate contribution','delta'),('What L197 adds and leaves open','after_l197'),('Refined hypothesis','hypothesis_refined'),('Matched controls','baseline'),('Estimand','estimand'),('Falsifier','falsifier_refined'),('Complete accounting','count_note'),('Cheapest next artifact','next_artifact')]:
  card+='**'+title+'.** '+c[key]+'\n\n'
 card+='**Declared matrix.** Every combination is listed in the machine-readable report; each axis below is frozen as a proposal.\n\n'
 for name,axes in c['matrices'].items():
  card+='- '+name.replace('_',' ')+': '+ ' × '.join(k+' ∈ {'+', '.join(map(str,v))+'}' for k,v in axes.items())+' = **'+str(o['counts'][name])+'**.\n'
 card+='\n**Unresolved before execution:**\n\n'+'\n'.join('- '+x for x in c['unresolved_execution_fields'])+'\n\n'
 card+='**Original priority rationale:**\n\n'+'\n'.join('- '+x for x in c['feasibility_reasons'])+'\n\n'
 card+='**Preparation effort:** '+str(c['human_hours'][0])+'–'+str(c['human_hours'][1])+' human hours (authored planning estimate). All six full-run cost phases are unknown; budget NOT_ESTABLISHED.\n\n'
 card+='**Closest work:** '+', '.join('['+x+']('+source_urls[x]+')' for x in c['related_work'])+'.\n\n'
 card+='**Original receipts (reports only):** '+', '.join('['+x+'](../labs/evidence/l198/packet/evidence/l189/packet/inherited/'+x+')' for x in c['evidence'])+'. These receipts are authenticated; their raw experiments are not all replayed. The full L197 packet is replayed separately.\n\n'
 card+='**State:** PROPOSAL_NOT_EXECUTION_READY; models NOT_RUN; novelty NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE.\n'
 cards.append(card)
all_cards='# L198 — three ranked research proposals\n\nAuthored examples for critique, not learner submissions. Default ranking concerns the next useful decision; final selection belongs to L199.\n\n'+'\n\n'.join(cards)
(E/'ranked-proposals.md').write_text(all_cards.replace('](../labs/evidence/l198/',']('))
card_html='\n'.join('<details class="proposal-card"><summary>'+str(i+1)+' · '+html.escape(c['short_title'])+' · proposed / unrun</summary>'+render(card)+'</details>' for i,(c,card) in enumerate(zip(cases,cards)))
p=r['landscape']['predictions'];g=p['regression']['test'];i=p['icl']['advantage'];ci=g['conditional_interval']
results=f'''| Reproduced evidence | Complete scope | Interpretation |
|---|---|---|
| Original research ranking | Three candidates, all 27 weight triples | Temporal leads 27/27 at original scores; changing impact can reverse this. |
| Published tables | 401 task cells + 90 summaries | Original 5 aggregate discrepancies and 34 rank differences retained. |
| L149 saved comparison | Five seeds per pipeline; all val/test keys | GNN advantage {g['gnn_advantage']:+.6f} MAE; original conditional interval [{ci['low']:.6f}, {ci['high']:.6f}]. |
| L182 saved comparison | Three arms × ten draws × 702 queries | RDB-PFN minus TabICL {i['mean']:+.6f} AUROC; positive {i['positive']}/10 draws. |
| L194 full task inventory | All 21 published reference rows | Zero fresh tasks; every fresh score remains null. |
| New proposal matrices | All three complete planned designs | Counts are planning artifacts; no new model run occurred. |

[Complete report](../labs/evidence/l198/report.json) · [Independent verification](../labs/_verify_l198_results.json) · [Source review](../labs/evidence/l198/source-review.md).
'''
status='<div class="evidence-status"><strong>Author audit: COMPLETE_SELECTED_PROPOSAL_AUDIT.</strong> All three proposals, all 27 ranking settings and the complete L197 evidence packet replayed. New model experiments: <code>NOT_RUN</code>. Novelty: <code>NOT_ESTABLISHED</code>. Your defense: <code>PENDING_WRITTEN_DEFENSE</code>.</div>'
prose=(R/'lessons/content'/(S+'.md')).read_text().replace('[[STATUS]]',status).replace('[[CARDS]]',card_html).replace('[[RESULTS]]',results)
figures={}
for name,caption in [('temporal','Invented scores isolate the change in a model contrast. Query identities and labeled information stay paired.'),('composite','Invented scores show positive conditional gains but negative interaction. The full proposed evaluation counts all support draws.'),('transfer','Invented scores show why extra-compute scratch is the decisive control. The target database is excluded from source preprocessing and training.')]:
 figure='<figure><div class="diagram-scroll" tabindex="0" role="region" aria-label="Scrollable '+name+' experiment"><img src="../labs/figures/l198/'+name+'.svg" alt="'+html.escape(caption)+'"></div><figcaption>'+caption+'</figcaption></figure>'
 figures[name]=(figure,caption);prose=prose.replace('[[FIG:'+name+']]',figure)
def doc(title,body,scripts=False):
 body=re.sub(r'<table([^>]*)>',r'<div class="table-scroll" tabindex="0" role="region" aria-label="Scrollable research table"><table\1>',body).replace('</table>','</table></div>')
 tags=''.join('<script src="../assets/'+x+'.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','teachback','research-priority','proposal-decisions','l198-lesson']) if scripts else ''
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in ['lesson','research-priority','proposal-decisions'])+'</head><body><article class="l198"><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html">Curriculum</a></nav>'+body+'</article>'+tags+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Lesson 198 — Three research directions',render(prose),True))
reference='''# Three research directions — proposal field guide

**Contract:** closest work → candidate contribution → falsifiable hypothesis → matched experiment → decision rule.

| Direction | Decisive comparison | Complete proposed counts |
|---|---|---|
| Temporal sensitivity | (PFN − tree) under availability minus the same contrast under timestamp filtering | 24 model/policy/task/seed evaluations |
| Prior × encoder | Composite gain under relational prior minus its gain under flat prior; also report conditional gain | 12 checkpoints; 72 primary, 6 released and 18 tree prediction batches |
| Compute-matched transfer | Each temporal-pretrained arm minus extra-compute scratch with whole target database excluded | 12 source fits + 24 target fits |

**Intervals:** for benefit, lower endpoint > +margin supports useful benefit; upper endpoint < +margin rules it out. For sensitivity, an interval strictly inside (−margin,+margin) rules out material change; wholly beyond either margin supports sensitivity. Otherwise inconclusive. This classifies an interval; it does not justify the estimator, resampling unit or coverage. Test synergy separately against zero. Margin 0.01 AUROC is an authored planning choice.

**Ranking:** impact × weighted mean(data, implementation, compute). Defaults: 18.67 / 15 / 10. All 27 weight triples preserve the original leader; temporal impact 3 gives 14 / 15 / 10. This ranks next decisions, not probabilities or full-run affordability.

**Run gate:** full source/data/key manifests; matched information; exact model/optimizer/schedule; validation-only selection; complete seeds/support draws; dependency-aware uncertainty; six cost phases. Unknown is not zero. No study here is execution-ready.

**Write three cards:** 250–400 words each; closest work, contribution, hypothesis, controls, matrix, estimand, useful effect, falsifier, uncertainty, cost, unresolved requirements and cheapest next artifact. Justify a changed ranking assumption and name its reversal condition. Final selection is L199.

'''+results+'''
[Lesson](../lessons/0198-three-research-directions.html) · [Complete cards](../labs/evidence/l198/ranked-proposals.md) · [Blank template](../labs/evidence/l198/proposal-template.md) · [Protocol](../labs/l198-reproduction.md) · [Student notebook](../labs/0198-three-research-directions.ipynb).

Primary reading: [temporal pretraining §4.4](https://arxiv.org/html/2609.35219v1#S4.SS4). Novelty NOT_ESTABLISHED; new model experiments NOT_RUN; learner PENDING_WRITTEN_DEFENSE. Full RDBLearn reproduction remains INCOMPLETE_SOURCE_PREPROCESSING_GATE.
'''
(R/'reference/three-research-directions.html').write_text(doc('Three research directions — field guide',render(reference)))
requirements='\n'.join(name+'=='+importlib.metadata.version(name) for name in ['numpy','scipy','beautifulsoup4'])+'\n';(E/'requirements-audit.txt').write_text(requirements)
entries={'requirements-audit.txt':requirements.encode(),'README.md':b'''# L198 complete proposal audit
Python 3.12 recommended. From this extracted directory:

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements-audit.txt
    .venv/bin/python labs/_budget_l198.py .venv/bin/python labs/_audit_l198.py
    .venv/bin/python labs/_budget_l198.py .venv/bin/python labs/_verify_l198.py

CPU only; no model calls. The budget wrapper caps its attempts together at 1800 seconds; account environment preparation separately in your own ledger. Output labs/evidence/l198/report.json must match the supplied reference. All original inputs and independent verifiers are included. Models NOT_RUN; novelty NOT_ESTABLISHED; learner defense pending.
''','labs/relkit/__init__.py':b''}
for name in ['_audit_l198.py','_verify_l198.py','_test_l198.py','_budget_l198.py','relkit/proposals_l198.py']:
 entries['labs/'+name]=(P/name).read_bytes()
for path in sorted(Q.rglob('*')):
 if path.is_file() and '__pycache__' not in str(path):entries['labs/evidence/l198/packet/'+str(path.relative_to(Q))]=path.read_bytes()
for name in ['input-manifest.json','protocol.json','proposals.json','report.json','proposal-template.md','source-review.md','ranked-proposals.md']:
 entries['labs/evidence/l198/'+name]=(E/name).read_bytes()
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for name,data in sorted(entries.items()):
  info=zipfile.ZipInfo(name,(2026,10,2,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
payload=buf.getvalue();(E/'reproducer.zip').write_bytes(payload)
portable=re.sub(r'<div id="[^"]+"></div>','',prose).replace(card_html,'\n\n'.join(cards))
for name,(figure,caption) in figures.items():
 portable=portable.replace(figure,'![L198 '+name+'](data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode()+')\n\n*'+caption+'*')
portable=portable.replace('](../','](https://avistian.github.io/relational/')
portable=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',portable)
source=(P/'relkit/proposals_l198.py').read_text();funcs={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
contracts=[('paired_contrast','scores, mode','Require exactly a,b for gain, and a,b,c,d for conditional or interaction. Scores must be finite non-boolean AUROC values in [0,1]. Gain = B−A; conditional = D−C; interaction = (D−C)−(B−A). Reject extra/missing arms and unknown modes.'),('interval_decision','low, high, margin, mode','Validate finite non-boolean numbers, low ≤ high, margin > 0 and mode benefit/sensitivity. Benefit: lower > margin → USEFUL_BENEFIT; upper < margin → BELOW_USEFUL_MARGIN. Sensitivity: wholly beyond either signed margin → MATERIAL_SENSITIVITY; strictly inside both → BELOW_USEFUL_MARGIN. Otherwise INCONCLUSIVE, including boundary touches. This classifies a supplied interval; it does not compute uncertainty.'),('expand_matrix','axes','Require a nonempty dict with nonblank string axis names and nonempty lists of distinct strings or integers (no booleans). Return every Cartesian combination as a dict, preserving axis/level order. Reject malformed or duplicate levels. Missing controls must not disappear silently.')]
# Every frozen scoring function is readable and executed inline. Run-only import wiring is excluded.
sources=[(Q/'relkit/gaps_l189.py','Original priority and complete-cost rules'),(Q/'_audit_l189.py','Original complete ranking and source audit'),(Q/'evidence/l197/packet/relkit/tracking_l191.py','Published-table parsing and comparison rules'),(Q/'evidence/l197/packet/_replay_l191.py','Every published task and summary cell'),(Q/'evidence/l197/packet/relkit/stress_l195.py','Keyed errors, intervals and AUROC'),(Q/'evidence/l197/packet/_replay_l195.py','All original saved predictions and reference rows'),(Q/'relkit/landscape_l197.py','Evidence coverage and scope rules'),(Q/'_audit_l197.py','Complete original landscape audit'),(P/'_audit_l198.py','Proposal audit driven by your functions')]
def notebook(solution):
 cells=[]
 def md(s):cells.append(nb.v4.new_markdown_cell(s))
 def code(s,tags=None):cells.append(nb.v4.new_code_cell(s,metadata={'tags':tags or []}))
 md('# Lesson 198 · Three research directions\n\nComplete proposal/evidence audit with full original input packet. CPU only; no model training or inference. Your three functions control the proposal matrices and decisions. Author examples are not learner submissions.')
 code('# @colab-bootstrap\nimport ast, base64, hashlib, io, itertools, json, math, os, subprocess, sys, tempfile, zipfile, importlib.util\nfrom pathlib import Path\nneeded = '+repr(requirements.splitlines())+'\nif any(importlib.util.find_spec(m) is None for m in ["numpy","scipy","bs4"]):\n    subprocess.run([sys.executable,"-m","pip","install","-q",*needed],check=True,timeout=300)\nimport numpy as np\nworkspace=Path(tempfile.mkdtemp(prefix="l198-proposals-"))')
 md(portable)
 code('payload=base64.b64decode('+repr(base64.b64encode(payload).decode())+')\nassert hashlib.sha256(payload).hexdigest()=='+repr(hashlib.sha256(payload).hexdigest())+'\nwith zipfile.ZipFile(io.BytesIO(payload)) as archive:\n    archive.extractall(workspace)\nevidence=workspace/"labs/evidence/l198"\nprint("Authenticated portable packet:",len(payload),"bytes")',['data-payload'])
 for name,args,contract in contracts:
  md('## TODO · '+name+'\n\n'+contract)
  code(funcs[name] if solution else 'def '+name+'('+args+'):\n    raise NotImplementedError("'+name+'")')
 md('## CHECK · Contrasts, intervals and complete run matrices\n\nCheck boundary cases, malformed inputs and every combination. These are scientific decision contracts, not measured model evidence.')
 test_source=(P/'_test_l198.py').read_text();node=next(n for n in ast.parse(test_source).body if isinstance(n,ast.FunctionDef))
 code(ast.get_source_segment(test_source,node)+'\nprint(check198(paired_contrast,interval_decision,expand_matrix))')
 for path,description in sources:
  text=path.read_text();nodes=ast.parse(text).body
  imports=[ast.get_source_segment(text,n) for n in nodes if isinstance(n,(ast.Import,ast.ImportFrom)) and not (isinstance(n,ast.ImportFrom) and ((n.module or '').startswith('relkit') or (n.module or '').startswith('_')))]
  functions=[n for n in nodes if isinstance(n,ast.FunctionDef) and n.name!='run198']
  md('## PROVIDED · '+description+'\n\nExact source: `'+str(path.relative_to(P))+'`. These visible functions run on the frozen packet below; they do not call a hidden course implementation.')
  if imports:code('\n'.join(imports))
  for node in functions:
   md('### '+node.name+'\n\n'+(ast.get_docstring(node) or 'Follow the input checks and returned fields; this function is part of the complete replay.'))
   code(ast.get_source_segment(text,node))
 md('## RUN · Your functions feed the complete audit')
 code('report=audit198(evidence,audit,admission,priority,sensitivity,audit197,replay191,replay195,paired_mae,interval_verdict,claim_scope,keyed_auc,cluster_interval,coverage_report,admit_claim,paired_contrast,interval_decision,expand_matrix)\nassert report==json.loads((evidence/"report.json").read_text())\nPath("l198-report.json").write_text(json.dumps(report,indent=2))\nprint(report["status"])\nfor proposal in report["proposals"]:\n    print(proposal["id"],proposal["counts"],proposal["budget"]["status"])')
 md('## CHECK · Independently implemented verification\n\nThis runs the included independent HTML/Fraction, SQL, pairwise-AUROC, bootstrap and matrix oracles in a disposable directory. Their source files are in the portable packet; the notebook calculations above must already match the full report.')
 code('subprocess.run([sys.executable,str(workspace/"labs/_budget_l198.py"),sys.executable,str(workspace/"labs/_verify_l198.py")],check=True,timeout=300)\nverification=json.loads((workspace/"labs/_verify_l198_results.json").read_text())\nassert verification["status"]=="PASS"\nprint(verification["status"],verification["complete_matrix_counts"])')
 md('## EXIT · Three cards and a defensible ranking\n\nWrite 250–400 words per proposal. Justify one changed priority assumption, one uncertainty plan and one ranking-reversal condition. Do not select the final direction yet. Structural checks cannot grade novelty or scientific reasoning; bring the work to the agent for review.')
 code('submission={"proposals":[{"id":name,"draft":""} for name in ["temporal","composite","transfer"]],"revised_ranking":[],"changed_assumption":"","reversal_condition":"","learner":"PENDING_WRITTEN_DEFENSE"}\nPath("l198-submission.json").write_text(json.dumps(submission,indent=2))\nprint("DRAFT: complete the three cards and request review. Prior exits remain unchanged.")')
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
 for idx,cell in enumerate(book.cells):cell.id=('solution' if solution else 'student')+'-l198-'+str(idx)
 path=P/'solutions'/(S+'.ipynb')
 if solution and path.exists():
  old=nb.read(path,4);book.metadata=old.metadata
  previous=[c for c in old.cells if c.cell_type=='code'];current=[c for c in book.cells if c.cell_type=='code']
  if [c.source for c in previous]==[c.source for c in current]:
   for prior,c in zip(previous,current):
    c.outputs=prior.outputs;c.execution_count=prior.execution_count;c.metadata=prior.metadata
 return book
nb.write(notebook(False),P/(S+'.ipynb'));nb.write(notebook(True),P/'solutions'/(S+'.ipynb'))
print('Built lesson, complete cards, reference, portable ZIP and both notebooks')
