"""Build one source into lesson, reference and self-contained visible-code notebooks."""
import ast,base64,hashlib,html,io,json,re,zipfile,importlib.metadata
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l199';Q=E/'packet';S='0199-select-primary-direction'
r=json.loads((E/'report.json').read_text());F=P/'figures/l199';F.mkdir(parents=True,exist_ok=True)
svg='''<svg xmlns="http://www.w3.org/2000/svg" width="680" height="780" viewBox="0 0 680 780" role="img" aria-labelledby="title desc"><title id="title">From a research shortlist to a defensible next step</title><desc id="desc">Compare three candidates, inspect ratings and evidence, then check all mandatory requirements. Missing arrival histories lead to a feasibility check, not experiment launch. Passing requirements leads to review, not automatic authorization.</desc><rect width="680" height="780" rx="20" fill="#eef5f1"/><style>text{font-family:Arial,sans-serif;fill:#173e37}.head{font-size:25px;font-weight:bold}.label{font-size:21px;font-weight:bold}.small{font-size:18px}.card{fill:white;stroke:#94b9a8;stroke-width:2}</style><text x="35" y="48" class="head">A CHOICE THAT CAN CHANGE</text><text x="35" y="78" class="small">One direction · explicit alternatives · observable stop rule</text>
<rect x="35" y="107" width="610" height="113" rx="12" class="card"/><text x="57" y="143" class="label">1 · Compare provisional candidates</text><text x="57" y="178" class="small">Availability / composite structure / joint constraints</text><text x="57" y="205" class="small">Priority organizes attention. It does not prove novelty.</text>
<path d="M340 223v27m-7 -7l7 7 7-7" fill="none" stroke="#3e7462" stroke-width="3"/>
<rect x="35" y="256" width="610" height="113" rx="12" class="card"/><text x="57" y="292" class="label">2 · Expose fragile judgments</text><text x="57" y="327" class="small">27 weight triples preserve the original first choice.</text><text x="57" y="354" class="small">Impact 4 → 3 changes the equal-weight leader.</text>
<path d="M340 372v27m-7 -7l7 7 7-7" fill="none" stroke="#3e7462" stroke-width="3"/>
<rect x="35" y="405" width="610" height="113" rx="12" class="card"/><text x="57" y="441" class="label">3 · Check mandatory requirements</text><text x="57" y="476" class="small">Data · baseline · design · aggregate budget</text><text x="57" y="503" class="small">Any FAIL or UNKNOWN → DO_NOT_LAUNCH</text>
<path d="M340 521v27m-7 -7l7 7 7-7" fill="none" stroke="#3e7462" stroke-width="3"/>
<rect x="35" y="554" width="610" height="134" rx="12" fill="#d8eadd" stroke="#659578"/><text x="57" y="590" class="label">4 · Take the first discriminating step</text><text x="57" y="625" class="small">Can usable measured arrival histories be obtained?</text><text x="57" y="652" class="small">If unavailable: stop this study; reopen alternatives.</text><text x="57" y="678" class="small">If all gates pass: submit the exact protocol for review.</text>
<text x="35" y="738" class="small">Author example ≠ learner commitment ≠ launch authorization</text></svg>'''
(F/'decision.svg').write_text(svg)
figure='<figure class="direction-figure"><div class="direction-scroll" tabindex="0" role="region" aria-label="Scrollable research decision diagram"><img src="../labs/figures/l199/decision.svg" alt="Four steps: compare provisional candidates, expose fragile ratings, check mandatory requirements, then investigate arrival-history feasibility."></div><figcaption>A model-specific result can motivate a question; this decision diagram governs whether that question is executable.</figcaption></figure>'
status='<div class="direction-status"><strong>Complete selected evidence replay:</strong> 30 runs, 21,060 predictions. <strong>Proposed study:</strong> DO_NOT_LAUNCH. <strong>Full model reproduction:</strong> INCOMPLETE_SOURCE_PREPROCESSING_GATE. <strong>Your choice:</strong> PENDING_WRITTEN_DEFENSE.</div>'
replay='''| Evidence | Complete scope | Result boundary |
|---|---|---|
| Saved predictions | 3 methods × 10 draws × 702 queries | Exact L190 report parity; no new inference |
| Literature | 4 frozen searches; 31 records / 30 unique papers | Only 2 searches complete; novelty not established |
| Ranking | All 27 weight triples; 21 one-rating perturbations | Authored decision aid, not measured research merit |
| Later receipts | Full L194/L197 reports authenticated | 21 L194 task slots have no fresh scores; numerics not replayed |
'''
prose=(R/'lessons/content'/(S+'.md')).read_text();memo=(E/'worked-memo.md').read_text()
for k,v in dict(STATUS=status,FIGURE=figure,REPLAY=replay,MEMO=memo).items():prose=prose.replace('[['+k+']]',v)
def doc(title,body,scripts=False):
    body=body.replace('<table>','<div class="direction-scroll" tabindex="0" role="region" aria-label="Scrollable comparison table"><table>').replace('</table>','</table></div>')
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/direction-selection.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html">Curriculum</a></nav>'+body+'</article>'+('<script src="../assets/direction-selection.js"></script>' if scripts else '')+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Lesson 199 — Select primary direction',render(prose),True))
reference='''# Select a primary direction · field guide

**Three distinct decisions:** priority for investigation; readiness of an exact protocol; authorization to execute. A high ranking never clears missing mandatory evidence.

**Priority:** impact × weighted mean of data, implementation and compute feasibility. Ordinal author ratings are assumptions, not probabilities. Retain ties. Check both weights and ratings. Here 27 weight triples retain availability, but impact 4 → 3 changes the equal-weight winner from 18.667 versus 15 to 14 versus 15.

**Four mandatory requirements:** measured usable data; healthy exact baselines; falsifiable locked design; complete aggregate cost forecast. FAIL or UNKNOWN → DO_NOT_LAUNCH. All PASS → READY_FOR_REVIEW, not automatic permission.

**Availability contrast:** D = (context − target-row control)event − (context − target-row control)arrival. Higher-is-better metric; positive D means context advantage shrinks. Match task, identities, splits, learner family and tuning budget. Related aggregates are already relational information.

**Interpretation:** justify δ before new evaluation. Interval above δ supports useful shrinkage; overlapping δ is inconclusive; below δ weakens that useful-shrinkage claim. Neither seeds nor support draws create independent databases. Missing history cannot be replaced by synthetic delays under the same claim.

'''+replay+'''
**Memo:** direction, hypothesis, contrast, baselines, metric, threshold, uncertainty, cost, stop, deferred alternative, evidence, revision. Automatic checks test presence only. Final choice and mastery require your written defense.

**First action:** obtain and assess an arrival-history sample within a predeclared feasibility effort. If unavailable, stop this study and reopen alternatives. Novelty and future cost remain NOT_ESTABLISHED.

[Lesson](../lessons/0199-select-primary-direction.html) · [Template](../labs/evidence/l199/memo-template.md) · [Author example](../labs/evidence/l199/worked-memo.md) · [Protocol](../labs/l199-reproduction.md) · [Report](../labs/evidence/l199/report.json).

Read [COS preregistration guidance](https://www.cos.io/initiatives/prereg) and [RelBench v2](https://arxiv.org/abs/2602.12606). Selected numerical source: [RDB-PFN v5 Table 9](https://arxiv.org/html/2603.03805v5). Old evidence informs exploration; a memo is not a public registration. Complete saved replay does not establish fresh training, historical identity, novelty, learner mastery or deployment.
'''
(R/'reference/select-primary-direction.html').write_text(doc('Primary direction field guide',render(reference)))
requirements='numpy=='+importlib.metadata.version('numpy')+'\n';(E/'requirements-audit.txt').write_text(requirements)
readme='''# L199 complete direction selection audit

Python 3.12 recommended. Extract this archive, create an environment and install requirements-audit.txt. Account installation/preparation in your own budget. Then from the extracted directory:

    python3 labs/_budget_l199.py python3 labs/_audit_l199.py
    python3 labs/_budget_l199.py python3 labs/_verify_l199.py

The wrapper caps aggregate subprocess execution at 1800 seconds, including failures, and kills the process group at cutoff. Add your preparation allowance to the ledger before execution. USD0 cloud/API. All inputs and source versions are frozen. Output report must equal the included report exactly. Independent verification runs in a disposable copy. No model training, inference or new literature collection; fresh model reproduction remains INCOMPLETE_SOURCE_PREPROCESSING_GATE. Complete replay is not independent replication. See protocol.json and l199-reproduction.md.
'''
entries={'README.md':readme.encode(),'requirements-audit.txt':requirements.encode(),'l199-reproduction.md':(P/'l199-reproduction.md').read_bytes()}
for name in ['_audit_l199.py','_verify_l199.py','_test_l199.py','_budget_l199.py','relkit/direction_l199.py']:entries['labs/'+name]=(P/name).read_bytes()
entries['labs/relkit/__init__.py']=b''
for path in sorted(Q.rglob('*')):
    if path.is_file():entries['labs/evidence/l199/packet/'+str(path.relative_to(Q))]=path.read_bytes()
for name in ['input-manifest.json','protocol.json','report.json','memo-template.md','worked-memo.md']:entries['labs/evidence/l199/'+name]=(E/name).read_bytes()
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
    for name,data in sorted(entries.items()):
        info=zipfile.ZipInfo(name,(2026,10,2,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
payload=buf.getvalue();(E/'reproducer.zip').write_bytes(payload)
def source(path):
    text=path.read_text()
    return '\n\n'.join(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef)))
text=(P/'relkit/direction_l199.py').read_text();functions={n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
contracts=[('priority','cases, weights','Require nonempty unique string IDs. Each case has impact and three feasibility integers in 1..5, excluding booleans. Three weights must be positive integers. Compute impact times weighted mean with Fraction; return scores as floats and sorted leaders, retaining exact ties. Invalid input raises ValueError.'),('launch_gate','gates','Require exactly data/baseline/design/budget with PASS, FAIL or UNKNOWN. Return state DO_NOT_LAUNCH if any is not PASS, else READY_FOR_REVIEW; blockers in that key order; authorization always NOT_GRANTED. Invalid input raises ValueError.'),('memo_readiness','sections','Require twelve textual fields in the displayed memo order. Return state DRAFT if any is whitespace-only else READY_FOR_REVIEW; missing field list in order; mastery always PENDING_WRITTEN_DEFENSE. Missing or nontext fields raise ValueError. Presence is not scientific review.')]
portable=prose.split('<details><summary>Read the worked author memo')[0]
portable=re.sub(r'<div id="[^"]+"></div>','',portable)
portable=portable.replace(figure,'![Research decision flow](data:image/svg+xml;base64,'+base64.b64encode(svg.encode()).decode()+')')
portable=portable.replace('](../','](https://avistian.github.io/relational/')
portable=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',portable)
def notebook(solution):
    cells=[]
    def md(s):cells.append(nb.v4.new_markdown_cell(s))
    def code(s,tags=None):cells.append(nb.v4.new_code_cell(s,metadata={'tags':tags or []}))
    md('# Lesson 199 · Select a primary direction\n\nComplete frozen replay, visible decision code, then your own memo. No model loading or training. L198 appeared after freeze; the approved L190 shortlist remains a provisional author example. Use your own L198 cards for the final memo.')
    code('# @colab-bootstrap\nimport ast,base64,hashlib,io,json,math,os,subprocess,sys,tempfile,time,zipfile,importlib.util\nfrom pathlib import Path\nfrom fractions import Fraction\nstarted=time.monotonic()\nif importlib.util.find_spec("numpy") is None:\n    subprocess.run([sys.executable,"-m","pip","install","-q",'+repr(requirements.strip())+'],check=True,timeout=180)\nimport numpy as np\nworkspace=Path(tempfile.mkdtemp(prefix="l199-direction-"))')
    md(portable)
    code('payload=base64.b64decode('+repr(base64.b64encode(payload).decode())+')\nassert hashlib.sha256(payload).hexdigest()=='+repr(hashlib.sha256(payload).hexdigest())+'\nwith zipfile.ZipFile(io.BytesIO(payload)) as archive: archive.extractall(workspace)\nevidence=workspace/"labs/evidence/l199"\nprint("Authenticated full portable packet:",len(payload),"bytes")',['data-payload'])
    for name,args,contract in contracts:
        md('## TODO · '+name+'\n\n'+contract)
        code(functions[name] if solution else 'def '+name+'('+args+'):\n    raise NotImplementedError("'+name+'")')
    md('## CHECK · Your decisions control the pipeline\n\nThese tests reject unsupported launch, discarded ties, malformed scores and false mastery. Keep them unchanged.')
    code(source(P/'_test_l199.py')+'\nprint(checks(priority,launch_gate,memo_readiness))')
    for name in ['relkit/checkpoint_l190.py','_replay_l190.py']:
        md('## PROVIDED · '+name+'\n\nExact frozen executable source. Follow the complete key join, support checks, tie-aware AUROC and raw literature reconstruction. This code executes below, not a hidden model API.')
        code(source(Q/name))
    md('## PROVIDED · Complete audit assembled with your three functions')
    code(source(P/'_audit_l199.py')+'\nreport=audit199(evidence,replay190,keyed_auc,claim_gate,rank_cases,priority,launch_gate,memo_readiness)\nassert report==json.loads((evidence/"report.json").read_text())\nPath("l199-report.json").write_text(json.dumps(report,indent=2))\nprint(report["status"],report["replay"]["prediction_rows"],report["default_priority"],report["launch"])')
    md('## CHECK · Independent verification\n\nPair-count AUROC, XML DOM and rational arithmetic run in an isolated copy; the verifier also corrupts inputs and substitutes wrong learner functions. Its complete source is included in the final appendix.')
    code('elapsed=time.monotonic()-started\nassert elapsed<1800,"INCOMPLETE_LOCAL_BUDGET_GATE"\nledger={"cap_seconds":1800,"cloud_usd":0,"attempts":[{"command":["notebook preparation and inline replay"],"seconds":elapsed,"status":"ACCOUNTED"}]}\n(evidence/"local-budget.json").write_text(json.dumps(ledger))\nsubprocess.run([sys.executable,str(workspace/"labs/_budget_l199.py"),sys.executable,str(workspace/"labs/_verify_l199.py")],check=True,timeout=max(1,1800-elapsed))\nverification=json.loads((workspace/"labs/_verify_l199_results.json").read_text())\nassert verification["status"]=="PASS"\nPath("l199-verification.json").write_text(json.dumps(verification,indent=2))\nprint(verification)')
    md('## EXERCISE · Challenge one rating\n\nPredict the effect of changing availability impact 4 to 3. Compute it with your priority function; explain why the launch decision does not change.')
    code('cases=json.loads((evidence/"packet/evidence/l190/packet/cases.json").read_text())\nchanged=json.loads(json.dumps(cases));changed[0]["impact"]=3\nprint(priority(changed,[1,1,1]))\nassert priority(changed,[1,1,1])["leaders"]==["composite"]')
    md('## EXERCISE · Remove one saved run\n\nUse a disposable copy. Rebuild the outer manifest for the edited receipt and explain why semantic run coverage still rejects it. Never edit the original evidence. The independent verifier contains an example of this intervention.')
    md('## EXIT · Write the memo before reading the author example\n\nUse 400–700 words plus the twelve fields. Explain D, the missing prerequisite, the strongest alternative and a stopping rule. Send your memo to the agent for review.')
    code('fields='+repr(['direction','hypothesis','contrast','baselines','metric','threshold','uncertainty','cost','stop','deferred','evidence','revision'])+'\nsections=dict.fromkeys(fields,"")\nsubmission={"sections":sections,"review":memo_readiness(sections),"memo":"","authorization":"NOT_GRANTED"}\nPath("l199-submission.json").write_text(json.dumps(submission,indent=2))\nprint(submission["review"])')
    md('## Author memo — compare after your draft\n\n'+memo)
    for path in [Q/'_verify_l190.py',P/'_verify_l199.py']:
        md('## Independent verifier source · '+path.name+'\n\n```python\n'+path.read_text()+'\n```')
    md('## Revisit and discuss\n\nTomorrow reconstruct the contrast without notes; in seven days defend the strongest alternative; in thirty days revisit the decision with new evidence. Ask the agent about any unclear step. Source: [COS preregistration guidance](https://www.cos.io/initiatives/prereg). Old replay remains exploratory evidence; no registration was submitted. Fresh model reproduction stays INCOMPLETE_SOURCE_PREPROCESSING_GATE; learner PENDING_WRITTEN_DEFENSE; live Colab NOT_CHECKED.')
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
    for i,c in enumerate(cells):c.id=('solution' if solution else 'student')+'-l199-'+str(i)
    return book
nb.write(notebook(False),P/(S+'.ipynb'));nb.write(notebook(True),P/'solutions'/(S+'.ipynb'))
print('Built lesson, reference, SVG, ZIP and two portable notebooks')
