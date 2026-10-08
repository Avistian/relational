"""Build connected lesson, reference, shareable original-source packet and portable notebooks."""
import ast,base64,hashlib,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l196';S='0196-community-engagement'
ticks=chr(96)*3
minimal="""import pandas as pd
from rdblearn.preprocessing import TabularPreprocessor
train = pd.DataFrame({'category': ['b', 'c', 'd'] * 4,
                      'number': list(range(12))})
known = pd.DataFrame({'category': ['b', 'c', 'd'], 'number': [1, 2, 3]})
for unseen in ['a', 'z', '0', 'e']:
    pipeline = TabularPreprocessor().fit(train)
    support = pipeline.transform(train)
    before = pipeline.transform(known)
    pipeline.transform(pd.DataFrame({'category': [unseen], 'number': [1]}))
    after = pipeline.transform(known)
    print(unseen, before.category.tolist(), after.category.tolist())
"""
draftpath=E/'question-draft.md'
draft=draftpath.read_text().replace('[[MINIMAL_CODE]]',ticks+'python\n'+minimal+ticks)
draftpath.write_text(draft)
(E/'minimal-question.py').write_text(minimal)
d=json.loads((E/'diagnostic.json').read_text())
table='| Unseen row | Known codes before | Known codes after | Query b alone / batched |\n|---|---|---|---|\n'
for c in d['observations']:
 table+='| '+str(c['unseen'])+' | '+str(c['before'])+' | '+str(c['after'])+' | '+str(c['query_alone'][0])+' / '+str(c['query_with_other'][1])+' |\n'
shortdraft='''<div class="community-draft"><p><strong>Goal:</strong> reproduce the published RDBLearn experiments using v0.1.2.</p><p><strong>Observation:</strong> after unseen a or 0, known codes change from [0,1,2] to [1,2,3]; z/e and numeric controls are unchanged.</p><p><strong>Question:</strong> how should support/query encodings stay aligned when the vocabulary expands, and which preprocessing revision/environment matches the published experiments?</p><p><strong>Boundary:</strong> this synthetic check measures no benchmark effect. A correction to the expectation or setup is welcome.</p></div>'''
prose=(R/'lessons/content'/(S+'.md')).read_text().replace('[[CASE_TABLE]]',table).replace('[[DRAFT]]',shortdraft)
def doc(title,body,scripts=False):
 body=re.sub(r'<table([^>]*)>',r'<div class="repro-table" tabindex="0" role="region" aria-label="Scrollable comparison table"><table\1>',body).replace('</table>','</table></div>')
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/multitask-reproduction.css"><link rel="stylesheet" href="../assets/community-question.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html">Curriculum</a></nav>'+body+'</article>'+('<script src="../assets/community-question.js"></script>' if scripts else '')+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Lesson 196 — Community engagement',render(prose),True))
reference='''# An answerable technical question

**Structure:** goal → pinned version → minimal complete reproducer → expected and observed behavior → one decision.

**Before posting:** identify the implementation owner; search open and closed threads; include both failing cases and unchanged controls; preserve environment and source hashes. Attach the packet rather than linking inaccessible local files.

**Observation versus inference:** changed encoder codes are measured; score damage, historical paper behavior and prevalence are not measured by that observation.

'''+table+'''
**Routing:** [RDBLearn source](https://github.com/HKUSHXLab/rdblearn/issues); [RelBench task/data](https://huggingface.co/relbench); [PyG framework Q&A](https://github.com/pyg-team/pytorch_geometric/discussions). Verified 2026-10-02; posting access and responses are not guaranteed.

**Feedback:** DRAFT_ONLY → AWAITING_RESPONSE → RESPONSE_UNVERIFIED → FEEDBACK_CHECKED. Record an actual thread, actual reply, a pinned check, and what changed. A checked answer can correct you. Silence is not agreement. These states do not certify scientific truth.

**Follow-up template:** “I ran command ___ on revision ___; observed ___; this supports/corrects ___; historical impact remains ___.”

**Current status:** full four-case original-preprocessor diagnostic complete. Model effect NOT_ESTABLISHED; full model reproduction INCOMPLETE_SOURCE_PREPROCESSING_GATE; actual question DRAFT_ONLY; learner PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/0196-community-engagement.html) · [Question draft](../labs/evidence/l196/question-draft.md) · [Response log](../labs/evidence/l196/response-log.md) · [Reproducer ZIP](../labs/evidence/l196/reproducer.zip) · [Protocol](../labs/l196-reproduction.md)
'''
(R/'reference/community-engagement.html').write_text(doc('Community engagement field guide',render(reference)))
readme='''# L196 complete original-preprocessor diagnostic

Python 3.12, CPU, no checkpoints or API calls. Internet is needed only to install pinned dependencies.
From this extracted directory:
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements-diagnostic.txt
    .venv/bin/python -m pip check
    .venv/bin/python diagnostic.py .

The script verifies every source hash, imports the original full preprocessor and runs all four cases.
It writes diagnostic.json (fresh). saved-diagnostic.json contains the author's measured output for comparison.
No changed encoder or simplified substitute is installed. This is not benchmark reproduction.
'''
entries={'diagnostic.py':(P/'_diagnostic_l196.py').read_bytes(),'requirements-diagnostic.txt':(E/'requirements-diagnostic.txt').read_bytes(),'README.md':readme.encode(),'saved-diagnostic.json':(E/'diagnostic.json').read_bytes(),'protocol.json':(E/'protocol.json').read_bytes()}
for p in sorted((E/'packet').rglob('*')):
 if p.is_file() and '__pycache__' not in str(p):entries[str(p.relative_to(E))]=p.read_bytes()
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for name,data in sorted(entries.items()):
  info=zipfile.ZipInfo(name,date_time=(2026,10,2,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
payload=buf.getvalue();(E/'reproducer.zip').write_bytes(payload);encoded=base64.b64encode(payload).decode()
source=(P/'relkit/community_l196.py').read_text()
funcs={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
diag=(P/'_diagnostic_l196.py').read_text()
run=next(ast.get_source_segment(diag,n) for n in ast.parse(diag).body if isinstance(n,ast.FunctionDef))
report_source=(P/'_report_l196.py').read_text()
report_func=next(ast.get_source_segment(report_source,n) for n in ast.parse(report_source).body if isinstance(n,ast.FunctionDef))
tests=(P/'_test_l196.py').read_text()
test_func=next(ast.get_source_segment(tests,n) for n in ast.parse(tests).body if isinstance(n,ast.FunctionDef))
contracts=[
 ('summarize_cases','cases','Require exactly one case each for a/z/0/e, finite numeric vectors of lengths 3/3/3/3/1/2 (before, after, numeric_before, numeric_after, query_alone, query_with_other). Return cases=4, changed_cases and batch_sensitive_cases in a/z/0/e order, numeric_controls_unchanged, and model_effect=NOT_ESTABLISHED. Compare query_alone[0] to query_with_other[1]. Reject incomplete/duplicate/malformed observations.'),
 ('route_question','owner','Map rdblearn to https://github.com/HKUSHXLab/rdblearn/issues, relbench to https://huggingface.co/relbench, and pyg to https://github.com/pyg-team/pytorch_geometric/discussions. Reject other owners. This is topic routing, not an external action.'),
 ('feedback_state','thread_url, response, verified','Require textual URL/response and an actual boolean verified. Nonempty thread URLs must have HTTPS scheme and a hostname. Blank thread returns DRAFT_ONLY. A valid thread with no response returns AWAITING_RESPONSE; response with no check returns RESPONSE_UNVERIFIED; otherwise FEEDBACK_CHECKED. Actual evidence must accompany the state outside this practice function.')]
portable=re.sub(r'<div id="community-feedback"></div>','',prose)
portable=portable.replace('](../','](https://avistian.github.io/relational/')
portable=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',portable)
def notebook(solution):
 cells=[]
 def md(x):cells.append(nb.v4.new_markdown_cell(x))
 def code(x,tags=None):cells.append(nb.v4.new_code_cell(x,metadata={'tags':tags or []}))
 md('# Lesson 196 · Community engagement\n\nPrepare an answerable question with a complete original-source diagnostic. Default mode **fresh** installs pinned CPU dependencies into an isolated temporary environment and executes all four cases. Choose **replay** for offline inspection of saved measurements; that is not fresh reproduction. No posting, model calls or cloud spend occurs.')
 code('# @colab-bootstrap\nimport ast, base64, hashlib, io, itertools, json, math, os, subprocess, sys, tempfile, zipfile, copy\nfrom pathlib import Path\nfrom urllib.parse import urlsplit\nMODE = "fresh"  # choose "replay" only for saved-evidence practice\nworkspace = Path(tempfile.mkdtemp(prefix="l196-community-"))')
 md(portable)
 code("payload = base64.b64decode("+repr(encoded)+")\nassert hashlib.sha256(payload).hexdigest() == "+repr(hashlib.sha256(payload).hexdigest())+"\nwith zipfile.ZipFile(io.BytesIO(payload)) as archive:\n    archive.extractall(workspace)\nprint('Authenticated embedded packet:', len(payload), 'bytes')",['data-payload'])
 md('## PROVIDED · Original diagnostic code\n\nThis function authenticates source files and invokes the complete released TabularPreprocessor. The full original preprocessing source is displayed at the end. No model scores are produced.')
 code('import importlib.metadata\n'+run)
 code("""if MODE == "fresh":
    env = workspace / "diagnostic-env"
    subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
    python = str(env / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    subprocess.run([python, "-m", "pip", "install", "-q", "-r", str(workspace / "requirements-diagnostic.txt")], check=True, timeout=300)
    subprocess.run([python, "-m", "pip", "check"], check=True, timeout=30)
    # Save the same visible function into a child script; a fresh process avoids stale imports.
    subprocess.run([python, str(workspace / "diagnostic.py"), str(workspace)], check=True, timeout=90)
    diagnostic = json.loads((workspace / "diagnostic.json").read_text())
    assert diagnostic == json.loads((workspace / "saved-diagnostic.json").read_text())
elif MODE == "replay":
    diagnostic = json.loads((workspace / "saved-diagnostic.json").read_text())
    (workspace / "diagnostic.json").write_text(json.dumps(diagnostic))
else:
    raise ValueError("Choose fresh or replay explicitly")
print(MODE, diagnostic["status"])
print(json.dumps(diagnostic["observations"], indent=2))""")
 for name,args,contract in contracts:
  md('## TODO · '+name+'\n\n'+contract)
  code(funcs[name] if solution else 'def '+name+'('+args+'):\n    raise NotImplementedError("'+name+'")')
 md('## CHECK · Independent behavioral contracts\n\nThese checks reject incomplete evidence and false participation claims. Explain why each rejected input is unsafe.')
 code(test_func+'\nprint(check())')
 md('## PROVIDED · Assemble your evidence report')
 code(report_func+'\nreport = make_report(workspace)\nprint(json.dumps(report, indent=2))\nPath("l196-report.json").write_text(json.dumps(report, indent=2))')
 md('## EXIT · Your question and reflection\n\nWrite your own question before reading the local draft. Keep three parts distinct: observed fact, unknown impact, and requested clarification. Fill the response log only after a real conversation. Neither these checks nor a successful notebook establishes mastery.')
 code('submission = {"question": "", "thread_url": "", "response": "", "verification_command": "", "what_changed": "", "remaining_unknown": "", "learner": "PENDING_WRITTEN_DEFENSE"}\nPath("l196-submission.json").write_text(json.dumps(submission, indent=2))\nprint("Actual participation:", feedback_state(submission["thread_url"], submission["response"], False))')
 md('## Original implementation appendix\n\nComplete frozen preprocessing.py follows. Read DynamicLabelEncoder.transform and trace why sorting after expansion moves b. This is original source, not a patched implementation.')
 md(ticks+'python\n'+(E/'packet/source/rdblearn/rdblearn/preprocessing.py').read_text()+'\n'+ticks)
 md('## Local draft for comparison\n\n'+draft)
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
 for index,cell in enumerate(book.cells):cell.id=('solution' if solution else 'student')+'-l196-'+str(index)
 path=P/'solutions'/(S+'.ipynb')
 if solution and path.exists():
  old=nb.read(path,4);book.metadata=old.metadata
  previous=[c for c in old.cells if c.cell_type=='code'];current=[c for c in book.cells if c.cell_type=='code']
  if [c.source for c in previous]==[c.source for c in current]:
   for prior,c in zip(previous,current):
    c.outputs=prior.outputs;c.execution_count=prior.execution_count;c.metadata=prior.metadata
 return book
nb.write(notebook(False),P/(S+'.ipynb'));nb.write(notebook(True),P/'solutions'/(S+'.ipynb'))
print('Built lesson, reference, packet and notebooks')
