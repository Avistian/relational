"""Deterministic B01 lesson, reference, archive and portable notebook builder."""
import ast,base64,hashlib,html,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b01';S='b01-architecture-coverage-honest-comparison'
r=json.loads((E/'report.json').read_text());families=json.loads((E/'families.json').read_text())
esc=html.escape
matrix='| Family / example | Pretraining | Representation | Adaptation | Task scope / information |\n|---|---|---|---|---|\n'
for f in families:
 matrix+=f"| [{f['name']} · {f['example']}]({f['source']}) | {f['prior']} | {f['representation']} | {f['adaptation']} | {f['scope']}. {f['information']} |\n"
explorer='<div class="family-explorer"><div class="family-controls"><label>Follow one computation <select aria-label="Architecture family">'+''.join(f'<option value="{f["id"]}">{esc(f["name"])}</option>' for f in families)+'</select></label></div>'
for f in families:
 explorer+=f'<section data-family="{f["id"]}"><h3>{esc(f["example"])}</h3><ol class="family-route">'+''.join('<li>'+esc(s.strip())+'</li>' for s in f['representation'].split('→'))+f'</ol><p><strong>Adaptation:</strong> {esc(f["adaptation"])}</p><p><strong>Information:</strong> {esc(f["information"])}</p></section>'
explorer+='</div>'
widget='<div class="comparison-board"><p><strong>Baseline:</strong> matching released inputs; complete saved replay; selected-score claim. This is a declared-contract simulator.</p><div class="comparison-controls">'
for name,title in [('features','Feature arrays'),('support','Support identities'),('visibility','Visibility evidence')]:
 widget+=f'<label>{title}<select name="{name}"><option value="same">Matched and documented</option><option value="different">Different between arms</option><option value="unknown">Unknown / unsupported</option></select></label>'
widget+='<label>Evidence<select name="evidence"><option>COMPLETE_REPLAY</option><option>INCOMPLETE</option></select></label><label>Claim<select name="claim"><option value="selected_score">Selected score difference</option><option value="architecture_cause">Architecture caused gain</option><option value="fresh_inference">New inference completed</option><option value="learner_mastery">Learner mastered comparison</option></select></label></div><button type="button">Reset contract</button><output aria-live="polite">MATCHED → SUPPORTED_DESCRIPTIVE_REPLAY. This does not establish architecture causality.</output></div>'
svg='''<svg xmlns="http://www.w3.org/2000/svg" width="480" height="680" viewBox="0 0 480 680" role="img" aria-labelledby="t d"><title id="t">Same customer, different information paths</title><desc id="d">Age-only inputs collapse Ada and Bo. Order sums separate them. Raw linked orders retain more detail than the sums. Prediction differences do not isolate architecture.</desc><rect width="480" height="680" rx="18" fill="#eef5f0"/><style>text{font-family:Arial,sans-serif;fill:#173f30;font-size:18px}.head{font-size:24px;font-weight:bold}.sub{font-size:16px}.card{fill:white;stroke:#a5c4b3;stroke-width:1.5}</style><text x="24" y="40" class="head">Trace the information first</text><text x="24" y="72">Course example · no trained model scores</text><rect x="20" y="95" width="440" height="112" rx="10" class="card"/><text x="36" y="127">Eligible rows at the same cutoff</text><text x="36" y="158">Ada: age 40; orders [10, 20]</text><text x="36" y="186">Bo: age 40; orders [20, 40]</text><text x="235" y="239">↓</text><rect x="20" y="255" width="440" height="91" rx="10" class="card"/><text x="36" y="285">A · Target-row-only predictor</text><text x="36" y="316">Ada [40] = Bo [40] → identical inputs</text><rect x="20" y="362" width="440" height="112" rx="10" class="card"/><text x="36" y="392">B · Aggregate → flat predictor</text><text x="36" y="424">Ada [40, 30] ≠ Bo [40, 60]</text><text x="36" y="451" class="sub">Order sum added information absent from A.</text><rect x="20" y="490" width="440" height="114" rx="10" class="card"/><text x="36" y="520">C · Linked rows → relational predictor</text><text x="36" y="550">Ada retains [10, 20], not only sum 30.</text><text x="36" y="578" class="sub">Replacing with [15, 15] changes C, not B.</text><text x="24" y="642" class="sub">B vs C: test which retained distinction matters.</text></svg>'''
F=P/'figures/b01';F.mkdir(exist_ok=True);(F/'information-flow.svg').write_text(svg)
flow='<figure class="b01-flow"><img src="../labs/figures/b01/information-flow.svg" alt="Ada and Bo have the same age but different orders; order sums retain information absent from age-only inputs, and raw linked rows retain more than sums."><figcaption>Inputs differ before any predictor computes a score. Control that difference before attributing a gain.</figcaption></figure>'
results='| Configuration | Replayed mean AUROC | Sample SD | Original rounded target |\n|---|---:|---:|---:|\n'
for name,m in r['models'].items():results+=f"| {name} | {m['mean']:.6f} | {m['sample_sd']:.6f} | {m['paper_target']:.4f} |\n"
d=r['paired_rdbpfn_minus_tabicl'];results+=f"\nRDB-PFN − TabICL: **{d['mean']:+.6f}** mean paired difference; positive in **{d['positive']}/10** draws. All 30 original results are retained.\n"
prose=(R/'lessons/content'/(S+'.md')).read_text()
for key,value in dict(FAMILY_EXPLORER=explorer,FAMILY_MATRIX=matrix,FLOW=flow,CONTRACT_WIDGET=widget,RESULTS=results).items():prose=prose.replace('[['+key+']]',value)
def doc(title,body,script=False):
 body=body.replace('<table>','<div class="b01-scroll" tabindex="0" role="region" aria-label="Scrollable comparison table"><table>').replace('</table>','</table></div>')
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/comparison-contract.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html#research-bridge">Bridge curriculum</a></nav>'+body+'</article>'+('<script src="../assets/comparison-contract.js"></script>' if script else '')+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Lesson B01 — Architecture coverage and honest comparison',render(prose),True))
reference='''# Comparison contract · field guide

A result compares complete configurations: information → representation → adaptation → selection → inference → metric. Match the fields needed for your claim; disclose what differs.

## Freeze before scoring

Task/labels/split; complete query keys; features/support; preprocessing/visibility; selection; metric/uncertainty unit; budget policy. Unknown is not matched. Declared equality still needs source evidence. Equal source records need not preserve equal information after aggregation.

## Four claims, four evidence requirements

Saved replay authenticates and rescores existing predictions. Fresh inference executes checkpoints again. Architecture causality needs a controlled intervention. Learner mastery requires your own explanation and teacher review. Passing one gate does not fill another.

## The worked result

'''+results+'''
All 30 runs and 21,060 predictions replayed. Matching released DFS inputs does not certify historical availability. Shared test rows make support draws dependent for cross-database claims. Full benchmark reproduction and fresh B01 inference/training NOT_RUN.

## Family coverage

'''+matrix+'''
## Before B23

Name baseline and comparison unit. Freeze the information and resource policies. Give two falsification tests and one practical decision threshold. Leave unresolved fields explicit. No execution is authorized by filling the template.

[Lesson](../lessons/b01-architecture-coverage-honest-comparison.html) · [Protocol](../labs/b01-reproduction.md) · [TabArena](https://arxiv.org/html/2506.16791v1) · [Fair RDB benchmark](https://arxiv.org/html/2607.03659v1).
'''
(R/'reference/b01-comparison-contract.html').write_text(doc('Comparison contract field guide',render(reference)))
entries={}
for name in ['_audit_b01.py','_test_b01.py','_verify_b01.py','relkit/comparison_b01.py']:
 entries['labs/'+name]=(P/name).read_bytes()
entries['labs/relkit/__init__.py']=b''
for name in ['source-lock.json','l200-reproducer.zip','inherited-l200-protocol.md','families.json','reading-ledger.json','report.json']:
 entries['labs/evidence/b01/'+name]=(E/name).read_bytes()
entries['requirements.txt']=b'numpy==2.5.0\n'
entries['README.md']=b'B01 saved-evidence audit. Install NumPy, then python labs/_audit_b01.py and python labs/_verify_b01.py. No model inference.\n'
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for name,data in sorted(entries.items()):
  info=zipfile.ZipInfo(name,(2026,10,3,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
payload=buf.getvalue();(E/'reproducer.zip').write_bytes(payload)
def functions(path):
 text=path.read_text();return '\n\n'.join(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef))
source=(P/'relkit/comparison_b01.py').read_text();tree=ast.parse(source)
funcs={n.name:ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
portable=prose.replace(explorer,'').replace(widget,'**Contract exercise:** change one field to a mismatch and one to unknown. Predict the status before running your checker.').replace(flow,'![Information paths](data:image/svg+xml;base64,'+base64.b64encode(svg.encode()).decode()+')')
portable=portable.replace('](../','](https://avistian.github.io/relational/')
portable=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',portable)
for solution in [False,True]:
 cells=[]
 def md(s):cells.append(nb.v4.new_markdown_cell(s))
 def code(s,tags=None):cells.append(nb.v4.new_code_cell(s,metadata={'tags':tags or []}))
 md('# B01 · Architecture coverage and honest comparison\n\nPROVIDED: frozen evidence and full visible audit. TODO: three functions below. CHECK: call your implementations on valid and adversarial cases, then the entire 30-run replay. EXIT: write your defense. This notebook performs saved replay only; no cloud/model calls. NumPy is the only dependency. Installation, if needed, is capped at 300 seconds.')
 code('# @colab-bootstrap\nimport ast,base64,copy,hashlib,io,json,subprocess,sys,tempfile,zipfile,importlib.util\nfrom pathlib import Path\nif importlib.util.find_spec("numpy") is None:\n    subprocess.run([sys.executable,"-m","pip","install","numpy==2.5.0"],check=True,timeout=300)\nimport numpy as np\nworkspace=Path(tempfile.mkdtemp(prefix="b01-notebook-"))')
 md(portable)
 code('payload=base64.b64decode('+repr(base64.b64encode(payload).decode())+')\nassert hashlib.sha256(payload).hexdigest()=='+repr(hashlib.sha256(payload).hexdigest())+'\nwith zipfile.ZipFile(io.BytesIO(payload)) as z:\n    z.extractall(workspace)\nevidence=workspace/"labs/evidence/b01"', ['data-payload'])
 code("FIELDS = ('task','query_keys','split','labels','features','support','preprocessing','visibility','selection','metric','budget_policy')")
 for name,src in funcs.items():
  md('## '+('SOLUTION' if solution else 'TODO')+' · '+name+'\n\n'+{'compare_contracts':'Return status, mismatches and unknown lists. Reject unexpected fields; absent, null and unknown values cannot establish a match. A known mismatch takes precedence.','paired_effect':'Join draw IDs 0–9, reject duplicates/incomplete or invalid scores. Return per-draw differences, mean, sample SD, positive count and the uncertainty unit.','claim_gate':'Limit the requested claim to the contract and evidence status. Replay never becomes fresh inference, causal isolation or learner mastery.'}[name])
  code(src if solution else src.split('\n')[0]+'\n    raise NotImplementedError("Implement '+name+' before CHECK")')
 md('## PROVIDED · behavioral feedback\n\nRead these cases; your functions must handle all of them, not merely reproduce one mean.')
 code(functions(P/'_test_b01.py'))
 code("assert checks(compare_contracts,paired_effect,claim_gate)=='PASS'\nprint('Contract, pairing and claim checks PASS')")
 md('## PROVIDED · full evidence audit\n\nThis function invokes the archived pairwise and independent rank scorers in isolated subprocesses. It authenticates the original archive before extraction; your three functions control the final interpretation.')
 code(functions(P/'_audit_b01.py'))
 code("report=replay(evidence,compare_contracts,paired_effect,claim_gate)\nexpected=json.loads((evidence/'report.json').read_text())\nassert report==expected\nPath('b01-report.json').write_text(json.dumps(report,indent=2)+'\\n')\nprint(json.dumps({k:report[k] for k in ['execution','runs','predictions','claims','paired_rdbpfn_minus_tabicl']},indent=2))")
 md('## PROVIDED · inspect the scoring implementation\n\nThese archived sources are visible here as well as inside the authenticated bundle; the original raw evidence stays unchanged.')
 with zipfile.ZipFile(E/'l200-reproducer.zip') as z:
  for name in ['labs/relkit/exit_l200.py','labs/_audit_l200.py','labs/_verify_l200.py']:
   md('### '+name+'\n\n```python\n'+z.read(name).decode()+'\n```')
 md('## EXIT · your comparison contract\n\nWrite 250–400 words: baseline rationale, complete comparison unit, information control, resource policy, two falsification tests and one revision condition. Use a new example. L200 remains INCOMPLETE; teacher review is required. Revisit in 1, 7 and 30 days. Ask the agent about unclear steps.')
 code("submission={'baseline_rationale':'','comparison_unit':'','information_control':'','resource_policy':'','falsification_tests':[],'revision_condition':'','defense':'PENDING_WRITTEN_DEFENSE'}\nPath('b01-submission.json').write_text(json.dumps(submission,indent=2)+'\\n')\nprint('Author checks do not grade the written defense.')")
 for i,c in enumerate(cells):c.id=hashlib.sha256((str(i)+c.source).encode()).hexdigest()[:12]
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
 nb.write(book,(P/'solutions' if solution else P)/(S+'.ipynb'))
print('Built B01 lesson, reference, two portable notebooks and frozen audit archive')
