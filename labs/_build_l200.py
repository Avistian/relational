"""Build lesson, field guide, portable notebooks and deterministic audit archive."""
import ast,base64,hashlib,html,io,json,re,zipfile,importlib.metadata
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l200';S='0200-year-5-exit-exam';F=P/'figures/l200';F.mkdir(exist_ok=True)
r=json.loads((E/'report.json').read_text())
# Portrait layout keeps actual operator labels readable at 375px.
svg='''<svg xmlns="http://www.w3.org/2000/svg" width="440" height="1130" viewBox="0 0 440 1130" role="img" aria-labelledby="t d"><title id="t">RDB-PFN released base checkpoint</title><desc id="d">Support features and labels plus query features enter feature and target encoders. Six blocks mix features then read support rows only. Query target tokens produce two logits. The attention mask has support columns visible and query columns hidden.</desc><rect width="440" height="1130" rx="18" fill="#edf4ef"/><style>text{font-family:Arial,sans-serif;fill:#183f33;font-size:17px}.h{font-size:23px;font-weight:bold}.b{font-size:18px;font-weight:bold}.small{font-size:15px}.card{fill:white;stroke:#9fbcad;stroke-width:1.5}.arrow{stroke:#347863;stroke-width:2.5;fill:none}</style><text x="22" y="36" class="h">RDB-PFN · inference path</text><text x="22" y="63">Released base: 6 blocks · width 96</text>
<rect x="20" y="85" width="400" height="115" rx="10" class="card"/><text x="35" y="113" class="b">512 support + 702 query rows</text><text x="35" y="140">Same numeric DFS features in all arms</text><text x="35" y="166">Support labels observed; query labels hidden</text><text x="35" y="189" class="small">Prior learned before this task; weights fixed here</text>
<path class="arrow" d="M220 202v23m-6 -6l6 6 6-6"/>
<rect x="20" y="231" width="400" height="122" rx="10" class="card"/><text x="35" y="260" class="b">Encode features + target token</text><text x="35" y="286">Support statistics → scalar projection</text><text x="35" y="312">Query target input = mean(support labels)</text><text x="35" y="339" class="small">B × 1214 × (F + 1) × 96</text>
<path class="arrow" d="M220 355v23m-6 -6l6 6 6-6"/>
<rect x="20" y="384" width="400" height="440" rx="10" fill="#d9ebe1" stroke="#578b72" stroke-width="2"/><text x="35" y="415" class="b">Repeat six times</text>
<rect x="35" y="432" width="370" height="75" rx="8" class="card"/><text x="48" y="460" class="b">1 · Mix features within each row</text><text x="48" y="488">4 heads × 24 coordinates</text>
<text x="35" y="539" class="b">2 · Each row reads support only</text>
<text x="180" y="568" class="small">keys / values →</text><text x="181" y="592" class="small">S1</text><text x="230" y="592" class="small">S2</text><text x="279" y="592" class="small">Q1</text><text x="328" y="592" class="small">Q2</text>
'''
for i,label in enumerate(['S1','S2','Q1','Q2']):
 y=603+26*i;svg+=f'<text x="131" y="{y+18}" class="small">{label}</text>'
 for j in range(4):
  x=175+49*j;fill='#32745e' if j<2 else '#f8faf8';svg+=f'<rect x="{x}" y="{y}" width="40" height="22" rx="3" fill="{fill}" stroke="#9fbcad"/>'
svg+='''<text x="35" y="730" class="small">Green = visible. Query columns stay hidden.</text><text x="35" y="755" class="small">Illustrative 2-support / 2-query mask</text><text x="35" y="786">3 · Feed-forward 96 → 192 → 96</text><text x="35" y="811" class="small">Residual additions and layer normalization</text>
<path class="arrow" d="M220 826v23m-6 -6l6 6 6-6"/>
<rect x="20" y="855" width="400" height="118" rx="10" class="card"/><text x="35" y="886" class="b">Read query target tokens</text><text x="35" y="914">96 → 192 → 2 logits → softmax</text><text x="35" y="941">Save 702 probabilities + complete keys</text><text x="35" y="963" class="small">Never substitute held-out labels as model input</text>
<text x="22" y="1013" class="b">Comparison protocol</text><text x="22" y="1041">3 configurations × 10 paired support draws</text><text x="22" y="1069">21,060 fresh predictions; one test population</text><text x="22" y="1101" class="small">Checkpoint width 96; paper appendix states 128.</text></svg>'''
(F/'architecture.svg').write_text(svg)
figure='<figure class="exit-figure"><img src="../labs/figures/l200/architecture.svg" alt="RDB-PFN portrait architecture: feature mixing then support-only row attention in six blocks, followed by query target decoding."><figcaption>Which paths could carry a held-out label? None: only support labels enter. The miniature mask illustrates the rule; the actual run has 512 support and 702 query rows. TabICL is a separate released comparator.</figcaption></figure>'
status='<div class="exit-status"><strong>Fresh selected reproduction:</strong> '+r['execution']+'. 30 evaluations · 21,060 predictions. <strong>Year 5 exit:</strong> INCOMPLETE; proposal and defense pending. <strong>RDBLearn:</strong> INCOMPLETE_SOURCE_PREPROCESSING_GATE.</div>'
results='| Configuration | Fresh mean AUROC ± sample SD | Paper target | Comparison |\n|---|---:|---:|---|\n'
for a,m in r['models'].items():results+=f"| {a} | {m['mean']:.6f} ± {m['sample_sd']:.6f} | {m['paper_target']:.4f} | {m['status']} |\n"
d=r['paired_rdbpfn_minus_tabicl'];results+=f"\nRDB-PFN minus TabICL: **{d['mean']:+.6f}** mean paired difference, positive in **{d['positive']}/10** support draws. [Every draw and full report](../labs/evidence/l200/report.json). These are new L200 predictions.\n"
prose=(R/'lessons/content'/(S+'.md')).read_text()
for key,value in dict(STATUS=status,ARCHITECTURE=figure,RESULTS=results).items():prose=prose.replace('[['+key+']]',value)
def doc(title,body,scripts=False):
 body=body.replace('<table>','<div class="exit-scroll" tabindex="0" role="region" aria-label="Scrollable comparison table"><table>').replace('</table>','</table></div>')
 js=''.join('<script src="../assets/'+name+'.js"></script>' for name in ['retrieval-pool','retrieval-bank','predict','teachback','year-five-exit']) if scripts else ''
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/year-five-exit.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html">Curriculum</a></nav>'+body+'</article>'+js+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Lesson 200 — Year 5 exit exam',render(prose),True))
reference='''# Year 5 exit · field guide

**One deliverable:** an auditable reproduction plus a defensible research proposal. Three separate gates: reproduction, proposal, defense. Author execution never fills learner work.

## Exact selected experiment

RDB-PFN v5 Table9,F1 driver-dnf;3fixed configurations×10paired support draws×702queries. 512support examples from training. All30runs required. Query identity=(driverId,date). Fresh checkpoint inference≠pretraining. Notebook default=complete replay of fresh L200 artifacts.

'''+results+'''
## Reject bad evidence

Require hashes, complete identities, original labels, paired support identities and finite probabilities. Recompute every AUROC. For positive-negative pairs, win=1,tie=½,loss=0. Preserve all draws. A complete grid can be OUTSIDE_TOLERANCE; a close mean can hide an incomplete grid. Both must be checked.

## Explain the boundary

One test population; support SD is not database uncertainty. Released label orientation and width96 discrepancy preserved. Full availability/historical identity unestablished; raw DFS reconstruction and fresh pretraining unrun. RDBLearn still source-gated. No broad superiority claim.

## Write the proposal

Three alternatives → justified ranking → one hypothesis. Specify complete arm matrix, data/arrival policy, full keys, splits, validation selection, metric, practical threshold, uncertainty unit, aggregate cost, stop condition and revision evidence. Unknown inputs block execution. Prior exploration must be disclosed. A plan is not a public preregistration; see [COS guidance](https://www.cos.io/initiatives/prereg).

## Defense

Explain support-only information flow; distinguish fresh inference from replay; name a conclusion unsupported by the score; state an observation that would change your proposal. Submit1,000–1,500words and artifacts. All three PASS means READY_FOR_TEACHER_REVIEW,not automatic mastery.

[Lesson](../lessons/0200-year-5-exit-exam.html) · [Template](../labs/evidence/l200/proposal-template.md) · [Protocol](../labs/l200-reproduction.md) · [Portable audit](../labs/evidence/l200/reproducer.zip) · [Primary paper](https://arxiv.org/html/2603.03805v5).
'''
(R/'reference/year-5-exit-exam.html').write_text(doc('Year 5 exit field guide',render(reference)))
entries={'README.md':b'L200 complete raw prediction audit. Python3.12, NumPy. Run python labs/_audit_l200.py and python labs/_verify_l200.py. Use labs/_budget_l200.py as a subprocess wrapper for aggregate 3600s cap. This is saved replay of fresh author execution, not fresh inference. For complete inference see l200-reproduction.md. Learner defense remains pending.\n','requirements.txt':('numpy=='+importlib.metadata.version('numpy')+'\n').encode(),'l200-reproduction.md':(P/'l200-reproduction.md').read_bytes()}
for name in ['_audit_l200.py','_verify_l200.py','_test_l200.py','_budget_l200.py','relkit/exit_l200.py']:entries['labs/'+name]=(P/name).read_bytes()
entries['labs/relkit/__init__.py']=b''
manifest=json.loads((E/'packet-manifest.json').read_text())
for name in [*manifest['files'],'packet-manifest.json','report.json','proposal-template.md']:entries['labs/evidence/l200/'+name]=(E/name).read_bytes()
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for name,data in sorted(entries.items()):
  info=zipfile.ZipInfo(name,(2026,10,2,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
payload=buf.getvalue();(E/'reproducer.zip').write_bytes(payload)
def funcs(path):
 text=path.read_text();return '\n\n'.join(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)))
contracts={n.name:ast.get_source_segment((P/'relkit/exit_l200.py').read_text(),n) for n in ast.parse((P/'relkit/exit_l200.py').read_text()).body if isinstance(n,ast.FunctionDef)}
portable=prose.replace(figure,'![RDB-PFN information flow](data:image/svg+xml;base64,'+base64.b64encode(svg.encode()).decode()+')')
portable=re.sub(r'<div id="[^"]+"></div>','',portable).replace('](../','](https://avistian.github.io/relational/')
portable=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',portable)
def notebook(solution):
 cells=[]
 def md(s):cells.append(nb.v4.new_markdown_cell(s))
 def code(s,tags=None):cells.append(nb.v4.new_code_cell(s,metadata={'tags':tags or []}))
 md('# Lesson 200 · Year 5 exit exam\n\nPROVIDED explains and supplies source. TODO implements three live contracts. CHECK calls your functions. EXIT writes a proposal for teacher review. The default replays all raw new L200 predictions; a fresh model run is a separate optional track. Complete narrative and portable architecture follow. Runtime budget: 3600s aggregate local work; package installation counts. No paid cloud calls in the default notebook.')
 code('# @colab-bootstrap\nimport ast,base64,hashlib,io,json,os,sys,tempfile,time,zipfile,subprocess,importlib.util,shutil\nfrom pathlib import Path\nstarted=time.monotonic()\nfor package in ["numpy","torch"]:\n    if importlib.util.find_spec(package) is None:\n        subprocess.run([sys.executable,"-m","pip","install",package],check=True,timeout=300)\nimport numpy as np\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F\ntorch.set_num_threads(1)\nworkspace=Path(tempfile.mkdtemp(prefix="l200-exam-"))')
 md(portable)
 code('payload=base64.b64decode('+repr(base64.b64encode(payload).decode())+')\nassert hashlib.sha256(payload).hexdigest()=='+repr(hashlib.sha256(payload).hexdigest())+'\nwith zipfile.ZipFile(io.BytesIO(payload)) as archive:archive.extractall(workspace)\nevidence=workspace/"labs/evidence/l200"\nprint("Authenticated full raw packet",len(payload),"bytes")',['data-payload'])
 md('## PROVIDED · Visible checkpoint-compatible model\n\nTrace the two attention axes, support statistics and target padding. This exact source defines the model below; it is not hidden inside an import. The two-table prior helper is explicitly a course illustration, not the released pretraining generator. The author performed checkpoint parity separately; the small live check here uses initialized weights.')
 code((E/'packet/rdbpfn_visible.py').read_text())
 code('torch.manual_seed(200)\nmodel=RDBPFN().double().eval()\nx=torch.randn(1,7,3,dtype=torch.float64)\ny=torch.tensor([[0.,1.,0.,1.]],dtype=torch.float64)\nwith torch.no_grad():\n    baseline=model((x,y),4)\n    changed=x.clone();changed[:,5,:]+=10\n    intervened=model((changed,y),4)\nassert baseline.shape==(1,3,2)\ntorch.testing.assert_close(baseline[:,0],intervened[:,0],atol=1e-10,rtol=1e-10)\nprint("Changing another query leaves the first query unchanged; query labels are absent.")')
 for name,args,contract in [('keyed_auc','keys, labels, prediction_keys, probabilities','Join two-column complete keys, reject duplicates/missing keys and invalid labels/probabilities; return tie-aware pairwise AUROC. Binary labels must contain both classes.'),('complete_grid','records','Require precisely three named arms × seeds0–9,702rows and512support each. Reject booleans as numeric counts. Return runs=30,predictions=21060 only for the complete grid.'),('exit_gate','reproduction, proposal, defense','Accept PASS/FAIL/PENDING only. Return ordered blockers and INCOMPLETE if any is not PASS; otherwise READY_FOR_TEACHER_REVIEW. Never grant mastery.')]:
  md('## TODO · '+name+'\n\n**Goal:** '+contract+'\n\n**Why:** incorrect evidence admission can invalidate an apparently close score. **Hint:** validate before aggregating; do not edit CHECK.')
  code(contracts[name] if solution else 'def '+name+'('+args+'):\n    raise NotImplementedError("'+name+'")')
 md('## CHECK · live learner contracts\n\nThese call your implementations. A constant score, accepting a partial grid, or automatic exam pass fails.')
 code(funcs(P/'_test_l200.py')+'\nprint(checks(keyed_auc,complete_grid,exit_gate))')
 md('## PROVIDED · Complete authenticated report\n\nThe report below joins each fresh prediction with the original query labels and supports. Your functions are passed into the actual report computation. All30runs are required.')
 code(funcs(P/'_audit_l200.py')+'\nreport=audit(evidence,score=keyed_auc,grid=complete_grid,gate=exit_gate)\nassert report==json.loads((evidence/"report.json").read_text())\nPath("l200-report.json").write_text(json.dumps(report,indent=2))\nprint(json.dumps(report,indent=2))')
 md('## PROVIDED · Independent verifier\n\nThis second AUROC algorithm sorts and assigns tied average ranks. It does not reuse pairwise comparisons. The full verifier also rejects seven evidence mutations in disposable copies.')
 code(funcs(P/'_verify_l200.py')+'\nprint(verify(evidence))')
 md('## EXIT · Write and defend your own proposal\n\nUse the rubric and template above. Blank or author examples are not learner evidence. Save your1,000–1,500word proposal separately and bring it to the teacher. The following record deliberately keeps both learner gates pending.')
 code('submission={"reproduction":report["comparison"],"proposal":"PENDING_WRITTEN_DEFENSE","defense":"PENDING_WRITTEN_DEFENSE","exam":exit_gate("PASS" if report["comparison"]=="CLOSE" else "FAIL","PENDING","PENDING")}\nPath("l200-submission.json").write_text(json.dumps(submission,indent=2))\nPath("l200-proposal-template.md").write_text((evidence/"proposal-template.md").read_text())\nassert time.monotonic()-started<3600\nprint(submission)')
 md('## Optional · Full fresh checkpoint execution\n\nThe author already executed all30fresh evaluations for L200. To run again, use a new output directory and a separate execution budget. Python3.11 with pinned [requirements](https://avistian.github.io/relational/labs/l166-requirements.txt) and Torch2.5.1; download the course repository, authenticate [protocol](https://avistian.github.io/relational/labs/l200-reproduction.md) and source ledger. This may require GPU resources; do not run it as part of the default notebook. No automatic download or paid dispatch occurs here.\n\n```sh\npython labs/_fetch_l166.py --out /tmp/l200-input\npython labs/_run_l166.py --input /tmp/l200-input --out /tmp/l200-fresh-evaluation\n```\n\nThe next cell exposes the exact complete evaluator source for inspection. It is stored as text, not executed in this replay session.')
 code('fresh_evaluator_source = '+repr((P/'_run_l166.py').read_text())+'\nprint(fresh_evaluator_source)')
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(book.cells):c['id']='l200-'+str(i)
 return book
nb.write(notebook(False),P/(S+'.ipynb'));nb.write(notebook(True),P/'solutions'/(S+'.ipynb'))
print('Built L200; archive bytes',len(payload))
