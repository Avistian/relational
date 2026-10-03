"""Build lesson, reference and portable notebooks from authenticated report inputs."""
import ast,base64,hashlib,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l194';F=P/'figures/l194';S='0194-open-fm-analysis-report'
F.mkdir(exist_ok=True,parents=True)
r=json.loads((E/'report.json').read_text());source=(P/'relkit/report_l194.py').read_text();funcs={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
replay=(P/'_replay_l194.py').read_text();rf={n.name:ast.get_source_segment(replay,n) for n in ast.parse(replay).body if isinstance(n,ast.FunctionDef)}
plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'l194','font.size':11})
fig,axes=plt.subplots(2,1,figsize=(4.3,6.4),layout='constrained');fig.patch.set_facecolor('#f3f7f3')
for ax,title,names,values,unit in [(axes[0],'Item churn\nHigher AUROC is better',['RDBLearn','AutoGluon+DFS'],[.8188,.7953],'AUROC'),(axes[1],'Item lifetime value\nLower MAE is better',['RDBLearn','AutoGluon+DFS'],[48.5044,57.0],'MAE in target units')]:
 ax.barh(names,values,color=['#216959','#496d98']);ax.invert_yaxis();ax.set_xlabel(unit);ax.set_title(title,loc='left',fontsize=10.5,weight='bold');ax.set_xlim(0,max(values)*1.25);ax.spines[['top','right']].set_visible(False)
 for i,v in enumerate(values):ax.text(v+max(values)*.025,i,f'{v:.4f}',va='center',fontsize=10)
fig.suptitle('Printed scores → oriented gaps\nTwo metrics, separate units',fontsize=13,weight='bold')
for ext in ['png','svg']:fig.savefig(F/('gaps.'+ext),dpi=160,metadata={'Date':None} if ext=='svg' else {'Software':'L194'})
plt.close(fig)
# Reuse the existing model recap; copy portable assets so this package's figures stay available.
for name in ['architecture-mobile.png','architecture-mobile.svg']:(F/name).write_bytes((P/'figures/l193'/name).read_bytes())
status='<div class="repro-status"><strong>Full reproduction: INCOMPLETE_SOURCE_PREPROCESSING_GATE.</strong> 0/21 measured tasks; 0/630 model evaluations. This lesson completes evidence analysis, not inference. $0 new cloud/API.</div>'
table='| Task | Metric | Published RDBLearn | Published AutoGluon+DFS | Oriented gap | Fresh result |\n|---|---|---:|---:|---:|---|\n'+'\n'.join(f"| {x['task']} | {x['metric']} | {x['paper_model']:.4f} | {x['paper_comparator']:.4f} | {x['gap']:+.4f} | NOT_RUN |" for x in r['task_results'])
counts='**Published reference signs: 17 favorable, 3 unfavorable, 1 equal at printed precision.**'
captions={'architecture':'RDBLearn recap from L193: relational records → DFS features → fitted map → frozen tabular predictor → validation selection and test scoring. Admission stopped at the fitted map.','gaps':'Published reference examples only. Item churn: +0.0235 AUROC. Item lifetime value: +8.4956 target units of MAE reduction. Positive gaps favor RDBLearn; units cannot be pooled.'}
def fill(text,portable=False):
 text=text.replace('[[STATUS]]',status).replace('[[TABLE]]',table).replace('[[COUNTS]]',counts).replace('[[GAP_CODE]]','```python\n'+funcs['oriented_gap']+'\n```')
 for name,caption in captions.items():
  stem='architecture-mobile' if name=='architecture' else name
  src='data:image/png;base64,'+base64.b64encode((F/(stem+'.png')).read_bytes()).decode() if portable else '../labs/figures/l194/'+stem+'.svg'
  text=text.replace('[[FIG:'+name+']]',f'<figure class="repro-figure"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 if portable:
  text=re.sub(r'<div id="[^"]+"></div>','',text).replace('<noscript>','<p>').replace('</noscript>','</p>')
  text=text.replace('](../','](https://avistian.github.io/relational/')
  text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def doc(title,body,scripts=False):
 body=body.replace('<table>','<div class="repro-table" tabindex="0" role="region" aria-label="Scrollable comparison table"><table>').replace('</table>','</table></div>')
 tail=''.join('<script src="../assets/'+n+'.js"></script>' for n in ['retrieval-pool','retrieval-bank','predict','teachback','reproduction-report']) if scripts else ''
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/multitask-reproduction.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html">Curriculum</a></nav>'+body+'</article>'+tail+'</body></html>'
prose=(R/'lessons/content'/(S+'.md')).read_text();(R/'lessons'/(S+'.html')).write_text(doc('Lesson 194 — Open FM analysis & report',render(fill(prose)),True))
ref='''# Reproduction analysis · field guide

**One question:** what is the strongest sentence this evidence permits?

| Evidence | Permitted statement | Missing evidence |
|---|---|---|
| Printed table | Descriptive comparison of those pipelines | Fresh scores, variability, mechanism |
| Synthetic source diagnostic | Invariant fails under recorded conditions | Real-task occurrence and score effect |
| Missing benchmark run | No performance conclusion | Complete valid predictions and scoring |
| Proposed intervention | Testable hypothesis | Execution and controlled interpretation |

**Metric direction:** AUROC gap = model − comparator; MAE gap = comparator − model. Positive favors the model. Missing is null, never zero. Zero at printed precision is not equivalence. Never pool raw MAEs in different units or mix AUROC and MAE.

**Worked example:** .8188 − .7953 = +.0235 AUROC; 57 − 48.5044 = +8.4956 MAE units. Both favor RDBLearn in printed rows, neither identifies a cause.

**Test an explanation:** specify what varies, what stays fixed, what you measure and what would challenge the hypothesis. More labeled support tests context sensitivity; removing a join path tests information in that path; cold/warm subgroup comparisons remain observational and can be confounded. Define cold start with past membership and retain label counts; single-class AUROC is undefined.

**Report structure:** question/protocol → full task table → uncertainty → diagnostic/competing explanations → limitations → next decisive experiment. Include all missing tasks. A hypothesis must never be filled in as the explanation of an unrun task.

**Current evidence:** all 21 fresh task results remain missing. 17/3/1 printed favorable/unfavorable/equal signs compare RDBLearn to AutoGluon+DFS. Full reproduction INCOMPLETE_SOURCE_PREPROCESSING_GATE; fresh inference NOT_RUN; historical identity and task-level attribution NOT_ESTABLISHED. Whole-paper result and learner defense remain separate.

[Lesson](../lessons/0194-open-fm-analysis-report.html) · [Notebook](../labs/0194-open-fm-analysis-report.ipynb) · [Complete report](../labs/evidence/l194/report.md) · [Protocol](../labs/l194-reproduction.md) · [Primary source](https://arxiv.org/html/2602.18495v1).
'''
(R/'reference/open-fm-analysis-report.html').write_text(doc('Reproduction analysis field guide',render(ref)))
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted((E/'packet').iterdir())+[E/'input-manifest.json']:
  info=zipfile.ZipInfo(str(p.relative_to(E)),date_time=(2026,10,2,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,p.read_bytes())
payload=base64.b64encode(buf.getvalue()).decode();digest=hashlib.sha256(buf.getvalue()).hexdigest()
contracts=[('oriented_gap','metric, model, comparator','Positive favors the model: AUROC uses model minus comparator; MAE reverses it. Accept nonnegative finite numbers (AUROC at most 1), reject bools and unknown metrics; null in either input returns null after validating the other.',"assert abs(oriented_gap('AUROC',.8188,.7953)-.0235)<1e-12\nassert abs(oriented_gap('MAE',48.5044,57)-8.4956)<1e-12\nassert oriented_gap('MAE',None,57) is None\ntry: oriented_gap('AUROC',1.1,.5)\nexcept ValueError: pass\nelse: raise AssertionError('Invalid AUROC accepted')"),('evidence_license','kind','Map PUBLISHED_TABLE → DESCRIPTIVE_REFERENCE_ONLY; SAVED_SOURCE_DIAGNOSTIC → SOURCE_INVARIANT_FAILURE_ONLY; UNRUN_BENCHMARK → NO_PERFORMANCE_CONCLUSION; PROPOSED_INTERVENTION → HYPOTHESIS_NOT_TESTED. Reject every other kind. Explain the reason for each restriction before coding.',"assert evidence_license('UNRUN_BENCHMARK')=='NO_PERFORMANCE_CONCLUSION'\nassert evidence_license('SAVED_SOURCE_DIAGNOSTIC')=='SOURCE_INVARIANT_FAILURE_ONLY'\ntry: evidence_license('PROVEN_BENCHMARK_CAUSE')\nexcept ValueError: pass\nelse: raise AssertionError('Unsupported claim accepted')"),('summarize_comparisons','rows, expected_ids','Require nonempty unique expected IDs and exactly one row per task. Count positive/negative/zero/null gaps as higher/lower/equal_at_printed_precision/missing. Reject nonnumeric, bool or nonfinite gaps. Return tasks, the four counts and scope="Descriptive signs; no pooled metric or significance claim". Never average gap magnitudes.',"toy=[dict(task='a',gap=.1),dict(task='b',gap=None),dict(task='c',gap=0)]\ns=summarize_comparisons(toy,['a','b','c'])\nassert (s['higher'],s['missing'],s['equal_at_printed_precision'])==(1,1,1)\ntry: summarize_comparisons(toy[:-1],['a','b','c'])\nexcept ValueError: pass\nelse: raise AssertionError('Missing task hidden')")]
def notebook(solution):
 cells=[]
 def md(s):cells.append(nb.v4.new_markdown_cell(s))
 def code(s,tags=None):cells.append(nb.v4.new_code_cell(s,metadata={'tags':tags or []}))
 md('# Lesson 194 · Analysis & report\n\nOffline Python 3 standard-library evidence lab. The authored reference figures and scores below are not your kernel measurements. Your functions generate a complete report from the embedded frozen packet.')
 code('# @colab-bootstrap\n# Offline: no checkout, package installation, download, model, or paid API.\nimport base64, hashlib, io, json, math, tempfile, zipfile\nfrom pathlib import Path\nfrom collections import Counter\nworkspace=Path(tempfile.mkdtemp(prefix="l194-report-"))')
 # Keep contextual prose but leave the implementation for the learner's TODO.
 md(fill(prose.replace('[[GAP_CODE]]','The signed-gap function is your first TODO below.'),True))
 md('## PROVIDED · Authenticated packet\nThe archive contains the complete inherited task and seed ledgers, paper HTML, recorded diagnostic and source identity. Hash equality establishes byte identity to this embedded snapshot, not historical truth or benchmark validity.')
 code('payload='+repr(payload)+'\nraw=base64.b64decode(payload)\nassert hashlib.sha256(raw).hexdigest()=='+repr(digest)+'\nwith zipfile.ZipFile(io.BytesIO(raw)) as z:\n    for name in z.namelist():\n        if not (workspace/name).resolve().is_relative_to(workspace.resolve()): raise ValueError("Unsafe archive path")\n    z.extractall(workspace)\npacket=workspace/"packet"\nmanifest=json.loads((workspace/"input-manifest.json").read_text())',['data-payload'])
 for name,args,contract,check in contracts:
  md('## TODO · '+name+'\n'+contract+'\n\nPredict one failure before running the CHECK. Your function is called by the full report operator.')
  code(funcs[name] if solution else 'def '+name+'('+args+'):\n    raise NotImplementedError('+repr(name)+')')
  md('### CHECK · Meaningful boundaries');code(check+'\nprint("Contract PASS")')
 md('## PROVIDED · Complete report operator\nRead how paper rows are matched by task identity, how missing scores are preserved, and how your three functions control the resulting report. No predictor is executed.')
 code(rf['build_report']);code(rf['report_markdown'])
 md('## CHECK · Full 21-task report\nExact agreement authenticates this analysis. It does not establish full model reproduction.')
 code('report=build_report(packet,manifest,oriented_gap,evidence_license,summarize_comparisons)\nassert report==json.loads('+repr(json.dumps(r,sort_keys=True))+')\nPath("l194-report.json").write_text(json.dumps(report,indent=2))\nPath("l194-report.md").write_text(report_markdown(report))\nprint(report_markdown(report))')
 md('## EXIT · Write and defend\nWrite a three-sentence abstract and a controlled follow-up. Specify varied factor, fixed alternatives, metric, falsifying observation and limitation. Explain why cold/warm strata are observational and why a source repair changes the protocol. Send your defense to the teacher; successful author execution does not certify learning.')
 code('submission=dict(learner="PENDING_WRITTEN_DEFENSE",abstract="",vary="",hold_fixed="",measure="",falsifying_observation="",limitation="",repair_boundary="")\nPath("l194-submission.json").write_text(json.dumps(submission,indent=2))')
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(cells):c.id=f'l194-{i:03}'
 return book
for solution in [False,True]:
 path=P/('solutions' if solution else '')/(S+'.ipynb');book=notebook(solution)
 if solution and path.exists():
  old=nb.read(path,4)
  if [c.source for c in old.cells]==[c.source for c in book.cells]:book=old
 nb.write(book,path)
print('Built lesson, reference, figures and student/solution notebooks')
