"""Build the checkpoint lesson, reference and fully portable evidence notebooks."""
import ast,base64,hashlib,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0170-fm-design-checkpoint';E=P/'evidence/l170'
report=json.loads((E/'report.json').read_text());manifest=json.loads((E/'input-manifest.json').read_text())
def definitions(path):
    text=path.read_text();return [(n.name,ast.get_source_segment(text,n)) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)]
def results_table():
    lines=['| Context K | F1 gain ± SD | Positive draws | Trial gain ± SD | Positive draws |','|---|---:|---:|---:|---:|']
    for k in [64,128,256,512,1024]:
        a=report['paired']['by_task_context']['rel-f1/'+str(k)];b=report['paired']['by_task_context']['rel-trial/'+str(k)]
        lines.append(f"| {k} | {a['mean']:+.5f} ± {a['sample_sd']:.5f} | {a['positive_seeds']}/10 | {b['mean']:+.5f} ± {b['sample_sd']:.5f} | {b['positive_seeds']}/10 |")
    return '\n'.join(lines)+'\n\nGain = RDB-PFN minus TabICL AUROC. SD describes support-draw variation within each fixed task. All numbers are authenticated author-reference evidence, not learner results.'
captions={
'paradigms':'Trace the target labels: Griffin updates target weights; the two ICL routes supply labels as inputs. RDBLearn is a pipeline approach, not an extra model arm in this replay. On narrow screens, scroll the diagram horizontally.',
'paired':'Each dot is one paired support draw; the white diamond is the mean. Zero and identical horizontal scales expose sign reversals. All 100 differences come from reused raw predictions. Scroll horizontally on narrow screens.',
'gates':'The replay authenticates computations. Missing pipeline and pretraining evidence still constrain a design. The exact-repeatability failure remains visible. Scroll horizontally on narrow screens.'}
def prose(portable=False):
    text=(R/'lessons/content'/(S+'.md')).read_text()
    for name,caption in captions.items():
        src='data:image/png;base64,'+base64.b64encode((P/'figures/l170'/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l170/'+name+'.svg'
        style=' style="overflow-x:auto"' if portable else ''
        imgstyle=' style="min-width:720px;width:100%"' if portable else ''
        text=text.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"{style}><img{imgstyle} src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
    widgets={
'WARMUP':('Recall without looking: why is a support seed not a new database? Why does zero-gradient adaptation still use labels?','<div id="warmup"></div>'),
'PREDICT':('Before inspecting the results, predict whether a positive mean gain at K=512 must persist at K=128. Explain your answer.','<div id="predict"></div>'),
'EXPLORER':('Authenticated baseline: artifacts and metrics are checked. Full DFS, temporal availability, exposure, fresh pretraining, matched pipelines, adequate database generalization and exact repeatability prerequisites are not all met. Use your claim_gate function below to explore hypothetical declarations.','<div id="design-explorer"></div><noscript>Authenticated baseline: only the artifact and metric prerequisites are met. Saved replay is ready for review; complete-pipeline, fresh-pretraining, general-advantage and exact-repeatability claims remain unestablished. The notebook computes the same checklist without JavaScript.</noscript>'),
'TEACHBACK':('Write your defense before consulting the template outline. Ask the teacher to challenge its weakest evidence.','<div id="teachback"></div>')}
    for token,(static,html) in widgets.items():text=text.replace('[['+token+']]',static if portable else html)
    code='''```python
# A two-seed arithmetic example, not the complete lab implementation.
a = {0: .72, 1: .68}
b = {0: .70, 1: .69}
paired = [a[seed] - b[seed] for seed in sorted(a)]
print(sum(paired) / len(paired))  # approximately +0.005
```'''
    text=text.replace('[[CODE]]','In the first TODO below, implement the complete-grid comparison, retaining task/context/seed identity.' if portable else code).replace('[[RESULTS]]',results_table())
    if portable:
        text=text.replace('](../','](https://avistian.github.io/relational/');text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
    return text

def doc(title,text,interactive=False):
    body=render(text).replace('<table>','<div class="route-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
    css=['lesson','atomic-route','checkpoint','lab-access','foundation-scope','design-evidence'];scripts=['retrieval-pool','retrieval-bank','predict','teachback','design-evidence','l170-evidence','l170-lesson'] if interactive else []
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in css)+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0169-scaling-laws-open-questions.html">Lesson 169</a></nav><header><p class="route-kicker">Year 5 · Quarter 1 · Lesson 170</p><h1>'+title+'</h1></header>'+body+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Defend a relational foundation-model design',prose(),True))
(R/'assets/l170-evidence.js').write_text('window.L170Evidence='+json.dumps(report['evidence'],separators=(',',':'))+';\n')
reference='''**One skill:** connect a proposed relational predictor to evidence that could justify or overturn the choice.

| Decision | Question | Examples |
|---|---|---|
| Representation | What information reaches the predictor? | Row/key graph or materialized relational aggregates |
| Pretraining | What learned preference precedes this task? | Supervised source tasks; synthetic relational tasks; existing tabular FM |
| Target adaptation | Where do target labels enter? | Weight updates or labeled input context |

Griffin's supervised graph transfer is not graph-native ICL. RDB-PFN uses a synthetic relational prior and a frozen target predictor. RDBLearn featurizes linked records before applying a pretrained tabular ICL model. Training-free target inference still uses labels and computational resources.

**Pairing rule:** align full query keys before AUROC; pair model scores by task, context and support seed. Average task gains with equal task weights. Support-draw SD is not uncertainty across unseen databases. Freeze context on validation data, not the observed test curves.

'''+results_table()+'''

**Evidence boundary:** complete 300-run / 229,050-prediction saved replay; zero fresh L170 inference. Original 240 fresh L169 +60 L166/L168 evaluations. Exact TabICL repeatability FAIL; Griffin INCOMPLETE_BUDGET_GATE; RDBLearn pipeline and fresh pretraining NOT_RUN. Full DFS regeneration NOT_RUN; historical availability, identity and target-schema exclusion NOT_ESTABLISHED. Checklist completeness only makes a claim READY_FOR_REVIEW.

**Design checklist:** task/cutoff/horizon; information contract; three candidate mechanisms; supporting and opposing evidence; matched experiment and validation selection; measurable falsifier; aggregate cost and stop rule; limitations. Rubric: five axes ×0–2, >=8 with no zero, computational checks and teacher review. Learner PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/0170-fm-design-checkpoint.html) · [Design template](../labs/l170-design-template.md) · [Protocol](../labs/l170-reproduction.md) · [Griffin](https://arxiv.org/html/2505.05568v1) · [RDB-PFN v5](https://arxiv.org/html/2603.03805v5) · [RDBLearn](https://arxiv.org/html/2602.18495v1).
'''
(R/'reference/fm-design-checkpoint.html').write_text(doc('FM design checkpoint — quick reference',reference))
(E/'report.md').write_text('# L170 complete saved-evidence replay\n\n'+results_table()+'\n\nAll 300 evaluations / 229,050 predictions are reused in L170. Exact TabICL repeatability FAIL; Griffin INCOMPLETE_BUDGET_GATE; RDBLearn pipeline and fresh pretraining NOT_RUN. Learner PENDING_WRITTEN_DEFENSE.\n')
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for name in sorted(manifest['files']):
        info=zipfile.ZipInfo(name);info.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(info,(P/name).read_bytes())
raw=buf.getvalue();payload=base64.b64encode(raw).decode();digest=hashlib.sha256(raw).hexdigest()
contracts={
'paired_design':('Pair before comparing paradigms',
'''Accept records with database,context,arm,seed,auc. Require exactly the full 2×5×3×10 grid (databases rel-f1/rel-trial; contexts64/128/256/512/1024; arms RDBPFN/RDBPFN_single/TabICLv1.1; integer seeds0–9). Reject duplicate, missing, nonfinite or out-of-range AUROC values and invalid identifiers; reject Boolean contexts/seeds.

Return by_task_context: keys database/context, each containing per_seed (RDBPFN−TabICLv1.1 for seeds0–9), mean, sample_sd (denominator9), positive_seeds. Return macro_by_context with string K keys and the equally weighted average of the two task means. Return uncertainty_unit="SUPPORT_DRAWS_WITHIN_FIXED_TASK". Both output dictionaries must use numeric ascending context order.''',
'''# CHECK: a complete synthetic grid, not a paper result.
example=[dict(database=db,context=k,arm=arm,seed=s,auc=.6+(.02 if arm=="RDBPFN" else 0)) for db in ["rel-f1","rel-trial"] for k in [64,128,256,512,1024] for arm in ["RDBPFN","RDBPFN_single","TabICLv1.1"] for s in range(10)]
assert abs(paired_design(example)["macro_by_context"]["512"]-.02)<1e-12
print("CHECK: complete paired comparison")'''),
'adaptation_mode':('Locate the target labels',
'''Accept nonnegative integer target_labels and gradient_steps; reject Booleans. This contract covers supervised target adaptation only: positive gradient_steps with zero target_labels is invalid. Return TARGET_WEIGHT_UPDATES for positive steps, otherwise LABELED_CONTEXT for positive labels, otherwise NO_TARGET_LABELS_OR_UPDATES. Do not call zero-gradient labeled context zero-shot.''',
'''assert adaptation_mode(512,0)=="LABELED_CONTEXT"
assert adaptation_mode(512,5)=="TARGET_WEIGHT_UPDATES"
print("CHECK: labeled context is distinguished from weight updates")'''),
'claim_gate':('Keep missing evidence attached to the claim',
'''Input claim and a complete evidence dictionary of nine Boolean fields: artifacts, metrics, dfs, temporal, exposure, fresh_training, matched_pipeline, heldout_databases, repeatability. Reject unknown claims, missing/extra fields and non-Boolean values. Required fields in output order:

- saved_replay: artifacts, metrics
- full_pipeline: artifacts, metrics, dfs, temporal
- fresh_pretraining: artifacts, metrics, fresh_training, exposure
- general_advantage: artifacts, metrics, matched_pipeline, heldout_databases, temporal, exposure
- exact_repeatability: artifacts, metrics, repeatability

Return status=NOT_ESTABLISHED if any required field is false, otherwise READY_FOR_REVIEW; return missing as an ordered list of those false required fields. This is a necessary-evidence checklist, not an automatic scientific verdict.''',
'''facts={k:False for k in ["artifacts","metrics","dfs","temporal","exposure","fresh_training","matched_pipeline","heldout_databases","repeatability"]}
facts.update(artifacts=True,metrics=True)
assert claim_gate("full_pipeline",facts)["missing"]==["dfs","temporal"]
print("CHECK: narrower passes do not erase broader gaps")''')}
for solution in [False,True]:
    cells=[nb.v4.new_markdown_cell('# Lesson 170 · Defend a relational FM design\n\nPortable CPU-only saved-evidence replay: Python3 + NumPy, no downloads or cloud calls. Tier B real relational evidence. Synthetic CHECK data only isolates mechanisms. PROVIDED code authenticates raw files; your three TODO functions determine the report. CHECK gives feedback; EXIT requires your own written defense. This is author preparation, not learner mastery.'),nb.v4.new_markdown_cell(prose(True)),nb.v4.new_code_cell('# @colab-bootstrap\nimport os\nos.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")\nimport base64,hashlib,io,json,zipfile\nfrom pathlib import Path\nimport numpy as np\nP=Path("l170-portable");P.mkdir(exist_ok=True)')]
    cells.append(nb.v4.new_markdown_cell('## PROVIDED · Original raw evidence\nAuthenticate source snapshots, released arrays, supports, original per-run receipts and all selected prediction files. The original saved report is included only as a parity target; all metrics below are recomputed from raw predictions. Original L169 diagnostics are retained separately from experiment cells.'))
    c=nb.v4.new_code_cell('encoded='+repr(payload)+'\nraw=base64.b64decode(encoded)\nassert hashlib.sha256(raw).hexdigest()=='+repr(digest)+'\nwith zipfile.ZipFile(io.BytesIO(raw)) as archive:\n    for name in archive.namelist():assert not Path(name).is_absolute() and ".." not in Path(name).parts\n    archive.extractall(P)\nmanifest='+repr(manifest)+'\nprint("Authenticated packet: 300 evaluation cells; 229,050 probabilities to rescore")');c.metadata['tags']=['data-payload'];cells.append(c)
    for filename,title in [('relkit/transfer_l167.py','Full-key AUROC'),('relkit/scaling_l169.py','Original support sampler and context curves'),('_audit_l169.py','Original source/receipt/query audit')]:
        cells.append(nb.v4.new_markdown_cell('## PROVIDED · '+title+'\nVisible canonical code from the aligned earlier experiment. Source checks compare these function bodies with their pinned originals. This checkpoint reuses the existing model evaluation and audits all outputs.'))
        for name,code in definitions(P/filename):
            if filename=='relkit/transfer_l167.py' and name!='keyed_auc':continue
            cells.append(nb.v4.new_code_cell(code))
    for name,code in definitions(P/'relkit/design_l170.py'):
        title,contract,check=contracts[name]
        cells.extend([nb.v4.new_markdown_cell('## TODO · '+title+'\n\n**Why:** this function feeds the final replay report; a hard-coded answer cannot pass the independent cases.\n\n**Contract:** '+contract),nb.v4.new_code_cell(code if solution else code.split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'),nb.v4.new_code_cell('# CHECK\n'+check)])
    for name,code in definitions(P/'_check_l170.py'):cells.append(nb.v4.new_code_cell(code))
    cells.append(nb.v4.new_code_cell('assert check170(paired_design,adaptation_mode,claim_gate)=="PASS"\nprint("All three live learner contracts passed")'))
    cells.append(nb.v4.new_markdown_cell('## PROVIDED · Recompute the complete checkpoint report\nEvery curve derives from raw keyed predictions. Your pairing, adaptation and gate functions drive the final output. Original evidence flags remain conservative: narrow timestamp checks do not establish full historical availability.'))
    for name,code in definitions(P/'_replay_l170.py'):cells.append(nb.v4.new_code_cell(code))
    cells.append(nb.v4.new_code_cell('report=replay170(P,manifest,audit169,keyed_auc,sample_context,scaling_curve,scaling_claim,paired_design,adaptation_mode,claim_gate)\nassert report["runs"]==300 and report["predictions"]==229050\nPath("l170-report.json").write_text(json.dumps(report,indent=2)+"\\n")\nprint(report["status"],"| fresh L170 inference:",report["l170_fresh_runs"])\nfor key,value in report["paired"]["by_task_context"].items():\n    print(key,"mean gain",round(value["mean"],6),"SD",round(value["sample_sd"],6),"positive draws",value["positive_seeds"])\nfor claim,result in report["claims"].items():print(claim,result)\nprint("Exact repeatability:",report["exact_repeatability"],"| learner:",report["learner"])'))
    cells.append(nb.v4.new_markdown_cell('## Experiment with a hypothetical declaration\nCopy the facts and change a missing prerequisite. Observe which gate changes. This does not authenticate the new declaration or modify the original report. Explain why even an all-true declaration only makes the claim ready for review.'))
    cells.append(nb.v4.new_code_cell('hypothetical=dict(report["evidence"])\nhypothetical.update(dfs=True,temporal=True)\nprint("Original:",claim_gate("full_pipeline",report["evidence"]))\nprint("Hypothetical:",claim_gate("full_pipeline",hypothetical))\nassert report["evidence"]["dfs"] is False'))
    cells.append(nb.v4.new_markdown_cell('## Separate fresh-run lane\nNo fresh inference or training is enabled here. The complete aligned evaluator and fixed-checkpoint model source are in the embedded `sources/l166/upstream` tree; see the [L169 portable lab](https://avistian.github.io/relational/labs/0169-scaling-laws-open-questions.ipynb) for visible inference execution. The [L169 managed protocol](https://avistian.github.io/relational/labs/l169-reproduction.md) supplies the explicit fresh command, environment and budget controls. A new run needs its own declared budget. Neither this notebook nor the design proposal dispatches it.'))
    cells.append(nb.v4.new_markdown_cell('## EXIT · Your design and defense\nWrite 500–800 words using the eight template sections above. Attach the computed report, discuss a context/seed reversal, state a fair falsifying experiment and an aggregate budget. Rubric: task/temporal contract, paradigm accuracy, evidence/uncertainty, fair experiment, cost/limitations; 0–2 each, >=8 and no zero plus teacher review. The reference solution intentionally leaves the personal defense blank.'))
    cells.append(nb.v4.new_code_cell('design={name:"" for name in ["task","information_contract","candidates","evidence","experiment","falsifier","budget","limitations"]}\n# Fill each field with your own reasoning; do not copy a reference outline.\nsubmission=dict(design=design,report_sha256=hashlib.sha256(Path("l170-report.json").read_bytes()).hexdigest(),teacher_review=None,learner="PENDING_WRITTEN_DEFENSE")\nPath("l170-submission.json").write_text(json.dumps(submission,indent=2)+"\\n")\nprint("Submit your report and design for teacher review. Learner PENDING_WRITTEN_DEFENSE.")'))
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    for i,c in enumerate(cells):c.id=f'l170-{i:03d}'
    path=P/('solutions' if solution else '')/(S+'.ipynb')
    if solution and path.exists():
        old=nb.read(path,4)
        if [(c.cell_type,c.source) for c in old.cells]==[(c.cell_type,c.source) for c in cells]:
            book.metadata=old.metadata
            for c,prior in zip(cells,old.cells):
                if c.cell_type=='code':c.outputs=prior.outputs;c.execution_count=prior.execution_count;c.metadata=prior.metadata
    nb.write(book,path)
print('Built L170 lesson, reference and both portable notebooks')
