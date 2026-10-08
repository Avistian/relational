"""Canonical prose, reference, portable raw evidence and visible learner code."""
import ast,base64,gzip,hashlib,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0167-tabular-to-relational-fm-transfer';E=P/'evidence/l167'
report=json.loads((E/'report.json').read_text());pins=json.loads((E/'input-manifest.json').read_text())
def definitions(path):
    text=path.read_text()
    return [(n.name,ast.get_source_segment(text,n)) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)]
def table():
    lines=['| Model | Paper AUROC | Audited mean ± sample SD | Verdict |','|---|---:|---:|---|']
    for arm,row in report['models'].items():lines.append(f"| {arm} | {row['paper']:.4f} | {row['mean']:.6f} ± {row['sample_sd']:.6f} | {row['status']} |")
    return '\n'.join(lines)
captions={'architecture':'Three model-specific routes from a cutoff-safe matrix to predictions. Relational preparation and the pretrained prior are separate; TabPFN is not a measured arm.',
          'cutoff':'The feature clock filters events and the label clock filters support. Both use the individual query cutoff.',
          'paired':'All ten paired AUROC differences from the original L166 runs, independently rescored in L167. Repeated support draws on one task do not establish cross-database performance.'}
def prose(portable=False):
    text=(R/'lessons/content'/(S+'.md')).read_text()
    for name,caption in captions.items():
        src='data:image/png;base64,'+base64.b64encode((P/'figures/l167'/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l167/'+name+'.svg'
        text=text.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
    widgets={'WARMUP':('Recall: what changes during ICL, and what changed during pretraining?','<div id="warmup"></div>'),
      'PREDICT':('Predict before proceeding: can identical count/mean summaries distinguish histories [2,8] and [5,5]? Explain why.','<div id="predict"></div>'),
      'EXPLORER':('Baseline: cutoff 10, final value 12 → count 2, mean 6, support A only. At cutoff 20 → count 4, mean 31, support A/B/C. Recreate and intervene using the code below.','<div id="transfer-explorer"></div><noscript>Baseline: cutoff 10, final value 12 → count 2, mean 6, support A only. At cutoff 20 → count 4, mean 31, supports A/B/C. The notebook runs the intervention without JavaScript.</noscript>'),
      'TEACHBACK':('Explain representation, prior, adaptation, time and evidence in your written transfer map.','<div id="teachback"></div>')}
    for key,(static,html) in widgets.items():text=text.replace('[['+key+']]',static if portable else html)
    source=dict(definitions(P/'relkit/transfer_l167.py'))['temporal_summary']
    text=text.replace('[[CODE]]','Implement the visible contracts in the TODO cells below; the solution notebook supplies reference implementations.' if portable else '### Visible implementation · strict-past aggregation\n\n```python\n'+source+'\n```')
    text=text.replace('[[RESULTS]]',table())
    if portable:
        text=text.replace('](../','](https://avistian.github.io/relational/')
        text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
    return text
def doc(title,text,interactive=False):
    body=re.sub(r'<table([^>]*)>',r'<div class="route-scroll" tabindex="0"><table\1>',render(text)).replace('</table>','</table></div>')
    css=['lesson','atomic-route','checkpoint','lab-access','foundation-scope','transfer-map']
    scripts=['retrieval-pool','retrieval-bank','predict','teachback','transfer-map','l167-lesson'] if interactive else []
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in css)+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0166-rdb-pfn-synthetic-relational-priors.html">Lesson 166</a></nav><header><p class="route-kicker">Year 5 · Quarter 1 · Lesson 167</p><h1>'+title+'</h1></header>'+body+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Tabular → relational FM: what transfers?',prose(),True))
reference='''## Three questions

What information reaches the predictor? What prior shaped its weights? How does it adapt to this task?

| Layer | Transfer | Required check |
|---|---|---|
| Representation | Numeric cell/row encoders | PK/FK joins and summaries preserve needed information |
| Prior | A learned task-solving procedure | Relational pretraining distribution differs from tabular-only tasks |
| Adaptation | Labeled context with frozen weights | Support/query labels remain separated |
| Time | Existing temporal evaluation principles | Events AND support labels available before each owner cutoff |
| Evidence | AUROC and paired comparisons | Complete entity/date identities, identical supports and label orientation |

## Worked invariants

At cutoff 10, customer 7 events(2,4),(8,8),(10,100),(14,12) yield count 2, mean 6. At cutoff 20 they yield count 4, mean 31. A label ready at 12 cannot be support at cutoff 10, even when its query time is 5. Fixture assumes immediate event availability and strict-before boundaries.

Histories [2,8] and [5,5] both yield count 2, mean 5; no predictor can recover their difference from those summaries alone. Tabular input can contain relational information. Relational priors and graph-native inference are different design choices.

## Complete selected saved-evidence audit

'''+table()+'''

30 original L166 runs, 21,060 predictions, 702 queries, 512 supports, ten draws, 72 DFS features. L167 independently recomputes scores; it does not rerun inference. Paired RDB-PFN minus TabICL is +0.00437 AUROC, positive in 6/10 draws on one task. No cross-database superiority follows. Released labels complement current raw DNF semantics. Historical identity NOT_ESTABLISHED; fresh pretraining/whole paper NOT_RUN; learner PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/0167-tabular-to-relational-fm-transfer.html) · [Transfer template](../labs/l167-transfer-map-template.md) · [Protocol and commands](../labs/l167-reproduction.md) · [RDB-PFN paper](https://arxiv.org/html/2603.03805v5) · [TabICL](https://proceedings.mlr.press/v267/qu25d.html) · [TabPFN](https://www.nature.com/articles/s41586-024-08328-6).
'''
(R/'reference/tabular-relational-transfer.html').write_text(doc('Tabular → relational transfer — quick reference',reference))
(E/'report.md').write_text('# L167 complete saved-evidence audit\n\n'+table()+'\n\n30 original runs; 21,060 predictions; 702 unique test queries. 41 file hashes checked against frozen pins and original source/input/run manifests. Support indices independently regenerated.\n\nRDB-PFN minus single-table: +.05794808, positive in 10/10 draws. RDB-PFN minus TabICL: +.00436930, positive in 6/10 draws. One task only.\n\nNew L167 inference, pretraining and whole paper NOT_RUN; historical identity NOT_ESTABLISHED. Additional paid compute USD 0.\n')
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for name in sorted(pins['files']):
        info=zipfile.ZipInfo(name);info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,(P/name).read_bytes())
raw=buf.getvalue();payload=base64.b64encode(raw).decode();digest=hashlib.sha256(raw).hexdigest()
tasks={
'temporal_summary':('Aggregate events at each owner cutoff.','Queries are unique numeric[entity,cutoff] pairs; events are numeric[entity,event_time,value] triples. Return[count,mean] per query in its original order. Keep only matching entities with event_time<cutoff. Empty groups become[0,0]. Reject malformed/nonfinite input or duplicate query keys with ValueError.',"np.testing.assert_allclose(temporal_summary([[7,10],[7,20]],[[7,2,4],[7,8,8],[7,10,100],[7,14,12]]),[[2,6],[4,31]])\nprint('CHECK: every query owns its cutoff')"),
'eligible_support':('Separate old features from available labels.','keys contains unique[entity,query_time] rows; label_ready gives their label availability times, never earlier than query_time. Return original indices where BOTH query_time and label_ready are strictly below cutoff. Reject bad shapes, duplicates, nonfinite values or impossible readiness with ValueError.',"np.testing.assert_array_equal(eligible_support([[7,2],[8,5],[9,10]],[9,12,10],10),[0])\nprint('CHECK: only support A is available')"),
'keyed_auc':('Score the exact prediction population after alignment.','expected_keys and prediction_keys are unique[entity,cutoff] pairs. Labels align to expected_keys; probabilities align to prediction_keys. Require identical complete key sets, both label classes 0/1, finite probabilities in[0,1]. Reorder by keys; compute AUROC with half-credit ties. Reject invalid inputs with ValueError.',"keys=[[1,10],[1,20],[2,10],[2,20]]\nassert keyed_auc(keys,[0,1,0,1],keys[::-1],[1.,.5,.5,.1])==.875\nprint('CHECK: shuffled predictions and tied scores align correctly')")}
for solution in [False,True]:
    cells=[nb.v4.new_markdown_cell('# Lesson 167 · Tabular → relational FM transfer\n\nPortable CPU lab. Requires Python 3 and NumPy; no model downloads, cloud calls or new inference. Original raw evidence and diagrams are embedded. Public course links require publication. Complete three functions, then defend your transfer map.'),nb.v4.new_markdown_cell(prose(True)),
    nb.v4.new_code_cell('# @colab-bootstrap\nimport os\nos.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")\nimport base64,hashlib,io,json,zipfile\nfrom pathlib import Path\nimport numpy as np\nP=Path("l167-portable");P.mkdir(exist_ok=True)')]
    cells.append(nb.v4.new_markdown_cell('## PROVIDED · Original evidence, not hidden computation\nThe packet contains all 30 raw prediction arrays, original run receipts, prepared data and selected original source bytes. The new audit below verifies them; it never imports the archived model code.'))
    c=nb.v4.new_code_cell('encoded='+repr(payload)+'\nraw=base64.b64decode(encoded)\nassert hashlib.sha256(raw).hexdigest()=='+repr(digest)+'\nwith zipfile.ZipFile(io.BytesIO(raw)) as archive:\n    for name in archive.namelist():\n        assert not Path(name).is_absolute() and ".." not in Path(name).parts\n    archive.extractall(P)\npins='+repr(pins)+'\nprint("Authenticated all original evidence files")');c.metadata['tags']=['data-payload'];cells.append(c)
    for name,code in definitions(P/'relkit/transfer_l167.py'):
        title,contract,check=tasks[name]
        cells.extend([nb.v4.new_markdown_cell('## TODO · '+title+'\n\n**Contract:** '+contract),nb.v4.new_code_cell(code if solution else code.split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'),nb.v4.new_code_cell('# CHECK\n'+check)])
    for name,code in definitions(P/'_check_l167.py'):cells.append(nb.v4.new_code_cell(code))
    cells.append(nb.v4.new_code_cell('assert check167(temporal_summary,eligible_support,keyed_auc)=="PASS"\n# Intervention: change only the future purchase; observe both query cutoffs.\nfor cutoff in [10,20]:\n    for value in [12,1200]:\n        features=temporal_summary([[7,cutoff]],[[7,2,4],[7,8,8],[7,10,100],[7,14,value]])[0]\n        support=eligible_support([[7,2],[8,5],[9,10]],[9,12,10],cutoff)\n        if cutoff==10:np.testing.assert_allclose(features,[2,6])\n        else:np.testing.assert_allclose(features,[4,(112+value)/4])\n        print(cutoff,value,"features",features.tolist(),"support indices",support.tolist())\n# Representation collision, regardless of predictor architecture.\na=temporal_summary([[7,10]],[[7,1,2],[7,2,8]])\nb=temporal_summary([[7,10]],[[7,1,5],[7,2,5]])\nnp.testing.assert_array_equal(a,b)\nprint("Different histories, identical count/mean inputs")'))
    cells.append(nb.v4.new_markdown_cell('## PROVIDED · Complete independent saved-evidence audit\nRead the code: it checks 41 file hashes against pinned inputs and original manifests/receipts, regenerates the support schedule, and calls **your keyed_auc** for all 30 original runs. Rank scoring here is independently implemented from L166; author tests additionally compare sklearn and explicit positive/negative pairs.'))
    for name,code in definitions(P/'_audit_l167.py'):cells.append(nb.v4.new_code_cell(code))
    cells.append(nb.v4.new_code_cell('report=audit167(P,pins,keyed_auc)\nassert report["runs"]==30 and report["predictions"]==21060\nPath("l167-report.json").write_text(json.dumps(report,indent=2)+"\\n")\nfor arm,row in report["models"].items():print(arm,round(row["mean"],6),"SD",round(row["sample_sd"],6),row["status"])\nfor arm,row in report["paired"].items():print("RDBPFN minus",arm,round(row["mean"],6),"positive draws",row["positive_seeds"])\nprint("Fresh metric audit COMPLETE; new inference/pretraining NOT_RUN; historical identity NOT_ESTABLISHED")'))
    cells.append(nb.v4.new_markdown_cell('## EXIT · Transfer map and written defense\nFill all five rows with carries_over, requires, counterexample, evidence_limit. Write 300–500 words. Rubric 0–2 each: representation, prior/adaptation, time, fair comparison, evidence limits. Target ≥8/10 and no zero; teacher review required. The solution provides illustrative rows, not proof of your mastery. For a new full inference run, use the separate [L166 lane](https://avistian.github.io/relational/labs/l166-reproduction.md); default execution does not run it.'))
    example={'representation':{'carries_over':'Numeric row inputs','requires':'FK joins and appropriate summaries','counterexample':'[2,8] and [5,5] collide under count/mean','evidence_limit':'Constructive fixture, not benchmark superiority'},
    'prior':{'carries_over':'Pretrained prediction procedure','requires':'Audit training distribution','counterexample':'Identical API, different synthetic tasks','evidence_limit':'No fresh pretraining'},
    'adaptation':{'carries_over':'Frozen-weight labeled context','requires':'Support/query label separation','counterexample':'Using a query target as support','evidence_limit':'Mask does not certify availability'},
    'time':{'carries_over':'Temporal evaluation discipline','requires':'Feature and label availability','counterexample':'Query5 label ready12 at cutoff 10','evidence_limit':'Immediate event availability is a fixture assumption'},
    'evaluation':{'carries_over':'AUROC and paired comparison','requires':'Full entity/date keys and matched supports','counterexample':'Same entity at two cutoffs scored as one key','evidence_limit':'Ten support draws on one task'}}
    empty={name:{key:'' for key in value} for name,value in example.items()}
    cells.append(nb.v4.new_code_cell('transfer_map='+repr(example if solution else empty)+'\ndefense="" # Write your own 300–500 word defense.\nPath("l167-submission.json").write_text(json.dumps(dict(transfer_map=transfer_map,defense=defense,review=None,learner="PENDING_WRITTEN_DEFENSE"),indent=2)+"\\n")\nprint("Learner PENDING_WRITTEN_DEFENSE")'))
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    for i,c in enumerate(cells):c.id=f'l167-{i:03d}'
    path=P/('solutions' if solution else '')/(S+'.ipynb')
    if solution and path.exists():
        old=nb.read(path,4);book.metadata=old.metadata
        previous=[c for c in old.cells if c.cell_type=='code'];current=[c for c in book.cells if c.cell_type=='code']
        if [c.source for c in previous]==[c.source for c in current]:
            for prior,c in zip(previous,current):
                c.outputs=prior.outputs;c.execution_count=prior.execution_count;c.metadata=prior.metadata
    nb.write(book,path)
print('Built L167 lesson, reference and both portable notebooks')
