"""Canonical prose and portable full-evidence notebook with visible live contracts."""
import ast,base64,hashlib,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0168-cross-database-generalization';E=P/'evidence/l168'
report=json.loads((E/'report.json').read_text());pins=json.loads((E/'audit-manifest.json').read_text())
def definitions(path):
    text=path.read_text()
    return [(n.name,ast.get_source_segment(text,n)) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)]
def table():
    lines=['| Database / evidence | Model | Paper AUROC | Measured mean ± sample SD |','|---|---|---:|---:|']
    for db,d in report['datasets'].items():
        for arm,row in d['models'].items():lines.append(f"| {db} / {d['evidence']} | {arm} | {row['paper']:.4f} | {row['mean']:.6f} ± {row['sample_sd']:.6f} |")
    return '\n'.join(lines)
def interpretation():
    f=report['paired']['rel-f1'];t=report['paired']['rel-trial'];m=report['macro']
    return f'''RDB-PFN minus TabICL averages **{f['mean']:+.5f} AUROC on F1** and **{t['mean']:+.5f} on trial**, positive in {f['positive_seeds']}/10 and {t['positive_seeds']}/10 support draws respectively. The equal-database mean gain is **{m['macro_gain']:+.5f}**. All three fresh trial means are within the predeclared .02 descriptive distance from the published values; this is not a statistical equivalence test or proof of historical data identity.

The new evidence is 30 complete trial evaluations and **24,750 fresh probabilities**. The comparison also independently rescores 21,060 reused F1 probabilities. Both selected task means favor RDB-PFN over TabICL, but seed-level reversals and only two databases prevent a broad-superiority claim. The small trial gain over its single-table ablation also cautions against attributing every gain to the relational prior.'''
captions={'architecture':'Same frozen RDB-PFN checkpoint across two DFS feature widths. F1 evidence is reused, trial evidence is fresh; schema-generator provenance is a separate question.',
 'protocol':'Pretraining exclusion and target adaptation are independent axes. Griffin uses graph computation and supervised adaptation; RDB-PFN uses DFS and labeled context.',
 'results':'All ten paired RDB-PFN minus TabICL AUROC differences per database. The horizontal scale is shared; seeds measure support variation, not the distribution of new databases.'}
def prose(portable=False):
    text=(R/'lessons/content'/(S+'.md')).read_text()
    for name,caption in captions.items():
        src='data:image/png;base64,'+base64.b64encode((P/'figures/l168'/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l168/'+name+'.svg'
        text=text.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
    widgets={'WARMUP':('Recall: how do representation, prior and adaptation differ? Why are ten support seeds not ten databases?','<div id="warmup"></div>'),
      'PREDICT':('Commit your prediction: 512 target labels and zero gradient steps is few-shot ICL. Explain why before reading on.','<div id="predict"></div>'),
      'EXPLORER':('Baseline: incomplete lineage + 512 labels + frozen weights → NOT_ESTABLISHED / FEW_SHOT_ICL. With a complete target-excluding inventory → HELD_OUT / FEW_SHOT_ICL. Test all interventions with the code below.','<div id="cross-explorer"></div><noscript>Baseline: incomplete lineage + 512 labels + frozen weights → NOT_ESTABLISHED / FEW_SHOT_ICL. A complete target-excluding inventory changes holdout to HELD_OUT. The notebook runs this exercise without JavaScript.</noscript>'),
      'TEACHBACK':('Write the cross-database defense; distinguish verified inference, reported pretraining and unknown schema novelty.','<div id="teachback"></div>')}
    for key,(static,html) in widgets.items():text=text.replace('[['+key+']]',static if portable else html)
    code=dict(definitions(P/'relkit/generalization_l168.py'))['transfer_regime']
    text=text.replace('[[CODE]]','Implement the exposure/adaptation contract in the TODO cells below.' if portable else '### Visible implementation · the claim follows the evidence\n\n```python\n'+code+'\n```')
    text=text.replace('[[RESULTS]]',table()).replace('[[INTERPRETATION]]',interpretation())
    if portable:
        text=text.replace('](../','](https://avistian.github.io/relational/')
        text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
    return text

def doc(title,text,interactive=False):
    body=render(text).replace('<table>','<div class="route-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
    css=['lesson','atomic-route','checkpoint','lab-access','foundation-scope','cross-database']
    scripts=['retrieval-pool','retrieval-bank','predict','teachback','cross-database','l168-lesson'] if interactive else []
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in css)+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0167-tabular-to-relational-fm-transfer.html">Lesson 167</a></nav><header><p class="route-kicker">Year 5 · Quarter 1 · Lesson 168</p><h1>'+title+'</h1></header>'+body+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Cross-database generalization',prose(),True))
reference='''## Name the unit of generalization

New row, later query, held-out database and unseen schema are different claims. A complete target-excluding pretraining inventory supports database holdout; an incomplete inventory leaves it NOT_ESTABLISHED. Schema novelty needs a separate matching rule and corpus audit.

## Declare adaptation

| Target labels | Target updates | Category |
|---:|---:|---|
| 512 | 0 | Few-shot ICL |
| 512 | 5 | Supervised adaptation |
| 0 | 0 | Zero-label, zero-gradient protocol category; not tested here |

## Reproduction and aggregation

RDB-PFN v5 Table9,512supports,10seeds,3fixed arms. Full825trial queries freshly predicted;702F1 queries reused. Identical per-task features/supports across arms. Pair by database/seed after exact entity/date identity checks. Mean seeds within each database, then mean database gains with equal weights. For gains +.10 and −.20, macro = −.05 regardless of task sizes.

'''+table()+'\n\n'+interpretation()+'''

Both releases complement their raw task labels. Complement labels and probabilities together to preserve AUROC while changing interpretation. Trial passes twelve exposed MAX-timestamp checks; full DFS regeneration is unrun and historical arrival times are unestablished. RDB-PFN reports synthetic predictor training, but LayerDAG uses real schema corpora. Checkpoint training lineage / exact schema exclusion NOT_ESTABLISHED. Whole-paper benchmark and fresh pretraining NOT_RUN. Griffin fresh comparison INCOMPLETE_BUDGET_GATE; learner PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/0168-cross-database-generalization.html) · [Protocol](../labs/l168-reproduction.md) · [Defense template](../labs/l168-transfer-template.md) · [RDB-PFN primary source](https://arxiv.org/html/2603.03805v5) · [Griffin](https://arxiv.org/html/2505.05568v1).
'''
(R/'reference/cross-database-generalization.html').write_text(doc('Cross-database evaluation — quick reference',reference))
(E/'report.md').write_text('# L168 selected cross-database evaluation\n\n'+table()+'\n\n'+interpretation()+'\n\nFresh pretraining and whole paper NOT_RUN. Historical identity, checkpoint lineage and exact target-schema exclusion NOT_ESTABLISHED.\n')
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for name in sorted(pins['files']):
        info=zipfile.ZipInfo(name);info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,(P/name).read_bytes())
raw=buf.getvalue();payload=base64.b64encode(raw).decode();digest=hashlib.sha256(raw).hexdigest()
tasks={
 'transfer_regime':('Classify two independent axes.','Inputs: iterable of known pretraining database names, nonempty target name, explicit Boolean inventory completeness, and nonnegative integer target label/update counts. Known inclusion wins even if the inventory is incomplete. Return database_holdout SEEN / HELD_OUT / NOT_ESTABLISHED and adaptation FEW_SHOT_ICL / SUPERVISED_ADAPTATION / ZERO_LABEL_ZERO_GRADIENT. Reject supervised updates without labels; this contract does not cover unsupervised adaptation.',"assert transfer_regime(['shop','forum'],'trial',True,512,0)=={'database_holdout':'HELD_OUT','adaptation':'FEW_SHOT_ICL'}\nassert transfer_regime(['shop'],'trial',False,512,0)['database_holdout']=='NOT_ESTABLISHED'\nprint('CHECK: incomplete exposure is not verified exclusion')"),
 'paired_gains':('Pair every intended run before averaging.','records contain database, seed, arm, auc, test_rows, metric. Exactly one task per named database; models RDBPFN and TabICLv1.1; unique expected databases/seeds. Require one of every expected pair, finite AUROC in[0,1], and fixed positive integer test-row counts within each database. Return per_seed differences in sorted seed order, mean, sample_sd (None for one seed), positive_seeds, test_rows, seeds and metric for every database. Raw key/support audits run separately before this function.',"toy=[dict(database='trial',seed=s,arm=a,auc=v,test_rows=825,metric='AUROC') for s in [1,0] for a,v in [('RDBPFN',.65),('TabICLv1.1',.60)]]\nnp.testing.assert_allclose(paired_gains(toy,['trial'],[0,1])['trial']['per_seed'],[.05,.05])\nprint('CHECK: pair by identity, not row order')"),
 'database_macro':('Give the database one vote.','Input is paired_gains output for one selected task per database. Reject empty input, non-AUROC metrics, invalid or nonfinite per_seed deltas outside[-1,1], and means inconsistent with those deltas (1e-12 tolerance). Return databases, macro_gain, positive_databases, inference="DESCRIPTIVE_SELECTED_DATABASES_ONLY". Do not weight by query count or seed count.',"toy={'small':dict(per_seed=[.1,.1],mean=.1,metric='AUROC'),'large':dict(per_seed=[-.2,-.2],mean=-.2,metric='AUROC')}\nassert abs(database_macro(toy)['macro_gain']+.05)<1e-12\nprint('CHECK: equal database weight')")}
for solution in [False,True]:
    cells=[nb.v4.new_markdown_cell('# Lesson 168 · Cross-database generalization\n\nDefault: portable CPU-only saved-evidence audit, Python 3 + NumPy. Embedded raw predictions, keys, receipts and diagrams; no downloads or cloud calls. Complete three contracts before interpreting the table. Fresh inference is a separate opt-in lane near the end.'),nb.v4.new_markdown_cell(prose(True)),
    nb.v4.new_code_cell('# @colab-bootstrap\nimport os\nos.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")\nimport base64,hashlib,io,json,zipfile\nfrom pathlib import Path\nimport numpy as np\nP=Path("l168-portable");P.mkdir(exist_ok=True)')]
    cells.append(nb.v4.new_markdown_cell('## PROVIDED · Authenticate the original evidence\nThis packet contains all 60 raw prediction arrays (30 fresh trial +30 reused F1), original receipts, full prepared matrices, and pinned original model source. Computation remains visible below; the payload is data and source files, not an opaque executable helper.'))
    c=nb.v4.new_code_cell('encoded='+repr(payload)+'\nraw=base64.b64decode(encoded)\nassert hashlib.sha256(raw).hexdigest()=='+repr(digest)+'\nwith zipfile.ZipFile(io.BytesIO(raw)) as archive:\n    for name in archive.namelist():\n        assert not Path(name).is_absolute() and ".." not in Path(name).parts\n    archive.extractall(P)\npins='+repr(pins)+'\nprint("Authenticated both databases: raw predictions, prepared data, receipts and model source")');c.metadata['tags']=['data-payload'];cells.append(c)
    cells.append(nb.v4.new_markdown_cell('## PROVIDED · Full-key AUROC from Lesson 167\nRank scores after exact entity/date alignment. Independent author checks use sklearn and explicit positive/negative pairs; ties receive half credit.'))
    cells.append(nb.v4.new_code_cell(dict(definitions(P/'relkit/transfer_l167.py'))['keyed_auc']))
    for name,code in definitions(P/'relkit/generalization_l168.py'):
        title,contract,check=tasks[name]
        cells.extend([nb.v4.new_markdown_cell('## TODO · '+title+'\n\n**Contract:** '+contract),nb.v4.new_code_cell(code if solution else code.split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'),nb.v4.new_code_cell('# CHECK\n'+check)])
    for name,code in definitions(P/'_check_l168.py'):cells.append(nb.v4.new_code_cell(code))
    cells.append(nb.v4.new_code_cell('assert check168(transfer_regime,paired_gains,database_macro)=="PASS"\nfor complete in [False,True]:\n    for source in [[],["trial"]]:\n        for labels,updates in [(512,0),(512,5),(0,0)]:\n            print(source,complete,labels,updates,transfer_regime(source,"trial",complete,labels,updates))'))
    cells.append(nb.v4.new_markdown_cell('## PROVIDED · Independent complete evidence audit\nAll hashes, original run receipts, support schedules, key sets and label orientations must pass before **your paired_gains and database_macro** produce the comparison. The final table is live output, not an embedded answer.'))
    for name,code in definitions(P/'_audit_l168.py'):cells.append(nb.v4.new_code_cell(code))
    cells.append(nb.v4.new_code_cell('report=audit168(P,pins,keyed_auc,paired_gains,database_macro)\nassert report["fresh_runs"]==30 and report["all_predictions_checked"]==45810\nPath("l168-report.json").write_text(json.dumps(report,indent=2)+"\\n")\nfor db,data in report["datasets"].items():\n    print(db,data["evidence"])\n    for arm,row in data["models"].items():print(arm,round(row["mean"],6),"SD",round(row["sample_sd"],6),row["status"])\nprint("Paired:",report["paired"])\nprint("Equal-database summary:",report["macro"])\nprint("Training lineage / historical identity / exact schema exclusion NOT_ESTABLISHED")'))
    cells.append(nb.v4.new_markdown_cell('## OPTIONAL · Rerun all 30 trial predictions\nDefault execution skips this section. The model source is pinned and embedded in `l168-portable/sources/l166/upstream/model_pretrain`; inspect `src/models.py` for the full released model and classifier. Lesson166 exposes the same model as readable notebook cells. The visible runner below loads the two fixed checkpoints and the default 32-estimator TabICL; there is no checkpoint search.\n\nFor fresh execution, use Python3.11 and install torch2.5.1, numpy1.26.4, pandas2.2.3, scikit-learn1.6.1, pydantic1.10.26, pyyaml6.0.2 and tabicl0.1.3 in a separate environment. Weights download only when explicitly enabled. Local execution has no billing guard; the repository Modal lane reserves costs before dispatch. Do not set RUN_FRESH on an unbudgeted paid instance.'))
    for name,code in definitions(P/'_fetch_l168.py'):cells.append(nb.v4.new_code_cell(code))
    for name,code in definitions(P/'_run_l168.py'):cells.append(nb.v4.new_code_cell(code))
    cells.append(nb.v4.new_code_cell('RUN_FRESH=False\nif RUN_FRESH:\n    import shutil,urllib.request,sys,time,random,torch\n    destination=P/"fresh-input"\n    fetch168(destination)\n    fresh=run168(destination,P/"sources/l166/upstream",Path("l168-fresh-predictions"),list(range(10)))\n    assert len(fresh["records"])==30\nelse:\n    print("Optional fresh-inference lane NOT_RUN in this notebook; author raw evidence was rescored above.")'))
    cells.append(nb.v4.new_markdown_cell('## EXIT · Transfer table and written defense\nWrite250–400words: what the second database adds; whether zero-gradient means zero-label; how to audit unseen-schema claims; why the equal-database macro is descriptive. Rubric0–2each: exposure, adaptation, pairing, aggregation, limits. Target≥8/10and no zero; teacher review required. The reference solution supplies an example outline, not a completed learner defense. Ask the teacher about any uncertainty.'))
    example={'exposure':'Synthetic predictor training reported; real schema corpora used by generator; lineage and exact schema exclusion not independently established',
     'adaptation':'512target labels,0gradient steps = few-shot ICL; supervised Griffin adaptation is a separate protocol',
     'pairing':'Raw hashes, full entity/date keys, same support identities,10seeds before aggregation',
     'aggregation':'Mean paired gains within database, then equal mean over two selected databases',
     'limits':'F1 reused; trial fresh; no broad superiority, historical identity, fresh pretraining or learner mastery inferred'}
    cells.append(nb.v4.new_code_cell('transfer_table='+repr(example if solution else {k:'' for k in example})+'\ndefense="" # Write your own defense.\nPath("l168-submission.json").write_text(json.dumps(dict(transfer_table=transfer_table,defense=defense,review=None,learner="PENDING_WRITTEN_DEFENSE"),indent=2)+"\\n")\nprint("Learner PENDING_WRITTEN_DEFENSE")'))
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    for i,c in enumerate(cells):c.id=f'l168-{i:03d}'
    path=P/('solutions' if solution else '')/(S+'.ipynb')
    if solution and path.exists():
        old=nb.read(path,4)
        if [(c.cell_type,c.source) for c in old.cells]==[(c.cell_type,c.source) for c in cells]:
            book.metadata=old.metadata
            for c,prior in zip(cells,old.cells):
                if c.cell_type=='code':c.outputs=prior.outputs;c.execution_count=prior.execution_count;c.metadata=prior.metadata
    nb.write(book,path)
print('Built L168 lesson, reference and both portable notebooks')
