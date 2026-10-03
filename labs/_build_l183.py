"""Deterministic HTML/reference and portable offline learner/solution notebooks."""
import ast,base64,hashlib,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l183';S='0183-graph-transformer-pretraining'
r=json.loads((E/'report.json').read_text());pins=json.loads((E/'input-manifest.json').read_text())
def defs(path):
    s=path.read_text();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
fns=defs(P/'relkit/pretraining_l183.py')
status='**Author evidence:** all **7,554 saved predictions** replayed; four live audit contracts checked. Full published experiments remain **INCOMPLETE**. Fresh model inference, training and hybrid pretraining: **NOT_RUN**. Cloud/API spend: **$0**. Learner: **PENDING_WRITTEN_DEFENSE**.'
captions={
'relgt':'RelGT source architecture and proposed schema adapter. B is batch size; K is row-token count; d is vector width. Local and global branches are fused inside each block. Model dimensions are the full F1 search, not the reduced saved course runs.',
'griffin':'Griffin source/release pathway: cell encoders, metadata/task attention, relation-aware messages and shared decoding. Lᵢ counts cells in row i. Pretraining initializes the same architecture before target fine-tuning; it does not produce RelGT-compatible weights.',
'access':'Synthetic cutoff-10 policy fixture. Only the past feature passes every access rule. The query target, unknown arrival, future event and unfinished outcome window are all withheld for different reasons.',
'factorial':'Invented single-seed MAE values. MP gain 0.4; GT gain 0.7; extra GT benefit +0.3. These values illustrate arithmetic and are not trained-model evidence.',
'replay':'Six saved L146 course runs, freshly rescored on complete query keys. Dots show three seed scores; error bars show mean plus/minus sample seed standard deviation, not confidence intervals. Both arms start from scratch.'}
def result_table():
    s='| Saved course arm | Validation MAE, mean ± seed SD | Test MAE, mean ± seed SD |\n|---|---:|---:|\n'
    for arm,label in [('gnn','Course GNN'),('relgt','Reduced RelGT')]:
        a=r['summary'][arm];s+=f"| {label} | {a['val']['mean']:.6f} ± {a['val']['sample_sd']:.6f} | {a['test']['mean']:.6f} ± {a['test']['sample_sd']:.6f} |\n"
    return s

def prose(portable=False):
    s=(R/'lessons/content'/(S+'.md')).read_text().replace('[[STATUS]]',status).replace('[[RESULTS]]',result_table())
    for name,caption in captions.items():
        src='data:image/png;base64,'+base64.b64encode((P/'figures/l183'/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l183/'+name+'.svg'
        s=s.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
    tags={'WARMUP':('From memory: name the five RelGT token elements; distinguish pretrained weights from a scratch control; explain why a cutoff belongs to the query key.','<div id="warmup"></div>'),
          'PREDICT':('Predict before reading: if RelGT won validation in all three pairs, must it win test? Write your answer before continuing.','<div id="predict"></div>'),
          'EXPLORER':('Notebook intervention: change only GT pretrained MAE to 3.5 and predict the interaction; then run the TRY cell below.','<div id="factorial"></div><noscript>Synthetic default: MP 4.0 → 3.6, GT 3.9 → 3.2. Gains 0.4 and 0.7; interaction +0.3. No trained-model evidence.</noscript>'),
          'TEACHBACK':('Write the one-page brief in the EXIT cell and ask the teacher for review.','<div id="teachback"></div>')}
    for tag,(plain,html) in tags.items():s=s.replace('[['+tag+']]',plain if portable else html)
    if portable:
        s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
    return s

def doc(title,body,interactive=False):
    html=render(body).replace('<table>','<div class="route-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
    css=['lesson','atomic-route','checkpoint','lab-access','factorial-contrast'];js=['retrieval-pool','retrieval-bank','predict','teachback','factorial-contrast','l183-lesson'] if interactive else []
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 183 — '+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in css)+'</head><body class="checkpoint l183"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0164-griffin-graph-centric-rdb-fm.html">Griffin prerequisite</a></nav><header><p class="route-kicker">Year 5 · Quarter 3 · Research gap 183</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in js)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Graph-Transformer + pre-training',prose(),True))
reference='''**Question:** does source pretraining help a graph-transformer backbone more than message passing? This is a specific interaction claim, not an established novelty claim.

| Factor | Level 1 | Level 2 |
|---|---|---|
| Backbone | Message passing | Graph transformer |
| Initialization | Scratch | Pretrained |

**Lower-is-better loss:** gain = scratch − pretrained. Interaction = GT gain − MP gain. **Higher-is-better score:** gain = pretrained − scratch. Calculate within matched seed blocks and task/label budgets. Keep all four absolute scores. Positive interaction alone does not make GT the best model. Seed SD is not database uncertainty.

**RelGT interface:** five vectors per token (type, hop, time, features, local graph structure) → concatenate/project → local/global attention → prediction head. Per-table encoders and indexed type embeddings need a declared transfer interface.

**Griffin interface:** unified cells and metadata → task-conditioned cross-attention → relation-aware message passing → shared decoder. Borrowing this training idea does not make its checkpoint compatible with RelGT.

**Access:** require legal event time, known arrival by cutoff, completed outcome windows and masked query target. Unknown arrival is missing evidence. This strict policy must be distinguished from the paper's policy.

**Observed:** all7554saved L146 predictions replay; GNN wins all3test pairs, RelGT all3validation pairs. Neither has pretraining. Raw SQL labels and full temporal caches are not rerun here.

**Full protocols:** RelGT9configs×100epochs retains temporal/budget blockers; inherited forecast80.41USD. Griffin20fits retains budget blocker; inherited adjusted compute63.89USD. Those are old pilot scenarios, not current quotes. Wholepaper/hybrid training NOT_RUN. Complete implementations and commands remain in the protocol/notebook appendices.

**Next probe:** fourarms×two F1 label budgets×five seeds=40targetfits,plus source pretraining. Freeze adapter,source corpus/schedule,selection and all-in cost first. A single F1 probe cannot prove cross-database superiority.

'''+status+'''

[Lesson](../lessons/0183-graph-transformer-pretraining.html) · [Student lab](../labs/0183-graph-transformer-pretraining.ipynb) · [Full protocol](../labs/l183-reproduction.md) · [Evidence](../labs/evidence/l183/report.json) · [RelGT](https://arxiv.org/html/2505.10960v1) · [Griffin](https://arxiv.org/html/2505.05568v1).
'''
(R/'reference/graph-transformer-pretraining.html').write_text(doc('Graph-transformer pretraining — quick reference',reference))
(E/'report.md').write_text('# L183 author evidence\n\n'+status+'\n\n'+result_table()+'\nComplete saved-prediction replay. The original training happened in L146; no source pretraining is represented by these scores. Full details in report.json and l183-reproduction.md.\n')
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for name in sorted(pins['files']):
        info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,(E/'packet'/name).read_bytes())
packet=base64.b64encode(buf.getvalue()).decode();digest=hashlib.sha256(buf.getvalue()).hexdigest()

def source_chunks(path):
    source=path.read_text();lines=source.splitlines(keepends=True);nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    points=[0]+[n.lineno-1 for n in nodes]+[len(lines)]
    labels=['Imports and setup']+[getattr(n,'name','source') for n in nodes]
    return [(label,''.join(lines[a:b])) for label,a,b in zip(labels,points,points[1:]) if b>a]

def make(solution):
    cells=[]
    def md(s):cells.append(nb.v4.new_markdown_cell(s))
    def code(s,tags=None):cells.append(nb.v4.new_code_cell(s,metadata={'tags':tags or []}))
    md('# Lesson 183 · Graph-Transformer + pre-training\n\nImplement four evidence contracts, replay every saved prediction, then write a gap brief. Tier B saved F1 evidence and Tier C arithmetic/access fixtures. Default execution is offline and CPU-only. Python3 + NumPy are the only runtime requirements; model source appendices are displayed, not executed. Ask the teacher whenever an assumption is unclear.')
    code('# @colab-bootstrap: standalone offline packet; no repository clone or cloud service required.\nfrom pathlib import Path\nimport json,copy,math\nimport numpy as np\nprint("L183: saved predictions and synthetic contracts; new training NOT_RUN; cloud/API $0")')
    md(prose(True))
    md('## PROVIDED · Authenticate and unpack the evidence\nThe archive includes every saved prediction and the retained model/trainer sources. The fixed manifest checks each file before analysis. Hashes establish byte identity, not historical legality.')
    code('import base64,hashlib,io,zipfile\nPACKET='+repr(packet)+'\nraw=base64.b64decode(PACKET)\nassert hashlib.sha256(raw).hexdigest()=='+repr(digest)+"\nP=Path('l183-packet');P.mkdir(exist_ok=True)\nwith zipfile.ZipFile(io.BytesIO(raw)) as z:\n    for name in z.namelist():assert not Path(name).is_absolute() and '..' not in Path(name).parts\n    z.extractall(P)\nmanifest="+repr(pins)+"\nprint('Authenticated packet:',len(manifest['files']),'files')",['data-payload'])
    tasks=[('temporal_mask','Separate four access rules','Input: a list of cell dictionaries and a numeric cutoff. Required fields are event_time, available_at, is_label, label_end, query_target. Flags must be actual booleans; times must be finite numbers or None. Reject missing fields and label windows ending before their event time. Return one boolean per cell: require known event/arrival no later than cutoff; labels also need a known completed window; query targets are never visible. Unknown time means withhold, not proof of leakage.',"c=dict(event_time=5,available_at=6,is_label=True,label_end=12,query_target=False)\nassert temporal_mask([c,dict(c,label_end=10)],10)==[False,True]\nprint('CHECK: horizon boundary distinguishes two otherwise identical cells')"),
    ('keyed_mae','Score complete query populations','Input rows are (entity, cutoff, value). Reject duplicates in either side, empty populations, nonfinite values and different key sets. Return mean absolute error after aligning complete keys; row order is irrelevant. A driver may occur at several cutoffs.',"assert keyed_mae([(7,10,2.),(7,20,5.)],[(7,20,3.),(7,10,2.)])==1.\nprint('CHECK: repeated entity and reordered cutoff rows align')"),
    ('factorial_effect','Calculate the extra pretraining benefit','Require exactly mp_scratch, mp_pretrained, gt_scratch and gt_pretrained, each mapping seed IDs to finite scores. Require identical seed sets and at least two seeds. higher_is_better must be boolean. Return seeds (sorted), mp_gain_mean, gt_gain_mean, interaction_per_seed, interaction_mean, interaction_sample_sd. Gains are improvements in the metric direction; interaction is GT gain minus MP gain. Sample SD uses n−1. Do not pool different tasks or metric units.',"toy={k:{0:v,1:v} for k,v in zip(['mp_scratch','mp_pretrained','gt_scratch','gt_pretrained'],[4.,3.6,3.9,3.2])}\nassert abs(factorial_effect(toy)['interaction_mean']-.3)<1e-12\nprint('CHECK: invented +0.3 interaction')"),
    ('transfer_gate','Keep readiness separate from evidence of benefit','Evidence fields, in order: checkpoint_architecture, parameter_shapes, schema_semantics, task_decoder, temporal_access, database_holdout, finite_gradients, validation_selection, full_cost. Each missing or non-PASS value creates an uppercase blocker. Return status BLOCKED if any blockers, otherwise READY_FOR_SEPARATELY_AUTHORIZED_PROBE; include blockers and transfer_gain always NOT_ESTABLISHED. A flag is only an inventory entry: the caller must supply supporting evidence.',"assert transfer_gate({})['status']=='BLOCKED'\nassert transfer_gate({})['transfer_gain']=='NOT_ESTABLISHED'\nprint('CHECK: absent prerequisites cannot establish transfer')")]
    for name,title,instructions,check in tasks:
        md('## TODO · '+title+'\n'+instructions)
        n=ast.parse(fns[name]).body[0];signature='def '+name+'('+ast.unparse(n.args)+'):'
        code(fns[name] if solution else signature+'\n    raise NotImplementedError("Implement '+name+'")')
        md('### CHECK · Use your function');code(check)
    md('## CHECK · Adversarial contracts\nMissing arms, mismatched seed blocks, nonfinite values, duplicate keys and malformed time metadata must fail. These tests call the live functions you wrote.')
    code(defs(P/'_check_l183.py')['checks']);code("assert checks(temporal_mask,keyed_mae,factorial_effect,transfer_gate)=='PASS'\nprint('Four live contracts PASS')")
    md('## PROVIDED · Full saved-evidence replay\nThis harness calls all four of your functions. Read the query joins, selection checks and cost scenario reconstruction. Time counts are inherited report fields; the code checks one saved future-token witness, not all original contexts.')
    code(defs(P/'_audit_l183.py')['audit183'])
    code("report=audit183(P,manifest,temporal_mask,keyed_mae,factorial_effect,transfer_gate)\nPath('l183-report.json').write_text(json.dumps(report,indent=2))\nprint(report['status'],report['verified_predictions'],'predictions')\nfor arm,splits in report['summary'].items():\n    for split,values in splits.items():print(arm,split,round(values['mean'],6),'±',round(values['sample_sd'],6))\nprint('Inherited cost scenarios:',report['inherited_cost_scenarios_usd'])\nprint('Hybrid gate:',report['hybrid_gate'])")
    md('## CHECK · An independent SQL oracle\nSQLite enforces composite primary keys, checks both population differences and computes absolute error through a database join. Small floating-point accumulation differences are allowed up to 1e−12.')
    code(defs(P/'_verify_l183.py')['sql_mae'])
    code("other=audit183(P,manifest,temporal_mask,sql_mae,factorial_effect,transfer_gate)\nfor arm in ['gnn','relgt']:\n    for split in ['val','test']:\n        for a,b in zip(report['summary'][arm][split]['values'],other['summary'][arm][split]['values']):assert abs(a-b)<1e-12\nprint('Independent SQLite scoring passed for all 7554 predictions')")
    md('## TRY · Right shape, wrong table meaning\nThis two-coordinate fixture is an invented registry example, not a learned embedding. Predict what happens when table IDs are reassigned. Aligning known names repairs a permutation; it does not define a representation for an unseen table.')
    code("source_ids={'drivers':0,'races':1}\ntarget_ids={'races':0,'drivers':1}\nfrozen={0:(1.,0.),1:(0.,1.)}\nwrong=frozen[target_ids['drivers']]\ncorrect=frozen[source_ids['drivers']]\nassert len(wrong)==len(correct) and wrong!=correct\nprint('Shape-compatible but semantically wrong:',wrong,'expected',correct)\nassert 'patients' not in source_ids\nprint('An unseen table still needs a declared semantic encoder')")
    md('## TRY · Be best while gaining less\nPredict the sign before running. These scores are invented: MP 5→4, GT 3→2.5. Then invent an example with negative transfer and explain which hypothesis it challenges.')
    code("scenario={k:{0:v,1:v} for k,v in zip(['mp_scratch','mp_pretrained','gt_scratch','gt_pretrained'],[5.,4.,3.,2.5])}\nassert factorial_effect(scenario)['interaction_mean']==-.5\nprint('GT has lower MAE but less pretraining gain:',factorial_effect(scenario))\nready={name:'PASS' for name in report['hybrid_evidence']}\nassert transfer_gate(ready)['transfer_gain']=='NOT_ESTABLISHED'\nprint('Hypothetical prerequisites pass; observed hybrid remains NOT_RUN')")
    md('## TRY · Reject corrupted provenance\nThe manifest itself is pinned in this notebook. Replacing an expected digest is a deliberate adversarial exercise, not a valid way to approve changed data.')
    code("bad=copy.deepcopy(manifest);bad['files'][next(iter(bad['files']))]='0'*64\ntry:audit183(P,bad,temporal_mask,keyed_mae,factorial_effect,transfer_gate)\nexcept ValueError:print('Changed evidence rejected')\nelse:raise AssertionError('Tampered evidence admitted')")
    md('## EXIT · One-page gap brief\nFill these fields in your own words, including a result that would make you abandon the idea. Passing code checks does not complete this defense. Paste the brief to the teacher for review.')
    code("submission={'learner':'PENDING_WRITTEN_DEFENSE','claim':'','interface_change':'','four_arms':'','held_fixed':'','database_holdout':'','temporal_access':'','validation_selection':'','falsification':'','all_in_cost':'','novelty_limit':''}\nPath('l183-submission.json').write_text(json.dumps(submission,indent=2))")
    md('## NEXT STEP · Full published experiments remain gated\nThe following complete protocols and source appendices preserve the original named experiments. No default cell trains either model. The retained repository entry point `labs/_run_l183.py --lane relgt` (or griffin) reports the blocker; `--run` deliberately refuses. Original full operators are documented below. Resolving a blocker requires new evidence and separate execution approval, not a boolean toggle in this notebook.')
    code("RUN_PAPER_REPRO=False\nif RUN_PAPER_REPRO:raise RuntimeError('STOP: temporal/budget gates unresolved; no paid training authorized')\nprint('RelGT:',report['relgt_full_reproduction'])\nprint('Griffin:',report['griffin_full_reproduction'])\nprint('Hybrid pretraining:',report['hybrid_pretraining'])")
    md((P/'l183-reproduction.md').read_text())
    md('## Source appendix · Complete models and trainers, displayed only\nRelGT token encoders, local/global attention and head are followed by its full search trainer. Griffin includes its readable implementation, released model, full source trainer and wrapper. Each definition is a separate readable chunk. These appendices do not import GPU packages into the audit kernel. Original comments, unused methods and source defects remain visible; seeing them is not execution validation.')
    for name in ['labs/relkit/relgt_l145.py','labs/_full_l145.py','labs/relkit/griffin_l164.py','labs/sources/l164/upstream/hmodel.py','labs/sources/l164/upstream/hmaintask_downsample_absolute_eval_sample.py','labs/_run_l164.py']:
        md('### '+name+'\nSHA256: `'+pins['files'][name]+'`. Source retained without behavioral edits.')
        for label,source in source_chunks(E/'packet'/name):md('**'+label+'**\n\n```python\n'+source+'\n```')
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    for i,c in enumerate(cells):c.id=f'l183-{i:03}'
    return book
for solution in [False,True]:
    book=make(solution);path=P/('solutions' if solution else '')/(S+'.ipynb')
    if solution and path.exists():
        old=nb.read(path,4)
        if [c.source for c in old.cells if c.cell_type=='code']==[c.source for c in book.cells if c.cell_type=='code']:
            book.metadata=old.metadata
            for a,b in zip([c for c in book.cells if c.cell_type=='code'],[c for c in old.cells if c.cell_type=='code']):a.outputs=b.outputs;a.execution_count=b.execution_count;a.metadata=b.metadata
    nb.write(book,path)
print('Built lesson, reference, student and solution notebooks')
