"""Build lesson, reference, portable notebooks and machine-readable explorer evidence."""
import ast,base64,hashlib,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0169-scaling-laws-open-questions';E=P/'evidence/l169'
report=json.loads((E/'report.json').read_text());pins=json.loads((E/'audit-manifest.json').read_text())
def definitions(path):
 text=path.read_text();return [(n.name,ast.get_source_segment(text,n)) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)]
def table():
 lines=['| Task / model | K=64 | 128 | 256 | 512 reused | 1024 |','|---|---:|---:|---:|---:|---:|']
 for db in ['rel-f1','rel-trial']:
  for arm in ['RDBPFN','RDBPFN_single','TabICLv1.1']:
   levels=report['curves'][db+'/'+arm]['levels'];lines.append('| '+db+' / '+arm+' | '+' | '.join(f"{r['mean']:.4f}" for r in levels)+' |')
 return '\n'.join(lines)
def interpretation():
 counts=sum(r['status']=='CLOSE' for r in report['comparisons']);sentences=[]
 for db in ['rel-f1','rel-trial']:
  c=report['curves'][db+'/RDBPFN'];sentences.append(f"On {db}, RDB-PFN changes from **{c['levels'][0]['mean']:.5f}** at 64 to **{c['levels'][-1]['mean']:.5f}** at 1024 ({c['levels'][-1]['mean']-c['levels'][0]['mean']:+.5f} AUROC).")
 negative=[f"{key} {d['start']}→{d['end']} ({d['mean_gain']:+.5f})" for key,c in report['curves'].items() for d in c['doublings'] if d['mean_gain']<0]
 return ' '.join(sentences)+f' **{counts}/30 published-mean comparisons** fall within the predeclared 0.02 AUROC descriptive tolerance. '+'Negative mean doubling contrasts: '+('; '.join(negative) if negative else 'none in this selected run')+'. These are measured responses under changing support draws, not evidence for a universal monotonic law.'
captions={'protocol':'Only context changes in the observed sweep. The parameter, data and diversity rows are proposed experiments with additional compute controls required.',
 'curves':'Same AUROC scale across both tasks. Solid means and sample SD bands are measured; dotted diamonds are published means. Hollow512 markers are reused evidence.',
 'gains':'Mean AUROC change for each context doubling with sample SD over ten seed-indexed contrasts. Negative gains remain visible; supports across sizes are not nested.'}
def prose(portable=False):
 text=(R/'lessons/content'/(S+'.md')).read_text()
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/'figures/l169'/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l169/'+name+'.svg'
  text=text.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 widgets={
 'WARMUP':('Recall: why are ten support seeds not ten databases? Why is zero-gradient inference not zero-label inference?','<div id="warmup"></div>'),
 'PREDICT':('Predict before reading: five context sizes with fixed weights can measure a context response. Explain why a parameter exponent is not identified.','<div id="predict"></div>'),
 'EXPLORER':('Baseline is K=64 for the selected task/model. Compare any observed level to that baseline using the live curve output below.512evidence is reused. Other contexts are fresh.','<div id="context-explorer"></div><noscript>The full curves and table above remain available without JavaScript.512evidence is reused; other contexts are fresh. The notebook computes every difference.</noscript>'),
 'CLAIM_EXPLORER':('Controlled five-level context sweep → CONTEXT_RESPONSE_ONLY. An attempted prediction beyond the range → EXTRAPOLATION_NOT_ESTABLISHED. Other axes here are hypothetical study declarations.','<div id="claim-explorer"></div><noscript>Controlled context variation supports CONTEXT_RESPONSE_ONLY. An unvalidated extrapolation remains NOT_ESTABLISHED. See the evidence-map table above.</noscript>'),
 'TEACHBACK':('Write your own falsifiable next experiment: axis, controls, exposure, uncertainty, falsifier and total cost.','<div id="teachback"></div>')}
 for name,(static,html) in widgets.items():text=text.replace('[['+name+']]',static if portable else html)
 text=text.replace('[[CODE]]','Implement the source-compatible sampler in the first TODO task below.' if portable else '```python\n'+dict(definitions(P/'relkit/scaling_l169.py'))['sample_context']+'\n```')
 text=text.replace('[[RESULTS]]',table()).replace('[[INTERPRETATION]]',interpretation())
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/');text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def doc(title,text,interactive=False):
 body=render(text).replace('<table>','<div class="route-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 css=['lesson','atomic-route','checkpoint','lab-access','foundation-scope','scaling-context'];scripts=['retrieval-pool','retrieval-bank','predict','teachback','scaling-context','l169-evidence','l169-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in css)+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0168-cross-database-generalization.html">Lesson 168</a></nav><header><p class="route-kicker">Year 5 · Quarter 1 · Lesson 169</p><h1>'+title+'</h1></header>'+body+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Scaling laws & open questions',prose(),True))
(R/'assets/l169-evidence.js').write_text('window.L169Evidence='+json.dumps(dict(curves=report['curves']),separators=(',',':'))+';\n')
reference='''**One question:** which scale changed, and which claim follows?

| Axis | Unit | Evidence here |
|---|---|---|
| Context K | Labeled target examples at inference | Five sizes; fixed checkpoints; two tasks |
| Parameters P | Learned coefficients | Not varied |
| Pretraining D | Training examples before target evaluation | Not varied |
| Schema diversity S | Distinct declared schema units | Not varied; lineage caveats |
| Compute C | Operations or resource expenditure | Budgeted; no compute-optimal law fitted |

A curve is observed response. A scaling law is a quantitative model with a stated domain and predictive validation. An illustrative loss law L(x)=L∞+A·x^(−α) needs a named loss, fitted floor and exponent, controlled axes, uncertainty and held-out-scale checks. AUROC is a ranking metric; this lesson does not fit a pretraining exponent.

For each doubling compute seed-indexed Δ=A(2K,s)−A(K,s), then mean and sample SD. Same-seed samples are not nested under the released sampler; these are descriptive contrasts. At each K the models do share exactly the same supports. A sample SD over ten support draws is not uncertainty across databases.

'''+table()+'\n\n'+interpretation()+'''

Complete selected experiment:300evaluations,240fresh+60reused;229,050probabilities.30table-cell means compared with0.02descriptive AUROC tolerance. Exact query keys, raw labels/supports, hashes and original receipts verified. Released labels complement raw task labels. Full DFS regeneration NOT_RUN; historical identity/availability, checkpoint lineage and exact schema exclusion NOT_ESTABLISHED. Exact TabICL repeatability failed in two diagnostic reruns (max probability difference .0002791, max AUROC difference .00000488); the first retained results are used without score selection. Whole paper and fresh pretraining NOT_RUN. Learner PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/0169-scaling-laws-open-questions.html) · [Protocol](../labs/l169-reproduction.md) · [Gap template](../labs/l169-gap-template.md) · [RDB-PFN v5](https://arxiv.org/html/2603.03805v5) · [Survey §5.2](https://arxiv.org/html/2506.16654v1#S5.SS2).
'''
(R/'reference/scaling-laws.html').write_text(doc('Scaling evidence — quick reference',reference))
(E/'report.md').write_text('# L169 selected context sweep\n\n'+table()+'\n\n'+interpretation()+'\n\n240fresh+60reused evaluations;229,050predictions. Source protocol preserves independently drawn contexts. Pretraining law NOT_ESTABLISHED; whole paper/fresh pretraining NOT_RUN.\n')
# Deterministic ZIP embeds original evidence, source and data, not hidden executable answers.
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as archive:
 for name in sorted(pins['files']):
  info=zipfile.ZipInfo(name);info.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(info,(P/name).read_bytes())
raw=buf.getvalue();payload=base64.b64encode(raw).decode();digest=hashlib.sha256(raw).hexdigest()
tasks={
 'sample_context':('Reproduce the released support draw.','Validate positive integer context≤n_rows (Booleans are not accepted) and a nonempty string seed_key. Convert the first4bytes of its SHA256digest to an unsigned big-endian integer. Use a new NumPy default_rng with that integer to draw context distinct indices from n_rows. Do not substitute a nested prefix. Return the indices.',"idx=sample_context(1500,64,'example:0')\nassert len(idx)==len(set(idx))==64\nassert min(idx)>=0 and max(idx)<1500\nprint('CHECK:64distinct support identities')"),
 'scaling_curve':('Require the entire grid before drawing a curve.','Input records have context,seed,auc. Require exactly one finite AUROC in[0,1] for all five context sizes and seeds0–9; reject invalid/duplicate/missing points. Return levels (context,per_seed,mean,sample_sd) in ascending context order and doublings (start,end,per_seed,mean_gain,sample_sd,positive_seeds). Pair by seed, use sample SD with denominator9. Also return interpretation="SUPPORT_DRAW_VARIATION_ON_ONE_FIXED_TASK".',"toy=[dict(context=k,seed=s,auc=.5+.01*i+.001*s) for i,k in enumerate([64,128,256,512,1024]) for s in range(10)]\nc=scaling_curve(toy)\nnp.testing.assert_allclose([d['mean_gain'] for d in c['doublings']],.01)\nprint('CHECK:complete paired curve')"),
 'scaling_claim':('Classify what a study declaration permits.','Validate axis in context/parameters/pretraining_data/schema_diversity; controlled and extrapolating must be Booleans; levels a positive integer. In priority order return CONFOUNDED if not controlled, INSUFFICIENT_LEVELS if levels<2, EXTRAPOLATION_NOT_ESTABLISHED if extrapolating, CONTEXT_RESPONSE_ONLY for context, otherwise CONTROLLED_SWEEP_NOT_LAW. Other-axis declarations are hypothetical, not measured experiments.',"assert scaling_claim('context',True,5,False)=='CONTEXT_RESPONSE_ONLY'\nassert scaling_claim('schema_diversity',False,5,False)=='CONFOUNDED'\nprint('CHECK:claims follow declared evidence')")}
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson169 · Scaling laws & open questions\n\nDefault: portable CPU-only full saved-evidence replay; Python3+NumPy. No downloads or cloud calls. TierB real relational evidence; synthetic checks only illustrate mechanics. PROVIDED code is complete; TODO contracts drive the audit; CHECK gives immediate feedback; EXIT requires your written defense. Fresh inference is a separate opt-in lane.'),nb.v4.new_markdown_cell(prose(True)),nb.v4.new_code_cell('# @colab-bootstrap\nimport os\nos.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")\nimport base64,hashlib,io,json,zipfile\nfrom pathlib import Path\nimport numpy as np\nP=Path("l169-portable");P.mkdir(exist_ok=True)')]
 cells.append(nb.v4.new_markdown_cell('## PROVIDED · Authenticate raw evidence\nThis packet includes all300 complete prediction files, original receipts, prepared task matrices, supports and released source.240runs are fresh L169 evidence;60are reused512-context evidence. The first interrupted batch is excluded.225saved runs from the second attempt plus9tail runs form an explicitly author-assembled receipt; original per-run files are also authenticated. Six diagnostic sentinel reruns (four RDB-PFN exact; two TabICL with small numerical differences) are not counted as new experiment cells.'))
 c=nb.v4.new_code_cell('encoded='+repr(payload)+'\nraw=base64.b64decode(encoded)\nassert hashlib.sha256(raw).hexdigest()=='+repr(digest)+'\nwith zipfile.ZipFile(io.BytesIO(raw)) as archive:\n    for name in archive.namelist():assert not Path(name).is_absolute() and ".." not in Path(name).parts\n    archive.extractall(P)\npins='+repr(pins)+'\nprint("Authenticated300complete raw prediction sets and original source")');c.metadata['tags']=['data-payload'];cells.append(c)
 cells.append(nb.v4.new_markdown_cell('## PROVIDED · Full-key AUROC\nRank after aligning exact entity/date identities. Independent author validation compares this with sklearn and pairwise positive/negative comparisons, including ties.'))
 cells.append(nb.v4.new_code_cell(dict(definitions(P/'relkit/transfer_l167.py'))['keyed_auc']))
 for name,code in definitions(P/'relkit/scaling_l169.py'):
  title,contract,check=tasks[name]
  cells.extend([nb.v4.new_markdown_cell('## TODO · '+title+'\n\n**Why:** this contract determines the support evidence, curve or permitted claim in the final report.\n\n**Contract:** '+contract),nb.v4.new_code_cell(code if solution else code.split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'),nb.v4.new_code_cell('# CHECK\n'+check)])
 for name,code in definitions(P/'_check_l169.py'):cells.append(nb.v4.new_code_cell(code))
 cells.append(nb.v4.new_code_cell('assert check169(sample_context,scaling_curve,scaling_claim)=="PASS"\nprint("All3live learner contracts passed")'))
 cells.append(nb.v4.new_markdown_cell('## PROVIDED · Audit every size, model and seed\nYour sampler authenticates the stored support schedule. Your curve function aggregates raw independently rescored predictions. Your claim function declares the result. The final report is computed here, not read from a saved answer.'))
 for name,code in definitions(P/'_audit_l169.py'):cells.append(nb.v4.new_code_cell(code))
 cells.append(nb.v4.new_code_cell('report=audit169(P,pins,keyed_auc,sample_context,scaling_curve,scaling_claim)\nassert report["fresh_runs"]==240 and report["reused_runs"]==60 and report["all_predictions_checked"]==229050\nPath("l169-report.json").write_text(json.dumps(report,indent=2)+"\\n")\nfor name,curve in report["curves"].items():\n    print(name,[(x["context"],round(x["mean"],6),round(x["sample_sd"],6)) for x in curve["levels"]])\n    print("Doubling gains:",[(x["start"],x["end"],round(x["mean_gain"],6),x["positive_seeds"]) for x in curve["doublings"]])\nprint("Support overlaps:",report["support_overlap"])\nprint("Paired RDBPFN minus TabICL:",report["paired_models"])\nprint(report["claim"],"pretraining law NOT_ESTABLISHED")'))
 cells.append(nb.v4.new_markdown_cell('## OPTIONAL · Fresh inference\nThe default skips this lane. The pinned full model is embedded at `l169-portable/sources/l166/upstream/model_pretrain/src/models.py`; the runner below visibly loads both fixed checkpoints and32-estimator TabICL, loops all contexts/seeds and saves every keyed prediction. For compatible GPU execution install Python3.11,torch2.5.1,numpy1.26.4,pandas2.2.3,scikit-learn1.6.1,pydantic1.10.26,pyyaml6.0.2,tabicl0.1.3.\n\nThis reruns all240fresh cells, retaining60authenticated512cells. Weights download only if enabled. Manual execution has no billing guard; the repository managed lane reserves each attempt. Do not enable on an unbudgeted paid instance. A new output directory is required.'))
 for filename in ['_fetch_l169.py','_run_l169.py']:
  for name,code in definitions(P/filename):cells.append(nb.v4.new_code_cell(code))
 cells.append(nb.v4.new_code_cell('RUN_FRESH=False\nif RUN_FRESH:\n    destination=P/"fresh-input"\n    fetch169(destination)\n    receipt=run169(destination,P/"sources/l166/upstream",Path("l169-fresh-predictions"),"all")\n    assert len(receipt["records"])==240\nelse:\n    print("Optional fresh inference NOT_RUN in this notebook; complete author evidence rescored above")'))
 cells.append(nb.v4.new_markdown_cell('## EXIT · Scaling-gap map and defense\nComplete the four-axis map; rank three gaps by decision value, tractability and falsifiability(1–3each, with reasons). Write250–400words defending the top experiment. Include the control variables, evaluation population, uncertainty unit, falsifier and total budget. Rubric0–2each: axes, protocol, uncertainty, falsifiability, feasibility; target≥8/10and no zero; teacher review required. The reference notebook gives a starting outline, not a completed learner defense.'))
 outline={'context':'Measured five sizes/two tasks; next compare a declared nested-support protocol','parameters':'Fixed here; needs matched fresh model-size sweep and compute policy','pretraining_data':'Fixed here; vary synthetic examples with model and schema mix held fixed','schema_diversity':'Not isolated; hold total rows fixed and audit target exclusion'}
 cells.append(nb.v4.new_code_cell('gap_map='+repr(outline if solution else {k:'' for k in outline})+'\nranked_questions=[] # Add scores and reasons.\ndefense="" # Write your own defense.\nPath("l169-submission.json").write_text(json.dumps(dict(gap_map=gap_map,ranked_questions=ranked_questions,defense=defense,review=None,learner="PENDING_WRITTEN_DEFENSE"),indent=2)+"\\n")\nprint("Learner PENDING_WRITTEN_DEFENSE; ask the teacher for feedback")'))
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(cells):c.id=f'l169-{i:03d}'
 path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
  old=nb.read(path,4)
  if [(c.cell_type,c.source) for c in old.cells]==[(c.cell_type,c.source) for c in cells]:
   book.metadata=old.metadata
   for c,prior in zip(cells,old.cells):
    if c.cell_type=='code':c.outputs=prior.outputs;c.execution_count=prior.execution_count;c.metadata=prior.metadata
 nb.write(book,path)
print('Built L169 lesson, reference, explorer data and both portable notebooks')
