"""Deterministic connected lesson, reference, measured explorer and portable notebooks."""
import ast,base64,hashlib,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l176';S='0176-few-shot-icl-evaluation'
r=json.loads((E/'report.json').read_text());old=json.loads((E/'published-replay.json').read_text());pins=json.loads((E/'audit-manifest.json').read_text())
def defs(path):
 text=path.read_text();return {n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
functions=defs(P/'relkit/few_shot_l176.py')
def table():
 lines=['| Task / frozen model | 64 | 128 | 256 | 512 | 1,024 |','|---|---:|---:|---:|---:|---:|']
 for name,c in r['curves'].items():lines.append('| '+name+' | '+' | '.join(f"{a['mean']:.4f}" for a in c['levels'])+' |')
 return '\n'.join(lines)+'\n\nMeasured mean AUROC over ten support seeds; every cell contains fresh L176 inference. Sample SD and all paired gains are in the notebook and explorer.'
def interpretation():
 negative=[f"{name}: {x['start']}→{x['end']} ({x['mean_gain']:+.5f})" for name,c in r['curves'].items() for x in c['doublings'] if x['mean_gain']<0]
 reversal=sum(any(v<0 for v in x['per_seed']) and any(v>0 for v in x['per_seed']) for c in r['curves'].values() for x in c['doublings'])
 return '**Observed negative mean doubling gains:** '+('; '.join(negative) if negative else 'none in this selected experiment')+f'. **{reversal} of 24 adjacent-size contrasts contain both positive and negative seed gains.** These observations describe two fixed tasks and do not establish a universal effect.'
captions={'architecture':'Actual RDB-PFN support/query computation: support-derived feature and label encodings, six frozen attention blocks, and query decoding. TabICL is a separate comparator.','supports':'Synthetic worked trace: one ordered draw, three nested prefixes. Baseline rows remain blue; additional rows are green.','curves':'Fresh measured means and sample SD across ten support draws. Both panels use the same AUROC scale. No best-k test selection.'}
status='''**Executed:** all 300 fresh nested evaluations and all 300 saved published-protocol evaluations independently rescored: **458,100 probabilities across the two tracks**. These are repeated predictions on the same two test populations, not 458,100 independent examples. Fresh pretraining and whole-paper reproduction remain `NOT_RUN`; learner status is `PENDING_WRITTEN_DEFENSE`.'''
def prose(portable=False):
 s=(R/'lessons/content'/(S+'.md')).read_text().replace('[[RESULTS]]',table()).replace('[[INTERPRETATION]]',interpretation()).replace('[[STATUS]]',status)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/'figures/l176'/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l176/'+name+'.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption} Scroll horizontally on narrow screens.</figcaption></figure>')
 s=s.replace('[[CODE]]','Complete the live nested-support function below.' if portable else '```python\n'+functions['nested_support']+'\n```')
 replacements={'WARMUP':('Recall why frozen weights do not exclude label access, and why lesson169 supports were not nested.','<div id="warmup"></div>'),'PREDICT':('Predict: must adding examples improve AUROC? Explain before looking at the measured curve.','<div id="predict"></div>'),'EXPLORER':('Use the computed per-seed curves below to compare every size with k=64.','<div id="support-explorer"></div><noscript>The complete measured means are in the table above. Nested supports retain all64baseline rows. Additional examples need not improve AUROC.</noscript>'),'TEACHBACK':('Write your own evaluation card; ask the teacher for feedback.','<div id="teachback"></div>')}
 for tag,(plain,html) in replacements.items():s=s.replace('[['+tag+']]',plain if portable else html)
 # Use portable text notation rather than requiring a remote math renderer.
 s=s.replace('\\(','').replace('\\)','')
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def doc(title,body,interactive=False):
 html=render(body).replace('<table>','<div class="route-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 css=['lesson','atomic-route','checkpoint','lab-access','few-shot-evaluation'];scripts=['retrieval-pool','retrieval-bank','predict','teachback','few-shot-evaluation','l176-evidence','l176-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in css)+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0175-zero-shot-evaluation.html">Lesson175</a></nav><header><p class="route-kicker">Year5 · Quarter2 · Lesson176</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Few-shot ICL evaluation',prose(),True))
(R/'assets/l176-evidence.js').write_text('window.L176Evidence='+json.dumps(dict(curves=r['curves']),separators=(',',':'))+';\n')
reference='''**Support**: labeled examples passed into a frozen predictor. **Query**: a case whose hidden label is predicted. **k**: total support rows per task, shared across queries; not per class.

**Nested schedule**: draw1,024distinct eligible training rows once per task/seed; use prefixes64/128/256/512/1,024. Share ordered supports across models. Same seed plus independently sized draws does not ensure nesting.

**Audit before inference**: unique complete(entity_id,cutoff)keys; support from train only; no support/query overlap; completed support-label horizons; identical query population; exact source/checkpoint hashes. Feature arrival, DFS construction and pretraining lineage require separate evidence.

**Paired contrast**: Δ_s=AUROC(2k,s)−AUROC(k,s); mean over10seeds; sampleSD with denominator9. Report reversals. Seeds quantify support-draw variation on one task, not variation over databases. Do not select the best k on test scores.

**Mechanism**: source imputation and normalization are fitted to support. Query target placeholders use the support-label mean. Frozen weights do not freeze those intermediates. Measure the response of the complete pipeline.

'''+table()+'\n\n'+interpretation()+'\n\n'+status+'''

**Reproduction boundary**: full L169 selected published experiment replayed; fresh nested grid is a course intervention. Released complemented labels preserved. Original DFS regeneration/fresh pretraining/wholepaper NOT_RUN; historical availability/identity NOT_ESTABLISHED. L175 temporal failure remains unchanged. Prior TabICL exact repeatability FAIL is not converted into a pass.

[Lesson](../lessons/0176-few-shot-icl-evaluation.html) · [Notebook](../labs/0176-few-shot-icl-evaluation.ipynb) · [Protocol](../labs/l176-reproduction.md) · [RDB-PFN Appendix A.3](https://arxiv.org/html/2603.03805v5#A3).
'''
(R/'reference/few-shot-icl-evaluation.html').write_text(doc('Few-shot ICL — quick reference',reference))
(E/'report.md').write_text('# L176 measured nested-support experiment\n\n'+table()+'\n\n'+interpretation()+'\n\n'+status+'\n')
# Full inherited and new raw evidence, with original source and receipts.
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for name in sorted(pins['files']):
  info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,(P/name).read_bytes())
packet=base64.b64encode(buf.getvalue()).decode();digest=hashlib.sha256(buf.getvalue()).hexdigest()
checks=defs(P/'_check_l176.py')['checks']

def make(solution):
 cells=[]
 def md(t):cells.append(nb.v4.new_markdown_cell(t))
 def code(t,tags=None):cells.append(nb.v4.new_code_cell(t,metadata={'tags':tags or []}))
 md('# Lesson176 · Few-shot ICL evaluation\n\nBuild an auditable nested-support experiment. Complete three TODOs, then rescore every prediction in both tracks. The default is a complete saved-evidence replay with no paid calls. Fresh author inference is reported separately. Your mastery remains PENDING_WRITTEN_DEFENSE.')
 code('''# @colab-bootstrap
import importlib.util, subprocess, sys
needed=[name for name,module in [('numpy','numpy'),('torch','torch'),('pyyaml','yaml')] if importlib.util.find_spec(module) is None]
if needed: subprocess.check_call([sys.executable,'-m','pip','install',*needed])
import numpy as np
from pathlib import Path
import json
''')
 md(prose(True))
 md('## PROVIDED · Authenticate the complete evidence packet\nIncludes both full300-evaluation tracks, source files, ordered supports, query keys, labels and raw probabilities. No scores are fetched from an answer service. Hashes detect changed packet bytes; historical source lineage still has the documented limits.')
 code('import base64,hashlib,io,zipfile\nPACKET='+repr(packet)+'\nraw=base64.b64decode(PACKET)\nassert hashlib.sha256(raw).hexdigest()=='+repr(digest)+'''\nP=Path('l176-packet');P.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    for name in z.namelist():assert not Path(name).is_absolute() and '..' not in Path(name).parts
    z.extractall(P)
'''+ 'pins='+repr(pins)+'\nprint("Authenticated full published and nested prediction tracks")',['data-payload'])
 tasks=[('nested_support','Construct the nested support schedule','Validate positive integer sizes, distinct contexts and a nonempty seed key. Use the first4SHA256bytes as an unsigned big-endian integer. Draw max(contexts) distinct indices with NumPy default_rng, then return a dict of ordered prefixes. Never redraw at smaller k.',"toy=nested_support(20,[2,4,8],'worked:0')\nassert np.array_equal(toy[2],toy[8][:2]) and len(set(toy[8]))==8\nprint('CHECK1: old examples retained',toy)"),('keyed_auc','Align complete query keys before ranking','Reject missing/duplicate two-column keys, invalid probabilities and labels lacking either binary class. Align to label keys. Compute the fraction of positive-negative pairs correctly ranked, with half credit for ties.',"keys=np.array([[7,100],[7,200],[8,100],[8,200]]);labels=np.array([0,1,1,0]);scores=np.array([.2,.9,.7,.7]);order=[2,0,3,1]\nassert keyed_auc(keys,labels,keys[order],scores[order])==.875\nprint('CHECK2: keyed synthetic AUROC=.875')"),('paired_curve','Require the full grid, then pair gains','Require exactly one finite[0,1]AUROC for every(context,seed). Reject missing/duplicate/extra cells. Return levels and doublings with ordered per_seed values, mean, sample_sd(ddof=1), and positive_seeds for gains. See the synthetic expected result below; include interpretation=SUPPORT_DRAW_VARIATION_ON_ONE_FIXED_TASK.',"rows=[dict(context=k,seed=s,auc=float(v)) for k,vs in [(64,[.60,.70,.80]),(128,[.65,.68,.83])] for s,v in enumerate(vs)]\nc=paired_curve(rows,[64,128],[0,1,2])\nassert abs(c['doublings'][0]['mean_gain']-.02)<1e-12\nnp.testing.assert_allclose(c['doublings'][0]['per_seed'],[.05,-.02,.03])\nassert set(c)=={'levels','doublings','interpretation'}\nprint('CHECK3:',c)")]
 for name,title,description,check in tasks:
  md('## TODO · '+title+'\n'+description)
  code(functions[name] if solution else functions[name].split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")')
  code(check)
 md('## PROVIDED / CHECK · Evidence rejection and budget admission\nThese checks use your functions. Missing rows must fail before a mean is reported. The budget guard is provided, not a learner permission bypass.')
 code(functions['reserve_cost']);code(checks);code("assert checks(nested_support,keyed_auc,paired_curve,reserve_cost)=='PASS'\nprint('All live learner contracts passed')")
 md('## PROVIDED · Complete published-protocol replay\nOriginal independent support draws remain unchanged. The complete source audit authenticates all inherited files, regenerates all original supports, aligns every query and checks all300run receipts. Its historical240fresh+60reused counts describe L169; every one is reused evidence in this notebook.')
 for fn in defs(P/'relkit/scaling_l169.py').values():code(fn)
 code(defs(P/'_audit_l169.py')['audit169'])
 code("oldpins=json.loads((P/'evidence/l176/inherited-manifest.json').read_text())\npublished=audit169(P,oldpins,keyed_auc,sample_context,scaling_curve,scaling_claim)\nassert published['all_predictions_checked']==229050\nPath('l176-published-replay.json').write_text(json.dumps(published,indent=2))\nprint('Published replay:',published['status'],published['all_predictions_checked'],'saved predictions')")
 md('## PROVIDED · Complete nested replay using your live functions\nYour support function verifies100task/seed/size schedules, your keyed scorer rescales all300prediction files, and your paired function builds every measured curve. Source/preprocessing limitations remain explicit.')
 code(defs(P/'_audit_l176.py')['audit176'])
 code("report=audit176(P,pins,nested_support,keyed_auc,paired_curve)\nassert report['fresh_evaluations']==300 and report['predictions']==229050\nPath('l176-report.json').write_text(json.dumps(report,indent=2))\nfor name,curve in report['curves'].items():\n    print(name,[(x['context'],round(x['mean'],6),round(x['sample_sd'],6)) for x in curve['levels']])\n    print('Paired doubling gains:',[(x['start'],x['end'],round(x['mean_gain'],6),x['positive_seeds']) for x in curve['doublings']])\nprint('All600evaluations rescored; no new inference in this notebook')")
 md('## EXIT · Defend a measured contrast\nFind a negative mean gain or mixed-sign seed contrast. State task/model/sizes, numerical evidence, what changed and what stayed fixed. Explain support-draw uncertainty, preprocessing dependence and why this is a declared course intervention. Include historical availability/DFS limitations. Ask the teacher for feedback; execution alone does not establish mastery.')
 code("submission={'learner':'PENDING_WRITTEN_DEFENSE','defense':''}\nPath('l176-submission.json').write_text(json.dumps(submission,indent=2))")
 md('## Appendix A · Original released predictor, visible end to end\nRead FeatureEncoder, TargetEncoder, TransformerEncoderLayer, NanoTabPFNModel, and NanoTabPFNClassifier. These are exact archived definitions; six blocks are selected by ModelConfig(num_layers=6) in the runner. Defining the classes does not train a model or load a checkpoint. Additional categorical variants remain visible as released but are not used by these two base checkpoints.')
 model=(P/'sources/l166/upstream/model_pretrain/src/models.py').read_text();tree=ast.parse(model);cut=next(n.lineno for n in tree.body if isinstance(n,ast.If) and '__name__' in ast.get_source_segment(model,n.test));code('\n'.join(model.splitlines()[:cut-1]))
 md('### Support-dependent preprocessing\nOriginal median imputation and prediction chunking; these functions reveal why increasing support can change the representation even with unchanged weights.')
 es=defs(P/'sources/l166/upstream/model_pretrain/src/eval_utils.py')
 code(es['fill_nans']);code(es['predict_proba_in_chunks'])
 md('## Appendix B · Complete opt-in fresh inference\nThis exact runner produced the300fresh author evaluations. It loads fixed checkpoints, calls the original model wrappers and writes all keyed probabilities. Python3.11,torch2.5.1+CUDA12.4,numpy1.26.4,pandas2.2.3,scikit-learn1.6.1,pydantic1.10.26,pyyaml6.0.2,tabicl0.1.3; TabICL32estimators. A fresh run needs a compatible GPU environment and its own explicit budget. The repository managed operator reserves attempts; this manual function does not enforce cloud billing limits.')
 code(defs(P/'_fetch_l176.py')['fetch176']);code(defs(P/'_run_l176.py')['run176'])
 code("RUN_FRESH=False\nif RUN_FRESH:\n    destination=P/'fresh-input'\n    fetch176(destination)\n    receipt=run176(destination,P/'sources/l166/upstream',Path('l176-fresh-predictions'),'all')\n    assert len(receipt['records'])==300\nelse:\n    print('Fresh inference NOT_RUN in this notebook; complete author evidence replayed above')")
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(cells):c.id=f'l176-{i:03}'
 return book
for solution in [False,True]:
 book=make(solution);path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
  prior=nb.read(path,4)
  if [c.source for c in prior.cells if c.cell_type=='code']==[c.source for c in book.cells if c.cell_type=='code']:
   book.metadata=prior.metadata
   for a,b in zip([c for c in book.cells if c.cell_type=='code'],[c for c in prior.cells if c.cell_type=='code']):a.outputs=b.outputs;a.execution_count=b.execution_count;a.metadata=b.metadata
 nb.write(book,path)
print('Built L176 lesson, reference, explorer, student and solution')
