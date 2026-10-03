"""Deterministic lesson, reference and offline notebook build from frozen B04 evidence."""
import ast,base64,hashlib,html,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b04';S='b04-tabicl-scalable-icl'
r=json.loads((E/'diagnostic-audit.json').read_text());source=(R/'lessons/content'/(S+'.md')).read_text()
table='| Configuration | Support | Correct / 64 | Log loss | Predict seconds | Peak RSS MiB |\n|---|---:|---:|---:|---:|---:|\n'+''.join(f"| {x['name']} | {x['support_n']} | {x['correct']} | {x['log_loss']:.6f} | {x['predict_seconds']:.3f} | {x['peak_process_rss_mib']:.2f} |\n" for x in r['rows'])
figs=dict(ARCHITECTURE=('architecture','Pinned v2 classifier: synthetic pretraining freezes parameters, support-only transforms and target-aware groups enter column embedding, four CLS tokens compress each row, then twelve dataset ICL blocks read support before class prediction.'),ATTENTION=('attention','An exact toy: fixed logit gap loses anchor mass as keys increase; illustrative log scaling preserves it. This is not model accuracy.'),RESULT_PLOT=('results','Course diagnostic only: log loss decreases on these three supports; missingness indicators cost more prediction time on this one CPU observation.'))
blocks={}
for key,(file,caption) in figs.items():blocks[key]=f'<figure class="b04-figure {"b04-architecture" if key=="ARCHITECTURE" else ""}"><img src="../labs/figures/b04/{file}.{ "svg" if key=="ARCHITECTURE" else "png"}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'
blocks['RESULTS']=table
blocks['TOY']='''<div class="b04-widget" data-b04="toy"><h3>Predict before changing the support</h3><p>Hold the anchor logit gap fixed. Will adding distractors make its weight smaller?</p><label>Support keys<select name="keys">'''+''.join(f'<option value="{n}" {"selected" if n==64 else ""}>{n}</option>' for n in [2,8,64,256,1024,15001])+'''</select></label><label>Anchor logit gap<select name="gap">'''+''.join(f'<option value="{n}" {"selected" if n==2 else ""}>{n}</option>' for n in [0,1,2,4])+'''</select></label><label>Scaling rule<select name="scaling"><option value="fixed">Fixed gap</option><option value="log">Illustrative log(n) scaling</option></select></label><button type="button">Reset toy</button><output aria-live="polite">Static example: 64 keys and gap2 give anchor weight 0.104975 under fixed scaling and 0.984852 under log(n) scaling. These are analytical weights, not model accuracies.</output><div class="b04-bar" aria-hidden="true"><span style="width:10.5%"></span></div><p class="b04-baseline">Toy keys and values stay fixed within a comparison. Neither rule runs the learned QASSMax networks. Low entropy can still focus on the wrong row.</p></div>'''
blocks['MEASURED']='''<div class="b04-widget" data-b04="measured"><h3>Inspect frozen measured evidence</h3><label>Diagnostic configuration<select>'''+''.join(f'<option value="{i}" {"selected" if i==1 else ""}>{x["name"]}</option>' for i,x in enumerate(r['rows']))+'''</select></label><button type="button">Reset evidence</button><output aria-live="polite">clean-128: 63/64 correct; log loss0.095919; prediction0.609 seconds; peak process RSS677.20MiB. The table retains every configuration without JavaScript.</output><script type="application/json">'''+json.dumps(r)+'''</script><p class="b04-baseline">Measured CPU checkpoint inference. Query-batching maximum probability difference0; tolerance10⁻⁶. This panel never substitutes toy weights for measured scores.</p></div>'''
blocks['QUIZ']='''<div class="b04-widget b04-quiz" data-b04="quiz"><h3>Check your prediction</h3><p>With a fixed checkpoint, what can we guarantee about adding support rows?</p>'''+''.join(f'<label><input type="radio" name="support-claim" value="{v}">{label}</label>' for v,label in [('must','Accuracy must increase'),('may','Accuracy may change'),('weights','Weights must update')])+'''<button type="button">Reset answer</button><output aria-live="polite">Choose an answer before revealing the explanation.</output><noscript><p>Feedback: accuracy may change; more support offers no guaranteed direction and does not update the frozen weights.</p></noscript></div>'''
def doc(title,body,interactive=False):
    body=body.replace('<table>','<div class="b04-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/scalable-icl.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html#research-bridge">Research bridge</a></nav>'+body+'</article>'+('<script src="../assets/scalable-icl.js"></script>' if interactive else '')+'</body></html>'
prose=source
for key,value in blocks.items():prose=prose.replace('[['+key+']]',value)
(R/'lessons'/(S+'.html')).write_text(doc('Lesson B04 — TabICL and scalable two-stage ICL',render(prose),True))
ref='''# Scalable ICL · field guide

**Trace:** support X/y + query X → support-fitted wrapper → repeated feature groups and early support-label embeddings → induced column attention → within-row feature attention and4 CLS tokens →512-wide row vectors → dataset ICL with support targets → class head → wrapper probabilities. Query targets enter the scorer only.

**Pinned classification shapes:** N=S+Q; feature representation N×F×128;3 column blocks with128 inducing vectors;3 row blocks;4×128 CLS outputs;12 ICL blocks,512 width,8 heads; ten-class head, task-specific outputs. Repeated grouping size3 with `same` mode preserves F positions. This is checkpoint-specific.

**One-query cost example:**128 support scores after compression versus30×128 repeated per-feature scores. This isolates a single attention operation; it is not total model FLOPs or measured memory.

**Attention oracle:** one logit gap g among n keys gives p_anchor=exp(g)/(exp(g)+n−1). H_normalized=−Σp log(p)/log(n), range[0,1] for n>1. For n=1 define the teaching entropy as0. Low entropy does not imply correctness.

**QASSMax:** q × base_MLP(log S) × [1+tanh(gate_MLP(q))], then scaled dot-product attention. Learned query scaling is not the toy's fixed log multiplier.

| Comparison | What changes | What it can establish |
|---|---|---|
| Nested support, fixed checkpoint | Evidence and context length | Behavior on those paired operating points |
| Fixed Q/K/V, scaling only | Local numerical operator | The isolated attention calculation |
| Matched pretrained variants | One training-time choice | Conditional architectural attribution, with matched prior and budget |
| Mean vs mean + flags | Information representation and width | Paired outcome for the declared missingness process |

**Information contract:** record support/query row identities, checkpoint/source hashes, seed, input type, preprocessing fit scope, view count, batch size, cache, selection and resource budget. Refit on query data only if the protocol explicitly allows it; label such a protocol separately. Hold scoring labels outside inference. Test actual batching behavior.

**Measured B04:** all384 course predictions scored. Batch delta0 at atol1e-6. MCAR flags preserve62/64 accuracy; log loss improves slightly and Brier worsens. Timings are one CPU observation per setting and RSS includes the whole worker. No large-context scaling or missingness-shift robustness claim.

'''+table+'''

**Reproduction gate:** Figure3 INCOMPLETE_SOURCE_PROTOCOL: original generator/seeds/full grid, matched ablation checkpoints and exact references are not authenticated. Executable preflight refuses dispatch. Fresh course inference COMPLETE; pretraining/full benchmark NOT_RUN; historical identity NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE.

**Three learner functions:** stable attention + entropy; support-only means; frozen imputation + flags. Revisit the shape trace, an attention calculation and a failure case after1/7/30 days. Ask the agent for feedback.

[Lesson](../lessons/b04-tabicl-scalable-icl.html) · [Protocol](../labs/b04-reproduction.md) · [Primary paper](https://arxiv.org/html/2602.11139v1#S3) · [Pinned source](https://github.com/soda-inria/tabicl/tree/0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3).
'''
(R/'reference/b04-scalable-icl.html').write_text(doc('Scalable ICL field guide',render(ref)))
entries={}
for name in ['_audit_b04.py','_source_b04.py','_verify_b04.py','_reproduce_b04.py','_diagnostic_b04.py','_prepare_b04.py','_mechanism_b04.py','_budget_b04.py','_test_b04.py','relkit/scalable_b04.py','b04-reproduction.md']:
    entries['labs/'+name]=(P/name).read_bytes()
for p in (P/'sources/b04').glob('*'):
    if p.is_file():entries[str(p.relative_to(R))]=p.read_bytes()
for name in ['inputs.json','diagnostic-protocol.json','checkpoint.json','model-config.json','source-gate.json','diagnostic-audit.json','mechanism-parity.json','environment.json']+[x['name']+'.json' for x in r['rows']]:
    entries['labs/evidence/b04/'+name]=(E/name).read_bytes()
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
    for name,b in sorted(entries.items()):
        info=zipfile.ZipInfo(name,(2026,10,3,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,b)
payload=buf.getvalue();(E/'reproducer.zip').write_bytes(payload)
mechanism=(P/'relkit/scalable_b04.py').read_text();functions={n.name:ast.get_source_segment(mechanism,n) for n in ast.parse(mechanism).body if isinstance(n,ast.FunctionDef)}
test_source=(P/'_test_b04.py').read_text();check=next(ast.get_source_segment(test_source,n) for n in ast.parse(test_source).body if isinstance(n,ast.FunctionDef))
portable=source.replace('[[RESULTS]]',table)
for key,(file,caption) in figs.items():portable=portable.replace('[['+key+']]',f'![{caption}](data:image/png;base64,'+base64.b64encode((P/'figures/b04'/(file+'.png')).read_bytes()).decode()+')')
portable=portable.replace('[[TOY]]','**Portable interaction:** change n and the logit gap in APPLY below; predict the attention weight before running.').replace('[[MEASURED]]','**Portable evidence:** the independent report below regenerates every table row.').replace('[[QUIZ]]','**Retrieval check:** does more support guarantee higher accuracy? Explain before checking the defense rubric.')
portable=portable.replace('](../','](https://avistian.github.io/relational/')
portable=re.sub(r'\]\(((?:b\d{2}|\d{4})-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',portable)
for solution in [False,True]:
    cells=[]
    def md(s):cells.append(nb.v4.new_markdown_cell(s))
    def code(s):cells.append(nb.v4.new_code_cell(s,metadata={'tags':['data-payload'] if s.startswith('payload=') else []}))
    md('# B04 · TabICL and scalable two-stage ICL\n\nOffline portable notebook: Python + NumPy. Three live learner functions drive the attention and missingness exercises. The embedded packet contains complete raw author evidence and upstream source; it replays saved predictions, not new inference. Full model source is visible in the appendices. Figure3 remains source-gated; learner PENDING_WRITTEN_DEFENSE.')
    code('import base64,hashlib,io,json,math,re,tarfile,tempfile,zipfile\nfrom pathlib import Path\nimport numpy as np\nworkspace=Path(tempfile.mkdtemp(prefix="b04-notebook-"))')
    md(portable)
    code('payload=base64.b64decode('+repr(base64.b64encode(payload).decode())+')\nassert hashlib.sha256(payload).hexdigest()=='+repr(hashlib.sha256(payload).hexdigest())+'\nwith zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(workspace)\nevidence=workspace/"labs/evidence/b04"\nraw=json.loads((evidence/"inputs.json").read_text())\nprotocol=json.loads((evidence/"diagnostic-protocol.json").read_text())\nrecords=[json.loads((evidence/(c["name"]+".json")).read_text()) for c in protocol["configs"]]')
    instructions={'attention_readout':'Implement stable dot-product attention, weighted value readout and entropy/log(n). Handle n=1. Hand oracle: logits log(3),0 and values2,10 should yield readout4.','fit_missingness':'Fit column means using only observed support entries. Freeze zero for entirely missing columns. Reject infinities and empty training data.','transform_missingness':'Apply frozen means without refitting. Optionally append one missing flag per original feature in the same order. Preserve the input arrays.'}
    for name,instruction in instructions.items():
        md('## '+('SOLUTION' if solution else 'TODO')+' · '+name+'\n\n'+instruction)
        code(functions[name] if solution else functions[name].split('\n')[0]+'\n    raise NotImplementedError("Implement '+name+'")')
    md('## CHECK · independent hand and intervention oracles')
    code(check);code('print(check_functions(attention_readout,fit_missingness,transform_missingness))')
    md('## APPLY · derive the distractor curve using your attention function\n\nBefore execution, predict whether entropy and anchor mass rise or fall. This is a numerical operator experiment with chosen Q/K/V, not QASSMax inference. Change the gap to0 and explain both curves.')
    code('gap=2.0\nfor n in [2,8,64,256,1024,15001]:\n    keys=np.zeros((n,1));keys[0]=gap\n    values=np.zeros((n,1));values[0]=1\n    for scale in [1.0,np.log(n)]:\n        out,weights,entropy=attention_readout([1.],keys,values,scale)\n        expected=1/(1+(n-1)*np.exp(-gap*scale))\n        assert abs(out[0]-expected)<1e-12\n        print(n,round(scale,4),round(weights[0],6),round(entropy,6))')
    md('## APPLY · your preprocessing reconstructs every measured configuration\n\nNo model runs here. Check frozen means for all six settings and show that changing query values cannot refit support means. Explicit missingness flags submit60 columns; the source wrapper may remove constant ones.')
    code('X=np.array(raw["X"]);missing=np.array(raw["missing_mask"]);query_ids=raw["query_ids"]\nfor record in records:\n    config=record["config"];table=X.copy()\n    if config["missing"]:table[missing]=np.nan\n    support=table[record["support_ids"]];query=table[query_ids]\n    means=fit_missingness(support)\n    np.testing.assert_allclose(means,record["means"],rtol=1e-12,atol=1e-12)\n    xs=transform_missingness(support,means,config["indicators"])\n    xq=transform_missingness(query,means,config["indicators"])\n    assert xs.shape[1]==record["transformed_features"] and np.isfinite(xq).all()\n    print(config["name"],xs.shape,xq.shape)')
    md('## PROVIDED · independent scalar scoring and source gate\n\nThis code does not import your functions. It independently validates identities, means, pairing and probabilities, then regenerates all metrics. The source audit authenticates archived bytes and reports exactly what the bounded search did not recover.')
    for name in ['_audit_b04.py','_source_b04.py']:
        t=(P/name).read_text();code('\n\n'.join(ast.get_source_segment(t,node) for node in ast.parse(t).body if isinstance(node,ast.FunctionDef)))
    code('assert hashlib.sha256((evidence/"inputs.json").read_bytes()).hexdigest()==protocol["inputs_sha256"]\nreport={"source":source_gate(workspace/"labs/sources/b04"),"diagnostic":audit_records(raw,protocol,records)}\nassert report["source"]==json.loads((evidence/"source-gate.json").read_text())\nassert report["diagnostic"]==json.loads((evidence/"diagnostic-audit.json").read_text())\nPath("b04-report.json").write_text(json.dumps(report,indent=2)+"\\n")\nprint(json.dumps(report,indent=2))')
    md('## CHECK · execute complete archived audit with ten corruptions\n\nYour live functions already passed the earlier checks. This separate subprocess checks the original evidence and archive independently.')
    code('import subprocess,sys\nverified=subprocess.run([sys.executable,str(workspace/"labs/_verify_b04.py")],cwd=workspace,check=True,capture_output=True,text=True)\nprint("Complete independent archive replay and mutation checks PASS")')
    md('## EXIT · defend the result\n\nTrace192×30 inputs to64×2 predictions. Compare fixed-checkpoint support sweeps, fixed-Q/K/V operator interventions and matched pretrained ablations. Explain the quality/resource tradeoffs, query-batching limits and exact Figure3 blockers. Give one falsifier and a revision condition. Revisit after1/7/30 days and ask the teaching agent for feedback.')
    code('submission={"shape_trace":"","cost_scope":"","three_comparisons":"","batching_limit":"","missingness_tradeoff":"","figure3_blockers":"","falsifier":"","revision_condition":"","status":"PENDING_WRITTEN_DEFENSE"}\nPath("b04-submission.json").write_text(json.dumps(submission,indent=2)+"\\n")')
    md('## Source appendix · complete visible computation\n\nThe following are the unmodified upstream model modules and the classifier/preprocessor used by the fresh diagnostic. They are provided for inspection, not executed as notebook globals. The complete repository archive and license are embedded in the packet. Start with `tabicl.py` for composition, `embedding.py` for grouping and support masking, `interaction.py` for CLS compression, `learning.py` for dataset ICL, and `ssmax.py` for the learned scaling equation. Other modules expose attention, residual/FFN blocks, RoPE and inference management.\n\nSource commit: `0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3`.')
    upstream=P/'sources/b04/upstream'
    for path in [*sorted((upstream/'src/tabicl/_model').glob('*.py')),upstream/'src/tabicl/_sklearn/classifier.py',upstream/'src/tabicl/_sklearn/preprocessing.py']:
        md('### '+str(path.relative_to(upstream))+'\n\n```python\n'+path.read_text()+'\n```')
    md('## Fresh diagnostic operator · complete source\n\nReplay above does not call this worker. A fresh run needs the external verified checkpoint, the archived source extracted with its top directory stripped, an isolated matching runtime and the budget procedure in the contract. One view and the small fixed split are deliberate course choices, not Figure3 settings.\n\n```python\n'+(P/'_diagnostic_b04.py').read_text()+'\n```\n\n## Source license\n\n```text\n'+(upstream/'LICENSE').read_text()+'\n```')
    for i,c in enumerate(cells):c.id=hashlib.sha256((str(i)+c.source).encode()).hexdigest()[:12]
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    nb.write(book,(P/'solutions' if solution else P)/(S+'.ipynb'))
print('Built B04 lesson, field guide, student/solution notebooks, source/evidence archive')
