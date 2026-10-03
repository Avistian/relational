"""Deterministic B04a HTML, reference, portable notebooks and evidence packet."""
import ast,base64,hashlib,html,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b04a';S='b04a-scaling-rows-features-classes'
audit=json.loads((E/'audit.json').read_text());rows=audit['rows'];source=(R/'lessons/content'/(S+'.md')).read_text()
summary='| Kernel / widening | F | Mean accuracy | Min–max over 3 seeds | Mean log loss |\n|---|---:|---:|---:|---:|\n'
for kernel in ['linear','softmax']:
 for mode in ['noise','copies']:
  for f in [8,64,256]:
   a=[x for x in rows if x['support']==256 and x['features']==f and x['kernel']==kernel and x['widening']==mode]
   summary+=f"| {kernel} / {mode} | {f} | {100*sum(x['accuracy'] for x in a)/3:.2f}% | {100*min(x['accuracy'] for x in a):.2f}–{100*max(x['accuracy'] for x in a):.2f}% | {sum(x['log_loss'] for x in a)/3:.6f} |\n"
blocks={'RESULTS':summary}
figs={'ARCHITECTURE':('architecture','TabFlex: synthetic pretraining learns encoders and blocks; support features and labels form row embeddings; each attention head builds a value summary and normalizer; queries read the summary; learned blocks and a class head finish prediction. The course kernel is a smaller untrained mechanism.'),'RESULT_PLOT':('results','Course S=256 comparison: mean and seed range for accuracy, plus warm pipeline time across widths and widening rules. This is not pretrained model evidence.')}
for key,(name,caption) in figs.items():blocks[key]=f'<figure class="b04a-figure {"b04a-architecture" if key=="ARCHITECTURE" else ""}"><img src="../labs/figures/b04a/{name}.{ "svg" if key=="ARCHITECTURE" else "png"}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'
def select(name,values,default):return '<label>'+name+'<select>'+''.join(f'<option value="{x}" {"selected" if x==default else ""}>{x}</option>' for x in values)+'</select></label>'
blocks['SCALING']='<div class="b04a-widget" data-b04a="scaling"><h3>Change one axis; predict which term grows</h3>'+select('Support rows S',[64,128,1024,32768],128)+select('Raw features F',[8,64,256,1000],64)+select('Classes C',[2,10,11],2)+'''<button type="button">Reset axes</button><output aria-live="polite">At S128, F64, C2: 8,192 explicit weights; 48 summary values; 196,608 input-projection multiply-accumulate terms. Capacity10 accepts two classes.</output><div class="b04a-bars" aria-hidden="true"><span style="width:100%"></span><span style="width:.586%"></span></div><p class="baseline">Analytical course readout only: Q64, d16, values are C-wide one-hot labels. Actual TabFlex values are learned dv-wide vectors. Bars compare stored weight/summary values, excluding other allocations; they are not GPU memory measurements.</p></div>'''
blocks['QUIZ']='''<div class="b04a-widget" data-b04a="quiz"><h3>A boundary check</h3><p>A wrapper merges 11 original labels into 10 outputs. What changed?</p>'''+''.join(f'<label><input type="radio" name="class-boundary" value="{v}"> {s}</label>' for v,s in [('memory','Only memory changed'),('task','Prediction task changed'),('nothing','Nothing meaningful changed')])+'''<button type="button">Reset answer</button><output aria-live="polite">Choose an answer before reading the feedback.</output><noscript><p>Feedback: prediction task changed. Count and authenticate original class identities.</p></noscript></div>'''
first=rows[0]
blocks['MEASURED']='''<div class="b04a-widget" data-b04a="measured"><h3>Inspect every measured setting</h3><label>Course operating point<select>'''+''.join(f'<option value="{i}">{x["name"]}</option>' for i,x in enumerate(rows))+'''</select></label><button type="button">Reset evidence</button><output aria-live="polite">'''+f'{first["name"]}: {first["correct"]}/64 correct; log loss {first["log_loss"]:.6f}. All108 records are in the portable packet.'+'''</output><script type="application/json">'''+json.dumps(audit)+'''</script><p class="baseline">Measured CPU data from this course experiment; warm median uses three complete calls. Every setting appears once, with no best-seed selection.</p></div>'''
def doc(title,body,widgets=False):
 body=body.replace('<table>','<div class="b04a-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/scaling-axes.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html#research-bridge">Research bridge</a></nav>'+body+'</article>'+('<script src="../assets/scaling-axes.js"></script>' if widgets else '')+'</body></html>'
prose=source
for k,v in blocks.items():prose=prose.replace('[['+k+']]',v)
(R/'lessons'/(S+'.html')).write_text(doc('Lesson B04a — Scaling rows, features and classes',render(prose),True))
reference='''# Scaling axes · field guide

**Name the dimensions:** S support rows; Q queries; F raw features; d head width; dv value width; C classes. Never substitute F for d in an attention-cost argument.

**Readout:** A=φ(K)ᵀV has shape d×dv; z=Σφ(k) has shape d. Output(q)=φ(q)A/(φ(q)z+ε). For ELU+1, φ(x)=x+1 when nonnegative and exp(x) otherwise. This is not softmax attention; reassociation is exact only for the chosen kernel. Hand oracle: q0, k0/1, v2/8 →18/(3+ε).

**Cost scope:** explicit query/support weights Q×S; summary d×dv+d. Input projection grows with (S+Q)×F×d in the course kernel. Full transformer projections, activations, FFNs and output heads add other terms. Flash-style algorithms can avoid materializing all pairwise scores. Formula counts are not process RSS or GPU allocation measurements.

**Actual TabFlex:** synthetic pretraining → learned feature/label embeddings → support-only keys/values in each block → linear-attention summary/readout → residuals/norm/FFN → class head and wrapper. Summaries are recomputed per block. The release selects S100/L100/H1K by row/feature shape and changes view count; beyond1000features it requests random projection.

**Compare interventions:** TabFlex changes attention; TabPFN-Wide adapts the prior by continued pretraining; BETA uses encoders and bagging. None implies unlimited rows, features and original classes simultaneously. Capacity is not an accuracy guarantee. Attention maps are not causal explanations.

**Class check:** with capacity10, the pinned preprocess method merges11labels into10effective classes. The course class encoder rejects the overflow. Preserve arbitrary original label IDs and probability-column order.

**Course contract:** three seeds × three row counts × three widths × two widening rules × two kernels =108settings/6,912predictions. Fixed random projection to16, one view, no pretraining or optimization. Identical F8 controls match exactly. Explicit-kernel reconstruction and scalar scoring audit all rows; ten corruptions are rejected. Query targets enter only the scorer. Counts and hashes authenticate recorded evidence, not historical availability.

**Operating point receipt:** identities, transforms and fit scope, checkpoint or no-checkpoint status, source/environment, seed/view count, selection budget, accuracy/logloss, preprocessing/cold/warm time, process/GPU memory scope, failures. Compare paper-era and current baselines separately. No current pretrained-model ranking is measured here.

**Paper lane:** Figure9 full grid2,880operator repetitions includes120reported failure slots. INCOMPLETE_SOURCE_PROTOCOL: original input generation, hardware/precision/kernel/timing contract and raw references unresolved. `--run` refuses dispatch. Pretraining/full benchmark NOT_RUN; learner PENDING_WRITTEN_DEFENSE. No paid runs.

**Retrieval:** tomorrow derive A/z with a new hand example; day7 explain a fifty-thousand-feature versus million-row bottleneck; day30 defend a paired operating point. Ask the teaching agent for feedback.

[Lesson](../lessons/b04a-scaling-rows-features-classes.html) · [Lab](../labs/b04a-scaling-rows-features-classes.ipynb) · [Contract](../labs/b04a-reproduction.md) · [TabFlex primary source](https://arxiv.org/html/2506.05584v1) · [Wide](https://arxiv.org/html/2510.06162v1) · [BETA](https://arxiv.org/html/2502.02527v1).
'''
(R/'reference/b04a-scaling-axes.html').write_text(doc('Scaling rows, features and classes · field guide',render(reference)))
entries={}
for name in ['_audit_b04a.py','_source_b04a.py','_verify_b04a.py','_prepare_b04a.py','_reproduce_b04a.py','_worker_b04a.py','_run_b04a.py','_mechanism_b04a.py','_budget_b04a.py','_test_b04a.py','relkit/scaling_b04a.py','b04a-reproduction.md']:
 entries['labs/'+name]=(P/name).read_bytes()
for p in (P/'sources/b04a').glob('*'):
 if p.is_file() and p.name!='upstream.tar.gz':entries[str(p.relative_to(R))]=p.read_bytes()
for name in ['inputs.json','protocol.json','execution-contract.json','source-gate.json','audit.json','mechanism.json','environment.json']:
 entries['labs/evidence/b04a/'+name]=(E/name).read_bytes()
for p in sorted((E/'runs').glob('*.json')):entries[str(p.relative_to(R))]=p.read_bytes()
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for name,b in sorted(entries.items()):
  info=zipfile.ZipInfo(name,(2026,10,3,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,b)
payload=buf.getvalue();(E/'reproducer.zip').write_bytes(payload)
module=(P/'relkit/scaling_b04a.py').read_text();funcs={n.name:ast.get_source_segment(module,n) for n in ast.parse(module).body if isinstance(n,ast.FunctionDef)}
portable=source.replace('[[RESULTS]]',summary)
for key,(name,caption) in figs.items():portable=portable.replace('[['+key+']]',f'![{caption}](data:image/png;base64,'+base64.b64encode((P/'figures/b04a'/(name+'.png')).read_bytes()).decode()+')')
for k,text in [('SCALING','**Predict:** at S128/Q64/d16/C2, compare8192explicit weights and48summary values. Double F: which term changes?'),('QUIZ','**Retrieval:** is an11→10label merge the original task? Explain before checking.'),('MEASURED','**Inspect:** the complete audit below exposes all108settings. Select a pair before interpreting any score.')]:portable=portable.replace('[['+k+']]',text)
portable=portable.replace('](../','](https://avistian.github.io/relational/')
portable=re.sub(r'\]\((b\d{2}[a-z]?-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',portable)
for solution in [False,True]:
 cells=[]
 def md(x):cells.append(nb.v4.new_markdown_cell(x))
 def code(x,payload=False):cells.append(nb.v4.new_code_cell(x,metadata={'tags':['data-payload'] if payload else []}))
 md('# B04a · Scaling rows, features and classes\n\nPortable notebook: Python + NumPy; no workspace dependency. The packet contains author evidence and complete executable source subset. Replay does not run pretrained TabFlex. Three live learner functions drive the exercises. Figure9 remains INCOMPLETE_SOURCE_PROTOCOL. Online course links may require later publication; figures, code and evidence below are embedded.')
 code('import base64,hashlib,io,json,math,re,tarfile,tempfile,zipfile,copy\nfrom pathlib import Path\nimport numpy as np\nworkspace=Path(tempfile.mkdtemp(prefix="b04a-notebook-"))')
 md(portable)
 code('payload=base64.b64decode('+repr(base64.b64encode(payload).decode())+')\nassert hashlib.sha256(payload).hexdigest()=='+repr(hashlib.sha256(payload).hexdigest())+'\nwith zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(workspace)\nevidence=workspace/"labs/evidence/b04a"\ninputs=json.loads((evidence/"inputs.json").read_text())\nprotocol=json.loads((evidence/"protocol.json").read_text())',True)
 for name,instruction in [('linear_readout','Implement ELU+1 and the associative summary/normalizer. Never form query×support weights. Return a Q×dv readout. Use the hand oracle18/(3+eps).'),('fit_scale','Return mean and population std fitted on support only. Replace zero std with1. Reject empty/nonfinite inputs.'),('class_values','Return one-hot values in the explicitly supplied class order. Reject unknown labels, duplicates or class-count overflow; never silently merge classes.')]:
  md('## '+('SOLUTION' if solution else 'TODO')+' · '+name+'\n\n'+instruction)
  code(funcs[name] if solution else funcs[name].split('\n')[0]+'\n    raise NotImplementedError("Implement '+name+'")')
 test=(P/'_test_b04a.py').read_text();md('## CHECK · independent hand and intervention oracles');code('\n\n'.join(ast.get_source_segment(test,n) for n in ast.parse(test).body if isinstance(n,ast.FunctionDef)));code('print(check_functions(linear_readout,fit_scale,class_values))')
 md('## APPLY · your functions reconstruct one full operating point\n\nChange the seed, support or feature count only after predicting the result. Query labels are read below only after prediction. Holding means fixed, change a query and show no refitting occurs.')
 code('config=next(c for c in protocol["configs"] if c["name"]=="s0-n256-f64-noise-linear")\nt=inputs["tables"][str(config["seed"])];n=config["support"];f=config["features"]\nx=np.column_stack([np.array(t["latent"]),np.array(t["noise"])])[:,:f]\nsupport=x[t["support_pool"][:n]];query=x[t["query_ids"]]\nmean,scale=fit_scale(support);projection=np.random.default_rng(4000+f).normal(size=(f,16))/np.sqrt(f)\nk=((support-mean)/scale)@projection;q=((query-mean)/scale)@projection\nv=class_values(np.array(t["y"])[t["support_pool"][:n]],protocol["classes"],10)\np=linear_readout(q,k,v);p/=p.sum(1)[:,None]\nsaved=json.loads((evidence/"runs"/(config["name"]+".json")).read_text())\nnp.testing.assert_allclose(p,saved["probabilities"],rtol=1e-11,atol=1e-12)\ny_query=np.array(t["y"])[t["query_ids"]]\nprint("Your functions:",sum(p.argmax(1)==y_query),"/64 correct")\nchanged=query.copy();changed[0]=1000\nnp.testing.assert_array_equal(mean,fit_scale(support)[0])\nprint("Frozen query transform:",((changed-mean)/scale).shape)')
 md('## PROVIDED · independent explicit-kernel audit\n\nThe following verifier does not import the learner functions or worker. It reconstructs every probability using a query×support matrix, scores original labels and rejects corrupted records. Timing values are structurally checked, not rerun.')
 for name in ['_source_b04a.py','_audit_b04a.py','_verify_b04a.py']:
  text=(P/name).read_text();code('\n\n'.join(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)))
 code('report=verify(workspace/"labs")\nPath("b04a-report.json").write_text(json.dumps(report,indent=2)+"\\n")\nprint(json.dumps(report,indent=2))\nprint(json.dumps(source_gate(workspace/"labs/sources/b04a"),indent=2))')
 md('## EXIT · written defense\n\nExplain F versus d, the kernel change versus reassociation, one information loss, and one matched operating point. State a falsifier and the exact Figure9 blockers. Source parity, saved replay and new training are different evidence. Revisit after1/7/30days and ask the teaching agent for feedback.')
 code('submission={"shape_trace":"","kernel_vs_reassociation":"","information_loss":"","paired_operating_point":"","class_boundary":"","falsifier":"","figure9_blockers":"","status":"PENDING_WRITTEN_DEFENSE"}\nPath("b04a-submission.json").write_text(json.dumps(submission,indent=2)+"\\n")')
 md('## Complete visible course inference worker\n\nThis worker produced the saved results in fresh processes; executing the replay above does not rerun it.\n\n```python\n'+(P/'_worker_b04a.py').read_text()+'\n```')
 md('## Complete released model modules\n\nUnmodified source from the pinned release; inspection appendix, not notebook globals. The complete114-file executable/config/license archive is in the embedded packet. This is source availability, not pretrained checkpoint execution.')
 for name in ['tabflex_model.py','linear_attention.py','tabflex_wrapper.py','tabpfn_wrapper.py']:
  md('### '+name+'\n\n```python\n'+(P/'sources/b04a'/name).read_text()+'\n```')
 md('## License\n\n```text\n'+(P/'sources/b04a/LICENSE.txt').read_text()+'\n```')
 for i,c in enumerate(cells):c.id=hashlib.sha256((str(i)+c.source).encode()).hexdigest()[:12]
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
 nb.write(book,(P/'solutions' if solution else P)/(S+'.ipynb'))
print('Built B04a lesson, reference, notebooks and',len(payload),'byte archive')
