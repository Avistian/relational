"""Build portable B06 lesson, reference, blank student and canonical solution."""
import ast,base64,hashlib,io,json,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b06';S='b06-mitra-prior-mixtures'
report=json.loads((E/'course-audit.json').read_text())
mixture='''<div class="b06-board" data-b06="mixture"><h3>Change the prior; keep the draws</h3><label>SCM probability<select><option value="0">0</option><option value=".25">0.25</option><option value=".5" selected>0.5</option><option value=".75">0.75</option><option value="1">1</option></select></label><div class="b06-tasks"><div class="b06-task" data-family="scm">u=0.1 → SCM</div><div class="b06-task" data-family="scm">u=0.4 → SCM</div><div class="b06-task" data-family="tree">u=0.6 → TREE</div><div class="b06-task" data-family="tree">u=0.9 → TREE</div></div><button type="button">Reset mixture</button><output aria-live="polite">Baseline p=0.5: 2 SCM + 2 tree tasks. Each task comes from one generator.</output></div>'''
selection='''<div class="b06-board" data-b06="selection"><h3>Which evidence chooses p?</h3><p>Hypothetical cross-entropy; lower is better.</p><table><thead><tr><th>SCM p</th><th>Development</th><th>Final</th></tr></thead><tbody><tr><td>0.3</td><td>0.42</td><td>0.41</td></tr><tr><td>0.5</td><td>0.40</td><td>0.39</td></tr><tr><td>0.7</td><td>0.38</td><td>0.43</td></tr></tbody></table><label>Mixture selection<select><option value="none">Prespecified</option><option value="development">Development results</option><option value="final">Final results</option></select></label><button type="button">Reset selection</button><output aria-live="polite">Prespecified p=0.5: no benchmark selected the mixture.</output></div>'''
quiz='''<div class="b06-board" data-b06="quiz"><h3>Isolate the generator effect</h3><p>Which intervention belongs in the prior comparison?</p><label><input type="radio" name="contract" value="prior">Change prior only</label><label><input type="radio" name="contract" value="learner">Change learner also</label><label><input type="radio" name="contract" value="test">Change queries also</label><button type="button">Reset answer</button><output aria-live="polite">Choose before revealing feedback.</output><noscript><p>Answer: change prior only; keep the matched learner and evaluation contract.</p></noscript></div>'''
def fig(name,caption,cls='b06-figure'):
 return f'<figure class="{cls}"><img src="../labs/figures/b06/{name}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'
table='| Evaluation family | Training prior | Accuracy | Cross-entropy |\n|---|---|---:|---:|\n'+''.join(f"| {x['family']} | {x['arm']} | {x['accuracy_mean']:.4f} ± {x['accuracy_sd']:.4f} | {x['ce_mean']:.6f} ± {x['ce_sd']:.6f} |\n" for x in report['aggregate'])
replacements={'[[MIXTURE]]':mixture,'[[SELECTION]]':selection,'[[QUIZ]]':quiz,'[[ARCHITECTURE]]':fig('architecture.svg','Mitra concept with explicit course dimensions; original scale is shown separately.'),'[[MASK]]':fig('mask.svg','Illustrative S=4,Q=2 read permissions. Filled cells allow attention; crosses forbid it.'),'[[RESULTS]]':table,'[[RESULT_FIGURE]]':fig('results.png','Measured course losses: points are training seeds; black bars are means; dashed line is uniform prediction.','b06-results')}
def doc(title,body):
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/prior-mixtures.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html#research-bridge">Research bridge</a></nav>'+body+'</article><script src="../assets/retrieval-pool.js"></script><script src="../assets/retrieval-bank.js"></script><script src="../assets/teachback.js"></script><script src="../assets/prior-mixtures.js"></script></body></html>'
text=(R/'lessons/content'/(S+'.md')).read_text()
for k,v in replacements.items():text=text.replace(k,v)
(R/'lessons'/(S+'.html')).write_text(doc('Lesson B06 — Mitra: the prior is part of the model',render(text)))
ref='''# Prior-mixture field guide

**One skill:** isolate the effect of training-task distributions.

| Term | Meaning |
|---|---|
| Prior | Distribution of generated tasks |
| Outer mixture | Draw one generator for an entire task |
| Hybrid mechanism | Combine mechanism families within one task |
| Support | Labeled context permitted as model input |
| Query | Unlabeled model input; label used only for loss/scoring |
| Forward ICL | Change context with frozen weights |
| Fine-tuning | Change weights using downstream training evidence |

**Control contract:** same learner, initial tensors per seed, task dimensions, optimizer/update budget, support transform and held-out query identities. Change the prior only. Freeze p before final evaluation. A benchmark used to choose p becomes development evidence.

**Worked trace:** u=.4,p=.5 selects SCM. Support[1,3] gives mean2/scale1; query5 transforms to3. Course[32rows,5tokens,32width] has column scores[4,5,5] and row scores[4,32,24]. Only24support rows supply row-attention keys/values. Paired loss effect is mixed minus baseline on matched seed/task/query identities; negative favors mixed.

**Read the evidence:**9fresh course fits/4320predictions; near chance and all family means worse than uniform CE=ln2. Checkpoints replay exactly, but replay is not new training. Course generators are proxies. Table12 original6settings remain INCOMPLETE_SOURCE_PROTOCOL; full pretraining/full benchmark NOT_RUN. Exact original trainer is unavailable in the authenticated package; only its source gate runs. Learner PENDING_WRITTEN_DEFENSE.

**Recover the scientific question:** low within-prior accuracy could mean difficult tasks or undertraining, not useful diversity. A v1-to-v2 improvement changes several factors and does not isolate prior design.

[Lesson](../lessons/b06-mitra-prior-mixtures.html) · [Lab](../labs/b06-mitra-prior-mixtures.ipynb) · [Protocol](../labs/b06-reproduction.md) · [Mitra](https://arxiv.org/html/2510.21204v1#S3) · [Mitra-v2](https://arxiv.org/html/2609.04540v1#S2).

Retrieve after1/7/30days following completion. Ask the agent to review your written defense.
'''
(R/'reference/b06-prior-mixtures.html').write_text(doc('B06 · Prior-mixture field guide',render(ref)))
# Small sealed standalone archive, including all saved course checkpoints and inputs.
files={}
for name in ['_run_b06.py','_audit_b06.py','_verify_b06.py','_source_b06.py','_test_b06.py','_model_test_b06.py','_reproduce_b06.py','_budget_b06.py','relkit/prior_b06.py','b06-reproduction.md']:
 files['labs/'+name]=(P/name).read_bytes()
files['labs/relkit/__init__.py']=b''
for root in [P/'sources/b06',E/'runs']:
 for p in sorted(root.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:files['labs/'+str(p.relative_to(P))]=p.read_bytes()
for name in ['course-protocol.json','course-audit.json','source-gate.json']:files['labs/evidence/b06/'+name]=(E/name).read_bytes()
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for name,data in sorted(files.items()):
  info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
archive=buf.getvalue();(E/'reproducer.zip').write_bytes(archive)
md=nb.v4.new_markdown_cell;code=nb.v4.new_code_cell
cells=[md('''# B06 · Mitra: the prior is part of the model

**Skill:** isolate generator effects while preserving a matched learner and evaluation contract. PROVIDED shows implementation; TODO is your work; CHECK gives immediate feedback; EXIT asks for a written defense.

The default executes your functions, a fresh one-update mechanism check, and inference from all nine saved course checkpoints. It does **not** rerun full pretraining or the nine author fits. A separate gated command regenerates all nine fits. Original Table12 is source-gated, not implemented as an exact trainer.

Python3.11+,NumPy,PyTorch. Author environment:NumPy2.5.0,PyTorch2.13.0+cpu. NoGPU required. The archive is embedded for portability; no repository checkout needed. Live Colab NOT_CHECKED. Learner PENDING_WRITTEN_DEFENSE.

## Concept recap

A prior is a distribution of whole tasks. A generator samples a task. An outer mixture draws one generator per table; a hybrid generator combines mechanisms within a table. Changing a prior can change what a fixed architecture learns.

Support rows supply labels; query labels only score outputs. With24support+8query rows and4features, the learner embeds32×5cells. Two blocks mix columns within rows and read support across rows. Query labels never enter its forward signature.

For u=.4 and p=.5 choose SCM. For support[1,3], mean2 and standard deviation1 transform query5 to3. Lower cross-entropy is better; constant p=.5 gives0.693147. Before viewing results, predict whether all controlled experiments necessarily learn useful ICL.
''')]
for name,caption in [('architecture','Mitra concept and course trace; original12×512 differs from course2×32.'),('mask','Toy attention permissions, S4/Q2. The measured course uses S24/Q8.')]:
 cells.append(md('![Architecture](data:image/png;base64,'+base64.b64encode((P/f'figures/b06/{name}.png').read_bytes()).decode()+')\n\n'+caption))
setup="""# @colab-bootstrap: install only missing dependencies; no repository needed.
import importlib.util, subprocess, sys
for module, package in [('numpy','numpy==2.5.0'),('torch','torch==2.13.0')]:
    if importlib.util.find_spec(module) is None:
        subprocess.check_call([sys.executable,'-m','pip','install',package])
import ast, base64, hashlib, io, json, math, statistics, zipfile, copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
"""
cells.append(code(setup))
payload="payload = "+repr(base64.b64encode(archive).decode())+"\nraw=base64.b64decode(payload)\nassert hashlib.sha256(raw).hexdigest()=="+repr(hashlib.sha256(archive).hexdigest())+"\nworkspace=Path('b06-portable'); workspace.mkdir(exist_ok=True)\nwith zipfile.ZipFile(io.BytesIO(raw)) as bundle: bundle.extractall(workspace)\nlab=workspace/'labs'; evidence=lab/'evidence/b06'\nconfig=json.loads((evidence/'course-protocol.json').read_text())\nprint('Authenticated portable course/source package')"
c=code(payload);c.metadata['tags']=['data-payload'];cells.append(c)
module=(P/'relkit/prior_b06.py').read_text();tree=ast.parse(module)
sections=[(['choose_prior'],'TODO 1 · Choose the whole-task generator','Implement interval validation and the two-family choice. Predict the p=0 and p=1 behavior.'),(['support_normalize'],'TODO 2 · Restrict information access','Fit means and population scales on support only. Preserve input arrays. A changed query must not change transformed support.'),(['paired_effect'],'TODO 3 · Pair by identity','Require identical nonempty seed-key sets; return left-minus-right in sorted identity order. Insertion order is not identity.')]
tests=(P/'_test_b06.py').read_text();testtree=ast.parse(tests)
for (names,title,body),check in zip(sections,['check_prior','check_normalize','check_pair']):
 cells.append(md('## '+title+'\n\n'+body));src='\n\n'.join(ast.get_source_segment(module,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names);cells.append(code(src))
 src='\n\n'.join(ast.get_source_segment(tests,n) for n in testtree.body if isinstance(n,ast.FunctionDef) and n.name==check)
 cells.append(code(src+'\n'+check+'('+names[0]+')\nprint("Learner contract passes")'))
for names,title,body in [(['make_task'],'PROVIDED · Generate task distributions','Course proxies: triangular nonlinear SCM, depth-two tree, additive hybrid. Support score median fixes class boundary. The hybrid family appears only at evaluation.'),(['CellBlock','CellLearner'],'PROVIDED · Cell-token learner','Follow x→feature embeddings and support_y→label tokens. A missing target uses token2. Column attention reads each row; row attention uses support-only keys/values. The query target token feeds the binary head.'),(['state_hash','train_one','predict_tasks'],'PROVIDED · Matched trainer and predictor','All three arms share initialization per seed.160updates×4tasks; balanced shuffled mixture counts. Query labels enter cross-entropy outside forward. Final checkpoint only; no test-based selection.')]:
 cells.append(md('## '+title+'\n\n'+body));cells.append(code('\n\n'.join(ast.get_source_segment(module,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names)))
cells.append(md('## CHECK · Your functions control real computation\n\nMonkey-patched counters verify that the visible trainer and generator call your functions. This fresh **one-update probe** is not the nine-fit result. Predictions from each saved course checkpoint must then match the immutable records.'))
cells.append(code("""counter={'prior':0,'normalize':0}
_saved_prior=choose_prior; _saved_normalize=support_normalize
def counted_prior(*args):
    counter['prior']+=1
    return _saved_prior(*args)
def counted_normalize(*args):
    counter['normalize']+=1
    return _saved_normalize(*args)
choose_prior=counted_prior; support_normalize=counted_normalize
probe_config=dict(config,steps=1)
probe,trace=train_one(probe_config,'mixed',0)
assert counter=={'prior':4,'normalize':4}
choose_prior=_saved_prior; support_normalize=_saved_normalize
assert trace['initial_sha256']!=trace['final_sha256']
tasks=json.loads((evidence/'runs/tasks.json').read_text())
maximum=0.0
for seed in config['seeds']:
    for arm in config['arms']:
        saved=json.loads((evidence/f'runs/run-{arm}-{seed}.json').read_text())
        model=CellLearner(config)
        model.load_state_dict(torch.load(evidence/f'runs/weights-{arm}-{seed}.pt',weights_only=True,map_location='cpu'))
        regenerated=predict_tasks(model,tasks,config)
        assert len(regenerated)==len(saved['records'])==480
        for a,b in zip(regenerated,saved['records']):
            assert (a['task_id'],a['query_id'],a['y'])==(b['task_id'],b['query_id'],b['y'])
            maximum=max(maximum,max(abs(x-y) for x,y in zip(a['p'],b['p'])))
assert maximum<=1e-7
print('4320 saved-checkpoint predictions regenerated; maximum error',maximum)
"""))
cells.append(md('## PROVIDED · Independent scalar audit\n\nRead the identity checks before the scorer. It does not import the model. Fixed equal query counts make the record mean equal the equal-task mean. This is saved-evidence verification, not fresh pretraining.'))
for name,fn in [('_audit_b06.py','audit_course'),('_source_b06.py','source_gate')]:
 src=(P/name).read_text();tr=ast.parse(src);cells.append(code('\n\n'.join(ast.get_source_segment(src,n) for n in tr.body if isinstance(n,ast.FunctionDef) and n.name==fn)))
cells.append(code("""course=audit_course(evidence)
source=source_gate(lab/'sources/b06')
assert course==json.loads((evidence/'course-audit.json').read_text())
assert source==json.loads((evidence/'source-gate.json').read_text())
for family in config['eval_families']:
    left={r['seed']:r['cross_entropy'] for r in course['rows'] if r['family']==family and r['arm']=='mixed'}
    right={r['seed']:r['cross_entropy'] for r in course['rows'] if r['family']==family and r['arm']=='scm'}
    print(family,'mixed-minus-SCM per seed:',paired_effect(left,right))
assert all(r['ce_mean']>math.log(2) for r in course['aggregate'])
Path('b06-report.json').write_text(json.dumps(dict(course=course,source=source),indent=2)+'\\n')
print('All family means worse than uniform CE',math.log(2))
print(source['status'])
"""))
cells.append(md('## Author-reference results\n\nThese values come from the nine author fits; the default notebook regenerated checkpoint predictions, not training. Three training-seed SDs are descriptive, conditional on fixed tasks. No real-data or paper parity.\n\n'+table))
cells.append(md('![Measured course results](data:image/png;base64,'+base64.b64encode((P/'figures/b06/results.png').read_bytes()).decode()+')\n\nEvery mean exceeds uniform CE. Near-chance performance limits any story about beneficial prior diversity.'))
cells.append(md('## EXIT · Written defense\n\nExplain: (1) what is held fixed and varied; (2) why an outer mixture differs from hybrid mechanisms; (3) why a benchmark used for prior selection is development evidence; (4) what the uniform baseline reveals; (5) which original Table12 artifacts remain missing. Predict how query-label input would invalidate the experiment. Submit your answer to the agent; passing author checks cannot satisfy it.'))
cells.append(code("""submission=dict(status='PENDING_WRITTEN_DEFENSE',prediction_replay_max_error=maximum,course_fits=course['fits'],paper_status=source['status'])
Path('b06-submission.json').write_text(json.dumps(submission,indent=2)+'\\n')
print(submission)
"""))
cells.append(md('## NEXT STEP · Two separate execution gates\n\nThe full fresh **course** command below runs the same visible source from the embedded package. It verifies that source matches your notebook definitions first. A new output directory preserves author evidence. Original **paper** preflight authenticates missing requirements and refuses dispatch. Replacing this gate with longer course training would not reproduce Mitra. USD10cap/8commitment stop;3600saggregate local numerical limit. Full paper training and liveColab NOT_RUN/NOT_CHECKED.'))
cells.append(code("""RUN_FRESH_COURSE=False
RUN_PAPER_REPRO=False
if RUN_FRESH_COURSE:
    # Confirm the extracted canonical file matches all live implementation definitions.
    namespace={};exec((lab/'relkit/prior_b06.py').read_text(),namespace)
    for name in ['choose_prior','support_normalize','paired_effect','make_task','train_one','predict_tasks']:
        assert globals()[name].__code__.co_code==namespace[name].__code__.co_code, name
    subprocess.check_call([sys.executable,str(lab/'_budget_b06.py'),sys.executable,str(lab/'_run_b06.py'),'--output',str(evidence/'fresh-course-runs')])
if RUN_PAPER_REPRO:
    subprocess.check_call([sys.executable,str(lab/'_reproduce_b06.py'),'--run'])
"""))
cells.append(md('## Source/protocol audit\n\nOriginal Mitra:12layers,width512,4heads,72Mparameters;45million synthetic tasks; support quantile transform then standardization. Course:2layers,width32,640tasks per fit,support standardization,proxy generators. Mitra-v2 Hybrid SCM is not the additive course hybrid. Fine-tuning changes downstream weights; forward ICL does not.\n\nThe complete protocol and pinned source inventory are in the extracted package. `python b06-portable/labs/_reproduce_b06.py` lists the missing original6checkpoint/training histories, TabRepo folds and original evaluator/predictions. The current finetuning source targets v2. Original source parity is NOT_ESTABLISHED.\n\nPrimary readings: [Mitra](https://arxiv.org/html/2510.21204v1#S3), [Mitra-v2](https://arxiv.org/html/2609.04540v1#S2). Ask the agent for feedback and revisit after1/7/30days following your completion.'))
solution=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
student=nb.reads(nb.writes(solution),as_version=4)
todo={'choose_prior','support_normalize','paired_effect'}
for cell in student.cells:
 if cell.cell_type!='code':continue
 tr=ast.parse(cell.source)
 if len(tr.body)==1 and isinstance(tr.body[0],ast.FunctionDef) and tr.body[0].name in todo:
  node=tr.body[0];header=cell.source.splitlines()[0];cell.source=header+'\n    raise NotImplementedError("TODO: '+node.name+'")';cell.metadata['tags']=['exercise']
for folder,book in [(P,student),(P/'solutions',solution)]:
 for i,cell in enumerate(book.cells):cell['id']='b06-'+str(i)+'-'+hashlib.sha256(cell.source.encode()).hexdigest()[:8]
 folder.mkdir(exist_ok=True);nb.write(book,folder/(S+'.ipynb'))
print('Built B06',len(cells),'cells;',len(archive),'archive bytes')
