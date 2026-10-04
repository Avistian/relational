"""Deterministic B05 lesson, reference, portable notebooks and source/evidence archive."""
import ast,base64,hashlib,html,io,json,zipfile
from pathlib import Path
import nbformat as nb
from _source_b05 import source_gate
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b05';S='b05-tabdpt-real-data-retrieval'
# Refuse to package contaminated source even if its checksum was updated.
source_gate(P/'sources/b05')
report=json.loads((E/'course-audit.json').read_text())
table='| Target column | Seed | Episode rows | Queries | MSE (target units²) |\n|---|---:|---:|---:|---:|\n'+''.join(f"| { {0:'Alcohol',6:'Flavanoids',12:'Proline'}[r['target']] } | {r['seed']} | {r['size']} | 8 | {r['mse']:.6f} |\n" for r in report['rows'])
widget='''<div class="b05-board" data-b05="retrieval"><h3>Choose neighbors before seeing the answer</h3><label>Distance features<select name="policy"><option value="excluded">Target removed</option><option value="included">Target included</option></select></label><label>Target values<select name="target"><option value="original">Original values</option><option value="changed">Changed values</option></select></label><label>Neighborhood size<select name="count"><option value="2">2</option><option value="3" selected>3</option><option value="4">4</option></select></label><button type="button">Reset neighborhood</button><output aria-live="polite">Selected row IDs: [0, 1, 2]. Target removed, k=3. Changing only target values leaves this selection unchanged.</output><table><thead><tr><th>Row ID</th><th>Feature x</th><th>Target y</th><th>d²</th></tr></thead><tbody><tr><td>0 · anchor</td><td>0</td><td>0</td><td>0</td></tr><tr><td>1</td><td>0.1</td><td>100</td><td>0.0200</td></tr><tr><td>2</td><td>0.2</td><td>0</td><td>0.0800</td></tr></tbody></table><p>Pretraining illustration: the anchor belongs to the candidate population. Standard deviations are fit to that population. Downstream inference fits statistics to eligible support instead.</p></div>'''
# Static table uses the same six-row input and declared scaling, without JavaScript.
xs=[0,.1,.2,.3,1,2];mean=sum(xs)/6;var=sum((x-mean)**2 for x in xs)/6
rows=''.join(f'<tr data-picked="{str(i<3).lower()}"><td>{i}{" · anchor" if i==0 else ""}</td><td>{x}</td><td>{[0,100,0,100,0,100][i]}</td><td>{x*x/var:.4f}</td></tr>' for i,x in enumerate(xs))
a=widget.index('<tbody>');b=widget.index('</tbody>');widget=widget[:a]+'<tbody>'+rows+widget[b:]
quiz='''<div class="b05-board" data-b05="quiz"><h3>Where must the target go?</h3><p>Before constructing a target-safe retrieval index:</p><label><input type="radio" name="target-quiz" value="keep">Targets enter retrieval</label><label><input type="radio" name="target-quiz" value="remove">Targets leave retrieval</label><label><input type="radio" name="target-quiz" value="predict">Targets become predictions</label><button type="button">Reset answer</button><output aria-live="polite">Choose before revealing feedback.</output><noscript><p>Feedback: targets leave retrieval. Deleting the target after selection cannot remove its influence on selected rows.</p></noscript></div>'''
fig='<figure class="b05-figure"><img src="../labs/figures/b05/architecture.svg" alt="TabDPT original architecture: target-free paper episode order, support label embeddings, support-only attention keys and values, classification and regression heads. Released training order differs."><figcaption>Original architecture and paper-order episode construction; the released sampler discrepancy is shown separately. Batch dimension omitted.</figcaption></figure>'
source=(R/'lessons/content'/(S+'.md')).read_text()
body=source.replace('[[RETRIEVAL]]',widget).replace('[[QUIZ]]',quiz).replace('[[ARCHITECTURE]]',fig).replace('[[RESULTS]]',table)
def doc(title,body):return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/retrieval-episodes.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html#research-bridge">Research bridge</a></nav>'+body+'</article><script src="../assets/retrieval-episodes.js"></script></body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Lesson B05 — TabDPT: real-data pretraining and retrieval',render(body)))
reference='''# Target-safe episode field guide

**Prediction contract:** choose target c → remove c from the distance representation → retrieve → split disjoint support/query → embed support labels only → query prediction → training loss outside model inputs.

At inference, fit transforms on eligible support and apply them to queries. Query labels never select neighbors. For relational tasks, eligibility also requires the correct entity/time boundary.

| Evidence | What it supports | What it cannot establish |
|---|---|---|
| Same dataset ID or file hash | Exact identity flag | Complete overlap audit |
| Different IDs and hashes | Different identifiers/bytes | Independence of derived tables |
| Unchanged IDs after changing targets | Tested selection invariance | Universal absence of leakage |
| Released-method counterexample | Dependency in tested code | Historical training or benchmark effect |
| Checkpoint SHA256 match | Download identity | Paper-score parity |

**Original shape trace:** S=24 + Q=8 rows, F12→100slots→768coordinates;16 blocks, 4 heads, head width 192; attention Q: 32 rows, K/V: 24 rows; score shape[4,32,24]. Classification and regression heads read query rows. Original training feature normalization includes the episode; inference uses support statistics.

**Source discrepancy:** release retrieves before target removal; the paper-order lab removes first. Released selected IDs[0,2,4]→[0,1,3]; safe IDs[0,1,2]→[0,1,2]. Exact L2 oracle replaces FAISS in the unchanged-method probe. No benchmark effect measured.

**B05 status:**18 course episodes / 144 predictions independently checked. Original banknote two-fold reproduction INCOMPLETE_SOURCE_PROTOCOL; full pretraining/full benchmark NOT_RUN. Learner PENDING_WRITTEN_DEFENSE. MSE differs in units across targets; query populations differ across episode sizes.

[Lesson](../lessons/'''+S+'''.html) · [Notebook](../labs/'''+S+'''.ipynb) · [Protocol](../labs/b05-reproduction.md) · [Primary reading](https://arxiv.org/html/2410.18164v3#S3).

Recall after 1/7/30 days following learner completion. Ask the agent to check a new target-intervention trace.
'''
(R/'reference/b05-retrieval-episodes.html').write_text(doc('B05 · Target-safe episode field guide',render(reference)))
# Archive is independently runnable. It excludes delivery/browser tools and mutable budget receipts.
files={}
for name in ['_audit_b05.py','_source_b05.py','_source_probe_b05.py','_test_b05.py','_verify_b05.py','_reproduce_b05.py','_run_b05.py','relkit/retrieval_b05.py','b05-reproduction.md']:
 files['labs/'+name]=(P/name).read_bytes()
files['labs/relkit/__init__.py']=b''
for f in sorted((P/'sources/b05').rglob('*')):
 if f.is_file() and '__pycache__' not in f.parts:files['labs/'+str(f.relative_to(P))]=f.read_bytes()
for name in ['inputs.json','course-protocol.json','episodes.json','course-audit.json','source-gate.json','sampler-probe.json']:
 files['labs/evidence/b05/'+name]=(E/name).read_bytes()
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for name,data in sorted(files.items()):
  info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
archive=buf.getvalue();(E/'reproducer.zip').write_bytes(archive)
md=nb.v4.new_markdown_cell;code=nb.v4.new_code_cell
cells=[md('''# B05 · TabDPT: real-data episodes and retrieval

**Goal:** build target-safe episodes and defend source/reproduction boundaries. Complete the three TODO functions before reading the solution. Author checks do not satisfy your written defense.

The portable default replays all saved course evidence, executes your episode construction and checks frozen source identities. It does not download model weights, train TabDPT or dispatch the gated banknote experiment. Optional fresh course regeneration and released-method probe are separate commands below. Dependencies: Python3.10+ and NumPy; default checks need no GPU. Live Colab NOT_CHECKED.
'''),md('![TabDPT original architecture](data:image/png;base64,'+base64.b64encode((P/'figures/b05/architecture.png').read_bytes()).decode()+')'),md('''**Retrieve:** identify S,Q,F and the K/V slice before opening the source. Why does deleting y after neighbor selection fail? Trace `[32,12] → [32,100] → [32,768]`, then the `[4,32,24]` attention matrices. Only pretraining updates weights. The original source uses whole-episode feature normalization in training and support statistics in inference.
''')]
setup="""import importlib.util, subprocess, sys
if importlib.util.find_spec('numpy') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy==2.2.6'])
import ast, base64, hashlib, io, json, math, itertools, zipfile
from pathlib import Path
import numpy as np
"""
cells.append(code(setup))
payload="payload = "+repr(base64.b64encode(archive).decode())+"\nraw = base64.b64decode(payload)\nassert hashlib.sha256(raw).hexdigest() == "+repr(hashlib.sha256(archive).hexdigest())+"\nworkspace = Path('b05-portable')\nworkspace.mkdir(exist_ok=True)\nwith zipfile.ZipFile(io.BytesIO(raw)) as bundle:\n    bundle.extractall(workspace)\nlab = workspace / 'labs'\nsys.path.insert(0, str(lab.resolve()))\nevidence = lab / 'evidence/b05'\nprint('Authenticated portable source/evidence package')"
c=code(payload);c.metadata['tags']=['data-payload'];cells.append(c)
module=(P/'relkit/retrieval_b05.py').read_text();cells.append(md('## TODO 1–3 · Implement the information contract\n\n`feature_view` removes c before fitting distances. `neighbors` fits scales on support, handles constant columns and breaks ties by identity. `overlap_status` never upgrades different hashes to independence. The provided `episode` function calls your implementations.'))
cells.append(code(module))
test_source=(P/'_test_b05.py').read_text();tree=ast.parse(test_source)
checks='\n\n'.join(ast.get_source_segment(test_source,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('check_'))
cells.append(code(checks+'\n\ncheck_feature_view(feature_view)\ncheck_neighbors(neighbors)\ncheck_overlap(overlap_status)\nprint("Three learner contracts pass")'))
cells.append(md('## CHECK · Your episode must drive measured identities\n\nAll18settings are checked. This calls the notebook functions, not a saved completion flag. The four-neighbor mean is a course readout, not a trained TabDPT.'))
cells.append(code("""inputs = json.loads((evidence/'inputs.json').read_text())
records = json.loads((evidence/'episodes.json').read_text())
for saved in records:
    ep = episode(inputs['X'], saved['target'], saved['seed'], saved['size'], saved['size']-8, saved['seed'])
    for key in ep:
        assert ep[key] == saved[key], key
    nn = neighbors(ep['support_x'], ep['query_x'], ep['support_ids'], 4)
    assert nn.tolist() == saved['neighbor_ids']
changed = np.array(inputs['X']); changed[:,0] = changed[::-1,0]
a = episode(inputs['X'],0,0,32,24,0)
b = episode(changed,0,0,32,24,0)
assert a['selected_ids'] == b['selected_ids']
print('18 episodes and 144 neighbor selections agree; target intervention passes')"""))
cells.append(md('## Independent audit · read the oracle before running\n\nThese scalar loops recompute distances and predictions from the frozen raw table. MSE is not pooled across target units. Larger episodes do not preserve query IDs.'))
for name in ['_audit_b05.py','_source_b05.py']:
 src=(P/name).read_text();tree=ast.parse(src)
 visible='\n\n'.join(ast.get_source_segment(src,n) for n in tree.body if not isinstance(n,ast.If))
 cells.append(code(visible))
cells.append(code("""course = audit_course(evidence)
source = source_gate(lab/'sources/b05')
assert course == json.loads((evidence/'course-audit.json').read_text())
assert source == json.loads((evidence/'source-gate.json').read_text())
report = dict(source=source, course=course)
Path('b05-report.json').write_text(json.dumps(report,indent=2)+'\\n')
print(source['status'], course['episodes'], 'episodes,', course['predictions'], 'predictions')
for row in course['rows']: print(row)
"""))
cells.append(md('## Source probe and optional fresh work\n\nThe released `use_knn` method selects [0,2,4] then [0,1,3]under a target-only intervention. The exact L2 test double establishes local method behavior, not FAISS performance or historical training. Full source is below. To rerun the probe, install PyTorch and run `python b05-portable/labs/_source_probe_b05.py`. To regenerate the 18 course episodes, install scikit-learn and run `_run_b05.py` then `_audit_b05.py` in the same directory. The benchmark command `_reproduce_b05.py --run` refuses while original identities are unresolved.'))
cells.append(code("""probe = json.loads((evidence/'sampler-probe.json').read_text())
assert probe['released_neighbors'] == [[0,2,4],[0,1,3]]
assert probe['paper_order_neighbors'] == [[0,1,2],[0,1,2]]
print(probe)
"""))
cells.append(md('## Version boundary\n\nOriginal TabDPT, v1.1, Turbo and v1.3 are separate artifacts. The recovered original has 4 attention heads; the later header has 8. Turbo changes target conditioning and uses a shared context without per-query retrieval in its described inference recipe. Its newer scores cannot validate the original selected paper result. [Turbo §3.4](https://arxiv.org/html/2608.01400v1#S3.SS4).'))
cells.append(md('## EXIT · Your written defense\n\nSupply a worked distance, the support-only attention slice, a renamed-table overlap counterexample, and a claim limited to the executed sampler probe. Explain the 4-head/8-head checkpoint issue and why two-fold banknote inference would not measure neighbor pruning. Leave your status pending until the tutor reviews this defense. Revisit after 1/7/30 days following completion. Ask follow-up questions whenever the contract is unclear.'))
cells.append(code("defense = {'distance_trace': '', 'attention_boundary': '', 'overlap_counterexample': '', 'source_claim': ''}\nsubmission = dict(status='PENDING_WRITTEN_DEFENSE', defense=defense)\nPath('b05-submission.json').write_text(json.dumps(submission,indent=2)+'\\n')\nprint(submission['status'])"))
cells.append(md('## Complete visible source appendix\n\nOriginal architecture, release preprocessing/retrieval and training sampler remain separate from the course functions. Source licenses accompany the archive. Primary reading: https://arxiv.org/html/2410.18164v3#S3 and https://arxiv.org/html/2608.01400v1#S3.'))
for f in [*sorted((P/'sources/b05/original/src/tabdpt').glob('*.py')),P/'sources/b05/training/dataset.py',P/'sources/b05/training/transformer_layer.py',P/'_source_probe_b05.py']:
 cells.append(md('### '+str(f.relative_to(P))+'\n\n```python\n'+f.read_text()+'\n```'))
book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
for i,c in enumerate(book.cells):c.id='b05-'+str(i)
(P/'solutions').mkdir(exist_ok=True);nb.write(book,P/'solutions'/(S+'.ipynb'))
student=nb.from_dict(json.loads(nb.writes(book)));tree=ast.parse(module);lines=module.splitlines()
for node in sorted([n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['feature_view','neighbors','overlap_status']],key=lambda n:n.lineno,reverse=True):
 lines[node.lineno-1:node.end_lineno]=[lines[node.lineno-1],f'    raise NotImplementedError("TODO: {node.name}")']
student.cells[6].source='\n'.join(lines)+'\n'
assert student.cells[6].source.count('NotImplementedError')==3
nb.write(student,P/(S+'.ipynb'));print('Built B05 lesson/reference/notebooks; archive',len(archive),'bytes')
