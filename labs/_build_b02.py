"""Deterministic lesson/reference/portable notebook builder."""
import ast,base64,hashlib,html,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b02';S='b02-numerical-embeddings-and-ensembles'
r=json.loads((E/'report.json').read_text())
results='**'+r['status']+'** · five evaluation seeds after a fresh 64-member search.\n\n| Evaluation seed | Independently scored test RMSE |\n|---|---:|\n'+''.join(f"| {x['run']} | {x['test']:.8f} |\n" for x in r['runs'][1:])
results+=f"\nMean **{r['test_mean']:.8f}**, sample SD **{r['test_sample_sd']:.8f}**. Released five-seed mean {r['released_mean']:.8f}. Independent scoring reconciles all {r['scored_prediction_rows']:,} validation/test predictions (search plus five evaluations), with maximum report difference {r['max_score_error']:.2g}. Seeds share one split, so this is conditional training variability, not cross-dataset uncertainty.\n"
results+='\n**Evidence-saving deviation:** upstream writes an ensemble name instead of prediction arrays. The first unmodified run is retained. A complete rerun uses a read-only observer of the original final in-memory member predictions; model/selection code stays unchanged. All reruns count toward the budget.\n'
widget='''<div class="b02-widget"><p>Fixed pool: A = [0,2], B = [2,0], C = [5,5]. Predict the chosen members before changing the labels.</p><label>Labels used for selection<select aria-label="Labels used for selection"><option value="validation">Validation labels [1,1]</option><option value="test">Test labels [5,5] — contamination</option></select></label><button type="button">Reset</button><output aria-live="polite">VALID SELECTION — A + B, MSE 0. Test labels must not select members.</output></div>'''
figure='<figure class="b02-architecture"><img src="../labs/figures/b02/architecture.svg" alt="Training-only representation feeds TabM shared weights or TabPack separate packed members; validation selects, frozen predictions are scored on test."><figcaption>Representation changes the coordinates; ensembling changes the prediction paths. Track both through selection and inference.</figcaption></figure>'
prose=(R/'lessons/content'/(S+'.md')).read_text().replace('[[RESULTS]]',results)
def doc(title,body,scripts=False):
 body=body.replace('<table>','<div class="b02-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/embedding-ensemble.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html#research-bridge">Bridge</a></nav>'+body+'</article>'+('<script src="../assets/embedding-ensemble.js"></script>' if scripts else '')+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Lesson B02 — Numerical embeddings and ensembles',render(prose.replace('[[ARCHITECTURE]]',figure).replace('[[WIDGET]]',widget)),True))
reference='''# Numerical embeddings and ensembles · field guide

**Representation:** scalar → vector before cross-feature mixing. Fit bin edges/normalization on training features only. PLE with edges [0,2,4]: 1→[.5,0], 3→[1,.5]; end bins extrapolate. Reject duplicate knots in this exercise.

**TabM:** shared W, member adapters r/s, member biases and separate heads. A layer computes ((x⊙r)W)⊙s+b. Mean member losses train; average predictions at inference.

**TabPack:** independent weights and member-specific hyperparameters packed into batched operations. Masks accommodate shape differences. Validation selects online ensembles and stopping; one run is not selection-free.

**Source formula:** α⊙concat(x,cos(2π(wx+b)))+β. The explicit 2π and truncated-normal initialization follow pinned code. Zero initial affine scales are intentional. This is distinct from PLR and piecewise embeddings.

**Factorial control:** A raw/single, B embedded/single, C raw/many, D embedded/many. RMSE interaction: (D−C)−(B−A). Match selection policy, disclose capacity and cost differences. Tuned tree and RealMLP assess practical usefulness; they do not isolate the mechanism.

**Evidence:** full selected release run ≠ full benchmark; source parity ≠ training identity; author checks ≠ learner defense. Upstream reports five evaluation seeds despite a ten-seed paper caption. An observer is required to retain final ensemble predictions because the upstream serializer stores names.

'''+results+'''\n[Lesson](../lessons/b02-numerical-embeddings-and-ensembles.html) · [Protocol](../labs/b02-reproduction.md) · [Numerical embeddings](https://arxiv.org/abs/2203.05556) · [TabM](https://arxiv.org/html/2410.24210v1) · [TabPack](https://arxiv.org/html/2607.05380v1).
'''
(R/'reference/b02-embeddings-ensembles.html').write_text(doc('Numerical embeddings and ensembles field guide',render(reference)))
# Portable evidence contains every final selected-member array, full dataset and split identities.
entries={}
for p in (E/'compact').rglob('*'):
 if p.is_file():entries['compact/'+str(p.relative_to(E/'compact'))]=p.read_bytes()
for name in ['_audit_b02.py','_test_b02.py','relkit/embeddings_b02.py']:
 entries[name]=(P/name).read_bytes()
entries['report.json']=(E/'report.json').read_bytes()
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for name,b in sorted(entries.items()):
  info=zipfile.ZipInfo(name,(2026,10,3,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,b)
payload=buf.getvalue();(E/'replay.zip').write_bytes(payload)
def funcs(path):
 s=path.read_text();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
f=funcs(P/'relkit/embeddings_b02.py');checks=funcs(P/'_test_b02.py')['check_functions'];auditor=funcs(P/'_audit_b02.py')['audit']
svg=(P/'figures/b02/architecture.svg').read_bytes()
portable=prose.replace('[[ARCHITECTURE]]','![End-to-end architecture](data:image/svg+xml;base64,'+base64.b64encode(svg).decode()+')').replace('[[WIDGET]]','**Try it in code:** call your selector with [1,1] and then [5,5]. Explain why only the first is a valid selection procedure.')
portable=portable.replace('](../','](https://avistian.github.io/relational/')
portable=re.sub(r'\]\(((?:b\d{2}|\d{4})-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',portable)
for solution in [False,True]:
 cells=[]
 def md(s):cells.append(nb.v4.new_markdown_cell(s))
 def code(s):cells.append(nb.v4.new_code_cell(s,metadata={'tags':['data-payload'] if s.startswith('payload=') else []}))
 md('# B02 · Numerical embeddings and ensembles\n\nPROVIDED: authenticated full selected-prediction evidence and visible scorer. TODO: three live functions. CHECK: behavioral cases, leakage example and independent score replay. EXIT: written defense. This portable lane replays fresh author evidence; it does not launch new training or paid work. Python 3.12 + NumPy; install NumPy if absent before starting.')
 code('import base64,copy,hashlib,io,json,math,statistics,tempfile,zipfile\nfrom pathlib import Path\nimport numpy as np\nworkspace=Path(tempfile.mkdtemp(prefix="b02-notebook-"))')
 md(portable)
 code('payload=base64.b64decode('+repr(base64.b64encode(payload).decode())+')\nassert hashlib.sha256(payload).hexdigest()=='+repr(hashlib.sha256(payload).hexdigest())+'\nwith zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(workspace)')
 for name in ['piecewise_linear','member_predictions','greedy_validation']:
  md('## '+('SOLUTION' if solution else 'TODO')+' · '+name+'\n\n'+{'piecewise_linear':'Return one coordinate per interval, extrapolate end bins, reject invalid knots. Train-fitted edges are provided as an argument.','member_predictions':'Return [batch,member,out] using shared W and member r/s/bias. Keep axes explicit; this is one affine layer.','greedy_validation':'Use member-by-row validation predictions only. Permit repeated members; choose first ties; stop when MSE does not strictly improve. Do not import a finished selector.'}[name])
  code(f[name] if solution else f[name].split('\n')[0]+'\n    raise NotImplementedError("Implement '+name+'")')
 md('## CHECK · feedback on your actual functions')
 code(checks)
 code("print(check_functions(piecewise_linear,member_predictions,greedy_validation))\npool=np.array([[0.,2.],[2.,0.],[5.,5.]])\nassert greedy_validation(pool,np.array([1.,1.]))==[0,1]\nassert greedy_validation(pool,np.array([5.,5.]))==[2]\nprint('Test-guided selection changes the predictor: reject that workflow.')")
 md('## PROVIDED · independent full prediction replay\n\nThe following visible function authenticates every compact input and selected-member prediction; checks the complete row partition, search-to-evaluation configurations, seed coverage and member/checkpoint identities; and recomputes every validation/test RMSE with a scalar summation oracle.')
 code(auditor)
 code("report=audit(workspace/'compact')\nassert report==json.loads((workspace/'report.json').read_text())\nPath('b02-report.json').write_text(json.dumps(report,indent=2)+'\\n')\nprint(json.dumps(report,indent=2))")
 md('## Read the training implementation\n\nFresh training uses the original pinned release, not these NumPy exercises. [Pinned model/training code](https://github.com/yandex-research/tabpack/blob/05a89e21b955f12de84889d662e15ca534019aaa/src/project/tabpack.py), [packed layers](https://github.com/yandex-research/tabpack/blob/05a89e21b955f12de84889d662e15ca534019aaa/src/project/nn.py), [optimizers](https://github.com/yandex-research/tabpack/blob/05a89e21b955f12de84889d662e15ca534019aaa/src/project/optim.py). The local source archive accompanies the lesson; use the reproduction protocol to rebuild the full environment. The selected code below makes the actual embedding and weight ownership visible.')
 source=(P/'sources/b02/upstream/src/project/nn.py').read_text();tree=ast.parse(source)
 for node in tree.body:
  if isinstance(node,ast.ClassDef) and node.name in ['CosineEmbeddings','LinearPack']:
   md('### Official '+node.name+'\n\n```python\n'+ast.get_source_segment(source,node)+'\n```')
 md('## EXIT · write before checking the solution\n\nExplain the three mechanisms, design four controlled cells plus tree/RealMLP baselines, state selection and compute budgets, give two falsification tests and one revision condition. Explain the five-versus-ten seed discrepancy and the observer. Revisit in 1, 7 and 30 days. Ask the agent for feedback. Author checks never grade your written defense.')
 code("submission={'mechanism_distinction':'','factorial_controls':'','baseline_rationale':'','selection_budget':'','falsification_tests':[],'revision_condition':'','source_discrepancies':'','status':'PENDING_WRITTEN_DEFENSE'}\nPath('b02-submission.json').write_text(json.dumps(submission,indent=2)+'\\n')")
 for i,c in enumerate(cells):c.id=hashlib.sha256((str(i)+c.source).encode()).hexdigest()[:12]
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
 nb.write(book,(P/'solutions' if solution else P)/(S+'.ipynb'))
print('Built B02 lesson, reference and portable notebooks')
