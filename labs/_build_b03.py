"""Deterministic B03 lesson, field guide and portable notebook builder."""
import ast,base64,hashlib,html,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b03';S='b03-pfn-tabpfn-generations'
r=json.loads((E/'diagnostic-audit.json').read_text());gate=json.loads((E/'source-gate.json').read_text())
versions=[
 ['v1','Numeric classification; ≤10 classes; 1k support rows / 100 features','Row tokens; no dedicated missingness channel','Released weights / inference','18-task numeric benchmark; 5 resamples; typically 32 views','https://arxiv.org/abs/2207.01848'],
 ['v2','Numeric + categorical; classification (≤10 classes) / regression; 10k / 500','Grouped cells; explicit missingness signals','Released weights; historical license','Nature: 10 repetitions; default 4 classifier / 8 regressor views; tuning separate','https://www.nature.com/articles/s41586-024-08328-6'],
 ['2.5','50k design; evaluated to 100k / 2k features','Deeper alternating attention; 3-feature groups; 64 learned rows; inherited missingness encoding','Released weights with separate terms; API','TabArena and additional tasks; base / fine-tuned / ensemble differ; exact run recipe NOT_AUTHENTICATED here','https://arxiv.org/html/2511.08667v1'],
 ['2.6','Later interim release; summary lists 100k / 2k','Do not infer recipe from the 2.5 name','Release-specific weights / access','Inventory only: B03 does not authenticate a distinct 2.6 benchmark recipe','https://arxiv.org/html/2609.17895v2'],
 ['3 base','Up to 1M rows / 2k features in report summary','Scaling, small cache, many-class decoder; missingness indicators','Released weights with separate terms; API','TabArena / TALENT; exact estimator count and split recipe NOT_AUTHENTICATED here','https://arxiv.org/html/2605.13986v2'],
 ['3.5 base / Fast','Current release: up to 1M rows / 20k features; not a B03 measurement','Fourier + support ECDF and missingness signals; shared classification/regression checkpoint per variant','Base/Fast weights; separate weight terms','Report: base 8 views, Fast 4; exact benchmark recipe NOT_AUTHENTICATED here','https://arxiv.org/html/2609.17895v2'],
 ['3 / 3.5 Plus, Thinking','Expanded system capabilities; version-specific modalities','Additional preprocessing / test-time computation; unreleased details stay unknown','Hosted offerings; base weights do not reproduce system','Pin API model, mode, time/selection budget; do not transfer headline scores to base','https://arxiv.org/html/2609.17895v2']]
(E/'versions.json').write_text(json.dumps(versions,indent=2)+'\n')
matrix='| Version / variant | Inputs and reported scale | Mechanism / missingness | Access boundary | Evaluation / computation contract |\n|---|---|---|---|---|\n'+''.join('| ['+v[0]+']('+v[5]+') | '+' | '.join(v[1:5])+' |\n' for v in versions)
results=f"**Measured:** maximum aligned change **{r['max_abs_delta']:.6f}**; accuracy **94.67%** for every permutation. All five nonidentity maps exceed tolerance. Repeated identical-query prediction changes by **{r['repeat_max_abs_delta']:.0f}**. The independent audit reconciles all **450** probability rows. [Raw evidence](../labs/evidence/b03/permutation-diagnostic.json) · [Scalar audit](../labs/evidence/b03/diagnostic-audit.json)."
figure='<figure class="b03-architecture"><img src="../labs/figures/b03/architecture.svg" alt="Historical TabPFN v2: synthetic pretraining freezes theta; support labels and query features enter four preprocessing views, grouped cell and target tokens, twelve feature/row attention blocks, a class head, label alignment and probability averaging."><figcaption>Historical v2 computation. B03 executes the released checkpoint for a separate diagnostic; original pretraining and the named blood benchmark are unrun.</figcaption></figure>'
widget='<div class="b03-widget"><h3>Rename labels; preserve their meaning</h3><p>Predict whether accuracy can stay fixed while probabilities move.</p><label>Support-label permutation<select>'+''.join('<option value="'+str(i)+'">'+str(x['permutation'])+'</option>' for i,x in enumerate(r['rows']))+'</select></label><button type="button">Reset</button><output aria-live="polite">Identity map: max change 0, accuracy 94.67%, log loss 0.097993. Other label maps change probabilities by up to0.071617.</output><p class="baseline">Fixed baseline: identity labels, same 75 query rows, same checkpoint and four views. Tolerance: absolute 10⁻⁶; no relative term.</p><script type="application/json">'+json.dumps(r)+'</script></div>'
prose=(R/'lessons/content'/(S+'.md')).read_text().replace('[[MATRIX]]',matrix).replace('[[RESULTS]]',results)
def doc(title,body,scripts=False):
 body=body.replace('<table>','<div class="b03-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/pfn-contracts.css"><link rel="stylesheet" href="../assets/l061-pfn.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html#research-bridge">Bridge</a></nav>'+body+'</article>'+('<script src="../assets/l061-pfn-viz.js"></script><script src="../assets/pfn-contracts.js"></script>' if scripts else '')+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Lesson B03 — PFN and the TabPFN generations',render(prose.replace('[[ARCHITECTURE]]',figure).replace('[[WIDGET]]',widget)),True))
reference='''# PFN contracts · field guide

**Three objects:** prior over tasks, observed support, learned inference parameters θ. Base in-context adaptation changes support-dependent computation; fine-tuning changes θ. View ensembling changes computation too.

**Posterior oracle:** normalize prior × support likelihood, then average query predictions. Two hypotheses (.25,.75), equal prior, three successes → posterior(1/28,27/28), predictive 41/56. This exact finite prior is a teaching oracle, not TabPFN's synthetic generator.

**Historical v2:** paired features + missingness → grouped 192-wide tokens + target token → 12 alternating attention/FFN blocks → query target head → temperature → class alignment → average four views. Do not confuse fixed weights with fixed representations or caches.

**Class remapping:** if old c becomes π(c), restore column c from new column π(c). Use a three-cycle, because a binary swap cannot expose inverse-permutation errors. Compare probabilities at a declared tolerance; equal accuracy is insufficient.

**Claim contract:** version, variant, checkpoint hash, inference recipe, ordered data/splits, selection and inference budget. Missing → INCOMPLETE; different → INCOMPARABLE; all equal → necessary match, still not a result reproduction.

'''+matrix+'\n\n'+results+'''

**Evidence boundary:** B03-TABPFNV2-BLOOD-OFFICIAL-SPLITS remains INCOMPLETE_SOURCE_PROTOCOL: absent archived loader, unauthenticated historical split/config/checkpoint linkage and reference. No benchmark runs. Iris is a separate fresh diagnostic. Full pretraining, full suite and later-version runs NOT_RUN.

**Missingness / class capacity:** record encoding and target class count per checkpoint. NaN acceptance does not prove missingness-shift robustness. A many-class wrapper changes the method and budget. Thinking-mode internals remain unknown where unreleased.

[Lesson](../lessons/b03-pfn-tabpfn-generations.html) · [Protocol](../labs/b03-reproduction.md) · [EquiTabPFN](https://arxiv.org/html/2502.06684v4).
'''
(R/'reference/b03-pfn-contracts.html').write_text(doc('PFN contracts field guide',render(reference)))
entries={}
for name in ['_audit_b03.py','_verify_b03.py','_reproduce_b03.py','_diagnostic_b03.py','_test_b03.py','relkit/pfn_b03.py','relkit/tabpfn_l064_v2.py','b03-reproduction.md']:
 entries['labs/'+name]=(P/name).read_bytes()
for name in ['TabPFN-main.zip','TabPFN-original-stale.zip','openml-task-10101.json','openml-task-145836.json','github-datasets.txt']:
 entries['labs/sources/b03/'+name]=(P/'sources/b03'/name).read_bytes()
for name in ['source-gate.json','source-lock.json','permutation-diagnostic.json','diagnostic-audit.json','versions.json']:
 entries['labs/evidence/b03/'+name]=(E/name).read_bytes()
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for name,b in sorted(entries.items()):
  info=zipfile.ZipInfo(name,(2026,10,3,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,b)
payload=buf.getvalue();(E/'reproducer.zip').write_bytes(payload)
source=(P/'relkit/pfn_b03.py').read_text();f={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
ts=(P/'_test_b03.py').read_text();checks=next(ast.get_source_segment(ts,n) for n in ast.parse(ts).body if isinstance(n,ast.FunctionDef))
portable=prose.replace('[[ARCHITECTURE]]','![Historical v2 architecture](data:image/png;base64,'+base64.b64encode((P/'figures/b03/architecture.png').read_bytes()).decode()+')').replace('[[WIDGET]]','**Portable interaction:** run the saved permutation panel below using your live alignment function.')
portable=re.sub(r'<details><summary>Recall the earlier conditioning visual</summary>.*?</details>','Recall L061: support changes the conditional prediction without redefining the prior.',portable,flags=re.S)
portable=portable.replace('](../','](https://avistian.github.io/relational/')
portable=re.sub(r'\]\(((?:b\d{2}|\d{4})-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',portable)
for solution in [False,True]:
 cells=[]
 def md(s):cells.append(nb.v4.new_markdown_cell(s))
 def code(s):cells.append(nb.v4.new_code_cell(s,metadata={'tags':['data-payload'] if s.startswith('payload=') else []}))
 md('# B03 · PFN and the TabPFN generations\n\nPortable offline notebook: Python + NumPy. The embedded packet contains raw author diagnostic predictions and authenticated upstream archives. This execution replays evidence; it does not run new TabPFN inference or dispatch the incomplete benchmark. Three live TODOs plus a written defense remain learner work.')
 code('import base64,hashlib,io,itertools,json,math,tempfile,zipfile\nfrom pathlib import Path\nimport numpy as np\nworkspace=Path(tempfile.mkdtemp(prefix="b03-notebook-"))')
 md(portable)
 code('payload=base64.b64decode('+repr(base64.b64encode(payload).decode())+')\nassert hashlib.sha256(payload).hexdigest()=='+repr(hashlib.sha256(payload).hexdigest())+'\nwith zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(workspace)\nevidence=workspace/"labs/evidence/b03"\nraw=json.loads((evidence/"permutation-diagnostic.json").read_text())')
 for name,instruction in [('posterior_predictive','Validate finite probabilities and weights; normalize prior × likelihood, then average query probabilities. Reject impossible support. Why does multiplying every likelihood by 10 leave the prediction fixed?'),('align_probabilities','Map each old column c to its new-label column π(c). Preserve row identities and reject non-bijections. Predict the three-cycle result before executing.'),('compare_variants','Check all seven contract fields. Missing/unknown → INCOMPLETE; any known difference → INCOMPARABLE; complete match → MATCHED_CONTRACT. Do not interpret matching as reproduced scores.')]:
  md('## '+('SOLUTION' if solution else 'TODO')+' · '+name+'\n\n'+instruction)
  code(f[name] if solution else f[name].split('\n')[0]+'\n    raise NotImplementedError("Implement '+name+'")')
 md('## CHECK · behavioral feedback on your functions')
 code(checks)
 code('print(check_functions(posterior_predictive,align_probabilities,compare_variants))')
 md('## PROVIDED · independent scalar evidence and archive audit\n\nThis code does not import your functions. It authenticates both archive hashes, checks the absent upstream loader, verifies complete permutation coverage and row identities, and recomputes diagnostic metrics. Expected report identities are frozen in the embedded packet.')
 code((P/'_audit_b03.py').read_text())
 code("report={'source':audit_sources(workspace/'labs/sources/b03'),'diagnostic':audit_predictions(raw)}\nassert report['source']==json.loads((evidence/'source-gate.json').read_text())\nassert report['diagnostic']==json.loads((evidence/'diagnostic-audit.json').read_text())\nPath('b03-report.json').write_text(json.dumps(report,indent=2)+'\\n')\nprint(json.dumps(report,indent=2))")
 md('## APPLY · your alignment controls every saved prediction\n\nThe author generated fresh predictions; you now replay them. For each map, use your function to restore class semantics and compare its errors with the independent scalar oracle. Never replace a failed map with the identity result.')
 code("baseline=np.array(raw['records'][0]['probabilities'])\ntargets=np.array(raw['labels'])[raw['test_ids']]\nfor record,oracle in zip(raw['records'],report['diagnostic']['rows']):\n    aligned=align_probabilities(record['probabilities'],record['permutation'])\n    delta=float(np.max(np.abs(aligned-baseline)))\n    loss=float(-np.log(np.clip(aligned[np.arange(len(targets)),targets],1e-15,1)).mean())\n    assert abs(delta-oracle['max_abs_delta'])<1e-12\n    assert abs(loss-oracle['log_loss'])<1e-12\n    print(record['permutation'], 'delta',round(delta,6),'log loss',round(loss,6))")
 md('## APPLY · claim boundary\n\nComplete the version/access matrix with one new primary-source observation. These examples exercise the contract, not a leaderboard.')
 code("measured=dict(version='2',variant='base',checkpoint=raw['checkpoint_sha256'],recipe='four-view-float32',data='Iris',splits='75/75-seed0',budget='fixed-no-HPO')\nclaimed=dict(measured,version='3.5',variant='Thinking')\nassert compare_variants(measured,claimed)=='INCOMPARABLE'\nprint(compare_variants(measured,claimed))\nassert posterior_predictive([.5,.5],[.25**3,.75**3],[.25,.75])>0.7")
 md('## PROVIDED · complete visible historical v2 educational model\n\nReused from L064; it implements the full 12-block checkpoint architecture and a deliberately restricted numeric wrapper. It is **not** a reimplementation of the current 3.5 model or the complete paper benchmark. B03 diagnostic predictions came from the pinned original 2.0.9 package, whose worker is shown next. Inspect the flow from grouping through both attention axes to class output.\n\n```python\n'+(P/'relkit/tabpfn_l064_v2.py').read_text()+'\n```')
 md('## PROVIDED · executable fresh diagnostic driver\n\nRun in the historical environment documented in the reproduction protocol, with the verified checkpoint. The portable default stays offline and only replays evidence. The source-gated blood benchmark is a different lane.\n\n```python\n'+(P/'_diagnostic_b03.py').read_text()+'\n```')
 md('## EXIT · submit your reasoning\n\nExplain fixed parameters versus support-dependent computation; trace the v2 model; interpret probability sensitivity with equal accuracy; give two falsification tests and a revision condition; identify source evidence needed to unblock the blood benchmark. Compare base, Plus and Thinking without inventing internal mechanisms. Revisit after 1, 7 and 30 days. Ask the agent for feedback; these automatic checks do not grade the defense.')
 code("submission={'fixed_vs_adaptive':'','model_trace':'','permutation_result':'','version_access_matrix_note':'','falsification_tests':[],'revision_condition':'','benchmark_unblocking_evidence':'','status':'PENDING_WRITTEN_DEFENSE'}\nPath('b03-submission.json').write_text(json.dumps(submission,indent=2)+'\\n')")
 for i,c in enumerate(cells):c.id=hashlib.sha256((str(i)+c.source).encode()).hexdigest()[:12]
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
 nb.write(book,(P/'solutions' if solution else P)/(S+'.ipynb'))
print('Built B03 lesson, field guide and portable notebooks')
