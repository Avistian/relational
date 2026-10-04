"""Build B18a lesson, reference and portable notebooks from measured evidence."""
import ast,base64,gzip,io,json,re,zipfile,tarfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;R=P.parent;S='b18a-context-state';F=P/'figures/b18a';E=P/'evidence/b18a'
r=json.loads((E/'summary.json').read_text())

def fig(name,caption,portable=False):
    src='data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/b18a/'+name+'.png'
    return f'<figure class="state-figure" tabindex="0" role="region" aria-label="{caption}"><img src="{src}" alt="{caption}"><figcaption>{caption} <span class="state-scroll-hint">On small screens, swipe the figure or focus it and use arrow keys.</span></figcaption></figure>'

rows=r['records'];fig1,axs=plt.subplots(1,2,figsize=(9,4.5));labels=['POT full','POT selected','TACO 4%'];arms=['POT-full','POT-selected','TACO4']
for i,(mode,color,label) in enumerate([('fit_preprocessors','#b58029','Predictor KV off'),('fit_with_cache','#28755a','Predictor KV on')]):
    for j,arm in enumerate(arms):
        rr=[q for q in rows if q['mode']==mode and q['arm']==arm]
        vals=[q['repeat_pass_seconds'] for q in rr];aucs=[q['auc'] for q in rr]
        for ax,v in zip(axs,[vals,aucs]):
            ax.scatter(np.full(3,j)+(i-.5)*.18,v,color=color,label=label if j==0 else None,s=30)
for ax in axs:ax.set_xticks(range(3),labels,rotation=12);ax.grid(axis='y',alpha=.2)
axs[0].set_ylabel('Repeated 285-query pass (seconds)');axs[0].set_ylim(bottom=0);axs[0].legend(fontsize=8)
axs[1].set_ylabel('AUROC on fixed 285 query rows');axs[1].set_ylim(.90,1)
fig1.suptitle('One dataset, one split, three inference seeds',weight='bold');fig1.tight_layout();fig1.savefig(F/'quality-cost.png',dpi=150);plt.close(fig1)

table='| Arm | KV | AUROC mean | Log loss mean | Fit s mean | Repeated pass s mean |\n|---|---|---:|---:|---:|---:|\n'
for arm in arms:
 for mode in ['fit_preprocessors','fit_with_cache']:
  rr=[q for q in rows if q['mode']==mode and q['arm']==arm]
  table+='| '+arm+' | '+('on' if mode=='fit_with_cache' else 'off')+' | '+' | '.join(f'{np.mean([q[k] for q in rr]):.4f}' for k in ['auc','log_loss','fit_seconds','repeat_pass_seconds'])+' |\n'
fails=[q for q in r['cache_agreement'] if q['status']=='FAIL']
table+=f"\nAll 18 fits completed. Cached/uncached agreement at the predeclared1e-5 tolerance: **{9-len(fails)}/9 pairs pass**. "
if fails:table+='Failed pairs: '+', '.join(q['arm']+' seed'+str(q['seed'])+' Δ='+format(q['max_probability_difference'],'.6g') for q in fails)+'. Keep this failure visible; these modes are not interchangeable to the declared tolerance.'
else:table+='Largest probability difference: '+format(max(q['max_probability_difference'] for q in r['cache_agreement']),'.3g')+'.'
table+=' Three repeated query passes measure warm reuse; they are not independent predictive trials.\n\n'+fig('quality-cost','Measured runtime and AUROC for all three inference seeds; no paper speedup claim.')
updates='| Model | State edit | Maximum probability change | Rebuild s |\n|---|---|---:|---:|\n'
for q in r['updates']:
 if q['intervention']!='baseline':updates+=f"| {q['arm']} | {q['intervention']} | {q['max_probability_difference']:.6f} | {q['fit_seconds']:.3f} |\n"
updates+='\nQuery missingness has no support rebuild; zero rebuild seconds means reuse, not a free prediction.'
widget='''<section class="state-panel" data-context-state><h3>Predict first: which artifacts need work?</h3><label for="state-edit">Change one dependency</label><select id="state-edit"><option value="none">No change</option><option value="support">Support labels or rows</option><option value="preprocess">Fitted preprocessing</option><option value="query">Query features only</option><option value="background">Explanation background only</option></select><output aria-live="polite">Support edits require rebuilding dependent caches; query-only edits preserve support caches; background-only edits recompute explanations.</output><button type="button">Reset</button><p class="state-note">Validity follows identity, even when the predicted number is unchanged.</p><noscript><p>Read the dependency rules below; the notebook executes the corresponding checks.</p></noscript></section>'''
boundaries='''<div class="state-evidence"><strong>Executed:</strong> 18 release-checkpoint fits, 12 state/query conditions and four explanation contrasts. Saved predictions independently rescored. <strong>Paper:</strong> historical Figure3 source-gated; pretraining NOT_RUN. <strong>Learner:</strong> PENDING_WRITTEN_DEFENSE. See each cache-agreement verdict above.</div>'''
body=(R/'lessons/content'/f'{S}.md').read_text()
for token,value in [('ARCHITECTURE',fig('architecture','TACO jointly learns a compressor and predictor; serving reuses derived context while its dependencies remain valid.')),('STATE_WIDGET',widget),('RESULTS',table),('UPDATES',updates),('BOUNDARIES',boundaries)]:body=body.replace('{{'+token+'}}',value)
body=body.replace('## 6 · Explain',fig('explanation-inputs','Support defines the predictor; query defines the case; background defines the comparison.')+'\n\n## 6 · Explain')

def document(title,text,interactive=False):
 h=render(text).replace('<table>','<div class="state-table" tabindex="0" role="region" aria-label="Scrollable evidence table"><table>').replace('</table>','</table></div>')
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/context-state.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../notebooks.html">Labs</a></nav>'+h+'</article>'+('<script src="../assets/context-state.js"></script>' if interactive else '')+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document('B18a · Context as deployed model state',body,True))
reference='''# Context-state contract · B18a

[Lesson](../lessons/b18a-context-state.html) · [Student lab](../labs/b18a-context-state.ipynb) · [Experiment contract](../labs/b18a-reproduction.md)

## Identify one prediction

Checkpoint SHA256 + fitted preprocessing + ordered support IDs/features/labels + retrieval policy + compression setting + inference seeds/transforms + dtype + implementation + query. Query memoization also needs query identity. An explanation adds the method/estimand, feature set and background identity.

## State dependencies

| Edit | Support cache | Prediction | Explanation | Distilled student |
|---|---|---|---|---|
| Support/preprocessing | Rebuild | Recompute | Recompute | Revalidate/retrain |
| Query only | Reuse | Recompute | Recompute | Reuse |
| Background only | Reuse | Reuse | Recompute | Reuse |

## Four mechanisms

KV caching retains attention state. Selection retains real rows. TACO jointly learns a compressor and predictor and retains latent context. Distillation learns a separate predictor from teacher outputs. The current TACO release already caches compressed context in fit_preprocessors; fit_with_cache additionally caches predictor attention.

## Chronology

Labeled support requires event≤cutoff AND arrival≤cutoff AND label_available≤cutoff. Student teacher targets must exclude the row's own label; chronological tasks require earlier-available teacher context. Hashing proves identity, not timestamp truth.

## Cost and explanation

Total = setup + all query calls + all rebuilds + all explanation calls. Break-even requires positive warm savings and enough requests before the next update. GPU peak allocation is not isolated KV storage.

Replacement contrast = f(x) − average_b f(x with x_j=b_j). It is not SHAP or a causal effect. Foreground16 × background8 requires16+128=144 prediction rows. Changing the background can change the contrast without changing the predictor.

## Evidence

'''+table+'\n'+boundaries+'''

Primary readings: [TACO §3–4](https://arxiv.org/html/2602.05649v2), [TabPFN-2.5 distillation](https://arxiv.org/html/2511.08667v2), [TabPFN interpretation](https://arxiv.org/html/2403.10923v2). Ask your teacher to review your state/update contract.
'''
(R/'reference'/f'{S}.html').write_text(document('B18a reference',reference))
# Portable evidence and full source/operator packet. No dependency on a repository checkout.
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for file in [P/'_experiment_b18a.py',P/'_cloud_b18a.py',P/'_reproduce_b18a.py',P/'b18a-reproduction.md',P/'relkit/state_b18a.py']:
  z.write(file,str(file.relative_to(P)))
 for folder in [P/'sources/b18a',E]:
  for file in folder.iterdir():
   if file.is_file() and file.name not in ['local-budget.json']:z.write(file,str(file.relative_to(P)))
packet=base64.b64encode(buf.getvalue()).decode()
source=(P/'relkit/state_b18a.py').read_text();tree=ast.parse(source);parts={node.name:ast.get_source_segment(source,node) for node in tree.body if isinstance(node,ast.FunctionDef)}
for solution in [False,True]:
 cells=[]
 def md(t):cells.append(nb.v4.new_markdown_cell(t))
 def py(t):cells.append(nb.v4.new_code_cell(t))
 md('# B18a · Context as deployed model state\n\nPortable '+('solution' if solution else 'student lab')+'. The default path reconstructs saved evidence and executes your own state/explanation functions. Fresh inference is a separate opt-in GPU cell. No paper-result reproduction or learner mastery is inferred. Allow 45–60 minutes.\n\n**Prerequisites:** NumPy arrays, support/query split, probability versus label. B18 showed why missing context matters; here we track how a changed context changes a deployed predictor.\n\n**Recall:** When is a label available? Why can fixed weights produce changed predictions?')
 md(fig('architecture','TACO compressor, latent context and predictor, with explicit query path.',True))
 md('## Setup\n\nPython3.10–3.12, NumPy and scikit-learn are needed for replay. If absent, install `numpy==2.2.6 scikit-learn==1.6.1`. Fresh inference uses the separately pinned GPU environment. Figures and the source/evidence packet are embedded; no repository or filesystem path is needed.')
 py('import base64,hashlib,io,json,zipfile\nfrom pathlib import Path\nimport numpy as np\nfrom sklearn.metrics import roc_auc_score,log_loss\npacket=Path("b18a-packet");packet.mkdir(exist_ok=True)\nwith zipfile.ZipFile(io.BytesIO(base64.b64decode('+repr(packet)+'))) as z:\n    z.extractall(packet)\nparts=[json.loads((packet/"evidence/b18a"/(p+".json")).read_text()) for p in ["pilot","matrix","updates"]]\ndata=parts[0]["data"]\nX=np.asarray(data["x"]);y=np.asarray(data["y"]);train=np.asarray(data["train"]);test=np.asarray(data["test"])\nassert not set(train)&set(test)\nprint(len(train), "support;",len(test),"query rows. Saved inference; no new model fitted.")')
 for name,description,check in [
 ('eligible_support','TODO1 · A row needs event, arrival AND label availability by the cutoff. Reject duplicate IDs.',"rows=[dict(id=1,event=1,available=2,label_available=5),dict(id=2,event=1,available=6,label_available=2)]\nassert [r['id'] for r in eligible_support(rows,5)]==[1]\ntry: eligible_support(rows+[rows[0]],9)\nexcept ValueError: pass\nelse: raise AssertionError('Duplicate IDs accepted')"),
 ('state_key','TODO2 · Hash ordered IDs, features including explicit NaN positions, labels, weights, fitted preprocessing identity and inference recipe. Return a deterministic hexadecimal SHA256.',"args=dict(ids=[1,2],x=np.array([[1.,np.nan],[2.,3.]]),y=[0,1],weights='abc',preprocessing={'scale':1},recipe={'seed':0})\na=state_key(**args)\nassert len(a)==64 and a==state_key(**args)\nfor key,value in [('y',[1,1]),('ids',[2,1]),('weights','def'),('preprocessing',{'scale':2}),('recipe',{'seed':1}),('x',np.nan_to_num(args['x']))]:\n    assert a!=state_key(**dict(args,**{key:value})),key"),
 ('replacement_contrast','TODO3 · Implement f(x)−mean_b f(x with feature j replaced by background b_j). Preserve other query features; one result per query.',"calls=[]\ndef oracle(q):\n    calls.append(len(q));return q[:,0]+2*q[:,1]\neffect=replacement_contrast(oracle,np.array([[4.,2.],[6.,1.]]),np.array([[1.,8.],[3.,9.]]),0)\nnp.testing.assert_allclose(effect,[2,4]);assert sum(calls)==6")]:
  md('## '+description)
  # Extract canonical function independently of the evidence variable in generated notebook.
  function=ast.get_source_segment(source,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))
  py(function if solution else function.split('\n',1)[0]+'\n    raise NotImplementedError("Implement '+name+'")')
  py(check+'\nprint("CHECK passed: '+name+'")')
 md('## Replay all 18 fits\n\nUse every seed/arm/cache condition. Do not select the best test score. Three seeds on one split do not create three independent datasets.')
 py("records=parts[0]['records']+parts[1]['records']\nassert len(records)==18\nfor row in records:\n    p=np.asarray(row['predictions']);assert p.shape==(4,285)\n    auc=roc_auc_score(y[test],p[0]);loss=log_loss(y[test],p[0])\n    assert abs(auc-row['auc'])<1e-12 and abs(loss-row['log_loss'])<1e-12\n    print(row['seed'],row['arm'],row['mode'],round(auc,6),round(loss,6))\nfor seed in [0,1,2]:\n    for arm in ['POT-full','POT-selected','TACO4']:\n        pair=[r for r in records if r['seed']==seed and r['arm']==arm]\n        delta=np.max(np.abs(np.asarray(pair[0]['predictions'])-np.asarray(pair[1]['predictions'])))\n        print('cache agreement',seed,arm,delta,'PASS' if delta<=1e-5 else 'FAIL')")
 md(fig('quality-cost','Author measurements: quality and repeated inference cost on one frozen split.',True))
 md('## Use your identity function on real support updates\n\nA change in identity requires checking dependent artifacts even if probabilities happen to be equal. Query missingness alone does not change support identity.')
 py("base=train[1:];ids=base.tolist()\nargs=dict(ids=ids,x=X[base],y=y[base],weights='fixed-checkpoint',preprocessing={'version':1},recipe={'seed':0})\noriginal=state_key(**args)\nchanged=y[base].copy();changed[0]=1-changed[0]\nassert original!=state_key(**dict(args,y=changed))\nassert original!=state_key(**dict(args,ids=ids[1:],x=X[base[1:]],y=y[base[1:]]))\nassert original==state_key(**args)\nfor row in parts[2]['updates']:\n    baseline=next(q for q in parts[2]['updates'] if q['arm']==row['arm'] and q['intervention']=='baseline')\n    delta=np.max(np.abs(np.asarray(row['predictions'])-baseline['predictions']))\n    print(row['arm'],row['intervention'],round(float(delta),6))")
 md(fig('explanation-inputs','Predictive support, query and explanation background have separate identities.',True))
 md('## Replay explanations using your replacement function\n\nThe lookup accepts only the exact saved query arrays; wrong feature replacement cannot accidentally return the right answer. This is a descriptive contrast, not SHAP or causality.')
 py("for explanation in parts[2]['explanations']:\n    def saved_predict(q):\n        for call in explanation['calls']:\n            if np.array_equal(q,np.asarray(call['x'])):return np.asarray(call['p'])\n        raise AssertionError('Requested unsaved/wrong intervention rows')\n    live=replacement_contrast(saved_predict,X[explanation['query_ids']],X[explanation['background_ids']],0)\n    np.testing.assert_allclose(live,explanation['effects'],atol=1e-12,rtol=0)\n    print(explanation['arm'],explanation['background'],float(live.mean()),'144 prediction rows')")
 md('## Visible full fresh-inference operator\n\nThe archived release provides the pretrained architecture; this complete operator defines all data selection, seeds, modes, timers, interventions and saved probabilities. It trains no new weights. Loading weights downloads two authenticated public checkpoint files. The source packet includes upstream model code/licenses, the exact Modal environment, and budget guard.')
 with tarfile.open(P/'sources/b18a/taco.tar.gz') as archive:
  member=next(n for n in archive.getmembers() if n.name.endswith('/src/taco/model/taco_model.py'))
  upstream=archive.extractfile(member).read().decode()
 model=next(n for n in ast.parse(upstream).body if isinstance(n,ast.ClassDef) and n.name=='TACO')
 excerpts='\n\n'.join(ast.get_source_segment(upstream,n) for n in model.body if isinstance(n,ast.FunctionDef) and n.name in ['_sample_keep_count','_compress_latents','_predict_with_compressor'])
 md('### Load-bearing released compressor code\n\nExact excerpts from the pinned source. Follow K selection → dummy features → emitted latent cells → residual projection → cached context → predictor. The complete cell-attention implementation is in the embedded source tarball. These are source excerpts, not newly trained course weights.\n\n```python\n'+excerpts+'\n```')
 runner=(P/'_experiment_b18a.py').read_text()
 md('```python\n'+runner+'\n```')
 md('## Optional fresh GPU run\n\nDefault false: replay does not trigger downloads or paid execution. On a compatible GPU runtime, install the packet tarball and pinned dependencies listed in `_cloud_b18a.py`; run the complete three phases below. This performs fresh inference and may take minutes. Local execution here has no Modal billing guard; use the provided budgeted cloud operator for paid execution. Preserve output files and count all attempts. No reduced dataset or fewer ensemble members is substituted.')
 py("RUN_FRESH=False\nif RUN_FRESH:\n    import sys,shutil\n    sys.path.insert(0,str(packet.resolve()))\n    shutil.copy(packet/'relkit/state_b18a.py',packet/'state_b18a.py')\n    from _experiment_b18a import run\n    for phase in ['pilot','matrix','updates']:\n        result=run(phase)\n        Path('fresh-'+phase+'.json').write_text(json.dumps(result))\n        assert result['status']=='COMPLETE',result.get('error')\nelse:\n    print('Fresh notebook inference NOT_RUN. Author GPU evidence is separately archived.')")
 md('## Historical Figure3 gate\n\nAudit success means source-byte identity only. Missing original generator, seeds, checkpoint mapping and timing identity prevent historical reproduction. Full pretraining and TabArena paper evaluations remain NOT_RUN.')
 py("import subprocess,sys\naudit=subprocess.run([sys.executable,str(packet/'_reproduce_b18a.py'),'--phase','audit'],capture_output=True,text=True,check=True)\nprint(audit.stdout)\nblocked=subprocess.run([sys.executable,str(packet/'_reproduce_b18a.py'),'--phase','paper'],capture_output=True,text=True)\nassert blocked.returncode!=0 and 'INCOMPLETE_SOURCE_PROTOCOL_GATE' in blocked.stderr\nprint(blocked.stderr)")
 md('## EXIT · written defense\n\nIn 150–200 words, identify the deployed predictor, trace a late label correction, explain why query-only and background-only changes differ, interpret one observed cache agreement result, and specify an update-frequency/quality/cost falsification test for B23/B24.\n\nTeacher targets for a distilled student must exclude each row’s own label and respect availability time; no student is trained here. Ask your teacher for feedback. Revisit tomorrow, in 7 days and 30 days. Learner status remains PENDING_WRITTEN_DEFENSE.\n\nPrimary sources: [TACO](https://arxiv.org/html/2602.05649v2), [TabPFN2.5](https://arxiv.org/html/2511.08667v2), [Interpretation](https://arxiv.org/html/2403.10923v2).')
 n=nb.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python3',language='python'),language_info=dict(name='python',version='3.12')))
 nb.write(n,(P/'solutions'/f'{S}.ipynb') if solution else P/f'{S}.ipynb')
print('Built lesson/reference, portable notebooks, measured plot')
