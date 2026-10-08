"""Deterministic connected lesson, reference and portable notebook package."""
import ast,base64,hashlib,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l182';S='0182-rdb-pfn-composite-message-passing'
r=json.loads((E/'report.json').read_text())
def definitions(path):
 source=path.read_text();return {n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
fns=definitions(P/'relkit/composite_l182.py');model=definitions(P/'relkit/rdbpfn_l166.py')
status='**Observed:** fresh complete selected reproduction: **30 evaluations, 21,060 predictions**, all three means round to their published targets. New hybrid training **NOT_RUN**; whole-paper reproduction **NOT_RUN**. Learner **PENDING_WRITTEN_DEFENSE**.'
rows=['| Released model | Fresh mean AUROC ± sample SD | Paper target |','|---|---:|---:|']
for arm,v in r['models'].items():rows.append(f"| {arm} | {v['mean']:.6f} ± {v['sample_sd']:.6f} | {v['paper']:.4f} |")
results='\n'.join(rows)+'\n\nAll three pass the predeclared descriptive mean-distance tolerance 0.02. This tolerance is not a statistical equivalence test.'
captions={'architecture':'Three complete paths: published synthetic pretraining, frozen released-checkpoint inference, and a proposed graph encoder plus ICL head. The two insertion points require different new training experiments.','trace':'Synthetic one-head composite calculation with identity projections and zero destination skip. A future purchase is excluded before fusion; the remaining output is [1.6698, 1.0000].','results':'Fresh full selected experiment. Ten support draws per model share one test population. Points show each draw; diamonds and bars show mean and sample SD, not confidence intervals.','factorial':'Fabricated four-arm AUROC example. D is best but the paired interaction is negative. No proposed hybrid arm was trained.'}
def prose(portable=False):
 s=(R/'lessons/content'/(S+'.md')).read_text().replace('[[STATUS]]',status).replace('[[RESULTS]]',results)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/'figures/l182'/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l182/'+name+'.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 slots={'WARMUP':('Recall: support labels versus query labels; foreign keys; validation-only selection. What did L181 leave incomplete?','<div id="warmup"></div>'),'PREDICT':('Predict before running: does duplicating only one legal message preserve softmax attention? Write your reason.','<div id="predict"></div>'),'EXPLORER':('In the live notebook below, compare baseline, permuted, future-value-changed, duplicated and merged-route states. Predict the unchanged states before executing.','<div id="composite-route"></div><noscript>Baseline [1.6698,1.0000]. Permuting or changing excluded future rows leaves it unchanged. Duplicating the first message gives [1.5035,1.0000]. Merging [8,0] from another route gives [7.8670,0.0210].</noscript>'),'TEACHBACK':('Write your hypothesis and falsification criteria in the EXIT ticket below, then ask the teaching agent to review them.','<div id="teachback"></div>')}
 for key,(plain,html) in slots.items():s=s.replace('[['+key+']]',plain if portable else html)
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def doc(title,body,interactive=False):
 html=re.sub(r'<table([^>]*)>',r'<div class="route-scroll" tabindex="0"><table\1>',render(body)).replace('</table>','</table></div>')
 css=['lesson','atomic-route','checkpoint','lab-access','composite-route'];scripts=['retrieval-pool','retrieval-bank','predict','teachback','composite-route-viz','l182-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 182 — '+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in css)+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0181-relbench-v2-autocomplete.html">Lesson 181</a></nav><header><p class="route-kicker">Year 5 · Research frontier · Lesson 182</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('RDB-PFN + composite message passing',prose(),True))
reference='''**Research question:** does a relational prior benefit more from composite graph messages than from conventional two-hop messages under matched legal context and training resources?

**Choose an insertion point.** Generator-side: change synthetic task dependencies, keep predictor fixed, retrain. Predictor-side: keep task stream fixed, change graph encoder, retrain with a common ICL head. Declare cross-schema parameter sharing. A graph embedding is not automatically compatible with a checkpoint trained on DFS features.

**Composite route:** resolve source→bridge→destination foreign keys; filter each message by its receiver cutoff; fuse projected source and bridge features; normalize attention within that route and receiver; aggregate. Multiple heads, skip connections and final projections belong to the full RelGNN. Our tested specialization uses one head and identity projections.

**Worked state:** fused messages [1,1] and [2,1], query [1,0], logits 0.7071/1.4142, weights 0.3302/0.6698, output [1.6698,1]. Row permutation and excluded future values preserve it; duplicating one message does not.

**Factorial contrast:** A=single prior/plain encoder; B=single prior/composite; C=relational prior/plain; D=relational prior/composite. Compute (D−C)−(B−A) on paired observations. D>A does not prove positive interaction. Fabricated .60/.65/.64/.66 gives −.03. All four proposed models remain NOT_RUN.

**Falsify:** predeclare a useful interaction, hold out schemas/databases and compare task-level uncertainty; check whether a two-hop, matched-context and matched-compute control closes the gap. An inconclusive result is not proof of no effect.

**Published evidence:** fixed released RDB-PFN/Table9/F1/512support, three models, ten support seeds, all702queries. Original evaluator, no checkpoint selection on test. Label orientation is the complement of current raw DNF. Full DFS reconstruction and historical feature availability remain unestablished. Ten support draws are one task, not ten databases.

'''+results+'\n\n'+status+'''

[Lesson](../lessons/0182-rdb-pfn-composite-message-passing.html) · [Lab](../labs/0182-rdb-pfn-composite-message-passing.ipynb) · [Protocol](../labs/l182-reproduction.md) · [Report](../labs/evidence/l182/report.json) · [RDB-PFN](https://arxiv.org/html/2603.03805v5) · [RelGNN](https://arxiv.org/html/2502.06784v2).
'''
(R/'reference/rdb-pfn-composite-message-passing.html').write_text(doc('Composite-message hypothesis — quick reference',reference))
(E/'report.md').write_text('# L182 fresh selected evidence\n\n'+status+'\n\n'+results+'\n')
# Self-contained evidence and inference inputs; large TabICL weights remain a hash-checked opt-in download.
files={'predictions.json':E/'predictions.json','input-manifest.json':E/'input-manifest.json','prepared.npz':P/'evidence/l166/prepared.npz','RDBPFN.pt':P/'evidence/l166/checkpoints/RDBPFN.pt','pfn-parity.json':E/'pfn-parity.json','_run_l166.py':P/'_run_l166.py'}
for path in sorted((P/'sources/l166/upstream/model_pretrain').rglob('*.py')):
 if '__pycache__' not in str(path):files['source/model_pretrain/'+str(path.relative_to(P/'sources/l166/upstream/model_pretrain'))]=path
buf=io.BytesIO();hashes={}
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for name,path in sorted(files.items()):
  content=path.read_bytes();hashes[name]=hashlib.sha256(content).hexdigest();info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,content)
encoded=base64.b64encode(buf.getvalue()).decode();digest=hashlib.sha256(buf.getvalue()).hexdigest()
(E/'notebook-packet-manifest.json').write_text(json.dumps(dict(zip_sha256=digest,files=hashes),indent=2)+'\n')
def make(solution):
 cells=[]
 def md(s):cells.append(nb.v4.new_markdown_cell(s))
 def code(s,tags=None):cells.append(nb.v4.new_code_cell(s,metadata={'tags':tags or []}))
 md('# Lesson 182 · RDB-PFN + composite message passing\n\nMirror scope: complete selected published-checkpoint evaluation and the RelGNN one-head composite equation. Tier B: all fresh F1 predictions and hashed inputs. Tier C: controlled route arithmetic and fabricated factorial examples. Default notebook execution replays evidence and runs the visible model on a small input; the author ran all30fresh evaluations separately. New hybrid pretraining is NOT_RUN.\n\nRequires Python3, NumPy and PyTorch. Images, evidence, source and one real checkpoint are embedded; default execution needs no repository clone or network.')
 code('# @colab-bootstrap\nimport os\nos.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")\nfrom pathlib import Path\nimport json,copy,hashlib,base64,io,zipfile\nimport numpy as np\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F\ntorch.set_num_threads(1)\nprint("Default: complete saved-evidence rescore + live mechanism. Fresh inference is an opt-in after EXIT.")',['colab-bootstrap'])
 md(prose(True))
 md('## PROVIDED · Authenticate the portable packet\nThe payload contains all30fresh prediction files as records, the prepared real dataset, one checkpoint, and pinned model/evaluation source. Authentication precedes scoring. The large string is data, not hidden model logic.')
 code('PACKET='+repr(encoded)+'\nraw=base64.b64decode(PACKET)\nassert hashlib.sha256(raw).hexdigest()=='+repr(digest)+'\nP=Path("l182-packet");P.mkdir(exist_ok=True)\nwith zipfile.ZipFile(io.BytesIO(raw)) as z:\n    for name in z.namelist():\n        assert not Path(name).is_absolute() and ".." not in Path(name).parts\n    z.extractall(P)\nhashes='+repr(hashes)+'\nfor name,h in hashes.items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h,name\npacket=json.loads((P/"predictions.json").read_text())\nprint(len(packet),"authenticated fresh author evaluation records")',['data-payload'])
 tasks=[('legal_fusion','Resolve and filter the route','Return fused messages and destination indices in retained row order. Each bridge row has one source index and one destination index. Keep time strictly less than that receiver cutoff; apply source and bridge matrices. Reject invalid FK indices, nonfinite values and incompatible shapes.'),('keyed_auc','Score the complete population','Join probabilities to labels using both entity and cutoff. Reject duplicate/missing keys, unknown identities, invalid probabilities and single-class truth. Compute positive-negative pair wins with half credit for ties.'),('factorial_interaction','Test complementarity','Input rows are paired seeds; columns A/B/C/D follow the four-arm table. Return one interaction per row. Require at least two finite AUROC rows and exactly four arms. D minus A is not the required contrast.')]
 immediate={'legal_fusion':"z,d=legal_fusion(np.array([[1.,0.]]),np.array([[0.,1.],[9.,9.]]),np.array([0,0]),np.array([0,0]),np.array([9.,11.]),np.array([10.]),np.eye(2),np.eye(2))\nnp.testing.assert_allclose(z,[[1,1]]);np.testing.assert_array_equal(d,[0])",'keyed_auc':"k=np.array([[1,10],[1,20],[2,20],[3,20]])\ny=np.array([0,1,0,1]);p=np.array([.2,.8,.8,.9]);order=[3,1,0,2]\nassert keyed_auc(k,y,k[order],p[order])==.875",'factorial_interaction':"illustration=np.array([[.60,.65,.64,.66],[.61,.66,.65,.67]])\nnp.testing.assert_allclose(factorial_interaction(illustration),[-.03,-.03])"}
 for name,title,instructions in tasks:
  md('## TODO · '+title+'\n'+instructions)
  node=ast.parse(fns[name]).body[0];code(fns[name] if solution else 'def '+name+'('+ast.unparse(node.args)+'):\n    raise NotImplementedError("Implement '+name+'")')
  md('### CHECK · Immediate feedback\nThis small case is necessary; the adversarial and complete-data checks below are stronger.');code(immediate[name]+'\nprint("Immediate CHECK passed")')
 md('## PROVIDED · Attention arithmetic\nThe load-bearing operation is visible here: dot products, stable softmax, then the weighted sum. The legal-fusion output supplied by your function determines its inputs.')
 code(fns['attend'])
 md('## CHECK · Adversarial contracts\nThese checks exercise duplicate entity IDs at distinct times, invalid foreign keys, receiver-specific cutoffs, probability range, ties and incomplete factorial arms.')
 code(definitions(P/'_check_l182.py')['checks']);code("assert checks(legal_fusion,keyed_auc,factorial_interaction)=='PASS'\nprint('Three live contracts PASS')")
 md('## TRY · Change one assumption\nPredict which outputs must remain unchanged. The experiment calls your fusion and interaction functions. No trained-model metric is measured here.')
 code(definitions(P/'_mechanism_l182.py')['mechanism182']);code("mechanism=mechanism182(legal_fusion,attend,factorial_interaction)\nfor name,state in mechanism['states'].items():print(name, np.round(state['output'],6))\nprint('Fabricated interaction:',mechanism['illustrative_interaction'])")
 md('## PROVIDED + CHECK · Reproduce the full fresh evidence table\nYour keyed AUROC is called for all30records. The harness requires every model/seed pair, complete702query populations and paired512row supports. This cell rescores fresh author inference; it does not run inference again.')
 code(definitions(P/'_report_l182.py')['summarize182']);code("report=summarize182(packet,keyed_auc)\nPath('l182-report.json').write_text(json.dumps(report,indent=2))\nprint(report['status'], report['predictions'],'predictions')\nfor arm,v in report['models'].items():print(arm, 'mean',round(v['mean'],6),'SD',round(v['sample_sd'],6),v['status'])")
 md('## CHECK · Independent rank-based AUROC\nYour pairwise definition and this sorting/search definition take different computational routes. Check every prediction against the prepared query and support keys, then compare all30scores.')
 code("prepared=np.load(P/'prepared.npz')\nfor row in packet:\n    np.testing.assert_array_equal(row['keys'],prepared['test_keys'])\n    np.testing.assert_array_equal(row['label'],prepared['y_test'])\n    np.testing.assert_array_equal(row['support_keys'],prepared['train_keys'][prepared['support'][row['seed']]])\n    y=np.array(row['label']);s=np.array(row['probability']);neg=np.sort(s[y==0]);pos=s[y==1]\n    lo=np.searchsorted(neg,pos,side='left');hi=np.searchsorted(neg,pos,side='right')\n    oracle=float(np.mean((lo+.5*(hi-lo))/len(neg)))\n    assert abs(oracle-keyed_auc(row['keys'],y,row['keys'],s))<1e-12\nprint('21,060 probabilities independently rescored; full query/support identities PASS')")
 md('## PROVIDED · Visible RDB-PFN implementation\nThese source-identical course blocks expose the full base numeric forward pass. Width96,sixblocks,fourheads. Library MultiheadAttention performs attention; support restriction, token layout, normalization, residuals and decoder are visible. This is a separate model from the proposed graph encoder.')
 for names,explain in [(['normalize_support','FeatureEncoder','TargetEncoder'],'Support-only moments and label padding; query labels never enter.'),(['BiAttention'],'Feature attention is followed by support-row attention and the feed-forward residual.'),(['Decoder','RDBPFN'],'The decoder consumes each query label token after six blocks.')]:
  md('### PROVIDED · '+explain);code('\n\n'.join(model[name] for name in names))
 md('## CHECK · A real checkpoint, a small forward pass\nLoad the authentic released RDB-PFN checkpoint into the visible class. Compare its logits with an independently executed original model on the fixed7row,3feature input in float64. The fresh full evaluator uses float32; precision is explicit for each lane. This is a mechanism check, not the full702query evaluation.')
 code("weights=torch.load(P/'RDBPFN.pt',map_location='cpu',weights_only=False)\nprint('Checkpoint container:',type(weights).__name__)\nparity=json.loads((P/'pfn-parity.json').read_text())['rows'][0]\nnet=RDBPFN().double().eval()\n# Released checkpoint is a state_dict or carries one under model_state_dict.\nstate=weights.get('model_state_dict',weights)\nnet.load_state_dict(state,strict=True)\nx=torch.tensor(parity['input'],dtype=torch.float64);y=torch.tensor(parity['support_labels'],dtype=torch.float64)\nwith torch.no_grad():logits=net((x,y),4)\nnp.testing.assert_allclose(logits.numpy(),parity['reference_logits'],atol=1e-9,rtol=1e-9)\nprint('Visible checkpoint forward PASS; parameters:',sum(p.numel() for p in net.parameters()))")
 md('## CHECK · Reject incomplete evidence and wrong learner work\nThe result must not become COMPLETE when a seed is absent. The checker also rejects three intentionally incorrect implementations.')
 code("for bad in [packet[:-1],packet+[packet[0]]]:\n    try:summarize182(bad,keyed_auc)\n    except ValueError:pass\n    else:raise AssertionError('Bad grid accepted')\nwrong=[(lambda *a:(np.zeros((2,2)),np.array([0,1])),keyed_auc,factorial_interaction),(legal_fusion,lambda *a:.5,factorial_interaction),(legal_fusion,keyed_auc,lambda a:a[:,3]-a[:,0])]\nfor fs in wrong:\n    try:checks(*fs)\n    except (ValueError,AssertionError):pass\n    else:raise AssertionError('Wrong learner work accepted')\nprint('Incomplete grids and three wrong implementations rejected')")
 md('## EXIT · Defend a hypothesis that could fail\nWrite your answer without copying the model defense. Explain why the published three-arm result cannot fill the proposed four-arm table. Ask the teaching agent to review your argument; code success alone does not award mastery.')
 code("submission={'learner':'PENDING_WRITTEN_DEFENSE','chosen_insertion_point':'','changed_operator':'','fixed_information_and_compute':'','cross_schema_weight_sharing':'','retraining_required':'','published_evidence_limit':'','falsification_one':'','falsification_two':''}\nPath('l182-submission.json').write_text(json.dumps(submission,indent=2))")
 md('## NEXT STEP · Fresh complete checkpoint evaluation\nDefault OFF. Run in a clean Python3.11 environment with Torch2.5.1 and the exact package versions below, preferably a GPU. This fetches two missing checkpoints by pinned URL and verifies every input hash, then calls the exact original evaluator for all30runs. The embedded source and prepared dataset need no course checkout. It costs compute on your selected runtime; no Modal dispatch occurs. A new directory prevents overwriting results. This path does not pretrain a hybrid. Live Colab remains NOT_CHECKED.')
 code("RUN_PAPER_REPRO=False\nif RUN_PAPER_REPRO:\n    import sys,subprocess,urllib.request,shutil\n    from importlib.metadata import version\n    expected={'torch':'2.5.1','numpy':'1.26.4','pandas':'2.2.3','scikit-learn':'1.6.1','pydantic':'1.10.26','tabicl':'0.1.3','PyYAML':'6.0.2'}\n    for name,want in expected.items():\n        assert version(name).split('+')[0]==want,(name,'Use the frozen reproduction environment',want)\n    manifest=json.loads((P/'input-manifest.json').read_text())\n    urls={'RDBPFN_single.pt':'https://raw.githubusercontent.com/MuLabPKU/RDBPFN/'+manifest['code_revision']+'/model_pretrain/checkpoints/RDBPFN_single/model_eval00360.pt','tabicl-classifier-v1.1-0506.ckpt':'https://huggingface.co/jingang/TabICL-clf/resolve/'+manifest['tabicl_revision']+'/tabicl-classifier-v1.1-0506.ckpt'}\n    for name,entry in manifest['files'].items():\n        dest=P/name\n        if not dest.exists():\n            with urllib.request.urlopen(urls[name],timeout=120) as response,dest.open('wb') as f:shutil.copyfileobj(response,f)\n        assert hashlib.sha256(dest.read_bytes()).hexdigest()==entry['sha256'],name\n    destination=Path('l182-fresh-evaluation')\n    assert not destination.exists(),'Choose a new immutable output directory'\n    subprocess.run([sys.executable,str(P/'_run_l166.py'),'--input',str(P),'--source',str(P/'source'),'--out',str(destination)],check=True)\nelse:\n    print('Fresh30run evaluation OFF. Author fresh experiment COMPLETE; hybrid training NOT_RUN.')")
 md('## Appendix · Original implementation and evaluator\nPinned original source for inspection, not executed by these Markdown blocks. RDB-PFN checkpoint evaluation needs no target-task trainer. The original model, source pretraining loop, and full RelGNN composite operator make the two training/inference boundaries inspectable. TabICL remains the explicitly library-backed comparison arm; this notebook does not reimplement its ensemble.')
 appendix=['sources/l166/upstream/model_pretrain/src/models.py','_run_l166.py','sources/l141/examples__relgnn_conv.py','sources/l166/upstream/model_pretrain/src/training.py']
 train=P/'sources/l166/upstream/model_pretrain/src/train.py'
 if train.exists():appendix.append(str(train.relative_to(P)))
 for name in appendix:md('### '+name+'\n```python\n'+(P/name).read_text()+'\n```')
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(cells):c.id=f'l182-{i:03}'
 return book
for solution in [False,True]:
 book=make(solution);path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
     old=nb.read(path,4);book.metadata=old.metadata
     previous=[c for c in old.cells if c.cell_type=='code'];current=[c for c in book.cells if c.cell_type=='code']
     if [c.source for c in previous]==[c.source for c in current]:
         for prior,c in zip(previous,current):
             c.outputs=prior.outputs;c.execution_count=prior.execution_count;c.metadata=prior.metadata
 nb.write(book,path)
print('L182 lesson,reference,student/solution notebook built')
