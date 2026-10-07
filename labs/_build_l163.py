"""Build HTML, figures and standalone notebooks from canonical L163 sources."""
import ast,base64,hashlib,json,re
from pathlib import Path
import numpy as np,nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l163';F=P/'figures/l163';S='0163-lm-encoders-for-rows'
report=json.loads((E/'report.json').read_text());packet=json.loads((E/'fixtures.json').read_text());traces=json.loads((E/'token-traces.json').read_text());arrays=dict(np.load(E/'embeddings.npz'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l163'})

def save(fig,name):
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 for ax in fig.axes:
  for t in ax.texts:
   b=t.get_window_extent(renderer);assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,(name,t.get_text())
 for ext in ['svg','png']:fig.savefig(F/(name+'.'+ext),dpi=150,metadata={'Date':None} if ext=='svg' else {'Software':'L163'})
 plt.close(fig)
 for f in F.glob('*.svg'):f.write_text('\n'.join(x.rstrip() for x in f.read_text().splitlines())+'\n')

def box(ax,x,y,w,h,title,body,color='#e2efec'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008',facecolor=color,edgecolor='#9ab7ad'))
 ax.text(x+.017,y+h-.027,title,fontsize=12,weight='bold',va='top',color='#174b41')
 ax.text(x+.017,y+h-.077,body,fontsize=10.7,va='top',linespacing=1.45,color='#233f36')

def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=2,color='#25806b'))

fig,ax=plt.subplots(figsize=(12,8));fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.91);fig.patch.set_facecolor('#f5f8f5');ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
fig.suptitle('One row, two encodings, the same prediction task',x=.04,ha='left',fontsize=18,y=.97)
box(ax,.02,.845,.96,.135,'ALLOW-LISTED INPUT · same information for both paths','price_usd=10.0   weight_kg=2.0   colour=red   condition=used\nRow ID and target are excluded. Presentation can change; values stay fixed.','#f8eddb')
arrow(ax,(.26,.835),(.26,.79));arrow(ax,(.75,.835),(.75,.79))
box(ax,.02,.595,.46,.19,'TEXT · pretrained and frozen','Ordered JSON → BART token IDs + attention mask\n6 encoder layers: B × L × 768 hidden states\nMasked mean, including BOS/EOS → B × 768')
box(ax,.54,.595,.44,.19,'TYPED · statistics fitted on train only','2 numeric columns → training mean / SD\n3 colours + 3 conditions → one-hot categories\nCanonical field lookup → B × 8 features')
arrow(ax,(.25,.585),(.25,.535));arrow(ax,(.76,.585),(.76,.535))
box(ax,.02,.365,.46,.165,'TEXT HEAD · fit only on training rows','Train-fitted standardization → ridge regression\n5 alphas compete on validation MAE\nFreeze selected scaler + head before test','#e5eaf6')
box(ax,.54,.365,.44,.165,'TYPED HEAD · same selection rule','Train-fitted standardization → ridge regression\nSame 144 train / 48 validation / 48 test rows\nSame alpha grid; different representation','#e5eaf6')
arrow(ax,(.25,.355),(.25,.295));arrow(ax,(.76,.355),(.76,.295))
box(ax,.02,.13,.96,.16,'TEST · unchanged heads predict the same target','Baseline / reversed order / opaque field names → predictions and MAE\nMeasure token counts, feature changes and paired prediction changes. Repeat all three fixed splits.','#f8eddb')
ax.text(.03,.055,'Frozen BART encoder: no decoder, no LM fine-tuning, no GNN. Only preprocessing and regression heads are fitted.',fontsize=10.8)
ax.text(.03,.015,'B = batch rows; L = padded token positions. Synthetic additive target deliberately favours typed numeric structure.',fontsize=10.8)
save(fig,'architecture')
fig,axes=plt.subplots(1,2,figsize=(11.5,4.8));fig.subplots_adjust(left=.08,right=.97,bottom=.23,top=.8,wspace=.35);fig.patch.set_facecolor('#f5f8f5')
fig.suptitle('Measured errors: numeric fit and presentation shift',x=.04,ha='left',fontsize=17,y=.97)
for ax in axes:ax.set_facecolor('#f5f8f5');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
for ax,groups in [(axes[0],[('typed','baseline'),('bart','baseline')]),(axes[1],[('bart','baseline'),('bart','reordered'),('bart','renamed')])]:
 for j,(method,variant) in enumerate(groups):
  vals=[v['test_mae'] for v in report['results'] if v['method']==method and v['variant']==variant]
  ax.scatter(np.array([-.08,0,.08])+j,vals,s=50,color='#23786b' if method=='typed' else '#b5742e',zorder=3)
  mean=np.mean(vals);ax.plot([j-.2,j+.2],[mean,mean],color='#213e36',lw=2)
  ax.annotate(f'{mean:.2f}',(j,max(vals)),xytext=(0,15),textcoords='offset points',ha='center',fontsize=10)
 ax.set_xticks(range(len(groups)),[('Typed' if m=='typed' else 'BART')+'\n'+v for m,v in groups]);ax.set_xlim(-.6,len(groups)-.4)
axes[0].set_ylim(0,21);axes[0].set_title('Unchanged input · linear scale',fontsize=12);axes[0].set_ylabel('Test MAE · target units')
axes[1].set_yscale('log');axes[1].set_ylim(8,1500);axes[1].set_title('Frozen BART head · logarithmic scale',fontsize=12);axes[1].set_ylabel('Test MAE · log scale')
fig.text(.08,.055,'Dots: all three fixed splits. Lines and labels: means. Split overlap means these dots are not independent datasets.',fontsize=10)
save(fig,'comparison')

captions={'architecture':'The implemented comparison: frozen BART encoder and a train-fitted typed feature map feed separately selected ridge heads. Both use identical splits and target information. This is a local synthetic comparison, not the historical BART_table+GNN model.', 'comparison':'Measured synthetic task: every dot is a split score. The left panel is linear; the right is logarithmic to show large schema-shift errors. No intervention retuning. The disclosed additive target favours typed linear features.'}
rows=[]
for i in range(3):
 row={'id':packet['rows'][i]['id']}
 for v in packet['variants']:
  a=arrays[v][i].astype(float);b=arrays['baseline'][i].astype(float)
  row[v]=dict(traces[v][i],cosine_distance=0. if v=='baseline' else float(1-np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))))
 rows.append(row)
widget={'rows':rows};(E/'widget-data.json').write_text(json.dumps(widget,ensure_ascii=False,indent=2)+'\n')

result_text='| Encoder | Input presentation | Mean MAE ± split SD | Mean absolute prediction change |\n|---|---|---:|---:|\n'
for q in report['summary']:result_text+=f"| {q['method']} | {q['variant']} | {q['mean_mae']:.3f} ± {q['sample_sd']:.3f} | {q['mean_prediction_change']:.3f} |\n"
enc=json.loads((E/'encoding-receipt.json').read_text());result_text+='\nAll 720 row encodings and all six head selections completed. Fresh encoding took '+f"{sum(enc['seconds'].values()):.1f}"+' seconds across the three variants; '+f"{enc['elapsed_with_fetch_seconds']:.1f}"+' seconds including fetch and checks. These are local CPU timings, not matched throughput benchmarks. Cloud spend: **$0**.'

timing=json.loads((E/'timing.json').read_text())
result_text+='\n\n| Encoding work | Local elapsed time | Features per row |\n|---|---:|---:|\n'
for event in timing['typed']:
 result_text+=f"| Typed fit/transform, split {event['seed']} | {1000*event['seconds']:.2f} ms | 8 |\n"
for variant,seconds in timing['bart_seconds_per_variant'].items():
 result_text+=f"| Frozen BART, {variant} | {seconds:.2f} s | 768 |\n"
result_text+='\nSingle local timings for all 240 rows per entry. Typed time includes train-moment/vocabulary fitting and transformation; BART time includes tokenization, forward computation and pooling after loading. Imports, process startup and head fitting are excluded. This is cost accounting, not a controlled speed benchmark.'

def prose(portable=False):
 t=(R/'lessons/content'/(S+'.md')).read_text().replace('[[RESULTS]]',result_text)
 for key,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((F/(key+'.png')).read_bytes()).decode() if portable else '../labs/figures/l163/'+key+'.svg'
  t=t.replace('[[FIG:'+key+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 for key,fallback,html in [
  ('WARMUP','Recall: why does row-wise text encoding alone not supply foreign-key context? Which preprocessing statistics may use held-out rows?','<div id="warmup"></div>'),
  ('PREDICT','Predict before calculating: vectors [1,3] and [3,7] are real; [900,900] is padding. What should the pooled row contain?','<div id="predict"></div>'),
  ('TOKENS','Portable trace: inspect token_traces below for baseline, reordered and renamed inputs. All values stay fixed; token counts and representations need not.','<div id="row-explorer"></div><noscript>Measured token traces for baseline, reversed order and opaque names are in the notebook. All variants preserve values; the result table below remains available without JavaScript.</noscript><script type="application/json" id="row-data">'+json.dumps(widget,ensure_ascii=False).replace('</','<\\/')+'</script>'),
  ('TEACHBACK','Explain the synthetic target bias, schema sensitivity, leakage controls and historical source gap before selecting an encoder.','<div id="teachback"></div>')]:t=t.replace('[['+key+']]',fallback if portable else html)
 if portable:
  t=t.replace('](../','](https://avistian.github.io/relational/').replace('href="../','href="https://avistian.github.io/relational/')
  t=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',t)
 return t

def doc(title,body,interactive=False):
 html=re.sub(r'<table([^>]*)>',r'<div class="route-scroll" tabindex="0"><table\1>',render(body)).replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','row-encoder','l163-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in ['lesson','atomic-route','checkpoint','lab-access','row-encoder'])+'</head><body class="checkpoint row-lesson"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0162-the-relational-fm-vision.html">Lesson 162</a></nav><header><p class="route-kicker">Year 5 · Quarter 1 · Lesson 163</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('LM encoders for rows',prose(),True))
ref='''## Two paths, one input contract
Text: allow-listed schema/value JSON → token IDs + attention mask → frozen BART encoder → B×L×768 hidden states → masked mean → B×768 features. Include BOS/EOS; exclude PAD. Reject overlength inputs instead of silently truncating.

Typed: training means/SD → numeric z-scores; training vocabularies → one-hot categories. Our two numbers plus two three-category fields give8 features. Unknown category→zero block; constant numeric SD→1. Canonical identity lookup gives presentation invariance by construction.

## Three load-bearing rules
1. Exclude target/row ID from inputs; preserve field/value association and string escaping.
2. Pool only real tokens: `(hidden*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]`.
3. Fit preprocessing/head on training rows, choose alpha on validation, freeze before test. Renamed/reordered inputs get the same frozen head.

## Reading the evidence
The synthetic target is numeric/additive and favours typed features. Short categories do not test rich text semantics. Equal split/grid rules do not equalize pretraining or feature capacity. Three overlapping split scores give descriptive SD, not a confidence interval. A changed embedding is not automatically a worse prediction; measure both.

'''+result_text+'''

## Reproduction boundaries
Local complete comparison:240 rows, three seeds, two encoders, three input variants. Fresh BART encoding and cached head replay have distinct receipts. Historical Vogel Table1 wikiTables reconstruction: NOT_RUN; fidelity NOT_ESTABLISHED because exact artifacts and training details are unresolved. No GNN or relational transfer is evaluated. Learner PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/0163-lm-encoders-for-rows.html) · [Comparison template](../labs/l163-comparison-template.md) · [Protocol](../labs/l163-reproduction.md) · [Paper §4](https://arxiv.org/html/2305.15321v1#S4) · [BART model](https://huggingface.co/facebook/bart-base).
'''
(R/'reference/row-encoders.html').write_text(doc('Row encoders — quick reference',ref))

def defs(path):
 t=path.read_text();return [(n.name,ast.get_source_segment(t,n)) for n in ast.parse(t).body if isinstance(n,ast.FunctionDef)]

payloads={name:base64.b64encode((E/name).read_bytes()).decode() for name in ['fixtures.json','embeddings.npz','pool-states.npz','token-traces.json']}
hashes={name:hashlib.sha256((E/name).read_bytes()).hexdigest() for name in payloads}
contracts={
 'serialize_row':('Allow-list and serialize the row','Require finite numeric price_usd/weight_kg and nonempty string colour/condition. Output compact JSON with table=products then these four fields. Never include id/target. baseline uses canonical order, reordered reverses only the four fields, renamed uses c0..c3 in canonical order. Reject bad inputs/variants with ValueError. Use JSON escaping.',"q=serialize_row(dict(id='private',price_usd=10.,weight_kg=2.,colour='red',condition='used',target=999))\nassert 'private' not in q and '999' not in q and 'target' not in q\nprint(q)"),
 'masked_mean':('Pool real token states','Accept finite [B,L,D] hidden states and binary [B,L] mask, at least one1 per row. Return [B,D] mean over included tokens, including BOS/EOS. Reject mismatched/nonbinary/all-zero masks with ValueError.',"np.testing.assert_allclose(masked_mean([[[1.,3.],[3.,7.],[900.,900.]]],[[1,1,0]]),[[2,5]])\nprint('CHECK: padding excluded')"),
 'choose_alpha':('Select without test labels','Accept nonempty aligned 1D positive alphas and finite nonnegative validation MAEs. Return the alpha with smallest error, first in supplied order on a tie. Reject invalid inputs with ValueError.',"assert choose_alpha([.01,.1,1.],[3,1,2])==.1\nassert choose_alpha([.01,.1],[1,1])==.01\nprint('CHECK: validation choice and stable tie')")}
for solution in [False,True]:
 cells=[]
 def md(t):cells.append(nb.v4.new_markdown_cell(t))
 def code(t,hidden=False):
  c=nb.v4.new_code_cell(t)
  if hidden:c.metadata['tags']=['data-payload']
  cells.append(c)
 md('# Lesson163 · LM encoders for rows\n\nStandalone student lab: three live TODOs, real frozen-encoder evidence and a full three-split head replay. Default execution needs only NumPy/scikit-learn; all inputs are embedded. Fresh BART encoding is a separate opt-in lane (~560 MB weights). Author execution is not learner mastery. Website links become available after publication.')
 md(prose(True))
 md('## PROVIDED · Environment\nThe author used NumPy2.5.0, SciPy1.18.0, scikit-learn1.9.0. The replay records your versions; numerical agreement, not identical bytes across platforms, is the portable criterion. If missing, the bootstrap installs NumPy and scikit-learn. Fresh encoding additionally requires torch, transformers4.57.1 and requests.')
 code("# @colab-bootstrap\nimport os,sys,subprocess,importlib.util\nos.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')\nif any(importlib.util.find_spec(k) is None for k in ['numpy','sklearn']):\n    subprocess.check_call([sys.executable,'-m','pip','install','numpy','scikit-learn'])\nimport io,json,base64,hashlib\nfrom pathlib import Path\nimport numpy as np\nfrom importlib.metadata import version\nprint({k:version(k) for k in ['numpy','scipy','scikit-learn']})")
 md('## PROVIDED · Pinned inputs and real model evidence\n240 complete rows and three partitions; all three768-feature arrays; token strings/IDs; final hidden states for the first8 rows in each variant. The encoded payload is only storage. Human-readable functions below perform the computation. Cache hashes guard accidental edits; the source/weight provenance is in the reproduction contract.')
 code('payloads='+repr(payloads)+'\nhashes='+repr(hashes)+'\nblobs={name:base64.b64decode(value) for name,value in payloads.items()}\nfor name,raw in blobs.items():\n    assert hashlib.sha256(raw).hexdigest()==hashes[name]\npacket=json.loads(blobs["fixtures.json"])\narrays=dict(np.load(io.BytesIO(blobs["embeddings.npz"])))\nstates=dict(np.load(io.BytesIO(blobs["pool-states.npz"])))\ntoken_traces=json.loads(blobs["token-traces.json"])\nprint("Verified complete 240-row evidence and three splits")',True)
 for name,source in defs(P/'relkit/rows_l163.py'):
  if name in contracts:
   title,contract,check=contracts[name];md('## TODO · '+title+'\n\n'+contract+'\n\nPredict an edge case before writing the implementation.')
   code(source if solution else source.split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")')
   code('# CHECK\n'+check)
  else:md('## PROVIDED · Train-only typed representation');code(source)
 for name,source in defs(P/'_check_l163.py'):code(source)
 code("assert check163(serialize_row,masked_mean,choose_alpha,typed_features)=='PASS'\n# Your serializer must recover the exact measured input for every cached row.\nfor variant in packet['variants']:\n    for row,trace in zip(packet['rows'],token_traces[variant]):\n        assert serialize_row(row,variant)==trace['text']\n    # Your pooling implementation supplies real features for the first8 rows.\n    repooled=masked_mean(states[variant+'_hidden'],states[variant+'_mask']).astype(np.float32)\n    np.testing.assert_allclose(repooled,arrays[variant][:8],atol=1e-6,rtol=1e-6)\n    arrays[variant][:8]=repooled\nprint('PASS: 720 serializations and 24 live real-state pooled rows')")
 md('## PROVIDED · Optional fresh frozen encoding\nDefault False replays the verified author cache. Set True only when you want to download the pinned model and recompute all720 representations on CPU; this is distinct from head replay and does not train BART. Dependencies/device affect runtime. No historical trainer is implied.')
 for name,source in defs(P/'_encode_l163.py'):code(source)
 code("RUN_FRESH_ENCODER=False\nif RUN_FRESH_ENCODER:\n    subprocess.check_call([sys.executable,'-m','pip','install','transformers==4.57.1','torch','requests'])\n    model_path=Path('l163-model')/packet['revision']\n    downloaded_artifacts=fetch_model(packet,model_path)\n    arrays,token_traces,fresh_receipt=encode_rows(packet,model_path,serialize_row,masked_mean)\n    Path('l163-fresh-receipt.json').write_text(json.dumps(dict(fresh_receipt,artifacts=downloaded_artifacts),indent=2))\n    print('Fresh frozen encoding complete')\nelse:\n    print('CACHE_REPLAY: no fresh encoding performed by this notebook run')")
 md('## PROVIDED · Fit on train, select on validation, freeze, then test\nYour choose_alpha implementation selects all six heads. Intervention inputs never retune the baseline heads. The stored selection receipt precedes test computation.')
 for name,source in defs(P/'_run_l163.py'):code(source)
 code("heads=fit_heads(packet,arrays,typed_features,choose_alpha)\nPath('l163-selection.json').write_text(json.dumps(heads,indent=2))\nreport=evaluate_heads(packet,arrays,heads,typed_features)\nassert len(report['predictions'])==864\nprint(render_report(report))\nPath('l163-report.json').write_text(json.dumps(report,indent=2)+'\\n')")
 md('## EXIT · Encoder-comparison table and written defense\nUse the lesson template. Include each split, representation shape, fitted/frozen parts, input cost, presentation sensitivity and a falsifiable follow-up. Explain the additive target bias, test-label isolation, cached versus fresh evidence and historical source gaps in200–300 words. Rubric: computation, legality, causality, evidence, next test;0–2 each;≥8/10 with no zero after review. The short author illustration is not a completed learner submission.')
 example='For this additive numeric task I would use the typed baseline. It exposes the actual numeric and category terms directly, whereas the frozen BART path adds a token/representation mapping whose linear readout has higher measured error. Canonical typed field lookup explains presentation invariance; it is not unseen-schema transfer. The experiment does not test rich text semantics. I would require a matched free-text task before generalizing the choice. Fresh encoding and cached head replay are distinct; the historical wikiTables trainer is source-gated.' if solution else ''
 code('defense='+repr(example)+'\nPath("l163-submission.json").write_text(json.dumps(dict(defense=defense,review=None,learner="PENDING_WRITTEN_DEFENSE"),indent=2))\nprint("Exported defense; learner PENDING_WRITTEN_DEFENSE")')
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(book.cells):c.id=f'l163-{i:03d}'
 path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
  old=nb.read(path,4);book.metadata=old.metadata
  previous=[c for c in old.cells if c.cell_type=='code'];current=[c for c in book.cells if c.cell_type=='code']
  if [c.source for c in previous]==[c.source for c in current]:
   for o,c in zip(previous,current):
    c.outputs=o.outputs;c.execution_count=o.execution_count;c.metadata=o.metadata
 nb.write(book,path)
print('Built L163 lesson, reference, two figures, measured token explorer and portable notebooks')
