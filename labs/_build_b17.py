"""Generate measured lesson, model-specific figures and portable live-code labs."""
import ast,base64,io,json,zipfile,re
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;R=P.parent;S='b17-reusable-representations';F=P/'figures/b17';E=P/'evidence/b17'
r=json.loads((E/'diagnostic.json').read_text())
small={'seeds':[{'seed':s['seed'],'records':[{k:v for k,v in x.items() if k in ['task','mode','embedding_delta','prediction_delta','cache_hit','q0_probability']} for x in s['records']]} for s in r['seeds']]}
(R/'assets/b17-evidence.js').write_text('/* Generated from the complete saved B17 diagnostic. */\nwindow.B17_EVIDENCE='+json.dumps(small,separators=(',',':'))+';\n')
plt.rcParams.update({'font.size':12,'figure.facecolor':'#fafcf9','axes.facecolor':'#fafcf9'})
def box(ax,x,y,w,h,title,body,color='#e8f2ef',edge='#277872'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008',fc=color,ec=edge,lw=1.4));ax.text(x+.025,y+h-.015,title,weight='bold',fontsize=13,va='top');ax.text(x+.025,y+.018,body,fontsize=11.5,va='bottom',linespacing=1.2)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#47666c',lw=1.8))
fig,ax=plt.subplots(figsize=(8,10));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
ax.text(.02,.985,'FlexTab · separate features from the target stream',weight='bold',fontsize=17,va='top')
box(ax,.02,.80,.62,.13,'Feature stream · no target column','4 support + 2 query rows, 2 features\nNumeric tokens + [ROW] → 6 × 3 × 16')
box(ax,.69,.80,.29,.13,'Label stream','A: 0,0,1,1\nB: 0,1,0,1',color='#fbefd9',edge='#9e701e')
box(ax,.02,.60,.62,.14,'Encoder · two distinct layers','Columns: [ROW] reads feature keys\nRows: all queries read support keys only\nResidual + normalization + feed-forward')
arrow(ax,(.33,.80),(.33,.75))
box(ax,.02,.43,.62,.11,'Collect intermediate [ROW] outputs','P₁(r₁) + P₂(r₂) → W → Z: 6 × 16')
arrow(ax,(.33,.60),(.33,.55))
box(ax,.02,.22,.96,.14,'Decoder · target mask + row features → support mixing','One row key: softmax weight = 1 → inject projected row value\nTarget stream reads 4 support tokens; query labels are MASK\nTwo decoder layers → binary head → 2 query probabilities',color='#fbefd9',edge='#9e701e')
arrow(ax,(.33,.43),(.33,.37));arrow(ax,(.835,.80),(.835,.37))
box(ax,.02,.04,.46,.11,'Task A','Same Z + labels A → pA')
box(ax,.52,.04,.46,.11,'Task B','Same Z + labels B → pB')
arrow(ax,(.26,.22),(.26,.16));arrow(ax,(.75,.22),(.75,.16))
fig.text(.5,.008,'Pictured dimensions are the reduced untrained course model; all targets bypass the encoder.',ha='center',fontsize=10.5)
fig.savefig(F/'flextab.png',dpi=150,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(8,9));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
ax.text(.02,.99,'GTAlign · locate the parameters that change',fontsize=18,weight='bold',va='top')
box(ax,.03,.75,.94,.16,'1 · Source graphs → universal graph encoder','Different feature widths → PCA common width 50\nPerturbed graph views → 3-layer GCN, hidden width 256\nContrastive objective updates graph encoder')
box(ax,.03,.49,.94,.19,'2 · Communities → pseudo-label episodes','Louvain partitions source adjacency at sampled resolution\nNode embeddings become rows for LimiX-16M\nSupport pseudo-labels → predict held-out pseudo-labels\nCross-entropy updates BOTH graph encoder and TFM')
box(ax,.03,.25,.94,.17,'3 · Labeled target support → adapted encoder','Class prototypes = mean labeled support embeddings\nPrototype loss updates graph encoder; TFM stays FROZEN\nOld encoder cache is now stale → recompute embeddings',color='#fbefd9',edge='#9e701e')
box(ax,.03,.02,.94,.17,'4 · New embeddings + support labels → prediction','Adapted graph encoder → node / pooled graph rows\nFrozen TFM performs in-context inference\nTarget labels already affected weights: adaptation occurred')
for y in [.75,.49,.25]:arrow(ax,(.5,y),(.5,y-.06))
fig.text(.5,.006,'Paper architecture, not executed in B17. A TFM is a tabular foundation model.',ha='center',fontsize=11)
fig.savefig(F/'gtalign.png',dpi=150,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(8,5.8));
for ax in axes:ax.axis('off')
axes[0].set_title('One key · no choice among values',loc='left',fontsize=16,weight='bold')
axes[0].text(.03,.65,'score [s] → softmax [1] → 1 × [2, −1]',fontsize=17)
axes[0].text(.03,.25,'Output [2, −1], regardless of query/key score s.\nA learned value/output projection can still transform it.',fontsize=13)
axes[1].set_title('Two keys · relative scores control the mixture',loc='left',fontsize=16,weight='bold')
axes[1].text(.03,.68,'[0, log 3] → [1/(1+3), 3/(1+3)] = [.25, .75]',fontsize=15)
axes[1].text(.03,.28,'.25 × [2, −1] + .75 × [0, 3] = [.5, 2]',fontsize=17)
fig.text(.5,.025,'Exact analytic example before learned value/output projections; not a trained-model result.',ha='center',fontsize=11)
fig.tight_layout(rect=[0,.05,1,1],h_pad=2);fig.savefig(F/'attention.png',dpi=150,bbox_inches='tight');plt.close(fig)
flex='''<section class="rb-board" aria-label="FlexTab model architecture"><h3>Two tasks, one feature stream</h3><div class="rb-flow"><div class="rb-step"><strong>1 · Features only → cell tokens + [ROW]</strong>4 support + 2 query rows; 2 columns; width 16.<small>6 × 2 numeric values → 6 × 3 × 16 tokens. Target labels bypass this path.</small></div><div class="rb-step"><strong>2 · Alternate attention across columns and rows</strong>[ROW] reads feature keys; cross-row keys come only from support.<small>Two layers; residual, normalization and feed-forward operations preserve each row’s own path.</small></div><div class="rb-step"><strong>3 · Collect every depth → shared representation</strong>P₁(r₁) + P₂(r₂) → final projection W → Z, shape 6 × 16.<small>One layer-specific projection per depth. Cache this result only while encoder dependencies are unchanged.</small></div><div class="rb-step rb-target"><strong>4 · Separate target stream → decoder</strong>Support label embeddings + query MASK; inject row values via cross-attention.<small>Then attend along the target stream to support rows; binary head returns two query probabilities.</small></div><div class="rb-branches"><div class="rb-step rb-target"><strong>Task A</strong>Labels [0,0,1,1] + same Z → predictions A</div><div class="rb-step rb-target"><strong>Task B</strong>Labels [0,1,0,1] + same Z → predictions B</div></div></div><p class="rb-baseline">Pictured dimensions belong to the reduced untrained lab. Published default: 12-layer encoder, width 768, 12 heads. Pretraining/decoder-training paths are explained below.</p></section>'''
gt='''<section class="rb-board" aria-label="GTAlign model architecture"><h3>Follow the trainable component</h3><div class="rb-flow"><div class="rb-step"><strong>1 · Graph pretraining</strong>Node features → PCA width 50 → 3-layer GCN width 256.<small>Perturbed graph views; contrastive objective updates the graph encoder.</small></div><div class="rb-step"><strong>2 · Graph-to-table alignment</strong>Graph embeddings become table rows. Louvain communities supply pseudo-labels.<small>Support/query episodes → LimiX-16M → pseudo-label cross-entropy. Both encoder and TFM update.</small></div><div class="rb-step rb-target"><strong>3 · Target-domain adaptation</strong>Labeled target support → class prototypes → prototype loss.<small>Graph encoder UPDATES; tabular predictor stays FROZEN. Recompute support and query embeddings afterward.</small></div><div class="rb-step"><strong>4 · In-context prediction</strong>Adapted embeddings + support labels → frozen TFM → query class.<small>Node embeddings or pooled graph embeddings enter the same prediction interface.</small></div></div><p class="rb-baseline">Paper architecture only; GTAlign training and evaluation are NOT_RUN in B17.</p></section>'''
att='''<section class="rb-board" data-single-key><h3>Can a query choose among one key?</h3><label>Number of row values <select><option value="one">One row value</option><option value="two">Two row values</option></select></label><p data-keys>KV length 1: changing Q or K cannot change the attention weight.</p><output aria-live="polite">Any single score → weight [1]. Output [2, −1].</output><button type="button">Reset attention</button><p class="rb-baseline">Baseline: one row value [2,−1]. Exact illustrative arithmetic before value/output projections.</p><noscript><p>With two values [2,−1],[0,3] and scores[0,log3], weights[.25,.75] produce[.5,2].</p></noscript></section>'''
boundary='''<section class="rb-board" data-representation-boundary><h3>What changes, and what can be reused?</h3><label>Initialization seed <select name="seed"><option>0</option><option>1</option><option>2</option></select></label><label>Task <select name="task"><option>A</option><option>B</option></select></label><label>Intervention <select name="mode">'''+''.join(f'<option value="{m}">{t}</option>' for m,t in [('baseline','No change'),('support-labels','Complement support labels'),('support-feature','Change one support feature'),('other-query','Change only the other query'),('support-order','Permute support with labels'),('query-order','Permute query rows')])+'''</select></label><p data-path>Same feature encoding → reuse Z → decode current support labels</p><output aria-live="polite">Maximum embedding change:0. Maximum probability change:0. Encoder cache:HIT.</output><p data-meaning>Reference state: no inputs changed.</p><button type="button">Reset boundary</button><p class="rb-baseline">Baseline is always the unmodified input for the selected seed and task. Outputs align by row ID. Only q1 is edited in “other query”; inspect the q0 readout separately.</p><noscript><p>Label-only changes preserve Z. Support-feature changes invalidate Z. Query1 edits do not affect query0. The table below records each seed’s measured label sensitivity.</p></noscript></section>'''
results='| Seed | Task | Label edit: max ΔZ | Label edit: max Δprobability | Cache |\n|---:|---|---:|---:|---|\n'
for s in r['seeds']:
 for x in s['records']:
  if x['mode']=='support-labels':results+=f'| {s["seed"]} | {x["task"]} | {x["embedding_delta"]:.0f} | {x["prediction_delta"]:.8f} | HIT |\n'
def document(title,text,interactive=False):
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','b17-evidence','representation-boundary'] if interactive else []
 html=render(text).replace('<table>','<div class="rb-table" role="region" aria-label="Scrollable data table" tabindex="0"><table>').replace('</table>','</table></div>')
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/representation-boundary.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html#research-bridge">Research bridge</a></nav>'+html+'</article>'+''.join('<script src="../assets/'+s+'.js"></script>' for s in scripts)+'</body></html>'
body=(R/'lessons/content'/f'{S}.md').read_text()
lesson=body
for k,v in dict(FLEX_ARCH=flex,GT_ARCH=gt,ATTENTION=att,BOUNDARY=boundary,RESULTS=results).items():lesson=lesson.replace('{{'+k+'}}',v)
(R/'lessons'/f'{S}.html').write_text(document('B17 · Reusable representations',lesson,True))
ref='''# B17 · Representation reuse reference

**Rule:** reuse a cached result only while every input to its function stays unchanged.

| Term | Meaning |
|---|---|
| Target-agnostic at inference | Current target labels do not enter the encoder |
| Contextual row embedding | A row representation that also depends on permitted support features |
| Adaptation boundary | Which parameters/state change for a new task or domain |
| Single-key attention | Softmax over one score is1; output is the transformed value |
| Multi-layer row aggregation | Apply a separate projection to each layer, sum, then final projection |
| Pseudo-label | A training label constructed from another signal, here graph community |
| Prototype | Mean support embedding for one class |

**FlexTab path:** numeric/semantic cells → column/row attention → layerwise [ROW] projections → Z → task decoder with support labels. Current labels bypass E; historical pretraining targets still influenced its weights. A different target column can change the feature set and invalidate reuse.

**Attention trace:** one V=[2,−1] gives[2,−1] for any score. Two values [2,−1],[0,3] and scores[0,log3] give weights[.25,.75], output[.5,2]. Residual and learned value/output projections still matter.

**Cache key:** features + ordered IDs + support membership + encoder weights + preprocessing + implementation/mask version + precision. Decoder caches additionally depend on support labels. A conservative key may miss on harmless reorderings; correctness matters before hit rate.

**GTAlign stages:** contrastive GNN pretraining → community episode alignment updates GNN and TFM → target-label prototype adaptation updates GNN with TFM frozen → recompute embeddings → in-context prediction. This is not a wholly zero-shot pipeline.

**Temporal check:** inspect shared encoder context and preprocessing as well as directly retrieved neighbors. Earlier BFS neighbors alone cannot prove no future information enters Z.

'''+results+'''

**Evidence:** complete 36 random-network forward records, independent NumPy replay. No learned accuracy claim. Selected FlexTab Table 5 F1-DNF target AUROC74.6% remains INCOMPLETE_SOURCE_PROTOCOL_GATE / NOT_RUN. Full pretraining and GTAlign experiments NOT_RUN. PaidUSD 0; learner defense pending.

[Lesson](../lessons/b17-reusable-representations.html) · [Notebook](../labs/b17-reusable-representations.ipynb) · [Contract](../labs/b17-reproduction.md) · [FlexTab](https://arxiv.org/html/2606.30336v2) · [GTAlign](https://arxiv.org/html/2607.11374v1)
'''
(R/'reference'/f'{S}.html').write_text(document('B17 · Representation reuse reference',ref))
code=(P/'relkit/representations_b17.py').read_text();tree=ast.parse(code)
parts={n.name:ast.get_source_segment(code,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
imports='\n'.join(ast.get_source_segment(code,n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)))
testcode=(P/'_test_b17.py').read_text();testtree=ast.parse(testcode);testclass=next(ast.get_source_segment(testcode,n) for n in testtree.body if isinstance(n,ast.ClassDef))
def figure(name):return '![B17 '+name+'](data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode()+')'
archive=io.BytesIO()
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for path in sorted((P/'sources/b17').glob('*')):z.write(path,str(path.relative_to(P)))
 for path in [P/'_reproduce_b17.py',P/'b17-reproduction.md',E/'environment.json']:z.write(path,str(path.relative_to(P)))
blob=base64.b64encode(archive.getvalue()).decode()
# Notebook explanation is self-contained; no relative image URLs or relkit imports.
recap=body.split('## Lab ·')[0]
recap=re.sub(r'<div id="b17-[^"]+"></div>','',recap)
recap=re.sub(r'</?noscript>','',recap)
for k,v in dict(FLEX_ARCH=figure('flextab'),GT_ARCH=figure('gtalign'),ATTENTION=figure('attention'),BOUNDARY='**Notebook intervention:** predict the changes now; the complete live grid is executed after the three TODOs below.',RESULTS='**Author reference, not current kernel output:**\n\n'+results).items():recap=recap.replace('{{'+k+'}}',v)
recap=re.sub(r'\]\(\.\./([^)]*)\)',r'](https://avistian.github.io/relational/\1)',recap)
recap=re.sub(r'\]\((b\d[^)]*\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',recap)
goals={'attention':'Compute a scaled QK transpose, normalize over the key axis, and mix values. Reject empty keys. Preserve batched inputs and gradients. Why: an attention mask is an information-access contract, not just a speed trick.',
'aggregate_layers':'Validate one projection per layer. Apply the corresponding linear map to every layer output, sum, then apply the final map. Why: averaging before different projections changes the operation.',
'cache_identity':'Return a SHA256 fingerprint of features, ordered unique IDs, support size, encoder state, preprocessing and implementation identity. Include tensor shape/dtype and raw bytes. Do not accept labels as inputs. Why: same shape or frozen weights alone cannot establish cache validity. Use implementation tag b17-encoder-v1 and preserve the order in the serialization contract printed below.'}
for solution in [False,True]:
 cells=[]
 def md(s):cells.append(nb.v4.new_markdown_cell(s))
 def py(s):cells.append(nb.v4.new_code_cell(s))
 md(recap)
 md('''## Implementation contract · PROVIDED / TODO / CHECK

This TierC fixture isolates information flow. No package is imported as the model; all reduced model code appears below. No trainer is hidden: training is NOT_RUN. The paper-specific tokenizers, pretrained weights and specialized relational decoder are not available in this packet. Our numerical forward pass is a mechanism mirror only.

Dimensions: six rows, two columns, width 16; two single-head encoder and decoder layers, FF width32, GELU, pre-LayerNorm eps1e-5; no dropout. Seeds0/1/2, taskA/B and all six interventions are fixed before execution. Tolerance1e-10 for independent arithmetic,1e-12 for invariance. Versions used for author execution appear in the evidence packet; this notebook requires PyTorch and NumPy (already installed in ordinary Colab runtimes). It performs no installs/downloads. The source packet below is embedded.

Make your prediction before running: can changing labels ever invalidate this encoder cache? What changes if the target column itself changes?
''')
 py('# @colab-bootstrap\n# No repository imports; CPU only. Tested with torch2.13.0 and NumPy2.5.0.\n'+imports+'\nimport unittest\ntorch.set_num_threads(1)')
 md('## PROVIDED · hand-computed behavioral checks\n\nThe tests check uniform attention, single-key output, gradients, layer projections and cache invalidation. They do not claim pretrained-model parity.')
 py(testclass)
 for name,test in [('attention','test_attention'),('aggregate_layers','test_aggregation'),('cache_identity','test_cache')]:
  md('## '+('SOLUTION' if solution else 'TODO')+' · '+name+'\n\n'+goals[name])
  if name=='cache_identity':md('For byte-exact report replay, serialize metadata as compact JSON `[ids, support_size, preprocessing, "b17-encoder-v1"]`. Then in order: features followed by sorted state entries. For each tensor append JSON `[name, shape, str(dtype)]` (default separators), then its contiguous CPU NumPy bytes. This fixes serialization, not which dependencies your implementation must validate.')
  py(parts[name] if solution else parts[name].split('\n')[0]+f'\n    raise NotImplementedError("Complete {name}")')
  md('### CHECK · run immediately\n\nFix the operation before continuing. This function remains live in the model/grid.')
  py(f"Boundaries('{test}').debug()\nprint('{name}: CHECK passed')")
 md('## PROVIDED · Q/K/V and feature encoder\n\nPaper §2.2 and A.1 mechanism: feature-only column keys, support-only row keys and collected layer outputs. The lab chooses numeric embeddings and pre-normalization; no upstream source parity is claimed. Watch where your attention and aggregation functions are called.')
 for names in [['Attention'],['EncoderBlock'],['Encoder']]:py('\n\n'.join(parts[n] for n in names))
 md('## PROVIDED · target decoder\n\nOne-key cross-attention injects each row embedding, then target-stream attention reads support labels indirectly through support tokens. Query tokens start with MASK index2. No query-label argument exists. The two tasks share this decoder as well as the encoder; the paper’s specialized task decoders are a separate training stage.')
 py(parts['DecoderBlock']+'\n\n'+parts['Decoder'])
 md('## PROVIDED · frozen fixture and complete intervention grid\n\nRead all six modes before execution. Save every state and prediction. No choice is made based on outputs. Query-label interventions enumerate all four external binary assignments; these cannot enter the interfaces. A cache stores the encoder output only.')
 py(parts['fixture']+'\n\n'+parts['initialized'])
 py(parts['run_experiment'])
 md('## RUN · your live model\n\nEvery TODO above contributes to this run. This table now comes from the current kernel, unlike the author-reference table earlier.')
 py("report=run_experiment()\nassert len(report['seeds'])==3\nprint('seed task mode                 max Δembedding   max Δprobability cache')\nfor s in report['seeds']:\n    for v in s['records']:\n        print(f\"{s['seed']:4} {v['task']:4} {v['mode']:20} {v['embedding_delta']:14.6g} {v['prediction_delta']:16.6g} {'HIT' if v['cache_hit'] else 'MISS'}\")\nfrom pathlib import Path\nPath('b17-diagnostic.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\\n')")
 md('## EXIT · defend the boundary\n\nIn150–200 words: explain why encoder output stays fixed under label edits but can change with support features. Derive single-key attention. Identify the GTAlign target-stage trainable component and explain why its old embeddings are stale. Distinguish the course evidence from the published 74.6% AUROC. Ask the teacher for feedback. Revisit in 1/7/30 days; execution is not learner mastery.')
 py("assert sum(len(s['records']) for s in report['seeds'])==36\nassert sum(len(s['checks']) for s in report['seeds'])==24\nprint('COMPLETE_MECHANISM_DIAGNOSTIC; no trained accuracy claim')\nprint('Paper: INCOMPLETE_SOURCE_PROTOCOL_GATE / NOT_RUN')\nprint('Learner: PENDING_WRITTEN_DEFENSE')")
 md('## Paper-results lane · authenticated source audit, blocked execution\n\nNamed target B17-FLEXTAB-MULTI-TABLE5-F1-DNF: full original test population, complete(driverId,date) keys, AUROC0.746. The complete reproduction contract below identifies the missing original assets, context/seeds and temporal scope. Enabling the gate refuses execution until these are recovered; it is not a benchmark launcher. Full pretraining, all other benchmarks and GTAlign experiments remain NOT_RUN. USD 0 paid; author aggregate numerical budget 3600s. Colab frontend NOT_CHECKED.')
 py("import base64,io,zipfile,subprocess,sys\npacket=Path('b17-source-packet');packet.mkdir(exist_ok=True)\nwith zipfile.ZipFile(io.BytesIO(base64.b64decode("+repr(blob)+"))) as archive:\n    archive.extractall(packet)\ncheck=subprocess.run([sys.executable,str(packet/'_reproduce_b17.py'),'--phase','audit'],capture_output=True,text=True,check=True)\nprint(check.stdout)\nprint((packet/'b17-reproduction.md').read_text())")
 py("RUN_PAPER_REPRO=False\nif RUN_PAPER_REPRO:\n    subprocess.run([sys.executable,str(packet/'_reproduce_b17.py'),'--phase','paper'],check=True)\nelse:\n    print('Paper NOT_RUN: original checkpoint, evaluator and temporal sampling protocol unresolved.')")
 n=nb.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python3',language='python'),language_info=dict(name='python',version='3.12')))
 nb.write(n,(P/'solutions'/f'{S}.ipynb') if solution else P/f'{S}.ipynb')
print('Built lesson, reference, three figures and live portable notebooks')
