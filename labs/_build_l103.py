"""Build lesson, portable standalone notebooks and reference from canonical sources."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0103-tgat';TITLE='TGAT: attention that knows when'
CAP={'architecture':'Trace each endpoint from its timestamped query through two recursive layers to the shared pair decoder. Shapes describe the released Wikipedia variant.', 'recursion':'A child reached through an event at time 5 must use history before 5, even when the root query is at time 8. The lower panel separately shows a released sampler defect.', 'kernel':'Illustrative fixed frequencies: paired sine/cosine inner products preserve time differences. A cosine-only inner product can change under a common shift of its arguments.'}
CAP['paper']='Per-seed AP in the complete released-protocol replay. Dashed paper targets refer to the original reported populations; numerical proximity does not resolve the protocol differences.'
CAP['comparison']='Each line pairs a common seed and negative schedule across models. Horizontal strokes are means; three seeds on one event slice do not establish benchmark superiority.'
def result_text(kind):
 if kind=='paper':
  path=P/'_paper_l103_results.json'
  if not path.exists():return '> **Execution status: RUNNING.** Ten full Wikipedia release-protocol runs have been launched. No completed mean or numerical verdict is claimed yet. Historical identity remains **INCOMPARABLE**; full-paper parity **NOT_ESTABLISHED**.'
  r=json.loads(path.read_text());s='| Released evaluation population | Mean AP | Seed sample SD | Original paper AP | Numerical verdict |\n|---|---:|---:|---:|---|\n'
  for k,label in [('all','All test events'),('new','New-node event subset')]:
   x=r['summary'][k];s+=f"| {label} | {x['mean_ap_percent']:.4f}% | {x['sample_sd_pp']:.4f} pp | {x['paper_ap_percent']:.2f}% | {x['numerical_verdict']} |\n"
  return s+f"\n**Execution: {r['status']}; {len(r['records'])}/10 seeds, {sum(x['epochs_completed'] for x in r['records'])} complete epochs.** Sample SD concerns initializations on this fixed split. Historical identity **INCOMPARABLE**; full-paper parity **NOT_ESTABLISHED**. [Per-seed traces and identities](../labs/_paper_l103_results.json)."
 r=json.loads((P/'_comparison_l103_results.json').read_text());s='| Seed | TGAT test AP | TGN test AP | TGN − TGAT |\n|---|---:|---:|---:|\n';values={arm:[] for arm in ['TGAT','TGN']}
 for seed in range(3):
  d={x['arm']:100*x['test']['ap'] for x in r['records'] if x['seed']==seed}
  for arm in values:values[arm].append(d[arm])
  s+=f"| {seed} | {d['TGAT']:.3f}% | {d['TGN']:.3f}% | {d['TGN']-d['TGAT']:+.3f} pp |\n"
 import statistics
 s+='\n'+ '; '.join(f'**{arm}: {statistics.mean(v):.3f}% ± {statistics.stdev(v):.3f} pp sample SD**' for arm,v in values.items())
 return s+'. These are author-reference measurements, separate from your fresh notebook run. **MEASURED course experiment; paper comparison INCOMPARABLE.** [Complete traces](../labs/_comparison_l103_results.json).'
def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text()
 source=(P/'relkit/tgat_l103.py').read_text();tree=ast.parse(source);cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='TGAT');method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='tem_conv')
 snippet='\n'.join(source.splitlines()[method.lineno-1:method.end_lineno]);s=s.replace('[[CODE:tem_conv]]','```python\n'+snippet+'\n```')
 for name,cap in CAP.items():
  if not (P/f'figures/l103/{name}.png').exists():
   s=s.replace('[[FIG:'+name+']]','');continue
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l103/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l103/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="tgat-figure" tabindex="0"><img src="{src}" alt="{cap}"><figcaption>{cap}</figcaption></figure>')
 for key,mount,plain in [('WARMUP','l103-warmup','**Retrieval first:** answer the prompts below before reading.'),('PREDICT','l103-predict','**Predict:** must an older event always receive less attention as query time increases?'),('TIME_WIDGET','l103-time','**Portable worked state:** at t=4 the two weights are approximately 0.178 and 0.822. Open the HTML lesson to change query time from 4 to 10 and see the older event regain attention.'),('TEACHBACK','l103-teachback','**Teach back:** explain the five points above without copying the lesson, then ask the tutor to challenge your reasoning.')]:s=s.replace('[['+key+']]',plain if portable else f'<div id="{mount}"></div>')
 s=s.replace('[[PAPER_RESULTS]]',result_text('paper')).replace('[[COMPARISON_RESULTS]]',result_text('comparison'))
 if portable:
  s=s.replace('href="../','href="https://avistian.github.io/relational/').replace('](../','](https://avistian.github.io/relational/')
  s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s
head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 103 — {TITLE}</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','tgat-lesson'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="0102-temporal-graph-networks.html">Lesson 102</a></nav><header><p class="tgat-kicker">Year 3 · Quarter 3 · Lesson 103</p><h1>'+TITLE+'</h1></header>'
body=render(prose()).replace('<table>','<div class="tgat-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
(R/'lessons'/f'{S}.html').write_text(head+body+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','teachback','temporal-attention-viz','l103-lesson'])+'</body></html>')
checks={
'encode_time':"x=torch.tensor([[0.,2.]],requires_grad=True);frequency=torch.tensor([1.,.5],requires_grad=True);phase=torch.zeros(2,requires_grad=True)\ny=encode_time(x,frequency,phase)\nassert y.shape==(1,2,2), 'Broadcast to B × N × D'\nassert torch.allclose(y[0,0],torch.ones(2)), 'Zero elapsed and zero phase must give cosine(0)'\nassert torch.allclose(y[0,1],torch.cos(torch.tensor([2.,1.]))), 'Frequency multiplies elapsed time'\ny.sum().backward();assert frequency.grad is not None and phase.grad is not None and frequency.grad.abs().sum()>0 and phase.grad.abs().sum()>0 and x.grad.abs().sum()>0, 'Keep the gradient path'\nprint('PASS: values, dimensions and gradients')",
'attention_weights':"q=torch.tensor([[[1.,0.]]],requires_grad=True);keys=torch.tensor([[[1.,0.],[0.,1.],[9.,9.]]],requires_grad=True);mask=torch.tensor([[[False,False,True]]])\na=attention_weights(q,keys,mask)\nassert a.shape==(1,1,3) and a[0,0,2]==0, 'Mask padding before normalizing'\nassert abs(a[0,0,0].item()-1/(1+math.exp(-1/math.sqrt(2))))<1e-6, 'Scale by sqrt(head width)'\nassert torch.allclose(a.sum(-1),torch.ones(1,1)), 'Normalize over neighbors'\nassert torch.allclose(attention_weights(q,keys,torch.ones_like(mask)),torch.full((1,1,3),1/3)), 'Preserve the declared finite-mask release convention'\na[0,0,0].backward();assert q.grad is not None and q.grad.abs().sum()>0 and keys.grad is not None and keys.grad.abs().sum()>0, 'Preserve query/key gradients'\nassert not keys.grad[0,2].any(), 'Padded keys must receive zero gradient'\nprint('PASS: scaling, padding, query/key gradients and all-padding counterexample')",
'strict_prefix':"for times in [[],[1],[1,1,2,4],[0,3,9]]:\n    for cutoff in [-1,0,1,2,3,4,10]:\n        expected=sum(t<cutoff for t in times)\n        assert strict_prefix(np.array(times),cutoff)==expected, 'Use strictly before; exclude all tied times'\nprint('PASS: empty, ties, exact cutoffs and end boundaries')"}
tgat=(P/'relkit/tgat_l103.py').read_text();tgn=(P/'relkit/tgn_l102.py').read_text()
# Keep all visible TGN definitions in a separate naming space; no imported hidden model.
for old in ['TimeEncoder','Merge','NegativeSampler','load_wikipedia','run_training','evaluate','batches','PAPER_AP','SOURCE_COMMIT','CLOSE_TOLERANCE_PP']:
 tgn=re.sub(r'\b'+old+r'\b','TGN_'+old,tgn)
tgn=tgn.replace("np.searchsorted(history[:, 2], cutoff, side='left')","strict_prefix(history[:, 2], cutoff)")
comparison=(P/'relkit/tgat_comparison_l103.py').read_text();comparison=re.sub(r'^from relkit\.[^\n]+\n','',comparison,flags=re.M)
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 103 — '+TITLE+'\n\n'+('Executed teacher solution; PENDING_WRITTEN_DEFENSE is the learner status.' if solution else 'Student edition: PROVIDED scaffolding, three live TODOs, immediate CHECKs, then an EXIT defense.')+'\n\n[HTML lesson](https://avistian.github.io/relational/lessons/0103-tgat.html). Figures are embedded; model and trainer code are fully visible.'),
 nbf.v4.new_code_cell("# @colab-bootstrap — standalone visible implementation; no course package needed.\nimport sys\nprint('Python',sys.version.split()[0])\n# If imports below are missing: %pip install torch numpy pandas scikit-learn"),
 nbf.v4.new_markdown_cell('## Runtime and data\n\nTier B: real Wikipedia interaction graph. First use downloads a 534 MiB hash-checked raw file. Allow about 2 GiB RAM for parsing. The default run uses a declared 600/120/120 event slice, one epoch, both models. It is not the historical experiment. The full released two-layer TGAT uses the complete dataset and ten seeds. GPU runtime pins: torch 2.8.0, numpy 2.2.6, pandas 2.3.2, scikit-learn 1.7.1; CPU author runtime is recorded in the evidence. Live Colab is NOT_CHECKED. After this lesson is published the course URLs work; local source files are supplied with the package.'),nbf.v4.new_markdown_cell(prose(True)),nbf.v4.new_markdown_cell('## Visible TGAT implementation\n\nRead each named chunk. The replay preserves documented source quirks; the corrected comparison uses the strict temporal-boundary function you implement. The three TODOs directly affect the model trained below.')]
 for chunk in re.split(r'^# %% ',tgat,flags=re.M)[1:]:
  title,code=chunk.split('\n',1);code=code.strip();task=next((k for k in checks if 'def '+k+'(' in code),None)
  cells.append(nbf.v4.new_markdown_cell('### '+('TODO' if task and not solution else 'PROVIDED')+' · '+title))
  if task and not solution:
   fn=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name==task)
   code=code.split('\n',1)[0]+'\n    """'+ast.get_docstring(fn)+'"""\n    # TODO: implement the contract; preserve shape and gradient requirements.\n    raise NotImplementedError("'+task+'")'
  cells.append(nbf.v4.new_code_cell(code))
  if task:cells.extend([nbf.v4.new_markdown_cell('### CHECK — do not edit'),nbf.v4.new_code_cell(checks[task])])
 cells.append(nbf.v4.new_markdown_cell('## Visible TGN baseline from lesson 102\n\nThe full one-layer model and its released trainer are included below for independent reading. The shared comparison uses `TGN` with the common trainer that follows. Auxiliary names receive a `TGN_` prefix to keep both implementations live in this notebook. The notebook routes TGN\'s strict cutoff through your live `strict_prefix` task too, replacing its equivalent NumPy lower-bound call. These are complete state updates, not a packaged baseline.'))
 for chunk in re.split(r'^# %% ',tgn,flags=re.M)[1:]:
  title,code=chunk.split('\n',1);cells.extend([nbf.v4.new_markdown_cell('### PROVIDED · TGN · '+title),nbf.v4.new_code_cell(code.strip())])
 for chunk in re.split(r'^# %% ',comparison,flags=re.M)[1:]:
  title,code=chunk.split('\n',1);cells.extend([nbf.v4.new_markdown_cell('### PROVIDED · '+title),nbf.v4.new_code_cell(code.strip())])
 cells.extend([nbf.v4.new_markdown_cell('## Predict, then execute your live comparison\n\nPredict whether one epoch can establish which architecture is generally better. It cannot: this is a small implementation experiment. Both models receive the same retained graph and negative destinations. Your corrected sampler, encoder and attention function are on the TGAT training path.'),
 nbf.v4.new_code_cell("torch.set_num_threads(1)\nDATA_DIRECTORY=Path('l103-cache')\nnode_features,edge_features,full_data,data_audit=load_wikipedia(DATA_DIRECTORY)\ndisplay(pd.DataFrame([data_audit['counts']]))\nteaching=common_data(full_data,train_count=600,eval_count=120)\nfresh=run_comparison(node_features,edge_features,teaching,seed=0,epochs=1,output=Path('l103-teaching'))\ndisplay(pd.DataFrame([{'model':r['arm'],'test_AP':r['test']['ap'],'selected_epoch':r['selected_epoch']} for r in fresh]))\nprint('MEASURED fresh teaching run; historical comparison INCOMPARABLE')"),
 nbf.v4.new_markdown_cell('## EXIT TICKET\n\nSubmit your three functions, the fresh result table and the five-part written defense in section 9. Explain one source defect without treating a corrected rerun as the same experiment. Prepared solution execution does not establish mastery. **PENDING_WRITTEN_DEFENSE**.'),
 nbf.v4.new_markdown_cell('## NEXT STEP — complete released Wikipedia replay\n\nThe same visible TGAT model and trainer run all ten independent full-data seeds below. No architecture downscale: two layers, two heads, 20 sampled neighbors, 172 feature width, up to 50 epochs with released early stopping. This can take hours; the unattended Modal command is in `labs/l103-reproduction.md`. The switch defaults OFF. Historical protocol remains INCOMPARABLE; full-paper parity NOT_ESTABLISHED even if the chosen AP cells are numerically close.'),
 nbf.v4.new_code_cell("RUN_PAPER_REPRO=False\nif RUN_PAPER_REPRO:\n    device='cuda' if torch.cuda.is_available() else 'cpu'\n    paper_records=[run_training(node_features,edge_features,full_data,seed=seed,epochs=50,output=Path('l103-paper')/f'seed-{seed}',device=device) for seed in range(10)]\n    display(pd.DataFrame([{'seed':r['seed'],'all_AP':r['test']['ap'],'new_AP':r['new_test']['ap']} for r in paper_records]))\nelse:\n    print('Full replay NOT_RUN in this kernel. See separately labeled author evidence.')")])
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
 for i,c in enumerate(cells):c.id=f'l103-{i:03d}'
 path=P/('solutions' if solution else '')/f'{S}.ipynb';path.parent.mkdir(exist_ok=True)
 if solution and path.exists():
  old=nbf.read(path,as_version=4);nb.metadata=old.metadata;oldcode=[c for c in old.cells if c.cell_type=='code'];newcode=[c for c in cells if c.cell_type=='code']
  if [c.source for c in oldcode]==[c.source for c in newcode]:
   for a,b in zip(newcode,oldcode):a.outputs=b.outputs;a.execution_count=b.execution_count;a.metadata=b.metadata
 nbf.write(nb,path)
ref='''# TGAT time-attention reference

**Query:** a node and a time. **History:** timestamped interactions. **Embedding:** a computed vector. **Persistent memory:** per-node recurrent state, present in TGN and absent in TGAT. **Parameters:** shared learned weights.

| Operation | Contract |
|---|---|
| Temporal eligibility | Event time strictly less than this query's cutoff |
| Recursive request | Child cutoff is the connecting interaction time, not the root time |
| Paired Fourier map | [cos(ωt), sin(ωt)] per frequency; paired inner products depend on time differences |
| Released time map | cos(ωΔt+b); learned frequency/phase, no explicit paired features |
| Root token | Previous-layer node representation, zero edge, Φ(0) |
| Neighbor token | Child representation at event time, edge features, Φ(elapsed) |
| Attention | softmax(qKᵀ/√head_width + mask) V; time changes keys and values |
| Two-layer Wikipedia | d=172, 20 neighbors, query/key width 516, two heads of width 258 |
| Endpoint prediction | Shared TGAT → concatenate two 172-vectors → pair MLP → sigmoid |
| Historical evidence | Original paper, released code and modern replay need separate identities |

**Counterexamples:** history [1,3,5] at cutoff 4 should return [1,3], but release returns [1]. A child at time 5 may not use an interaction at 6 even when root time is 8. All-padding finite-mask attention is uniform, not zero; fixed time-zero padding can break whole-model clock-shift invariance. Cosine time features do not enforce monotone decay.

[Full lesson](../lessons/0103-tgat.html) · [Protocol](../labs/l103-reproduction.md) · [Paper](https://arxiv.org/html/2002.07962v1)
'''
(R/'reference/tgat-time-attention.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TGAT reference</title><link rel="stylesheet" href="../assets/lesson.css"></head><body><article>'+render(ref)+'</article></body></html>')
print('Built lesson 103, student/solution notebooks and reference')
