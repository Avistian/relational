"""Build aligned HTML, standalone student/solution notebooks, and reference card."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0102-temporal-graph-networks';TITLE='Temporal Graph Networks: memory without time travel'

def results(kind):
 path=P/('_paper_l102_results.json' if kind=='paper' else '_experiment_l102_results.json')
 if not path.exists():return '**Full ten-run experiment: RUNNING.** No completed mean is claimed.'
 r=json.loads(path.read_text())
 if kind=='paper':
  tab='| Evaluation population | Measured mean AP | Seed sample SD | Paper AP | Numerical verdict |\n|---|---:|---:|---:|---|\n'
  for lane,label in [('all','All test events'),('new','New-node event subset')]:
   x=r['summary'][lane];sd=x['sample_sd_pp'];tab+=f"| {label} | {x['mean_ap_percent']:.4f}% | {sd:.4f} pp | {x['paper_ap_percent']:.2f}% | {x['numerical_verdict']} |\n" if sd is not None else f"| {label} | {x['mean_ap_percent']:.4f}% | unavailable | {x['paper_ap_percent']:.2f}% | {x['numerical_verdict']} |\n"
  tab+=f"\n**Execution: {r['status']} — seeds {r['complete_seeds']}.** Sample SD describes variation across initializations on this fixed dataset/split, not uncertainty across datasets. Historical identity: **INCOMPARABLE**. Full paper: **NOT_ESTABLISHED**."
 else:
  tab='| Seed | Short-run all-event AP | Short-run new-node AP |\n|---|---:|---:|\n'
  for x in r['records']:tab+=f"| {x['seed']} | {100*x['test']['ap']:.4f}% | {100*x['new_test']['ap']:.4f}% |\n"
  tab+='\n**MEASURED teaching evidence; INCOMPARABLE to paper scores.** The subsets have different event populations from the complete target.'
 return tab

captions={'architecture':'Trace the prediction path and the delayed state path. This is the complete released one-layer TGN-attn shape contract.',
          'gru-trace':'A computed scalar GRU illustration: old state 0.4, candidate 0.8005, new state 0.6002 after rounding.',
          'batch-timeline':'Two events share batch-1 memory; batch 2 applies their stored messages through the trainable updater.',
          'results':'Author-reference per-seed AP. Dashed lines are published targets; seeds share one fixed split.'}
def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text()
 for name,cap in captions.items():
  path=P/f'figures/l102/{name}.png'
  if not path.exists():s=s.replace('[[FIG:'+name+']]','');continue
  src='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode() if portable else f'../labs/figures/l102/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="tgn-figure" tabindex="0"><img src="{src}" alt="{cap}"><figcaption>{cap}</figcaption></figure>')
 s=s.replace('[[PAPER_RESULTS]]',results('paper')).replace('[[SMOKE_RESULTS]]',results('smoke'))
 for key,mount,alt in [('WARMUP','l102-warmup','**Retrieval first:** write your answers before opening the feedback.'),('PREDICT','l102-predict','**Commit a prediction:** which messages may update memory before the next batch is scored? Explain the gradient path.'),('TIMELINE','temporal-graph','| Event 2 | Batch size 2 | Batch size 1 |\n|---|---:|---:|\n| Candidate A memory | 0 | 0.3808 |\n| Candidate C memory | 0 | 0 |\n| Illustrative score | 0.5000 | 0.5941 |\n\nUse the linked HTML lesson for the interactive intervention.')]:
  s=s.replace('[['+key+']]',alt if portable else f'<div id="{mount}"></div>')
 if portable:
  s=s.replace('href="../','href="https://avistian.github.io/relational/').replace('](../','](https://avistian.github.io/relational/')
  s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s
head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 102 — {TITLE}</title>'+''.join(f'<link rel="stylesheet" href="../assets/{s}.css">' for s in ['lesson','temporal-lesson','tgn-lesson'])+f'</head><body><article><nav><a href="../index.html">Course</a> · <a href="0101-static-vs-temporal.html">Lesson 101</a></nav><header><p class="tgn-kicker">Year 3 · Quarter 3 · Lesson 102</p><h1>{TITLE}</h1></header>'
(R/'lessons'/f'{S}.html').write_text(head+render(prose()).replace('<table>','<div class="temporal-scroll" tabindex="0" role="region" aria-label="Scrollable evidence table"><table>').replace('</table>','</table></div>')+'<div id="l102-teachback"></div></article>'+''.join(f'<script src="../assets/{s}.js"></script>' for s in ['retrieval-pool','retrieval-bank','predict','teachback','temporal-graph-viz','l102-lesson'])+'</body></html>')
checks={
'last_message_indices':"assert last_message_indices(np.array([2,1,2,1])).tolist()==[3,2], 'Return the final occurrence in sorted node order'\nassert last_message_indices(np.array([],dtype=int)).dtype==np.int64\nassert last_message_indices(np.array([7,7,7])).tolist()==[2]\nprint('PASS: repeated IDs, single node, empty batch')",
'raw_messages':"enc=lambda t: torch.stack([t,t+1],dim=-1)\nown=torch.tensor([[1.,2.]]); other=torch.tensor([[3.,4.]])\nmsg=raw_messages(own,other,torch.tensor([[5.,6.]]),torch.tensor([7.]),enc)\nassert msg.shape==(1,8), 'Two memories, edge and encoded time must all survive'\nassert torch.equal(msg,torch.arange(1.,9.)[None]), 'Check endpoint order and elapsed encoding'\nprint('PASS: message order and shape')",
'update_state':"torch.manual_seed(3)\ngru=nn.GRUCell(3,2); old=torch.zeros(4,2); clock=torch.zeros(4)\nmsg=torch.tensor([[1.,2.,3.],[3.,1.,2.]])\nnew,times=update_state(old,clock,[1,3],msg,torch.tensor([2.,4.]),gru)\nassert torch.equal(old,torch.zeros_like(old)) and torch.equal(clock,torch.zeros_like(clock)), 'Inputs were mutated'\nassert torch.equal(new[[0,2]],old[[0,2]])\nassert times.tolist()==[0.,2.,0.,4.]\nassert torch.allclose(new[[1,3]],gru(msg,old[[1,3]]))\nnew.sum().backward();assert gru.weight_ih.grad.abs().sum()>0, 'Do not detach the candidate update'\nprint('PASS: values, clocks, untouched nodes, mutation safety, GRU gradient')"}
checks['update_state'] += "\ntry:\n    update_state(old,torch.tensor([0.,5.,0.,0.]),[1],msg[:1],torch.tensor([4.]),gru)\nexcept (AssertionError,ValueError):\n    pass\nelse:\n    raise AssertionError('Reject an update older than the node clock')\nempty,empty_clock=update_state(old,clock,[],None,None,gru)\nassert torch.equal(empty,old) and torch.equal(empty_clock,clock)\nprint('PASS: backward-time rejection and empty message set')"
source=(P/'relkit/tgn_l102.py').read_text();chunks=re.split(r'^# %% ',source,flags=re.M)
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 102 — '+TITLE+'\n\n'+('Executed teacher solution. Learner status remains PENDING_WRITTEN_DEFENSE.' if solution else 'Student edition. Three live functions control the model trained below.')),
 nbf.v4.new_code_cell("# @colab-bootstrap — standalone; no course package is required.\nimport sys\nprint('Python',sys.version.split()[0])"),
 nbf.v4.new_markdown_cell('## Runtime and data\n\nCPU author runs use Python 3.12, PyTorch 2.13.0+cpu, NumPy 2.5.0, pandas 3.0.3 and scikit-learn 1.9.0. See the downloadable requirements file. A missing package can be installed with `%pip install torch numpy pandas scikit-learn`; that creates a different, explicitly recorded runtime. First execution downloads a 534 MiB hash-checked data file and writes a processed cache. Allow about 2 GiB RAM for the notebook. Live Colab and fresh installation are NOT_CHECKED.\n\nThe default is a real-data teaching run. The complete named experiment uses the same visible model and trainer, gated after EXIT.'),nbf.v4.new_markdown_cell(prose(True)),nbf.v4.new_markdown_cell('## Visible implementation\n\nSource-informed one-layer TGN-attn port. The original Apache-2.0 release is pinned at `e38cdf85998c6ca077167610dc4e769a688efa95`. This focused implementation exposes all computations used below. Source parity checks are a separate artifact, not a substitute for result reproduction.')]
 for chunk in chunks[1:]:
  title,code=chunk.split('\n',1);code=code.strip();task=next((x for x in checks if 'def '+x+'(' in code),None)
  cells.append(nbf.v4.new_markdown_cell('### '+('TODO' if task and not solution else 'PROVIDED')+' · '+title))
  if task and not solution:
   tree=ast.parse(code);fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==task)
   doc=ast.get_docstring(fun)
   signature=code.split('\n',1)[0]
   code=signature+'\n    """'+doc+'"""\n    # TODO: implement the contract and retain the gradient path where required.\n    raise NotImplementedError("'+task+'")'
  cells.append(nbf.v4.new_code_cell(code))
  if task:cells.extend([nbf.v4.new_markdown_cell('### CHECK — do not edit'),nbf.v4.new_code_cell(checks[task])])
 cells.extend([nbf.v4.new_markdown_cell('## Real data and the live training loop\n\nPredict whether two epochs on 600 events can justify the complete-paper target. The answer does not depend on whether the score looks impressive.'),
 nbf.v4.new_code_cell("torch.set_num_threads(1)\nnode_features,edge_features,data,split_audit=load_wikipedia(Path('l102-cache'))\ndisplay(pd.DataFrame([split_audit['counts']]))\nsmall={k:{f:x[:(600 if k=='train' else 200)] for f,x in d.items()} if k!='full' else d for k,d in data.items()}\nfresh=run_training(node_features,edge_features,small,seed=0,epochs=2,output=Path('l102-teaching'))\ndisplay(pd.DataFrame([{'population':name,'AP':fresh[key]['ap'],'ROC_AUC':fresh[key]['auc']} for name,key in [('all events','test'),('new-node subset','new_test')]]))\nprint('MEASURED teaching run; paper comparison INCOMPARABLE')"),
 nbf.v4.new_markdown_cell('## EXIT TICKET\n\nSubmit your three functions, `l102-teaching/seed-0.json` and the five-part written defense in section 9. Identify exactly which state is restored by the release checkpoint and which state remains from the stopping epoch. Running this prepared solution does not establish mastery. **PENDING_WRITTEN_DEFENSE**.'),
 nbf.v4.new_markdown_cell('## Complete selected paper experiment — explicit long-run gate\n\nTen independent seeds, complete Wikipedia splits, up to 50 epochs and patience five. No training subset. CPU can take hours; a GPU runtime is recommended. This executes the visible implementation above, including your functions. Use a new output directory after any code change. Compare both means against the frozen 0.5 percentage-point tolerance, then separately report the historical-identity gap. All other paper experiments remain NOT_RUN.'),
 nbf.v4.new_code_cell("RUN_PAPER_REPRO = False\nif RUN_PAPER_REPRO:\n    device='cuda' if torch.cuda.is_available() else 'cpu'\n    paper_records=[run_training(node_features,edge_features,data,seed=s,epochs=50,output=Path('l102-paper'),device=device) for s in range(10)]\n    rows=[]\n    for lane,key in [('all','test'),('new','new_test')]:\n        a=np.array([r[key]['ap']*100 for r in paper_records])\n        rows.append({'population':lane,'mean_AP_percent':a.mean(),'sample_SD_pp':a.std(ddof=1),'paper_AP':PAPER_AP[lane],'numerical_verdict':'CLOSE' if abs(a.mean()-PAPER_AP[lane])<=CLOSE_TOLERANCE_PP else 'OUTSIDE_TOLERANCE'})\n    display(pd.DataFrame(rows))\n    print('Selected experiment COMPLETE; historical identity INCOMPARABLE; full paper NOT_ESTABLISHED')\nelse:\n    print('Full selected replay NOT_RUN in this kernel. See separately labeled author-reference evidence.')")])
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'}})
 for i,c in enumerate(cells):c.id=f'l102-{i:03d}'
 path=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and path.exists():
  previous=nbf.read(path,as_version=4);nb.metadata=previous.metadata
  old=[c for c in previous.cells if c.cell_type=='code'];new=[c for c in cells if c.cell_type=='code']
  if [c.source for c in old]==[c.source for c in new]:
   for a,b in zip(new,old):a.outputs=b.outputs;a.execution_count=b.execution_count;a.metadata=b.metadata
 nbf.write(nb,path)
ref='''# TGN memory: a reference card

**Memory** is per-node historical state. **Embedding** is a query-time representation. **Parameters** are shared learned weights. **Pending message** is an observed event waiting for a later differentiable update.

| Operation | Contract |
|---|---|
| Message | Concatenate own state, other state, event features, encoded time since own last update |
| Aggregation | Select the last stream occurrence per node, in sorted-node order |
| GRU | Updated memory = (1 − keep gate) × proposed content + keep gate × old state |
| Training | Recompute from past messages → predict current pairs → queue current positives → detach |
| Neighborhood | Strictly earlier edges; latest ten; mask node-zero padding |
| Evaluation | Freeze weights; consume real events chronologically; reset state between branches |
| Negative pairs | Score against current state; never write hypothetical events into history |
| Checkpoint | Save weights, memory, clocks, pending messages and stream/RNG/optimizer state together |
| Release metric | Equal mean of per-batch AP, one sampled destination per real interaction |
| Evidence | Source parity, numerical CLOSE, historical identity and full-paper coverage are distinct |

**Shapes for Wikipedia:** message 688 → GRU memory 172; query 344, key/value input 516 → attention output 344 → merged embedding 172; pair 344 → decoder 172 → logit 1.

**Release caveats:** all-event test includes new nodes; new-node negative destinations come from the subset’s destination pool; original early-stop checkpoints omit pending messages; explicit seed-per-run modern replay is not the original continuous RNG stream.

[Full lesson](../lessons/0102-temporal-graph-networks.html) · [Protocol and measured coverage](../labs/l102-reproduction.md) · [Rossi et al. v3](https://arxiv.org/html/2006.10637v3)
'''
(R/'reference/tgn-memory.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TGN memory reference</title><link rel="stylesheet" href="../assets/lesson.css"></head><body><article>'+render(ref)+'</article></body></html>')
print('Built L102 lesson, notebooks and reference')
