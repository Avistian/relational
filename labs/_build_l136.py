"""One source narrative, live canonical functions, portable figures and visible training."""
import ast,base64,gzip,hashlib,json,re,textwrap
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;SLUG='0136-leaderboard-literacy';TITLE='Leaderboard literacy: reproduce the score, audit the setup'
source=(P/'relkit/leaderboard_l136.py').read_text();functions={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
checks_source=(P/'_check_l136.py').read_text();checks={n.name:ast.get_source_segment(checks_source,n) for n in ast.parse(checks_source).body if isinstance(n,ast.FunctionDef)}
lb=json.loads((P/'evidence/l136/leaderboard.json').read_text());train=json.loads((P/'evidence/l136/training.json').read_text());provenance=json.loads((P/'_sources_l136.json').read_text())
lbtext='**Complete archive evaluation replay.** Independently rescored **'+f"{lb['predictions_rescored']:,}"+' predictions across 27 task-entry pairs. Exact row-alignment agreement with pinned upstream functions; all task scores agree within `1e−10`.\n\n| Entry | Recomputed mean NMAE | Tasks |\n|---|---:|---:|\n'
for row in sorted(lb['entries'].values(),key=lambda x:x['mean']):lbtext+=f"| {row['name']} | {row['mean']:.12f} | {row['tasks']}/9 |\n"
lbtext+='\n[Full task-by-task audit](../labs/evidence/l136/leaderboard.json). These are measured evaluation results from submitted files, not newly trained models.'
training='**Five fresh full-data fits completed.**\n\n| Split | Fresh mean ± sample SD (MAE) | Paper mean | Descriptive verdict |\n|---|---:|---:|---|\n'
for split,m in train['metrics'].items():training+=f"| {split} | {m['mean']:.6f} ± {m['sample_sd']:.6f} | {m['paper_target']:.3f} | {m['verdict']} |\n"
training+=f"\nIndependently rescored **{train['predictions_independently_rescored']:,}** saved pilot/final predictions. Maximum port-versus-original output difference on identical sampled batches: **{train['maximum_original_output_error']:.3g}**. Primary measured worker resource estimate: **USD {train['worker_resource_usd']:.4f}**; invoice total is not itemized. [Training audit](../labs/evidence/l136/training.json)."
if (P/'_notebook_gpu_l136_results.json').exists():
 g=json.loads((P/'_notebook_gpu_l136_results.json').read_text());training+=f"\n\n**Portable full-training check: {g['status']}.** All {g['code_cells']} code cells and five additional full fits ran in an isolated pinned GPU namespace; their {g['independent_rescore']['predictions']:,} predictions were independently rescored. They validate the runnable notebook and are not pooled into the primary result. Combined measured worker resource estimate: USD {train['worker_resource_usd']+g['resource_usd']:.4f}. Live Colab and deployment remain NOT_CHECKED."
captions={'alignment':'Synthetic keyed-row example: preserve both entity and cutoff, then recover official order.','normalization':'Synthetic fixed-error example: the scale changes NMAE while predictions remain identical.','coverage':'Synthetic two-task board: omitting the harder task produces an invalid partial average.','leaderboard':'Measured archive replay: three complete nine-task entries; no equal-training-budget claim.','regimes':'Synthetic query at time 14: event 12 is hidden by a snapshot frozen at 10 but visible in strictly past rolling history.','training':'Measured primary fresh historical RDL fits, seeds 0–4; sample SD measures fitting variability on one split.'}
def prose(portable=False,student=False):
 text=(R/'lessons/content'/f'{SLUG}.md').read_text().replace('[[KAPSO_COMMIT]]',provenance['kapso_commit']).replace('[[LEADERBOARD_RESULTS]]',lbtext).replace('[[TRAINING_RESULTS]]',training)
 for marker,name in [('ALIGN_CODE','align_predictions'),('METRIC_CODE','regression_score'),('BOARD_CODE','complete_board')]:text=text.replace('[['+marker+']]', 'Implement the contract in the TODO cell below, then run its CHECK.' if student else '```python\n'+functions[name]+'\n```')
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l136/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l136/{name}.svg'
  text=text.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0" style="overflow-x:auto"><img src="{src}" alt="{caption}" style="max-width:100%;height:auto'+(';min-width:580px' if portable else '')+'"><figcaption>'+caption+'</figcaption></figure>')
 for marker,id,fallback in [('SCALE_WIDGET','l136-scale','Baseline MAE 1 / scale 2 = 0.500. Change scale to 4 → NMAE 0.250, with identical predictions.'),('COVERAGE_WIDGET','l136-coverage','Both tasks [0.1, 0.5] → valid mean 0.300. Omit the second → partial mean 0.100, REJECT as incomplete.')]:text=text.replace('[['+marker+']]',fallback if portable else f'<div class="tuning-control" id="{id}"></div><noscript>{fallback}</noscript>')
 text=text.replace('[[WARMUP]]','' if portable else '<div id="warmup"></div>').replace('[[TEACHBACK]]','Write your EXIT audit before consulting the model answer; paste it to the teacher for feedback.' if portable else '<div id="l136-teachback"></div>')
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/');text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def doc(title,text,interactive=False):
 body=re.sub(r'<table([^>]*)>',r'<div class="stream-scroll" tabindex="0"><table\1>',render(text)).replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','teachback','leaderboard-contract-viz'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{s}.css">' for s in ['lesson','tuning-budget'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0135-tuning-on-reg.html">Lesson 135</a></nav><header><p class="stream-kicker">Year 4 · Quarter 2 · Lesson 136</p><h1>'+title+'</h1></header>'+body+'</article>'+''.join(f'<script src="../assets/{s}.js"></script>' for s in scripts)+'</body></html>'
(R/'lessons'/f'{SLUG}.html').write_text(doc(TITLE,prose(),True))
ref='''## Score reproduction contract

Pin entry issue, source commit, task/data revision, prediction bytes, query keys, metric and normalization constants. A valid file has exactly one finite prediction per official query. Align by entity and cutoff, never by row position alone.

## Regression arithmetic

MAE = mean absolute error. NMAE = MAE / official train-target scale (sample SD, ddof=1). Current board = arithmetic mean of exactly nine task NMAEs. Tasks have equal weight; query rows do not. A partial mean is diagnostic only. Record discrepancies between hosted constants and recomputed train statistics.

## Configuration difference report

Compare data/schema, historical feature availability, frozen versus rolling inputs, model, pretraining, selection, tuning budget, seeds and evaluator. Label unavailable information UNKNOWN; do not fill it with assumptions from the current default script.

## Three different claims

1. Evaluation replay: submitted predictions recover the published score.
2. Training reproduction: an identified pipeline regenerates predictions under a declared protocol.
3. Controlled comparison: competing procedures use comparable information and resources.

Each statement requires additional evidence. A validated submission does not prove equal compute or legal features. A selected experiment does not reproduce the entire paper.

## EXIT template

We reproduced ___ from ___ under ___. We did not establish ___. Before attributing the gap to the model, hold ___ fixed and measure ___.

[Lesson](../lessons/0136-leaderboard-literacy.html) · [Exact operators](../labs/l136-reproduction.md) · [Pinned submission source](https://github.com/stanford-star/relbench/blob/584a03d518b2b655580ea8e1cfbbb26bec0a2841/relbench/submit.py)
'''
(R/'reference/leaderboard-literacy.html').write_text(doc('Leaderboard audit field guide',ref))
prior=nb.read(P/'solutions/0135-tuning-on-reg.ipynb',4);bootstrap=next(c.source for c in prior.cells if c.cell_type=='code' and '@colab-bootstrap' in c.source)
# Compressed JSON keeps evidence portable without enormous Python AST literals.
def embedded_json(raw):return repr(base64.b64encode(gzip.compress(raw,mtime=0)).decode())
f1=(P/'evidence/l136/portable-f1.json').read_bytes()
replay='''# PROVIDED: full F1 submitted files in their original order, plus the full author board audit.
import base64,gzip,hashlib,json,io
raw=gzip.decompress(base64.b64decode('''+embedded_json(f1)+'''))
assert hashlib.sha256(raw).hexdigest()=='''+repr(hashlib.sha256(f1).hexdigest())+'''
F1=json.loads(raw)
AUTHOR_AUDIT=json.loads(gzip.decompress(base64.b64decode('''+embedded_json(json.dumps(lb,separators=(',',':')).encode())+''')))
truth=F1['truth'];entry_rows=[];predictions=0
for issue in ['393','380','397']:
    entry=F1[issue]
    aligned=align_predictions(truth['keys'],entry['keys'],entry['pred'])
    score=regression_score(truth['target'],aligned,truth['train_std'])
    assert abs(score['nmae']-entry['reported'])<1e-10
    # Row permutation must preserve the metric when keys travel with predictions.
    reverse=align_predictions(truth['keys'],entry['keys'][::-1],entry['pred'][::-1])
    assert aligned==reverse
    board=complete_board({t:AUTHOR_AUDIT['tasks'][t]['entries'][issue]['nmae'] for t in AUTHOR_AUDIT['canonical_tasks']},AUTHOR_AUDIT['canonical_tasks'])
    assert abs(board-AUTHOR_AUDIT['entries'][issue]['reported'])<1e-12
    entry_rows.append(dict(entry=entry['name'],f1_mae=score['mae'],f1_nmae=score['nmae'],board_mean=board));predictions+=score['count']
display(pd.DataFrame(entry_rows))
print('Directly rescored F1 predictions:',predictions)
print('Board reconstruction uses full author audit; enable full replay below to independently score all tasks.')
Path('l136-report.json').write_text(json.dumps(dict(status='PASS',direct_f1_predictions=predictions,board_task_scores_reaggregated=27,learner='PENDING_WRITTEN_DEFENSE'),indent=2))
'''
payload={str(path.parent.relative_to(P/'evidence/l136')):dict(base64=base64.b64encode(path.read_bytes()).decode(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),result=json.loads((path.parent/'result.json').read_text())) for path in sorted((P/'evidence/l136').glob('*/lr005-full/seed-*/predictions.npz'))}
training_replay='''# PROVIDED: independently rescore saved fresh-fit evidence, not a new training run.
TRAINING_PAYLOAD=json.loads(gzip.decompress(base64.b64decode('''+embedded_json(json.dumps(payload).encode())+''')))
training_rows=[];count=0
for key,item in TRAINING_PAYLOAD.items():
    raw=base64.b64decode(item['base64']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
    a=np.load(io.BytesIO(raw));r=item['result']
    assert r['selected_epoch']==min(r['trace'],key=lambda t:t['val_mae'])['epoch']
    for split,expected in r['scores'].items():
        score=regression_score(a[split+'_target'],a[split+'_pred'],1.)
        assert abs(score['mae']-expected)<1e-12;count+=score['count']
    if key.startswith('final/'):
        assert len(r['trace'])==10 and all(x['train_queries']==7453 for x in r['trace'])
        training_rows.append(dict(seed=r['seed'],**r['scores']))
assert count==6794 and len(training_rows)==5
display(pd.DataFrame(training_rows));print('Fresh primary fit/pilot predictions rescored:',count)
'''
# The full replay is genuinely standalone; downloads come from frozen source URLs and hashes.
fullreplay='''RUN_FULL_LEADERBOARD_REPLAY = False
if RUN_FULL_LEADERBOARD_REPLAY:
    import urllib.request
    P=Path('l136-replay');S=P/'sources/l136';S.mkdir(parents=True,exist_ok=True)
    FROZEN_SOURCES = '''+repr(provenance)+'''
    for name,meta in FROZEN_SOURCES['files'].items():
        path=S/name
        if not path.exists():path.write_bytes(urllib.request.urlopen(meta['url'],timeout=90).read())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256'],name
    (P/'_sources_l136.json').write_text(json.dumps(FROZEN_SOURCES,indent=2))
'''
operator=(P/'_replay_l136.py').read_text().replace('from relkit.leaderboard_l136 import align_predictions,regression_score,complete_board\n','').replace("P=Path(__file__).resolve().parent;S=P/'sources/l136';D=P/'results/l136/leaderboard-data';D.mkdir(parents=True,exist_ok=True)","P=Path('l136-replay');S=P/'sources/l136';D=P/'results/l136/leaderboard-data';D.mkdir(parents=True,exist_ok=True)")
fullreplay+=textwrap.indent(operator,'    ')+'\nelse:\n    print("Full nine-task replay NOT_RUN in this kernel; full author audit is provided above.")\n'
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson 136 · '+TITLE+'\n\n'+('Teacher reference' if solution else 'Student lab')+' · Three live audit functions, full author evidence, optional complete replay and full five-seed training. Default notebook scoring is explicitly separated from fresh training.'),nb.v4.new_code_cell(bootstrap),nb.v4.new_code_cell('import math,json,hashlib\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\nimport torch\nfrom IPython.display import display\ntorch.set_num_threads(1)\n'+checks['rejects'])]
 for section in re.split(r'(?=^## )',prose(True,not solution),flags=re.M):
  if not section.strip():continue
  cells.append(nb.v4.new_markdown_cell(section))
  for prefix,name,check in [('## 2','align_predictions','check_align'),('## 3','regression_score','check_metric'),('## 4','complete_board','check_board')]:
   if section.startswith(prefix):
    body=functions[name] if solution else functions[name].splitlines()[0]+'\n    raise NotImplementedError("TODO: '+name+'")'
    cells.extend([nb.v4.new_code_cell(body),nb.v4.new_code_cell(checks[check]+'\n'+check+'('+name+')\nprint("PASS: '+name+'")')])
  if section.startswith('## 5'):cells.extend([nb.v4.new_markdown_cell('### CHECK · Use your three functions on actual submitted evidence\n\nF1 predictions are embedded with hashes. All other task scores below come from the explicitly labeled complete author audit.'),nb.v4.new_code_cell(replay)])
  if section.startswith('## 7'):cells.append(nb.v4.new_code_cell(training_replay))
 cells.extend([nb.v4.new_markdown_cell('## Optional · Complete nine-task evaluation replay\n\nOFF by default. This downloads official prediction archives and pinned train/test task tables, scores all 27 task-entry pairs, checks upstream alignment and metrics, and audits normalization. It uses your live functions. Allow several minutes and several hundred MB of download/cache space; it does not download full databases or train any model. The operator is visible below.'),nb.v4.new_code_cell(fullreplay)])
 cells.append(nb.v4.new_markdown_cell('## PROVIDED · Full selected RDL model and trainer\n\nThis is the released basic RDL port, not the architecture of Kapso or RT-PluRel. Read typed feature encoding → relative-time addition → two relation-specific GraphSAGE layers → seed-only head. The blocks below are executable code, not a hidden import. Source: RelBench commit 9aa346267c2e1c560bd92da07d6f4ad1ca2f0639, MIT.'))
 for chunk in re.split(r'^# %% ',(P/'relkit/rdl_l117.py').read_text(),flags=re.M)[1:]:
  heading,body=chunk.split('\n',1)
  if heading.startswith('Full released-protocol training loop'):continue
  cells.extend([nb.v4.new_markdown_cell('### PROVIDED · '+heading),nb.v4.new_code_cell(body.strip())])
 cells.append(nb.v4.new_code_cell((P/'relkit/batch_audit_l123.py').read_text()))
 trainer=(P/'relkit/tuning_train_l135.py').read_text().replace('from relkit.rdl_l117 import Model, get_node_train_table_input, NeighborLoader, seed_everything, CONFIG as BASE_CONFIG','from torch_geometric.loader import NeighborLoader\nfrom torch_geometric.seed import seed_everything\nBASE_CONFIG = '+repr(dict(channels=128,num_layers=2,batch_size=512,lr=.005,epochs=10,fanout=[128,64],aggr='sum',temporal_strategy='uniform'))).replace('    from relkit.batch_audit_l123 import audit_batch\n','')
 cells.extend([nb.v4.new_markdown_cell('### PROVIDED · Full validation-selected trainer\n\nTrace the optimizer, seed-only L1 loss, full-query epoch, first-best checkpoint, restored evaluation, temporal batch audit and original-model parity. Here the configuration is fixed; this lesson does not repeat L135’s tuning search.'),nb.v4.new_code_cell(trainer)])
 constants='TEXT_SPEC = '+repr(json.loads((P/'sources/l117/text_model.json').read_text()))+'\nORIGINAL_MODEL_SOURCE = '+repr((P/'sources/l117/model.py').read_text())
 cells.extend([nb.v4.new_markdown_cell('## Optional · Full five-seed historical training reproduction\n\nOFF by default. Requires the exact documented GPU runtime. Runs all five seeds on full data through the visible model/trainer. Your `regression_score` checks every resulting prediction. Use the Modal operator for enforced dollar reservations; this notebook switch alone does not enforce a dollar cap. The independent GPU check of this notebook is separate from the primary fits.'),nb.v4.new_code_cell(constants),nb.v4.new_code_cell((P/'_notebook_gate_l136.py').read_text())])
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}})
 dest=P/('solutions' if solution else '')/f'{SLUG}.ipynb'
 if dest.exists():
  old=nb.read(dest,4);oldcode=[c for c in old.cells if c.cell_type=='code'];newcode=[c for c in notebook.cells if c.cell_type=='code']
  if [c.source for c in oldcode]==[c.source for c in newcode]:
   for a,b in zip(newcode,oldcode):a.outputs=b.outputs;a.execution_count=b.execution_count
 for i,c in enumerate(notebook.cells):c.id=f'l136-{i:03d}'
 nb.write(notebook,dest)
print('Built lesson/reference and standalone student/solution notebooks')
