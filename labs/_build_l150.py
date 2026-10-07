"""Canonical checkpoint lesson and portable live-function notebooks."""
import ast,base64,hashlib,json,re,zlib
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l150';S='0150-q3-reproduction-checkpoint';TITLE='Q3 checkpoint: defend a full reproduction'
def defs(path):
 s=path.read_text();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
functions=defs(P/'relkit/checkpoint_l150.py');checks=defs(P/'_check_l150.py');provided=defs(P/'relkit/reproduction_l143.py');summary=json.loads((E/'summary.json').read_text())
results='| Track | Validation MAE | Test MAE | Historical score gate |\n|---|---:|---:|---|\n| Paper Table 2 | — | 3.798 | Published target |\n'
replay=next(r for r in summary['records'] if r['track']=='replay')
results+=f"| Checkpoint replay | {replay['val_mae']:.6f} | {replay['test_mae']:.6f} | Separate inference lane |\n"
for t in ['reference','selected']:
 r=summary['tracks'][t];results+=f"| {t.title()}, five full fits | {r['val_mean']:.6f} ± {r['val_sample_sd']:.6f} | {r['mean']:.6f} ± {r['sample_sd']:.6f} | {r['gates']['paper_score']} |\n"
results+='\nMean ± sample SD. All **15,346 predictions** across ten primary fits, three validation-only search fits and one replay were independently aligned and rescored. **Protocol: PASS · Historical identity: NOT_ESTABLISHED · Current competitive standing: NOT_ESTABLISHED · Whole paper: NOT_RUN · Learner: PENDING_WRITTEN_DEFENSE.**\n\n<details><summary>Inspect every primary seed</summary>\n\n| Track | Seed | Test MAE |\n|---|---:|---:|\n'
for r in summary['records']:
 if r['track'] in ['reference','selected']:results+=f"| {r['track']} | {r['seed']} | {r['test_mae']:.6f} |\n"
results+='\n</details>\n'
search='| Learning rate | First minimum validation MAE |\n|---:|---:|\n'+''.join(f"| {r['lr']} | {r['selection_mae']:.6f} |\n" for r in summary['search']['candidates'])+f"\nFrozen winner: **{summary['search']['lr']}**. These are checkpoint-selection scores, not the final resampled validation scores.\n"
diagnosis=json.loads((E/'diagnosis.json').read_text());gradients=f"The independent real first-batch audit matched **{sum(diagnosis['matched_nonfinite_gradients'].values())} nonfinite gradient entries**, with maximum finite-entry difference **{diagnosis['max_finite_gradient_error']:.2g}**. This supports source agreement on that batch while exposing a gradient-health limitation; it does not establish healthy training."
captions={'gates':'Four independent questions: complete protocol, historical score closeness, current comparable performance and learner defense.','architecture':'RelGNN end to end: query-owned temporal sampling, row and time encoders, atomic routes, destination attention and scalar driver prediction.','selection':'Only validation enters candidate selection. A frozen choice precedes the five fresh selected fits.','results':'Every primary seed and its track mean with sample SD, separate from published mean and checkpoint replay.'}
def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results).replace('[[SEARCH]]',search).replace('[[GRADIENTS]]',gradients)
 for name,caption in captions.items():
  folder='l143' if name=='architecture' else 'l150'
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/{folder}/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/{folder}/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption} Scroll horizontally on narrow screens.</figcaption></figure>')
 fallback='Illustrative baseline: five complete seeds, mean3.90 → protocol PASS, historical score CLOSE, competitive standing NOT_ESTABLISHED, learner PENDING_WRITTEN_DEFENSE.'
 s=s.replace('[[GATE_WIDGET]]',fallback if portable else '<div class="route-widget" id="l150-gates"></div><noscript>'+fallback+'</noscript>')
 s=s.replace('[[WARMUP]]','Recall the three opening questions before continuing.' if portable else '<div id="warmup"></div>')
 s=s.replace('[[PREDICT]]','Write your prediction before reading the table: which evidence establishes fresh training, and which establishes historical identity?' if portable else '<div id="predict"></div><noscript>Collecting all fresh seeds supports complete training; neither lane establishes historical identity.</noscript>')
 s=s.replace('[[TEACHBACK]]','Complete the written EXIT and submit it for review.' if portable else '<div id="teachback"></div>')
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s
def doc(title,body,interactive=False):
 body=re.sub(r'<table([^>]*)>',r'<div class="route-scroll" tabindex="0"><table\1>',render(body)).replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','checkpoint-gates'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 150 — '+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/atomic-route.css"><link rel="stylesheet" href="../assets/checkpoint.css"></head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0149-weakest-relbench-tasks.html">Lesson 149</a></nav><header><p class="route-kicker">Year 4 · Quarter 3 · Lesson 150</p><h1>'+title+'</h1></header>'+body+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(doc(TITLE,prose(),True))
reference='''## Four questions, four verdicts
1. **Protocol:** all planned fits, complete populations, validation-only choice and key/temporal/source checks.
2. **Historical score:** five-seed mean within ±0.20 of raw MAE3.798. Descriptive tolerance, not current SOTA.
3. **Competitive standing:** comparable current metric, data-state and tuning/refit protocol. Current nMAE leaderboard is not directly comparable.
4. **Learner defense:** independently reviewed explanation and artifacts. Author execution cannot pass this gate.

## Frozen selection
Three learning rates, seed100, ten full epochs each. First validation minimum per fit; lower rate breaks candidate ties. No test split in search. Freeze result hashes and seed IDs before five selected refits.

'''+search+'\n## Measured evidence\n'+results+'\n'+gradients+'''

## Report and limits
Report every seed, mean and sample SD, source/data hashes, deviations and cost. Source parity is not gradient health. Temporal filtering does not reconstruct feature-arrival histories. Earlier exposure to this task makes the course search exploratory. Preserve counter-evidence and propose one falsifiable next experiment.

[Lesson](../lessons/0150-q3-reproduction-checkpoint.html) · [Protocol](../labs/l150-reproduction.md) · [Report template](../labs/l150-report-template.md). Submit the 250–400-word defense to the teaching agent.
'''
(R/'reference/q3-reproduction-checkpoint.html').write_text(doc('Checkpoint reference',reference))
source_files={p.name:p.read_text() for p in (P/'sources/l141').iterdir() if p.is_file()};raw=(E/'portable.npz').read_bytes()
packet='''# PROVIDED: author evidence and pinned source oracle; not learner execution.
PACKET=base64.b64decode('''+repr(base64.b64encode(raw).decode())+''')
assert hashlib.sha256(PACKET).hexdigest()=='''+repr(hashlib.sha256(raw).hexdigest())+'''
DATA=np.load(io.BytesIO(PACKET))
SOURCES=json.loads(zlib.decompress(base64.b64decode('''+repr(base64.b64encode(zlib.compress(json.dumps(source_files).encode())).decode())+''')))
SOURCE_ROOT=Path('sources/l141');SOURCE_ROOT.mkdir(parents=True,exist_ok=True)
for name,text in SOURCES.items():(SOURCE_ROOT/name).write_text(text)
AUTHOR_SUMMARY='''+repr(summary)+'\n'
bootstrap='''# @colab-bootstrap — portable CPU mechanism and evidence audit by default.
import os,sys,subprocess
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable,'-m','pip','install','-q','pytorch-frame==0.2.3','torch-geometric==2.6.1','relbench==1.1.0','sentence-transformers==3.3.1'])
from pathlib import Path
import base64,hashlib,io,json,zlib,math,statistics
import pandas as pd
from IPython.display import display
RUN_FULL_REPRODUCTION=False
VALIDATE_ONE=False;PREPARED_ROOT=None
'''
core=(P/'relkit/relgnn_l143.py').read_text();imports=core.split('def get_atomic_routes',1)[0];visible=core[core.index('def get_atomic_routes'):]
parity=(P/'_parity_l143.py').read_text().split("if __name__=='__main__':",1)[0].replace('    from relkit.relgnn_l143 import RelGNNConv\n','')
fixture=(P/'_fixture_l143.py').read_text().split("if __name__=='__main__':",1)[0]
trainer='\n'.join(line for line in (P/'_full_l150.py').read_text().splitlines() if line.strip() not in ['from relkit.relgnn_l143 import RelGNN_Model,get_atomic_routes,keyed_mae','from _parity_l143 import original_modules','from relkit.reproduction_l143 import first_validation_min','from relbench.modeling.graph import make_pkey_fkey_graph,get_node_train_table_input'])+'\n';ast.parse(trainer)
compat=(P/'_compat_l143.py').read_text().split("if __name__",1)[0].replace('from _full_l143 import sha,full_run\n','').replace('from relkit.reproduction_l143 import verify_numeric_layout\n','')
scoring='''# CHECK: live functions evaluate every real author prediction vector.
rows=[];rng=np.random.default_rng(150)
for record in AUTHOR_SUMMARY['records']:
    phase=record['phase'];splits=['val'] if record['track']=='search' else ['val','test']
    for split in splits:
        prefix=phase+'_'+split+'_';keys=list(zip(DATA[prefix+'entity'],DATA[prefix+'time']))
        order=rng.permutation(len(keys))
        score=keyed_mae(keys,DATA[prefix+'target'],[keys[i] for i in order],DATA[prefix+'pred'][order])
        assert abs(score-record[split+'_mae'])<1e-12
        rows.append(dict(lane=phase,split=split,mae=score,queries=len(keys)))
assert select_candidate(AUTHOR_SUMMARY['search']['candidates'])==AUTHOR_SUMMARY['search']['lr']
report_tracks={}
for track,seeds in [('reference',list(range(5))),('selected',list(range(10,15)))]:
    records=[dict(r,test_mae=next(x['mae'] for x in rows if x['lane']==r['phase'] and x['split']=='test')) for r in AUTHOR_SUMMARY['records'] if r['track']==track]
    report_tracks[track]=summarize_track(records,track,seeds)
    assert report_tracks[track]['mean']==AUTHOR_SUMMARY['tracks'][track]['mean']
    report_tracks[track]['gates']=checkpoint_verdict(True,report_tracks[track]['mean'],False,False)
    assert report_tracks[track]['gates']==AUTHOR_SUMMARY['tracks'][track]['gates']
for phase,history in AUTHOR_SUMMARY['histories'].items():
    if history:
        assert len(history)==10 and all(x['queries']==7453 for x in history)
        assert first_validation_min(history)==1+int(np.argmin([x['val_mae'] for x in history]))
assert sum(r['queries'] for r in rows)==15346
display(pd.DataFrame(rows))
checkpoint_report=dict(evidence_owner='AUTHOR_PACKET_INDEPENDENTLY_RESCORED',tracks=report_tracks,selected_lr=select_candidate(AUTHOR_SUMMARY['search']['candidates']),historical_identity='NOT_ESTABLISHED',competitive='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE',written_defense=None)
Path('l150-checkpoint-report.json').write_text(json.dumps(checkpoint_report,indent=2))
print(checkpoint_report)
'''
gate='''# PROVIDED: full reproduction is explicit, pinned and separate from author-packet scoring.
if RUN_FULL_REPRODUCTION:
    import importlib.metadata as md
    for name,version in {'torch':'2.5.1','torch-geometric':'2.6.1','pytorch-frame':'0.2.3','relbench':'1.1.0','numpy':'1.26.4','pandas':'2.2.3','sentence-transformers':'3.3.1'}.items():
        assert md.version(name).split('+')[0]==version,(name,'requires pinned runtime')
    assert torch.cuda.is_available(),'Pinned GPU environment required'
    root=Path('l150-runs');root.mkdir(exist_ok=False)
    prepared=Path(PREPARED_ROOT) if PREPARED_ROOT else Path('l150-prepared')
    if PREPARED_ROOT is None:materialize(prepared,SOURCE_ROOT)
    replay_root=Path('l150-compatible');prepare_checkpoint_compatibility(prepared,replay_root)
    replay_result=full_run(root/'replay-compatible',replay_root,SOURCE_ROOT,replay=True,track='replay')
    if VALIDATE_ONE:
        validation_result=full_run(root/'seed-1000',prepared,SOURCE_ROOT,seed=1000,lr=AUTHOR_SUMMARY['search']['lr'],track='notebook')
        print('Execution validation only: one full extra fit plus replay; excluded from primary means')
    else:
        fresh_reference=[full_run(root/f'ref-{seed}',prepared,SOURCE_ROOT,seed=seed,track='reference') for seed in range(5)]
        candidate_results=[full_run(root/f'search-{int(lr*1000):03d}',prepared,SOURCE_ROOT,seed=100,lr=lr,evaluate_test=False,track='search') for lr in [.001,.003,.005]]
        candidates=[dict(lr=r['lr'],seed=100,epochs=r['epochs'],complete=r['status']=='COMPLETE',split='val',selection_mae=r['selection_mae']) for r in candidate_results]
        selected_lr=select_candidate(candidates)
        frozen=dict(lr=selected_lr,candidates=candidates,seeds=list(range(10,15)),files={str(p):sha(p) for p in root.glob('search-*/result.json')})
        (root/'frozen.json').write_text(json.dumps(frozen,indent=2))
        for path,digest in frozen['files'].items():assert sha(Path(path))==digest
        fresh_selected=[full_run(root/f'selected-{seed}',prepared,SOURCE_ROOT,seed=seed,lr=selected_lr,track='selected') for seed in range(10,15)]
        own_tracks={}
        for track,seeds,outputs in [('reference',list(range(5)),fresh_reference),('selected',list(range(10,15)),fresh_selected)]:
            records=[dict(seed=r['seed'],track=track,complete=r['status']=='COMPLETE',epochs=r['epochs'],test_mae=r['scores']['test']['mae']) for r in outputs]
            own_tracks[track]=summarize_track(records,track,seeds)
            own_tracks[track]['gates']=checkpoint_verdict(True,own_tracks[track]['mean'],False,False)
        Path('l150-own-run-report.json').write_text(json.dumps(dict(tracks=own_tracks,selection=frozen,historical_identity='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE'),indent=2))
else:print('Default lane: full fresh training NOT_RUN; author evidence independently rescored.')
'''
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson 150 · '+TITLE+'\n\n'+('Teacher solution' if solution else 'Student lab')+' · PROVIDED / TODO / CHECK / EXIT. Tier B full relational task. Three live functions separate selection, aggregation and verdict. Default audit is distinct from the explicit full GPU reproduction.'),nb.v4.new_code_cell(bootstrap),nb.v4.new_code_cell(imports),nb.v4.new_code_cell(packet,metadata={'tags':['data-payload']}),nb.v4.new_code_cell('# PROVIDED: audited compatibility and first-minimum helpers\n'+provided['verify_numeric_layout']+'\n'+provided['first_validation_min'])]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if not section.strip():continue
  cells.append(nb.v4.new_markdown_cell(section))
  tasks=[]
  if section.startswith('## 4'):tasks=[('select_candidate','check_select')]
  if section.startswith('## 5'):tasks=[('summarize_track','check_summary'),('checkpoint_verdict','check_verdict')]
  for name,check in tasks:
   code=functions[name] if solution else functions[name].split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'
   cells.append(nb.v4.new_code_cell('# TODO — used by the real evidence audit and full reproduction.\n'+code))
   cells.append(nb.v4.new_code_cell('# CHECK — run unchanged.\n'+checks['rejects']+'\n'+checks[check]+'\n'+check+'('+name+')\nprint("PASS '+name+'")'))
  if section.startswith('## 3'):
   for part in re.split(r'(?=^# %% )',visible,flags=re.M):
    if part.strip():cells.append(nb.v4.new_code_cell('# PROVIDED — inspect this model stage.\n'+part))
   cells.append(nb.v4.new_code_cell('# PROVIDED/CHECK: complete model fixture and source differential\n'+fixture+'\n'+parity+'\nfixture_report=neural_fixture(RelGNN_Model,get_atomic_routes)\nparity_report=check(SOURCE_ROOT)\nprint(fixture_report);print(parity_report)'))
  if section.startswith('## 6'):cells.append(nb.v4.new_code_cell(scoring))
 cells.append(nb.v4.new_markdown_cell('## Post-EXIT: full-data reproduction appendix\n\nAll load-bearing model, graph construction, preprocessing, compatibility and training code is visible. Set RUN_FULL_REPRODUCTION only in the pinned GPU runtime after reviewing the protocol and cost. Default execution uses saved author predictions. VALIDATE_ONE is an author execution check; leave False for all thirteen training fits. Existing output directories are rejected. Export the report alongside your defense.'))
 cells.append(nb.v4.new_code_cell('# PROVIDED: released graph construction\n'+(P/'sources/l141/relbench__modeling__graph.py').read_text()))
 for part in re.split(r'(?=^def (?:sha|materialize|full_run)\()',trainer,flags=re.M):
  if part.strip():cells.append(nb.v4.new_code_cell('# PROVIDED: frozen full-data procedure\n'+part))
 cells.append(nb.v4.new_code_cell('# PROVIDED: checkpoint feature reconstruction\n'+compat));cells.append(nb.v4.new_code_cell(gate))
 cells.append(nb.v4.new_code_cell("report=dict(status='PASS',predictions=sum(r['queries'] for r in rows),fixture=fixture_report,operator=parity_report,full_training=('VALIDATION_ONE' if VALIDATE_ONE else 'ALL_TRACKS') if RUN_FULL_REPRODUCTION else 'NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')\nPath('l150-report.json').write_text(json.dumps(report,indent=2))\nprint(report)"))
 for i,c in enumerate(cells):c.id=f'l150-{i:03d}'
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}});path=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and path.exists():
  old=nb.read(path,4);notebook.metadata=old.metadata
  previous=[c for c in old.cells if c.cell_type=='code'];current=[c for c in cells if c.cell_type=='code']
  if [c.source for c in previous]==[c.source for c in current]:
   for before,after in zip(previous,current):
    after.outputs=before.outputs;after.execution_count=before.execution_count;after.metadata=before.metadata
 nb.write(notebook,path)
print('Built L150 lesson, reference and both standalone notebooks')
