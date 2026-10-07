"""Build aligned lesson/reference and portable notebooks from canonical source."""
import ast,base64,hashlib,json,re,zlib
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l151';S='0151-classification-portfolio';TITLE='Portfolio entry 1: entity classification'
def defs(path):
 s=path.read_text();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
functions=defs(P/'relkit/portfolio_l151.py');checks=defs(P/'_check_l151.py');summary=json.loads((E/'summary.json').read_text())
results='| Procedure | Validation AUROC points | Test AUROC points | Fixed-recipe score gate |\n|---|---:|---:|---|\n| Published RDL | 68.18 ± 0.49 | 68.60 ± 1.01 | Published target |\n'
for track in ['reference','selected']:
 r=summary['tracks'][track];results+=f"| {track.title()}, five fresh fits | {100*r['val_mean']:.6f} ± {100*r['val_sample_sd']:.6f} | {100*r['mean']:.6f} ± {100*r['sample_sd']:.6f} | {r['verdict']['paper_score']} |\n"
results+='\nMean ± sample SD; the published standard deviation uses the authors’ reporting convention. All primary fits use the full released task.\n\n| Track | Seed | Validation | Test |\n|---|---:|---:|---:|\n'
for r in summary['records']:
 if r['track'] in ['reference','selected']:results+=f"| {r['track']} | {r['seed']} | {100*r['val_auc']:.6f} | {100*r['test_auc']:.6f} |\n"
search='| Learning rate | Checkpoint-selection validation AUROC points |\n|---:|---:|\n'+''.join(f"| {r['lr']} | {100*r['selection_auc']:.6f} |\n" for r in summary['search']['candidates'])+f"\nFrozen course choice: **{summary['search']['lr']}**. Search seed 100 is excluded from both primary means.\n"
if summary['search']['lr']==.0001:search+='\nThe search returned the reference learning rate. Both refit groups therefore use the same model recipe. Their separate seed sets retain distinct reporting roles; a score difference is not a tuning gain.\n'
g=summary['gradient_audit'];counts=summary['first_batch_nonfinite_counts'];audit=f"All **13,779 labels** were independently reconstructed and compared with released SQL. All **{summary['verified_predictions']:,} saved predictions** across ten primary fits and three validation-only candidates were aligned to archive query keys and rescored. The sampler audited **{summary['query_occurrences']:,} query occurrences** with zero future-timestamp violations. Six load-bearing source definitions match the pinned implementation; each fit also passed a real-batch original-model comparison.\n\nFirst-training-batch nonfinite-gradient counts range from **{min(counts.values())} to {max(counts.values())}** across runs. A separate source differential at seed0’s selected checkpoint checked **{g['gradient_tensors']} gradient tensors**, matching **{sum(g['matched_nonfinite_gradients'].values())} nonfinite entries**, with maximum finite-entry difference **{g['max_finite_gradient_error']:.3g}**. These observations must travel with the result: source agreement is not evidence of gradient health.\n"
captions={'contract':'Six entry fields keep the measured result attached to its procedure, audits and claim.','architecture':'Released RDL pipeline, including a worked one-coordinate temporal mean. Illustrative arithmetic is not a measured learned activation.','selection':'Three validation-only candidates, a hash-frozen choice, then five fresh selected fits.','results':'Measured individual seed AUROCs and mean with sample SD. The tolerance band applies only to the reference protocol.'}
def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results).replace('[[SEARCH]]',search).replace('[[AUDIT]]',audit)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l151/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l151/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption} Scroll horizontally on narrow screens.</figcaption></figure>')
 fallback='Illustrative A=.70, B=.71, C=.71: B wins by lower-rate tie break. With any candidate missing, the choice is INCOMPLETE.'
 s=s.replace('[[SELECTION_WIDGET]]',fallback if portable else '<div class="route-widget" id="l151-selection"></div><noscript>'+fallback+'</noscript>')
 s=s.replace('[[WARMUP]]','Answer the following retrieval questions before continuing.' if portable else '<div id="warmup"></div>')
 s=s.replace('[[PREDICT]]','Predict before reading: what evidence supports complete fresh training?' if portable else '<div id="predict"></div><noscript>Five complete fresh fits establish fresh execution; historical identity remains separate.</noscript>')
 s=s.replace('[[TEACHBACK]]','Submit your explanation to the teaching agent for review.' if portable else '<div id="teachback"></div>')
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s
def doc(title,body,interactive=False):
 body=re.sub(r'<table([^>]*)>',r'<div class="route-scroll" tabindex="0"><table\1>',render(body)).replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','portfolio-selection'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 151 — '+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/atomic-route.css"><link rel="stylesheet" href="../assets/checkpoint.css"></head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0150-q3-reproduction-checkpoint.html">Lesson 150</a></nav><header><p class="route-kicker">Year 4 · Quarter 4 · Lesson 151</p><h1>'+title+'</h1></header>'+body+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(doc(TITLE,prose(),True))
reference='''## Classification entry contract
Task/query population → pinned procedure → frozen validation selection → complete seed table → verification/cost → bounded claim.

AUROC is positive-negative rank concordance with half credit for ties. Store it in [0,1]; display percentage points only with units. Five training seeds describe stochastic training variation on one task, not variation across databases.

## Procedure
Reference: seeds 0–4, lr .0001,20 epochs. Course search: lr .00005/.0001/.0002, seed 100,20 epochs; first validation maximum per run, maximum across candidates, lower-rate tie break. Freeze hashes before seeds 10–14. Search never reads test task rows. Final resampled validation can differ from selection validation.

'''+search+'\n## Measured evidence\n'+results+'\n## Audits\n'+audit+'''

## Claim boundaries
Reference tolerance±1point around68.60 is descriptive. A tuned score cannot pass this fixed-protocol gate. Historical identity and feature arrival legality NOT_ESTABLISHED; fresh FE comparison and whole paper NOT_RUN; learner PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/0151-classification-portfolio.html) · [Protocol](../labs/l151-reproduction.md) · [Entry template](../labs/l151-entry-template.md). Submit your report and250–400 word defense to the teaching agent.
'''
(R/'reference/classification-portfolio.html').write_text(doc('Classification portfolio reference',reference))
raw=(E/'portable.npz').read_bytes();source_files={p.name:p.read_text() for p in sorted((P/'sources/l151').iterdir()) if p.is_file()}
packet='''# PROVIDED: author evidence, not your current kernel's fresh training.
PACKET=base64.b64decode('''+repr(base64.b64encode(raw).decode())+''')
assert hashlib.sha256(PACKET).hexdigest()=='''+repr(hashlib.sha256(raw).hexdigest())+'''
DATA=np.load(io.BytesIO(PACKET))
SOURCES=json.loads(zlib.decompress(base64.b64decode('''+repr(base64.b64encode(zlib.compress(json.dumps(source_files).encode())).decode())+''')))
SOURCE_ROOT=Path('sources/l151');SOURCE_ROOT.mkdir(parents=True,exist_ok=True)
for name,text in SOURCES.items():(SOURCE_ROOT/name).write_text(text)
AUTHOR_SUMMARY='''+repr(summary)+'''
LABEL_EXAMPLES='''+repr(json.loads((E/'prepared/samples.json').read_text()))+'\n'
bootstrap='''# @colab-bootstrap — default CPU evidence audit; full training is explicitly gated.
import os,sys,subprocess
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable,'-m','pip','install','-q','pytorch-frame==0.2.3','torch-geometric==2.6.1','relbench==1.1.0','sentence-transformers==3.3.1'])
from pathlib import Path
import base64,hashlib,io,json,zlib,math,statistics
import numpy as np,pandas as pd
from IPython.display import display
RUN_FULL_REPRODUCTION=False
VALIDATE_ONE=False;PREPARED_ROOT=None
'''
core=(P/'relkit/rdl_l117.py').read_text();helpers=(P/'relkit/trial_l139.py').read_text();fixture=(P/'_fixture_l139.py').read_text().split("if __name__=='__main__':",1)[0]
trainer=(P/'_full_l151.py').read_text().replace('from relkit.rdl_l117 import Model,make_pkey_fkey_graph,get_node_train_table_input\n','')
scoring='''# CHECK: your live contracts process all author predictions, not fixture-only inputs.
rows=[];rng=np.random.default_rng(151)
for record in AUTHOR_SUMMARY['records']:
    phase=record['phase'];splits=['val'] if record['track']=='search' else ['val','test']
    for split in splits:
        prefix=phase+'_'+split+'_';keys=list(zip(DATA[prefix+'study'],DATA[prefix+'time']))
        order=rng.permutation(len(keys))
        score=keyed_auc(keys,DATA[prefix+'target'],[keys[i] for i in order],DATA[prefix+'pred'][order])
        assert abs(score-record[split+'_auc'])<1e-12
        rows.append(dict(lane=phase,split=split,auc=score,queries=len(keys)))
assert select_candidate(AUTHOR_SUMMARY['search']['candidates'])==AUTHOR_SUMMARY['search']['lr']
report_tracks={}
for track,seeds in [('reference',list(range(5))),('selected',list(range(10,15)))]:
    records=[dict(r,test_auc=next(x['auc'] for x in rows if x['lane']==r['phase'] and x['split']=='test')) for r in AUTHOR_SUMMARY['records'] if r['track']==track]
    report_tracks[track]=summarize_track(records,track,seeds)
    assert report_tracks[track]['mean']==AUTHOR_SUMMARY['tracks'][track]['mean']
    report_tracks[track]['verdict']=portfolio_verdict(track,True,report_tracks[track]['mean'],False)
    assert report_tracks[track]['verdict']==AUTHOR_SUMMARY['tracks'][track]['verdict']
for phase,history in AUTHOR_SUMMARY['histories'].items():
    assert len(history)==20 and all(x['queries']==11994 for x in history)
for examples in LABEL_EXAMPLES.values():
    for row in examples:assert trial_target(row['start'],row['analyses'],0)==(True,row['target'])
assert sum(r['queries'] for r in rows)==AUTHOR_SUMMARY['verified_predictions']
display(pd.DataFrame(rows))
portfolio_entry=dict(protocol=json.loads(SOURCES['protocol.json']),candidate_selection=AUTHOR_SUMMARY['search'],artifact_hashes=AUTHOR_SUMMARY['hashes'],audit=dict(labels=13779,predictions=AUTHOR_SUMMARY['verified_predictions'],query_occurrences=AUTHOR_SUMMARY['query_occurrences'],gradient=AUTHOR_SUMMARY['gradient_audit']),task='rel-trial/study-outcome',evidence_owner='AUTHOR_PACKET_INDEPENDENTLY_RESCORED',tracks=report_tracks,selected_lr=select_candidate(AUTHOR_SUMMARY['search']['candidates']),metric='AUROC in [0,1]',source_commit='9aa346267c2e1c560bd92da07d6f4ad1ca2f0639',historical_identity='NOT_ESTABLISHED',feature_arrival_legality='NOT_ESTABLISHED',fresh_fe_comparison='NOT_RUN',whole_paper='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE',written_defense=None)
Path('l151-portfolio-entry.json').write_text(json.dumps(portfolio_entry,indent=2))
print(portfolio_entry)
'''
gate='''# PROVIDED: complete fresh reproduction; review host memory, runtime and cost first.
if RUN_FULL_REPRODUCTION:
    import importlib.metadata as md
    for name,version in {'torch':'2.5.1','torch-geometric':'2.6.1','pytorch-frame':'0.2.3','relbench':'1.1.0','numpy':'1.26.4','pandas':'2.2.3','sentence-transformers':'3.3.1'}.items():
        assert md.version(name).split('+')[0]==version,(name,'requires pinned runtime')
    assert torch.cuda.is_available(),'Pinned GPU environment required; author used64 GiB host memory'
    root=Path('l151-runs');root.mkdir(exist_ok=False)
    prepared=Path(PREPARED_ROOT) if PREPARED_ROOT else Path('l151-prepared')
    if PREPARED_ROOT is None:materialize(prepared)
    if VALIDATE_ONE:
        validation_result=full_run(root/'seed-1000',seed=1000,prepared_root=prepared,lr=AUTHOR_SUMMARY['search']['lr'],track='notebook')
        print('One full extra execution-validation fit; excluded from primary means')
    else:
        fresh_reference=[full_run(root/f'ref-{seed}',seed=seed,prepared_root=prepared,track='reference') for seed in range(5)]
        candidate_results=[full_run(root/f'search-{int(lr*1000000):03d}',seed=100,prepared_root=prepared,lr=lr,track='search') for lr in [.00005,.0001,.0002]]
        candidates=[dict(lr=r['lr'],seed=100,epochs=r['epochs'],complete=r['status']=='COMPLETE',split='val',selection_auc=r['selection_auc']) for r in candidate_results]
        selected_lr=select_candidate(candidates)
        frozen=dict(lr=selected_lr,candidates=candidates,seeds=list(range(10,15)),files={str(p):sha(p) for p in root.glob('search-*/result.json')})
        (root/'frozen.json').write_text(json.dumps(frozen,indent=2))
        for path,digest in frozen['files'].items():assert sha(Path(path))==digest
        fresh_selected=[full_run(root/f'selected-{seed}',seed=seed,prepared_root=prepared,lr=selected_lr,track='selected') for seed in range(10,15)]
        own_tracks={}
        for track,seeds,outputs in [('reference',list(range(5)),fresh_reference),('selected',list(range(10,15)),fresh_selected)]:
            records=[dict(seed=r['seed'],track=track,complete=r['status']=='COMPLETE',epochs=r['epochs'],test_auc=r['scores']['test']['roc_auc']) for r in outputs]
            own_tracks[track]=summarize_track(records,track,seeds)
            own_tracks[track]['verdict']=portfolio_verdict(track,True,own_tracks[track]['mean'],False)
        Path('l151-own-run-entry.json').write_text(json.dumps(dict(evidence_owner='OWN_FRESH_RUN',tracks=own_tracks,selection=frozen,historical_identity='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE'),indent=2))
else:print('Default lane: full fresh training NOT_RUN; author evidence independently rescored.')
'''
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson 151 · '+TITLE+'\n\n'+('Teacher solution' if solution else 'Student lab')+' · PROVIDED / TODO / CHECK / EXIT. Tier B: full relational task. Three live functions select, summarize and bound the claim. Default execution independently audits saved author evidence; the full GPU gate runs all thirteen fresh fits.'),nb.v4.new_code_cell(bootstrap),nb.v4.new_code_cell(packet,metadata={'tags':['data-payload']}),nb.v4.new_code_cell('# PROVIDED: label/key/rank helpers\n'+helpers)]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if not section.strip():continue
  cells.append(nb.v4.new_markdown_cell(section))
  if section.startswith('## 2'):
   for part in re.split(r'(?=^# %% )',core,flags=re.M):
    if part.strip():cells.append(nb.v4.new_code_cell('# PROVIDED: inspect this pipeline stage.\n'+part))
   cells.append(nb.v4.new_code_cell('# PROVIDED/CHECK: complete synthetic neural computation, separate from benchmark training\n'+fixture+'\nfixture_report=neural_fixture(Model)[0]\nprint(fixture_report)'))
  tasks={'3':('select_candidate','check_select'),'4':('summarize_track','check_summary'),'5':('portfolio_verdict','check_entry')}
  for number,(name,check) in tasks.items():
   if section.startswith('## '+number):
    code=functions[name] if solution else functions[name].split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'
    cells.append(nb.v4.new_code_cell('# TODO: used in the real evidence report and fresh full gate.\n'+code))
    cells.append(nb.v4.new_code_cell('# CHECK: run unchanged.\n'+checks['rejects']+'\n'+checks[check]+'\n'+check+'('+name+')\nprint("PASS '+name+'")'))
  if section.startswith('## 5'):cells.append(nb.v4.new_code_cell(scoring))
 cells.append(nb.v4.new_markdown_cell('## Post-EXIT: full-data reproduction appendix\n\nThe complete model above and preprocessing/trainer below implement the same released lane. Enable RUN_FULL_REPRODUCTION only in the pinned GPU runtime. Default PREPARED_ROOT=None builds a fresh graph; an explicit root verifies its hash before use. The author used a T4 with64 GiB host RAM; ordinary free Colab may not fit. VALIDATE_ONE is an execution check only. Full notebook execution has no billing guard; the Modal operator reserves cost before dispatch. Source reuse and its license are preserved in the pinned packet.'))
 for part in re.split(r'(?=^def (?:sha|materialize|full_run)\()',trainer,flags=re.M):
  if part.strip():cells.append(nb.v4.new_code_cell('# PROVIDED: full released data/training procedure\n'+part))
 cells.append(nb.v4.new_code_cell(gate))
 cells.append(nb.v4.new_code_cell("report=dict(status='PASS',predictions=sum(r['queries'] for r in rows),fixture=fixture_report,full_training=('VALIDATION_ONE' if VALIDATE_ONE else 'ALL_TRACKS') if RUN_FULL_REPRODUCTION else 'NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')\nPath('l151-report.json').write_text(json.dumps(report,indent=2))\nprint(report)"))
 for i,c in enumerate(cells):c.id=f'l151-{i:03d}'
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}});path=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and path.exists():
  old=nb.read(path,4);notebook.metadata=old.metadata
  previous=[c for c in old.cells if c.cell_type=='code'];current=[c for c in cells if c.cell_type=='code']
  if [c.source for c in previous]==[c.source for c in current]:
   for before,after in zip(previous,current):
    after.outputs=before.outputs;after.execution_count=before.execution_count;after.metadata=before.metadata
 nb.write(notebook,path)
print('Built lesson, reference and both portable notebooks')
