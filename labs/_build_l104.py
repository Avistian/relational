"""Single-source lesson, standalone notebook pair and reference card."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0104-information-leakage-in-time';TITLE='Information leakage in time: audit the prediction'
CAP={'clocks':'Same historical event, different arrival. At query day 8, the day-9 correction must be excluded even though its event date is day 3.', 'recursion':'Trace 8 → 5 → 2. A record at 6 is illegal for the child requested at 5, even though it predates the root query.', 'maturity':'Targets must be complete when fitting starts. Both queries are old, but only query A has a mature target at day 10.', 'results':'Fixed weights and prediction questions; each dot is one seed’s paired pooled-AP difference from the corrected strict baseline. Both illegal arms remain invalid regardless of score direction.'}
def measured():
 path=P/'_analysis_l104_results.json'
 if not path.exists():return '**Full author evaluation: RUNNING.** No full-population intervention result is claimed yet. The pilot is excluded from scientific results. Upstream seed 8 was pending at launch.'
 r=json.loads(path.read_text());s=f'**Fresh evaluation: {r["status"]}, {len(r["records"])}/10 checkpoints.** Training was reused from L103.\n\n'
 s+='| Population | Strict pooled AP | Same-time AP change | One-day AP change |\n|---|---:|---:|---:|\n'
 for lane,label in [('all','All test events'),('new','New-node subset')]:
  a=r['summary'][lane];s+=f'| {label} | {a["strict_mean_ap_percent"]:.4f}% | {a["inclusive"]["mean_delta_pp"]:+.4f} ± {a["inclusive"]["sample_sd_pp"]:.4f} pp | {a["lookahead"]["mean_delta_pp"]:+.4f} ± {a["lookahead"]["sample_sd_pp"]:.4f} pp |\n'
 s+='\n± is sample SD across seeds, not a confidence interval. '+f'Released replay checked **{r["replayed_positive_events"]:,} positive events**, each with one negative, with maximum probability error **{r["max_prediction_error"]:.3g}**. Strict sampled nonpast accesses: **0**.\n\n'
 s+='**Paper-target check (separate aggregation):**\n\n| Released population | Fresh batch-mean AP ± seed SD | Original paper target | Numerical verdict |\n|---|---:|---:|---|\n'
 for lane,label in [('all','All test events'),('new','New-node subset')]:
  a=r['summary'][lane];s+=f'| {label} | {a["release_mean_ap_percent"]:.4f}% ± {a["release_sample_sd_pp"]:.4f} pp | {a["paper_target_percent"]:.2f}% | {a["numerical_verdict"]} |\n'
 s+='\nThe predeclared ±0.5 pp numerical tolerance does not establish matching historical populations or full-paper parity. **Historical identity INCOMPARABLE; full-paper parity NOT_ESTABLISHED.** [Machine-readable evidence](../labs/_analysis_l104_results.json).'
 return s

def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text()
 for name,cap in CAP.items():
  if not (P/f'figures/l104/{name}.png').exists():s=s.replace('[[FIG:'+name+']]','');continue
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l104/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l104/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="audit-figure" tabindex="0"><img src="{src}" alt="{cap}"><figcaption>{cap}</figcaption></figure>')
 for key,id_,fallback in [('WARMUP','l104-warmup','**Answer the cold retrieval prompts before reading.**'),('CLOCK_WIDGET','l104-clock','**Portable state:** event day 3, arrival day 9, query day 8 → BLOCK. Change arrival to 7 → ALLOW. The interactive lesson lets you vary arrival time.'),('PREDICT','l104-predict','**Predict first:** must access to future information increase AP for fixed weights?'),('TEACHBACK','l104-teachback','**Teach back:** explain the corrected number and the provenance limits in your own words before consulting the solution.')]:
  s=s.replace('[['+key+']]',fallback if portable else f'<div id="{id_}"></div>')
 source=(P/'relkit/leakage_l104.py').read_text();tree=ast.parse(source)
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AuditFinder')
 fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_temporal_neighbor')
 snippet='\n'.join(source.splitlines()[fn.lineno-1:fn.end_lineno])
 s=s.replace('[[CODE:sampling]]','```python\n'+snippet+'\n```')
 s=s.replace('[[RESULTS]]',measured())
 if portable:
  s=re.sub(r'<details><summary>(.*?)</summary>',r'**\1**\n\n',s).replace('</details>','')
  s=s.replace('href="../','href="https://avistian.github.io/relational/').replace('](../','](https://avistian.github.io/relational/')
  s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s
head='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 104 — '+TITLE+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','temporal-leakage'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="0103-tgat.html">Lesson 103</a></nav><header><p class="audit-kicker">Year 3 · Quarter 3 · Lesson 104</p><h1>'+TITLE+'</h1></header>'
body=render(prose()).replace('<table>','<div class="audit-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
(R/'lessons'/f'{S}.html').write_text(head+body+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','teachback','availability-audit-viz','l104-lesson'])+'</body></html>')

checks={
'eligible_history':"event=np.array([1.,3.,3.,5.]);arrival=np.array([1.,3.,4.,5.])\nnp.testing.assert_array_equal(eligible_history(event,arrival,3),[True,False,False,False])\nnp.testing.assert_array_equal(eligible_history(event,arrival,3,inclusive=True),[True,True,False,False])\nassert not eligible_history(np.array([1.]),np.array([6.]),5).any()\nassert len(eligible_history(np.array([]),np.array([]),5))==0\nprint('PASS: ties, late arrivals, empty history and explicit inclusive intervention')",
'admissible_training':"queries=np.array([2,4,7,10]);maturity=np.array([9,11,9,10])\nfit=10\nallowed=admissible_training(queries,maturity,fit)\nnp.testing.assert_array_equal(allowed,[True,False,True,False])\ndisplay(pd.DataFrame({'query_day':queries,'label_available_day':maturity,'fit_day':fit,'eligible':allowed}))\nprint('PASS: query age alone does not establish label availability')",
'paired_ap':"a={'e':np.array([1,2]),'negative':np.array([7,8]),'batch':np.array([0,0]),'p':np.array([.9,.8]),'n':np.array([.2,.1])}\nb={**a,'p':np.array([.1,.2]),'n':np.array([.8,.9])}\nassert paired_ap(a,a)['delta_pp']==0\nassert abs(paired_ap(a,b)['changed_ap']-5/12)<1e-12\ntry:\n    paired_ap(a,{**b,'negative':np.array([8,7])})\nexcept ValueError:\n    print('PASS: changed candidates rejected; reversed ranking AP = 5/12')\nelse:\n    raise AssertionError('Reject unpaired negatives before comparing AP')"}
intent={
'eligible_history':'Produce a boolean policy for event and availability arrays. This policy is called inside every AuditFinder request below. Specify both inequalities and the deliberately illegal inclusive mode.',
'admissible_training':'Produce a mask for targets that were available when fitting began. The table below is a synthetic diagnostic fixture with supplied maturity times, which may reflect different horizons; it is not an extra Wikipedia experiment.',
'paired_ap':'Verify identity alignment before comparing pooled AP. This function computes the actual intervention result table below; do not let candidate changes masquerade as a leakage effect.'}
inputs=json.loads((P/'evidence/l104/notebook-inputs.json').read_text())
bootstrap="""# PROVIDED — standalone environment, no relkit import required.
import sys
print('Python',sys.version.split()[0])
# If needed in a fresh runtime: %pip install torch==2.8.0 numpy==2.2.6 pandas==2.3.2 scikit-learn==1.7.1
"""
loadcode="""# PROVIDED — verify the checkpoint before deserializing it.
import urllib.request
INPUT_HASHES=__HASHES__
INPUT_DIR=Path('evidence/l104');INPUT_DIR.mkdir(parents=True,exist_ok=True)
for name,expected in INPUT_HASHES.items():
    path=INPUT_DIR/name
    if not path.exists():
        urllib.request.urlretrieve('https://raw.githubusercontent.com/avistian/relational/main/labs/evidence/l104/'+name,path)
    assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==expected, 'Input hash mismatch; stop and inspect'
torch.set_num_threads(1)
node_features,edge_features,data,data_audit=load_wikipedia(Path('l103-cache'))
EXPECTED_PROCESSED=__PROCESSED__
with np.load(Path('l103-cache')/'processed.npz') as cached:
    for field,record in EXPECTED_PROCESSED.items():
        assert hashlib.sha256(cached[field].tobytes()).hexdigest()==record['sha256'], 'Processed cache mismatch: '+field
checkpoint=torch.load(INPUT_DIR/'seed-0-selected.pt',map_location='cpu',weights_only=False)
archive=np.load(INPUT_DIR/'seed-0-questions.npz')
model=TGAT(NeighborFinder(data['full'],len(node_features),release=True),node_features,edge_features)
missing=model.load_state_dict(checkpoint['weights'],strict=False)
assert set(missing.missing_keys)=={'n_feat_th','e_feat_th','edge_raw_embed.weight','node_raw_embed.weight'} and not missing.unexpected_keys
print('Checkpoint: reused complete L103 seed 0; no training in the default lab')
""".replace('__HASHES__',repr(inputs)).replace('__PROCESSED__',repr(json.loads((P/'_inputs_l104.json').read_text())['processed_arrays']))
labcode="""# PROVIDED — same checkpoint, same question pairs; only information access changes.
questions={k:v[:120] for k,v in records_from_archive(archive,'all').items()}
arm_predictions={};access={}
for mode in ['strict','inclusive','lookahead']:
    finder=AuditFinder(data['full'],len(node_features),mode)
    model.ngh_finder=finder
    arm_predictions[mode]=score_fixed_questions(model,data['test'],questions,rng_seed=104)
    access[mode]=finder.audit.copy()
assert access['strict']['nonpast_records']==0
rows=[]
for mode in arm_predictions:
    rows.append({'arm':mode,**paired_ap(arm_predictions['strict'],arm_predictions[mode]),**access[mode]})
display(pd.DataFrame(rows))
Path('l104-fresh.json').write_text(json.dumps(rows,indent=2))
print('Fresh 120-question checkpoint intervention; not the full author evaluation or fresh training')
"""
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 104 — '+TITLE+'\n\n'+('Teacher solution: executed reference, learner status PENDING_WRITTEN_DEFENSE.' if solution else 'Student edition: implement three TODOs, run each CHECK, then defend your corrected metric.')+'\n\nTier B: real Wikipedia interaction graph. Small synthetic fixtures isolate clock rules. Full model and trainer are visible below; default execution uses an explicitly provided pretrained checkpoint. Embedded figures work without the lesson page.'),nbf.v4.new_code_cell(bootstrap),nbf.v4.new_markdown_cell('## Runtime and evidence\n\nThe pinned author GPU runtime uses Python 3.12, torch 2.8.0, NumPy 2.2.6, pandas 2.3.2, scikit-learn 1.7.1. CPU author execution has separately recorded versions. First execution downloads about 534 MiB raw Wikipedia data and a 10 MiB checkpoint; allow roughly 2 GiB RAM for parsing. Raw and checkpoint hashes are verified. Course input URLs become available when the lesson is published; until then use this local checkout. Live Colab remains NOT_CHECKED. Full author evaluation and notebook execution are distinct.'),nbf.v4.new_markdown_cell(prose(True)),nbf.v4.new_markdown_cell('## PROVIDED · Complete TGAT model and training recipe\n\nThis is the unchanged canonical L103 implementation, including its released neighbor-boundary, batch and checkpoint-selection quirks. Read the recursion carefully. The intervention does not call the release sampler; it replaces it with AuditFinder after restoring the trained weights. Each named chunk is visible, so there is no hidden model import.')]
 for chunk in re.split(r'^# %% ',(P/'relkit/tgat_l103.py').read_text(),flags=re.M)[1:]:
  title,code=chunk.split('\n',1);title=re.sub(r'^Task \d+: ', 'L103 mechanism · ', title);cells.extend([nbf.v4.new_markdown_cell('### PROVIDED · '+title),nbf.v4.new_code_cell(code.strip())])
 audit=re.sub(r'^from relkit\.[^\n]+\n','',(P/'relkit/leakage_l104.py').read_text(),flags=re.M)
 cells.append(nbf.v4.new_markdown_cell('## Your temporal audit\n\nThe following three contracts connect the prediction-time argument to executable checks. Hints specify the required behavior; write your own implementation.'))
 for chunk in re.split(r'^# %% ',audit,flags=re.M)[1:]:
  title,code=chunk.split('\n',1);code=code.strip();task=next((k for k in checks if 'def '+k+'(' in code),None)
  cells.append(nbf.v4.new_markdown_cell('### '+('TODO' if task and not solution else 'PROVIDED')+' · '+title+ ('\n\n'+intent[task] if task else '')))
  if task and not solution:
   fn=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name==task)
   code=code.split('\n',1)[0]+'\n    """'+ast.get_docstring(fn)+'"""\n    raise NotImplementedError("'+task+'")'
  cells.append(nbf.v4.new_code_cell(code))
  if task:cells.extend([nbf.v4.new_markdown_cell('### CHECK · do not edit'),nbf.v4.new_code_cell(checks[task])])
 cells.extend([nbf.v4.new_markdown_cell('## PROVIDED · Reuse a verified full-data checkpoint\n\nCheck the hash before loading the course checkpoint. Its trained weights are a provided input, not evidence that this notebook trained the model. The teacher solution executes locally from the same distributed files.'),nbf.v4.new_code_cell(loadcode),nbf.v4.new_markdown_cell('## Predict, then run\n\nWrite whether each illegal arm will raise or lower pooled AP. All questions and negative candidates remain fixed. A higher score does not prove validity, and a lower score does not erase a temporal violation.'),nbf.v4.new_code_cell(labcode),nbf.v4.new_markdown_cell('## EXIT TICKET\n\nSubmit your functions, measured table, one forbidden path and the written defense in the lesson. State what is still unknown about data availability. A prepared notebook is not learner mastery. **PENDING_WRITTEN_DEFENSE**.'),nbf.v4.new_markdown_cell('## NEXT STEP · full seed-0 checkpoint evaluation\n\nThe same visible model can replay every all/new test question using the saved release RNG state, then run the three complete intervention arms. Default OFF: the small task stays quick. This is fresh inference with reused training. All ten seeds are orchestrated by `labs/_run_l104.py` and `modal/l104_replay.py`, with hashes and budget controls. See the reproduction contract for checkpoint retrieval and per-seed commands.'),nbf.v4.new_code_cell("RUN_COMPLETE_EVALUATION=False\nif RUN_COMPLETE_EVALUATION:\n    model.ngh_finder=NeighborFinder(data['full'],len(node_features),release=True)\n    release_report,_=replay_release(model,data,checkpoint,archive)\n    print(release_report)\n    for lane,split in [('all','test'),('new','new_test')]:\n        predictions={}\n        for mode in ['strict','inclusive','lookahead']:\n            model.ngh_finder=AuditFinder(data['full'],len(node_features),mode)\n            predictions[mode]=score_fixed_questions(model,data[split],records_from_archive(archive,lane),104)\n        print(lane,{mode:paired_ap(predictions['strict'],predictions[mode]) for mode in ['inclusive','lookahead']})\nelse:\n    print('Full-population replay NOT_RUN in this notebook; see separate author evidence.')"),nbf.v4.new_markdown_cell('## OPTIONAL · regenerate the complete trained checkpoints\n\nThis is the full visible L103 trainer, with no architecture or dataset downscale. It can take hours and is outside the paid-compute budget authorized for this lesson. Default OFF. Original split, release quirks, up to 50 epochs and ten independent seeds are retained. This does not establish exact historical identity or reproduce every experiment in TGAT.'),nbf.v4.new_code_cell("RUN_FRESH_TRAINING=False\nif RUN_FRESH_TRAINING:\n    device='cuda' if torch.cuda.is_available() else 'cpu'\n    fresh_training=[run_training(node_features,edge_features,data,seed=s,epochs=50,output=Path('l104-retrained')/f'seed-{s}',device=device) for s in range(10)]\nelse:\n    print('Fresh training NOT_RUN; L103 training reused.')")])
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
 for i,c in enumerate(cells):c.id=f'l104-{i:03d}'
 path=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and path.exists():
  old=nbf.read(path,as_version=4);nb.metadata=old.metadata;before=[c for c in old.cells if c.cell_type=='code'];after=[c for c in nb.cells if c.cell_type=='code']
  if [c.source for c in before]==[c.source for c in after]:
   for a,b in zip(after,before):a.outputs=b.outputs;a.execution_count=b.execution_count;a.metadata=b.metadata
 nbf.write(nb,path)
ref='''# Temporal leakage audit · reference

**Prediction query:** an entity or pair and a time. **Event time:** when something happened. **Availability time:** when the predictor could read that record/version. **Label maturity:** when the entire future target becomes knowable.

| Boundary | Before-event contract | Failure fixture |
|---|---|---|
| Input event | s < query cutoff | Current event or timestamp tie |
| Input availability | a ≤ query cutoff | Day-3 correction arriving day 9 for a day-8 prediction |
| TGAT child | cutoff = connecting event time | Root 8 → child 5 reads event 6 |
| Fitting target | query < fit; label available ≤ fit | Query 4 + horizon 5 + delay 2 > fit 10 |
| Selection | Validation outcomes available before deployment | Day-25 labels select day-20 model |
| Paired comparison | Same event IDs, negatives, weights and metric | Candidate changes attributed to sampler repair |

A snapshot task can allow committed records at its cutoff; document the tie convention. Online observations may become legal history after an earlier prediction. Feature values and caches need their own historical validity. Unknown availability cannot be certified by event timestamps.

**Metric:** pooled AP is computed over the whole evaluated population; batch-mean AP averages batch APs equally. These differ. Percentage-point change = 100 × (changed AP − strict AP). AP direction cannot determine whether information access is legal.

**L104 evidence:** full-data L103-trained TGAT checkpoints, fresh released evaluation and a separate three-arm inference intervention. Same-time and one-day lookahead arms are deliberately invalid. Raw Wikipedia lacks ingestion timestamps. Reused training is not fresh training; selected Wikipedia replay is not full-paper reproduction.

[Lesson](../lessons/0104-information-leakage-in-time.html) · [Protocol](../labs/l104-reproduction.md) · [Kapoor & Narayanan](https://arxiv.org/abs/2207.07048) · [Fey et al.](https://proceedings.mlr.press/v235/fey24a.html)
'''
(R/'reference/temporal-leakage-audit.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Temporal leakage audit reference</title><link rel="stylesheet" href="../assets/lesson.css"></head><body><article>'+render(ref)+'</article></body></html>')
print('Built L104 lesson, notebook pair and reference')
