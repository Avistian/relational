"""Build B14 HTML, reference, portable notebooks and architecture figures."""
import ast,base64,hashlib,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;R=P.parent;S='b14-flattening-challenge';E=P/'evidence/b14';F=P/'figures/b14';SRC=P/'sources/b14';F.mkdir(parents=True,exist_ok=True)
INK='#183b4c';TEAL='#15786f';PURPLE='#71609d'
def flow(name,title,subtitle,steps,footer):
 f,a=plt.subplots(figsize=(11,6.8));f.patch.set_facecolor('#fbfcfd');a.set(xlim=(0,1),ylim=(0,1));a.axis('off')
 a.text(.03,.95,title,fontsize=21,weight='bold',color=INK);a.text(.03,.895,subtitle,fontsize=11,color='#50646e')
 for i,(head,body) in enumerate(steps):
  col=i%3;row=i//3;x=.03+col*.33;y=.54-row*.32
  a.add_patch(FancyBboxPatch((x,y),.285,.25,boxstyle='round,pad=.012',facecolor=['#eaf5f2','#eff0fb','#fff4df'][col],edgecolor='#abc3cb'))
  a.text(x+.008,y+.205,str(i+1)+' · '+head,fontsize=12,weight='bold',color=INK,va='top');a.text(x+.008,y+.15,body,fontsize=10.7,color=INK,va='top',linespacing=1.5)
  if col<2:a.annotate('',xy=(x+.313,y+.125),xytext=(x+.29,y+.125),arrowprops=dict(arrowstyle='->',color=INK,lw=1.6))
 a.annotate('',xy=(.17,.50),xytext=(.83,.52),arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=-90,angleB=0,rad=8',color=INK,lw=1.6))
 a.text(.03,.10,footer,fontsize=10.7,color=TEAL,wrap=True);f.savefig(F/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(f)
steps_r=[('Visible database','Query = (entity, time)\nKeys + typed values\nPin visibility rules'),('Compose DFS','Follow foreign keys\nAggregate linked values\nDepth 2 / 3 / 4'),('Materialize table','Support: Nₛ × F\nQueries: Nq × F\nSame feature definitions'),('Choose backbone','TabPFN v2 / v2.5\nor LimiX-16M\nSupport cap 10,000'),('In-context inference','Feature tokens + labels\nPretrained attention\nQuery labels hidden'),('Select → score','Nine validation candidates\nSelected configuration\n→ held-out predictions')]
flow('rdblearn','RDBLearn · construct first, predict second','Original toolkit paper v1; conceptual backbone block, not a shared architecture for all three models.',steps_r,'DFS constructs relational information. The backbone consumes a table; validation chooses depth and backend.')
steps_t=[('Freeze the phase','Inner DB → val cutoff\nOuter DB → test cutoff\nDFS depths 2 / 3 / 4'),('Build contexts','Text-free feature frame\nRecent pool M = 4K\nK = 100,000; 8 estimators'),('Encode distributions','TabPFN-3 column groups\nInduced attention reads\nvalue distributions'),('Aggregate features','Per-row feature attention\nSummary tokens combine\ncolumn-group information'),('Read support labels','ICL attention over rows\nKnown support labels\nHidden query labels'),('Decode → average','Class probabilities\nCombine estimator outputs\nValidation selects depth')]
flow('tabpfnrel','TabPFN-Rel OSS · a feature harness around TabPFN-3','Pinned release: e890022 · local, text-free · symbolic shapes depend on the selected DFS depth.',steps_t,'Final fit: train + validation. F1 support = 11,977 < K, so the recency-pool branch is inactive.')
flow('snapshot','Two snapshots · match tuning to evaluation','F1 release cutoffs: 2005-01-01 for validation; 2010-01-01 for test.',[
 ('Inner support','11,411 training rows\nKnown training labels\nDB capped at 2005'),('Inner queries','566 validation rows\nEvaluate all 3 depths\nNo validation-label input'),('Select once','Maximum validation AUROC\nDeclared candidate order\nNo test-label selection'),('Outer support','11,977 train + val rows\nKnown final support labels\nDB capped at 2010'),('Outer queries','702 test rows\nLater anchors may change\ntime-since-event features'),('Score independently','Complete (driverId,date)\nNo post-2010 DB events\nAUROC from saved scores')], 'A later query timestamp does not advance the frozen database. Arrival-time validity remains a separate audit.')
r=json.loads((E/'diagnostic.json').read_text())
table='| Features | Predictor | Fixed-snapshot MSE, mean ± SD | Rolling-policy MSE, mean ± SD |\n|---|---|---:|---:|\n'
for feature in ['entity-only','relational']:
 for model in ['ridge','rbf']:
  rows=[x for x in r['conditions'] if (x['features'],x['backbone'])==(feature,model)];a=[x['mse'] for x in rows];b=[x['rolling_mse'] for x in rows]
  table+=f'| {feature} | {model} | {np.mean(a):.5f} ± {np.std(a,ddof=1):.5f} | {np.mean(b):.5f} ± {np.std(b,ddof=1):.5f} |\n'
table+='\nAuthor measurements: 12 fits; 3,072 fixed + 3,072 rolling predictions. Lower MSE is better. Rolling uses later events and is a different information policy; it reuses fitted support.'
f,axes=plt.subplots(1,2,figsize=(10.5,4.5));f.patch.set_facecolor('#fbfcfd')
for ax,feature in zip(axes,['entity-only','relational']):
 for seed in range(3):
  a=[next(x['mse'] for x in r['conditions'] if (x['features'],x['seed'],x['backbone'])==(feature,seed,m)) for m in ['ridge','rbf']]
  ax.plot([0,1],a,'o-',label=f'Seed {seed}',color=[TEAL,PURPLE,'#a76624'][seed])
 ax.set(xticks=[0,1],xticklabels=['Linear ridge','RBF kernel ridge'],title=feature,ylim=(0,1.9),ylabel='Fixed-snapshot test MSE');ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=9)
f.suptitle('Change the predictor on the same feature table',color=INK);f.tight_layout();f.savefig(F/'results.png',dpi=150,bbox_inches='tight');plt.close(f)
paths={'rdblearn':steps_r,'tabpfnrel':steps_t,'snapshot':list(zip(['Inner support','Inner queries','Select once','Outer support','Outer queries','Score independently'],['Freeze the inner database at 2005; train on 11,411 queries.','Score three depths on 566 validation queries.','Select by validation AUROC only.','Refit on 11,977 train + validation queries; freeze the outer database at 2010.','Predict all 702 test queries without advancing the database.','Match full (driverId, date) keys before independently scoring.']))}
def fig(name,caption):
 h=f'<figure class="flat-figure flat-wide"><img src="../labs/figures/b14/{name}.png" alt="{caption}"><figcaption>{caption}</figcaption></figure>'
 if name in paths:h+='<div class="flat-mobile"><strong>'+caption+'</strong><ol>'+''.join('<li><strong>'+a+'</strong><br>'+b.replace('\n',' · ')+'</li>' for a,b in paths[name])+'</ol></div>'
 else:h+='<div class="flat-mobile" data-mobile-results><strong>Every fixed-snapshot MSE</strong><ul>'+''.join(f'<li>Seed{x["seed"]} · {x["features"]} · {x["backbone"]}: {x["mse"]:.5f}</li>' for x in r['conditions'])+'</ul></div>'
 return h
paper=json.loads((E/'paper-status.json').read_text());source=json.loads((E/'rdblearn-preprocessing.json').read_text())
status='**'+paper['status']+'** · '+paper['summary']
if (E/'full-audit.json').exists():
 audit_result=json.loads((E/'full-audit.json').read_text())
 status+='\n\n| Depth | Fresh validation AUROC | Fresh test AUROC | Role |\n|---:|---:|---:|---|\n'
 full=json.loads((E/'full/result.json').read_text())
 for trial in full['trials']:
  test='NOT_EVALUATED' if trial['test_score'] is None else f"{trial['test_score']:.6f}"
  role=('selected' if trial['config_id']==full['selected'] else 'unselected')+(' / default' if trial['config_tag']=='default' else '')
  status+=f"| {trial['config']['max_depth']} | {trial['val_score']:.6f} | {test} | {role} |\n"
 status+='\nFresh single-seed released-checkpoint inference. No uncertainty across seeds is estimated. Unselected nondefault settings have no test score by design.'

source_status='Fresh execution of the original full RDBLearn preprocessor '+('reproduces' if source['status']=='FAIL' else 'does not reproduce')+' the known-category inconsistency in the recorded environment. After fitting b/c/d, an unseen a changes known codes from 0/1/2 to 1/2/3; numeric controls remain unchanged. This is a synthetic source diagnostic, not a measured F1 score effect. The original toolkit reproduction retains its separate source gate.'
time_widget='''<div class="flat-board" data-flat-time><h3>Predict the mean before changing the cutoff</h3><p>Query A/day14 and values [2,6,100,20] remain fixed.</p><label>Database snapshot<select name="snapshot"><option value="8">Day8</option><option value="10" selected>Day10: fixed default</option><option value="14">Day14: rolling</option></select></label><label>Arrival policy<select name="arrival"><option value="check">Check available time</option><option value="ignore">Ignore available time</option></select></label><button type="button">Reset trace</button><output aria-live="polite">At fixed day10, read A1 and A2: count 2, mean 4, last 6.</output><noscript><p>Controls need JavaScript. At day 14 all four records are visible; mean 32. Ignoring arrival at day 10 wrongly includes A3: mean 36.</p></noscript></div>'''
swap_widget='''<div class="flat-board" data-flat-swap><h3>What did the comparison change?</h3><label>Change alongside the predictor<select><option value="backbone">Nothing: predictor only</option><option value="features">More relational features</option><option value="context">More labeled support rows</option><option value="snapshot">More recent database events</option></select></label><button type="button">Reset comparison</button><output aria-live="polite">Only the predictor changes: a controlled backbone comparison.</output><noscript><p>Changing features, support or snapshots prevents attributing the whole difference to the predictor.</p></noscript></div>'''
repl=dict(RDBLEARN=fig('rdblearn','RDBLearn: deterministic relational features feed a selected tabular ICL backbone.'),TABPFNREL=fig('tabpfnrel','TabPFN-Rel OSS: phase-aware DFS, context construction and TabPFN-3 prediction.'),SNAPSHOT=fig('snapshot','The inner and outer phases use separate frozen database states.'),RESULT_FIGURE=fig('results','All three seeds, paired across predictors; both panels use the same MSE scale.'),RESULTS=table,PAPER_STATUS=status,SOURCE_STATUS=source_status,TIME_WIDGET=time_widget,SWAP_WIDGET=swap_widget,WARMUP='<div id="b14-warmup"></div>',PREDICT='<div id="b14-predict"></div>',TEACHBACK='<div id="b14-teachback"></div>')
source_text=(R/'lessons/content'/f'{S}.md').read_text();body=source_text
for k,v in repl.items():body=body.replace('{{'+k+'}}',v)
def document(title,text,interactive=False):
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','flattening'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/flattening.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html#research-bridge">Research bridge</a></nav>'+render(text)+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document('B14 · The flattening challenge',body,True))
ref='''# B14 · Time-safe flattening and fair predictor swaps

**Recipe:** query key → permitted database rows → key-path aggregates → support/query feature table → predictor → keyed scores.

| Item | Contract |
|---|---|
| Query identity | Complete(entity_id,prediction_time), never entity alone |
| Visibility | Course:event time<min(query,snapshot);arrival time≤that boundary |
| Inner phase | Training support; validation-censored database; validation selects |
| Outer phase | Declared final support; test-censored database; hidden test labels |
| Backbone ablation | Same feature bytes, support/query keys, preprocessing and label access |
| Pipeline comparison | May change several components; cannot identify one component's effect |

**Worked trace:**event/arrival/value=(2,3,2),(6,7,6),(7,12,100),(10,10,20). Query14,snapshot10 reads[2,6], so count2,mean4,last6. Arrival checks exclude100; strict event boundary excludes20.

**RDBLearn original:**depths2/3/4 × TabPFNv2/v2.5/LimiX=9candidates;10k support cap. **RelArena RDBLearn:**six candidates, omits LimiX, adapted preprocessing. **TabPFN-Rel OSS:**three depths,TabPFN-3,no text,100k context cap,train+validation refit. API text-enabled scores are separate.

**F1 release:**seed0;11,411train/566validation/702test;inner2005/outer2010;reference target0.7145AUROC. Recent-pool branch inactive on11,977final support rows. A matched code/data release is not historical checkpoint identity.

'''+status+'\n\n'+table+'''

**Evidence checks:**future-row and held-out-label interventions; exact keyed coverage; independent scoring; complete candidate grid and validation-only selection; source/data/checkpoint hashes; aggregate budget accounting. Full suite, pretraining and learner mastery are separate.

[Lesson](../lessons/b14-flattening-challenge.html) · [Student lab](../labs/b14-flattening-challenge.ipynb) · [Full contract](../labs/b14-reproduction.md) · [RDBLearn paper](https://arxiv.org/html/2602.18495v1) · [RelArena paper](https://arxiv.org/html/2608.16319v2)
'''
(R/'reference'/f'{S}.html').write_text(document('B14 · Flattening reference',ref))
# Archive original sources, licenses and exact operators. Avoid caches/compiled code.
payload={}
for path in sorted(SRC.rglob('*')):
 if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ['.zip','.whl','.pyc']:
  payload[str(path.relative_to(R))]=path.read_bytes()
for name in ['b14-reproduction.md','_reproduce_b14.py','_preflight_b14.py','_warm_b14.py','_download_b14.py','_source_b14.py']:
 payload['labs/'+name]=(P/name).read_bytes()
for name in ['_budget_b14.py','_prepare_cloud_b14.py','_worker_b14.py','_audit_paper_b14.py','_keys_b14.py']:payload['labs/'+name]=(P/name).read_bytes()
payload['modal/b14_tabpfn_rel.py']=(R/'modal/b14_tabpfn_rel.py').read_bytes()
hashes={n:hashlib.sha256(data).hexdigest() for n,data in payload.items()};payload['source-manifest.json']=(json.dumps(hashes,indent=2)+'\n').encode();buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for name,data in sorted(payload.items()):
  info=zipfile.ZipInfo(name,date_time=(2026,10,3,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
(E/'portable-sources.zip').write_bytes(buf.getvalue());(E/'source-manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
module=(P/'relkit/flatten_b14.py').read_text();funcs={n.name:ast.get_source_segment(module,n) for n in ast.parse(module).body if isinstance(n,ast.FunctionDef)}
checks={
'flatten':"entities={1:4.};events=[(0,1,2,3,2.),(1,1,6,7,6.),(2,1,7,12,100.),(3,1,10,10,20.)]\nnp.testing.assert_array_equal(flatten(entities,events,[(1,14)],10),[[4,2,4,6]])\nnp.testing.assert_array_equal(flatten(entities,events+[(4,1,11,12,999.)],[(1,14)],10),[[4,2,4,6]])",
'standardize':"a,b=standardize(np.array([[1.,5],[3.,5]]),np.array([[100.,5.]]))\nnp.testing.assert_array_equal(a,[[-1,0],[1,0]])\nnp.testing.assert_array_equal(b,[[98,0]])",
'predict':"x=np.array([[-1.],[1.]]);y=np.array([-1.,1.]);q=np.array([[0.],[1.]])\nnp.testing.assert_allclose(predict(x,y,q,'ridge'),[0,2/3])\nr=np.exp(-4);np.testing.assert_allclose(predict(x,y,q,'rbf'),[0,(1-r)/(2-r)],atol=1e-12)"}
def embedded(name):return '!['+name+'](data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode()+')'
for solution in [False,True]:
 cells=[]
 def md(text):cells.append(nb.v4.new_markdown_cell(text))
 def code(text):cells.append(nb.v4.new_code_cell(text))
 md('# B14 · The flattening challenge\n\n**PROVIDED:** generator, original source, evidence contract. **TODO:** time-visible aggregation, normalization, predictor swap. **CHECK:** arithmetic, interventions, all 12 fits. **EXIT:** defend one controlled claim.\n\nThe default notebook runs synthetic CPU diagnostics. The paper lane uses the actual released TabPFN-Rel source and is separately gated. Displayed author results do not count as your kernel output or your mastery.')
 code('''# @colab-bootstrap — portable numerical lab; no repository imports
import importlib.util,subprocess,sys,os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
if importlib.util.find_spec('numpy') is None:subprocess.check_call([sys.executable,'-m','pip','install','numpy'])
import numpy as np
import base64,hashlib,io,json,math,tempfile,zipfile
from pathlib import Path
workspace=Path(tempfile.mkdtemp(prefix='b14-lab-'))
print('NumPy',np.__version__,'; work directory',workspace)
''')
 narrative=source_text[source_text.index('In B13'):source_text.index('## 6 ·')]
 for key,value in repl.items():
  if key in ['RDBLEARN','TABPFNREL','SNAPSHOT','RESULT_FIGURE']:value=embedded({'RDBLEARN':'rdblearn','TABPFNREL':'tabpfnrel','SNAPSHOT':'snapshot','RESULT_FIGURE':'results'}[key])
  elif key in ['WARMUP','PREDICT','TEACHBACK','TIME_WIDGET','SWAP_WIDGET']:value='**Predict before reading:** which records may this query read? Which factors must stay fixed to isolate the predictor?'
  narrative=narrative.replace('{{'+key+'}}',value)
 narrative=re.sub(r'\[([^\]]+)\]\(\.\./[^)]+\)',r'\1 (see the included contract)',narrative)
 for part in narrative.split('\n## '):md(part if part.startswith('In B13') else '## '+part)
 instructions=[('flatten','Build one row per complete query key. Reject duplicate events/keys and unknown entities. Use the declared event/arrival boundary; return local,count,mean,last. Empty histories return zero aggregates.'),('standardize','Return normalized support and query arrays. Fit mean and population SD on support only; use scale1 for a constant column.'),('predict','Solve fixed-alpha1 linear ridge or RBF kernel ridge. Center support labels around their mean. RBF gamma=1/feature_count; use the same support and query arrays in both arms.')]
 for i,(name,text) in enumerate(instructions,1):
  md(f'## TODO {i} · {name}\n\n'+text);code(funcs[name] if solution else funcs[name].split('\n')[0]+f'\n    raise NotImplementedError("Complete {name}")');code(checks[name]+'\nprint("CHECK passed: '+name+'")')
 md('## PROVIDED · Query-key scoring and the generated database\n\nThe generator uses your aggregation function to construct outcomes. The predictor does not receive held-out labels. Query keys contain entity and date; repeated entities are intentional.')
 code(funcs['keyed_mse']+'\n\n'+funcs['make_world'])
 md('## CHECK · Complete paired experiment\n\nYour three live functions are called below. We fit all 12 conditions and retain every prediction under fixed and rolling query information. Rolling predictions reuse the same fitted support.')
 code(funcs['run_diagnostic']+'\nreport=run_diagnostic()\nfor row in report["conditions"]:print(row["seed"],row["features"],row["backbone"],round(row["mse"],6))\n(workspace/"course-results.json").write_text(json.dumps(report,indent=2))')
 md('## CHECK · Independent complete-key scorer\n\nThis second scorer uses scalar math and a fresh reconstruction of labels. It refuses missing queries and duplicated identities.')
 verify=(P/'_verify_b14.py').read_text();audit=next(ast.get_source_segment(verify,n) for n in ast.parse(verify).body if isinstance(n,ast.FunctionDef) and n.name=='audit')
 code(audit+'\nassert audit(report)==3072\nprint("All 3,072 fixed and 3,072 rolling predictions independently scored")')
 md('## EXIT · Written defense\n\nExplain the two-clock trace; give a feature-collision example; name the exact factor the swap isolates; explain why a rolling-snapshot score is incomparable to fixed-snapshot results. Interpret the paper status separately from course success. Ask the teacher for feedback; revisit after 1/7/30 days. Learner status remains PENDING_WRITTEN_DEFENSE.')
 if (E/'full-audit.json').exists():
  md('## CHECK · Replay the complete selected-paper evidence\n\nThese are saved author predictions from fresh GPU inference, not inference in this notebook. Authenticate the packet, match every official validation/test key and label, independently score all five evaluations, then verify that validation selected the tested winner.')
  packet=io.BytesIO()
  with zipfile.ZipFile(packet,'w',zipfile.ZIP_DEFLATED) as z:
   for file in [E/'official-keys.json',E/'checkpoint.json',E/'paper-preflight.json',E/'cloud-input-lock.json',*sorted((E/'full').glob('*'))]:
    if file.is_file():z.writestr(str(file.relative_to(E)),file.read_bytes())
  raw=packet.getvalue();(E/'paper-replay.zip').write_bytes(raw)
  code('packet=base64.b64decode('+repr(base64.b64encode(raw).decode())+')\nassert hashlib.sha256(packet).hexdigest()=='+repr(hashlib.sha256(raw).hexdigest())+'\nwith zipfile.ZipFile(io.BytesIO(packet)) as z:z.extractall(workspace/"paper-evidence")')
  text=(P/'_audit_paper_b14.py').read_text();fs={n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
  code(fs['auroc']+'\n\n'+fs['audit'].replace('def audit(', 'def audit_paper(')+'\nexpected=json.loads((workspace/"paper-evidence/official-keys.json").read_text())\npaper_replay=audit_paper(workspace/"paper-evidence/full",expected)\nprint(paper_replay)')
 md('## NEXT STEP · Exact source and selected-paper operator\n\nThis authenticated archive includes the original harness, complete TabPFN-3 architecture, original RDBLearn source and licenses. Extraction does not run a model. The contract describes the separate environments and full target. The following gate is OFF by default.')
 archive=buf.getvalue();code('archive=base64.b64decode('+repr(base64.b64encode(archive).decode())+')\nassert hashlib.sha256(archive).hexdigest()=='+repr(hashlib.sha256(archive).hexdigest())+'\nwith zipfile.ZipFile(io.BytesIO(archive)) as z:z.extractall(workspace)\nmanifest=json.loads((workspace/"source-manifest.json").read_text())\nfor name,h in manifest.items():assert hashlib.sha256((workspace/name).read_bytes()).hexdigest()==h,name\nprint("Authenticated",len(manifest),"source files")')
 code('RUN_PAPER_REPRO=False\nif RUN_PAPER_REPRO:\n    subprocess.run([sys.executable,str(workspace/"labs/_reproduce_b14.py"),"--show-plan"],check=True)\n    print("The displayed plan is not inference. Follow b14-reproduction.md for the isolated runner and admission checks.")')
 md('## Source appendix · Read the actual released operations\n\nThe sections below are complete original files, displayed as source listings rather than executed notebook cells. Course ridge/kernel code above is not substituted for these models. Neural pretraining is not executed here; this lesson targets released-checkpoint inference.')
 listings=[('RDBLearn original estimator','original-rdblearn/rdblearn/rdblearn/estimator.py'),('RDBLearn original preprocessing','original-rdblearn/rdblearn/rdblearn/preprocessing.py'),('TabPFN-Rel feature harness','relarena/src/relarena/models/tabpfn_rel/model.py'),('Context selection','relarena/src/relarena/models/tabpfn_rel/context.py'),('Relational DFS','relarena/src/relarena/featurization/dfs.py'),('Temporal evaluation runner','relarena/src/relarena/runner.py'),('TabPFN-3 complete architecture','tabpfn/tabpfn/architectures/tabpfn_v3.py')]
 for title,path in listings:
  text=(SRC/path).read_text();md('### '+title+'\n\nPinned file: `'+path+'`. Read alongside the corresponding architecture step; full dependencies and licenses are in the archive.')
  if 'tabpfn_v3' in path:
   nodes=ast.parse(text).body;chunks=[];begin=0
   for node in nodes:
    if isinstance(node,ast.ClassDef):
     start=node.lineno-1
     if start>begin:chunks.append('\n'.join(text.splitlines()[begin:start]));begin=start
     chunks.append('\n'.join(text.splitlines()[begin:node.end_lineno]));begin=node.end_lineno
   if begin<len(text.splitlines()):chunks.append('\n'.join(text.splitlines()[begin:]))
   for chunk in chunks:
    if chunk.strip():md('```python\n'+chunk+'\n```')
  else:md('```python\n'+text+'\n```')
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}});dest=P/('solutions' if solution else '')/(S+'.ipynb');dest.parent.mkdir(exist_ok=True);nb.write(notebook,dest)
print('Built lesson, reference, figures and both notebooks')
