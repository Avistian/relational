"""Single-source lesson, reference and portable notebook builder."""
import ast,base64,hashlib,json,re,textwrap
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l148';S='0148-ablation-discipline';TITLE='Ablation discipline: what earns the improvement?'
s=json.loads((E/'summary.json').read_text());arms=['full','encoder','messages','history','combined']
results='| Arm | Validation MAE ± seed SD | Test MAE ± seed SD | Parameters |\n|---|---:|---:|---:|\n'
for a in arms:
 v=s['means']['val'][a];t=s['means']['test'][a];run=next(r for r in s['runs'] if r['arm']==a)
 results+=f"| {a} | {v['mean']:.4f} ± {v['sample_sd']:.4f} | {t['mean']:.4f} ± {t['sample_sd']:.4f} | {run['parameter_count']:,} |\n"
results+='\nAll 25 fresh fits completed; **31,475** held-out predictions independently rescored. Positive paired differences favor the full baseline.\n\n'
for split in ['val','test']:
 results+=f"**{split.capitalize()} paired changes:** "+'; '.join(f"{a} {s['paired'][split][a]['mean']:+.4f} ± {s['paired'][split][a]['sample_sd']:.4f}" for a in arms[1:])+'.\n\n'
inter=s['interaction'];results+=f"Encoder–message interaction: validation **{inter['val']['mean']:+.4f} ± {inter['val']['sample_sd']:.4f}**, test **{inter['test']['mean']:+.4f} ± {inter['test']['sample_sd']:.4f}** MAE. These are mean ± sample seed SD.\n"
probe=json.loads((E/'real-probe.json').read_text());nonfinite={a:r['nonfinite_gradient_entries'] for a,r in probe['gradient_probe'].items()}
evidence=f"""**Executed:** 25 complete ten-epoch fits on the full task; independent key/label audits; baseline original-model output replay; owner-window and intervention checks. Baseline descriptive comparisons: validation **{s['baseline_closeness']['val']}**, test **{s['baseline_closeness']['test']}**. Paper means are 3.193 / 4.022 MAE.

**Gradient boundary:** one real initial-batch diagnostic counted nonfinite gradient entries {nonfinite}. The released missing-value numerical path is preserved. Finite predictions and source parity do not prove gradient health; this is a limit on interpreting these fits.

**Cost:** conservative reservations including two pre-worker mount failures and overhead fit the approved USD 10 cap. Measured worker-body estimates and the exact reservations are in the cost ledger; they are not an itemized invoice. Browser and notebook results are recorded separately in the delivery reports.
"""
if (E/'cost-summary.json').exists():
 cost=json.loads((E/'cost-summary.json').read_text())
 evidence+=f"\n**Final delivery:** default solution and pinned notebook PASS; pinned validation adds one full history seed100 fit, excluded from primary statistics, with1,259 independently checked predictions. Desktop/mobile, keyboard/reset, print/noJS and clean Git-index Pages PASS. Conservative all-attempt reservations plus overhead **USD{cost['conservative_reservations_plus_overhead']:.6f}**; known worker-body estimate **USD{cost['known_worker_body_usd']:.6f}**, excluding unitemized costs. Live Colab/deployment NOT_CHECKED.\n"
captions={'interventions':'Trace the exact information path each arm changes. Width128 is retained; shallow encoder capacity is not matched.','history':'Illustrative day units: each query has a different inclusive365-day window. The day100 row changes eligibility with its owner.','results':'Fresh full-data measurements. Five dots per arm are paired seed differences; diamond is their mean. No confidence intervals.','interaction':'Measured interaction on the MAE scale, calculated within seed. Data-access interactions were not included in this five-arm design.'}
fallback={'WARMUP':'Recall: full query identity includes entity and cutoff. A complete architecture comparison does not isolate one component. A validation rule selects checkpoints before test scoring.','PREDICT':'Predict: if removing two parts separately costs1and2MAE, must their joint removal cost3? No; their contributions can interact.','HISTORY_WIDGET':'At365days, day100is legal for cutoff400but too old for cutoff500. Day450is future for cutoff400and legal for cutoff500.','INTERACTION_WIDGET':'Hypothetical F4,E5,M6,EM6.5 gives interaction−.5. Changing only EMto7makes the interaction0.','TEACHBACK':'Write your intervention, fixed protocol, paired evidence, confound and falsifier. Explain what the interaction can and cannot establish. Ask the teaching agent for feedback.'}
def prose(portable=False):
 text=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results).replace('[[EVIDENCE]]',evidence)
 for name,cap in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l148/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l148/{name}.svg'
  text=text.replace('[[FIG:'+name+']]',f'<figure class="ablation-figure" tabindex="0"><img src="{src}" alt="{cap}"><figcaption>{cap} Scroll for detail on narrow screens.</figcaption></figure>')
 for name,words in fallback.items():text=text.replace('[['+name+']]',words if portable else f'<div id="{name.lower()}"></div><noscript>{words}</noscript>')
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/');text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def document(title,body,interactive=False):
 html=re.sub(r'<table([^>]*)>',r'<div class="ablation-scroll" tabindex="0"><table\1>',render(body)).replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','ablation-viz','l148-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/ablation.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../notebooks.html">Labs</a></nav><header><p class="eyebrow">Year 4 · Quarter 3 · Lesson 148</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
ref='''## Design before measuring
Name the intervention in code. Freeze target population, splits, training and selection. Report what also changes: parameters, random-number consumption, sampling coverage, computation. Retraining an ablation differs from test-time corruption.

## Analysis contract
Require unique complete (entity,cutoff) keys. Compute MAE per seed. Paired effect = arm MAE − full MAE. Positive is worse. Report mean and sample seed SD. Interaction = combined − encoder − messages + full, within each seed. A negative interaction means a smaller combined penalty than additivity predicts on this error scale. No additive credit allocation follows.

## History contract
For dated non-root rows: owner cutoff −365days ≤ row time ≤ owner cutoff. Keep undated rows and legal query roots. Prune nodes and incident edges before encoding. This implementation prunes after sampling without refill; it tests reduced sampled context as well as shorter history.

## Author evidence
'''+results+'\n'+evidence+'''\nWhole paper NOT_RUN; historical identity NOT_ESTABLISHED; extensions exploratory; learner PENDING_WRITTEN_DEFENSE. [Lesson](../lessons/0148-ablation-discipline.html) · [Protocol](../labs/l148-reproduction.md) · [RelBench v1](https://arxiv.org/html/2407.20060v1).
'''
(R/'reference/ablation-discipline.html').write_text(document('Ablation discipline reference',ref))
(R/'assets/l148-lesson.js').write_text('''AblationViz.history(document.getElementById('history_widget'));
AblationViz.interaction(document.getElementById('interaction_widget'));
RetrievalBank.mount(document.getElementById('warmup'),{upTo:148,count:3});
Predict.mount(document.getElementById('predict'),{prompt:'Separate removals cost 1 and 2 MAE. Must their combined removal cost 3 MAE?',options:[{label:'Their penalties always add',value:'add'},{label:'Their penalties can interact',value:'interact'}],correct:'interact',reveal:'Removing one component can change how useful the other is. Calculate the paired interaction before claiming additive contributions.'});
Teachback.mount(document.getElementById('teachback'),{prompt:'Defend one ablation: intervention, fixed quantities, paired evidence, confound and falsifier. Then explain the interaction sign.',points:['Specify the actual code intervention.','Report paired seed differences and the scale.','Acknowledge capacity or sampling changes.','Distinguish a course extension from the paper baseline.','Propose a follow-up that could refute your explanation.'],model:'The history arm removes dated non-root rows older than 365 days after sampling. It holds labels, splits and training fixed, but leaves sampling slots unfilled. A paired MAE change therefore does not identify the value of historical information alone. A follow-up would compare pre-sampling restriction with the same budget of legal recent neighbors. Interaction is combined minus the two individual scores plus full; it is conditional on this task and MAE scale.'});
''')

def functions(path):
 text=path.read_text();return {x.name:ast.get_source_segment(text,x) for x in ast.parse(text).body if isinstance(x,ast.FunctionDef)}
fns=functions(P/'relkit/ablation_l148.py');checks=functions(P/'_check_l148.py')
bootstrap='''# @colab-bootstrap: portable NumPy analysis; GPU training is explicit opt-in.
import os,sys,subprocess
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable,'-m','pip','install','-q','numpy==1.26.4'])
import base64,hashlib,io,json
from pathlib import Path
import numpy as np
from IPython.display import display,Markdown
RUN_FULL_REPRODUCTION = os.environ.get('L148_FULL_TRAINING') == '1'
ARMS=['full','encoder','messages','history','combined']
'''
def payload(path,var):
 raw=path.read_bytes();return f"raw=base64.b64decode({base64.b64encode(raw).decode()!r})\nassert hashlib.sha256(raw).hexdigest()=={hashlib.sha256(raw).hexdigest()!r}\n{var}=np.load(io.BytesIO(raw),allow_pickle=False)\n"
audit='''# Your live functions audit every saved author prediction, deliberately shuffled.
scores={sp:{a:{} for a in ARMS} for sp in ['val','test']};verified=0
for arm in ARMS:
    for seed in range(5):
        for split in ['val','test']:
            k=f'{arm}_{seed}_{split}_';ref=split+'_'
            keys=list(zip(DATA[ref+'entity'],DATA[ref+'time']))
            pkeys=list(zip(DATA[k+'entity'],DATA[k+'time']))
            order=np.random.default_rng(148+seed).permutation(len(pkeys))
            score=keyed_mae(keys,DATA[ref+'target'],[pkeys[i] for i in order],DATA[k+'pred'][order])
            scores[split][arm][seed]=score;verified+=len(keys)
assert verified==31475
interactions={sp:interaction_summary(d['full'],d['encoder'],d['messages'],d['combined']) for sp,d in scores.items()}
for sp in interactions:
    np.testing.assert_allclose(interactions[sp]['differences'],AUTHOR['interaction'][sp]['differences'],atol=1e-12,rtol=0)
table='| Arm | Validation mean | Test mean |\\n|---|---:|---:|\\n'
for a in ARMS:table+=f"| {a} | {np.mean(list(scores['val'][a].values())):.4f} | {np.mean(list(scores['test'][a].values())):.4f} |\\n"
display(Markdown(table))
print('Verified author predictions:',verified)
for sp,d in interactions.items():print(sp,'interaction',d['mean'],'sample SD',d['sample_sd'])
'''
history='''# Your live mask drives the real sampled-history exercise.
removed=0;by_table={}
for key in SAMPLE.files:
    if not key.endswith('_times'):continue
    kind=key[:-6]
    mask=history_mask(SAMPLE['cutoffs'],SAMPLE[key],SAMPLE[kind+'_owners'],SAMPLE[kind+'_undated'],SAMPLE[kind+'_roots'],365*86400)
    removed+=int((~mask).sum());by_table[kind]=int((~mask).sum())
assert removed==REAL_PROBE['removed'],(removed,REAL_PROBE['removed'])
print('Removed dated non-root nodes from the real saved batch:',removed,by_table)
# Predict how the30-day window changes the retained set; it cannot admit future rows.
for key in SAMPLE.files:
    if key.endswith('_times'):
        k=key[:-6];args=[SAMPLE['cutoffs'],SAMPLE[key],SAMPLE[k+'_owners'],SAMPLE[k+'_undated'],SAMPLE[k+'_roots']]
        assert np.all(~history_mask(*args,30*86400)|history_mask(*args,365*86400))
'''
# Full lane remains visible, split by computation. Definitions execute only on opt-in.
base=(P/'relkit/rdl_l117.py').read_text();base=base[:base.index('# %% Full released-protocol training loop')]
model=(P/'relkit/ablation_model_l148.py').read_text();model=re.sub(r'^from relkit\.[^\n]+\n','',model,flags=re.M)
trainer=(P/'_train_l148.py').read_text();trainer=re.sub(r'^from relkit\.[^\n]+\n','',trainer,flags=re.M);trainer="CONFIG="+repr(dict(channels=128,num_layers=2,batch_size=512,lr=.005,epochs=10,fanout=[128,64],aggr='sum',temporal_strategy='uniform'))+'\n'+trainer
prep=(P/'_prepare_l148.py').read_text();prep=re.sub(r'^from relkit\.[^\n]+\n','',prep,flags=re.M);prep=prep.replace('    from relkit.rdl_l117 import foreign_key_edges\n','');prep=prep.replace("P=Path(__file__).resolve().parent", "P=Path.cwd()/'l148-full-runtime';P.mkdir(exist_ok=True)");prep=prep.replace("hashlib.sha256((P/'relkit/rdl_l117.py').read_bytes()).hexdigest()",repr(hashlib.sha256((P/'relkit/rdl_l117.py').read_bytes()).hexdigest()))
sourcefiles={p.name:base64.b64encode(p.read_bytes()).decode() for p in (P/'sources/l117').iterdir() if p.is_file() and p.suffix in ['.py','.json']}
full_setup='''# PROVIDED: pinned full-training environment assertions; no cloud calls.
if RUN_FULL_REPRODUCTION:
    import importlib.metadata as md
    for package,version in {'torch-geometric':'2.6.1','pytorch-frame':'0.2.3','relbench':'1.1.0','numpy':'1.26.4','pandas':'2.2.3','sentence-transformers':'3.3.1'}.items():
        assert md.version(package)==version,(package,md.version(package))
    import torch
    assert torch.__version__.startswith('2.5.1') and torch.cuda.is_available()
    assert sys.version_info[:2]==(3,11)
'''
full_execute='''# PAPER-RESULTS: opt-in fresh preparation and all 25 fits on your CUDA runtime.
if RUN_FULL_REPRODUCTION:
    if (P/'runs').exists():raise RuntimeError('Existing run namespace: archive it before rerunning')
    (P/'sources/l117').mkdir(parents=True,exist_ok=True)
    for name,data in SOURCE_FILES.items():(P/'sources/l117'/name).write_bytes(base64.b64decode(data))
    prepare(P/'prepared')
    g=torch.load(P/'prepared/graph.pt',weights_only=False)
    sys.path.insert(0,str(P/'sources/l117'));from model import Model as OriginalModel
    for arm in ARMS:
        for seed in range(5):
            fit_ablation(g['data'],g['stats'],g['task'],seed,P/'runs'/f'{arm}-{seed}',10,'cuda',OriginalModel if arm=='full' else None,arm)
else:
    print('Full training NOT_RUN in this notebook execution. Analysis uses saved author artifacts.')
'''
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson 148 · '+TITLE+'\n\n'+('Reference solution' if solution else 'Student lab')+' · PROVIDED / TODO / CHECK / EXIT. Tier B real full-task predictions and a saved real sampled context; small synthetic fixtures test failure handling. No learner mastery is inferred from author evidence.'),nb.v4.new_code_cell(bootstrap)]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if section.strip():cells.append(nb.v4.new_markdown_cell(section))
 code=payload(E/'audit-inputs.npz','DATA')+payload(E/'sampled-history.npz','SAMPLE')+'AUTHOR=json.loads('+repr(json.dumps(s))+')\nREAL_PROBE=json.loads('+repr(json.dumps(probe))+')'
 c=nb.v4.new_code_cell(code);c.metadata['tags']=['data-payload'];cells.append(c)
 for name,check,why in [('history_mask','check_history','Return one boolean per sampled node, using its owner cutoff; retain undated nodes and legal roots, reject future dated nodes, include the lower boundary.'),('keyed_mae','check_keyed','Reject duplicate/missing/extra keys and nonfinite values; align full entity/cutoff keys before calculating MAE.'),('interaction_summary','check_effect','Require identical sets of at least two seeds; return combined−encoder−messages+full per seed, mean and sample SD.')]:
  cells.append(nb.v4.new_markdown_cell('## TODO · '+name+'\n\n'+why+' Predict each fixture before running it. This function is used in the evidence audit below.'))
  cells.append(nb.v4.new_code_cell(fns[name] if solution else fns[name].splitlines()[0]+'\n    raise NotImplementedError("TODO: '+name+'")'))
  cells.append(nb.v4.new_code_cell(checks[check]+'\n'+check+'('+name+')\nprint("PASS: '+name+'")'))
 cells+=[nb.v4.new_markdown_cell('## CHECK · actual sampled context\n\nSame query owner and time units as the training intervention. Changing your live mask changes this audit.'),nb.v4.new_code_cell(history),nb.v4.new_markdown_cell('## CHECK · independent author-result replay\n\nEach fit is complete, with its own validation-selected checkpoint. These are saved author predictions, not fresh notebook training.'),nb.v4.new_code_cell(audit)]
 cells.append(nb.v4.new_markdown_cell('## Complete visible model and training lane\n\nThe following annotated cells contain the full row encoders, message passing, graph construction, interventions and trainer. They are gated by `RUN_FULL_REPRODUCTION` so the default NumPy analysis stays portable. To execute all 25 fits locally, use the exact pinned Python 3.11 CUDA environment in the reproduction contract, set `L148_FULL_TRAINING=1` **before** running the notebook, and use a fresh working directory. This local lane has no cloud billing guard; the repository Modal launcher enforces the approved aggregate budget.'))
 for primitive in ['resnet.py','sage_conv.py']:
  cells.append(nb.v4.new_markdown_cell('### Pinned primitive reference · '+primitive+'\n\nVerbatim source from the pinned dependency (MIT); the executable lane uses the installed matching package. This makes the row residual blocks and neighbor/root computation inspectable. The broader model and trainer below execute on opt-in.\n\n```python\n'+(P/'sources/l117/primitives'/primitive).read_text()+'\n```'))
 cells.append(nb.v4.new_code_cell(full_setup))
 for chunk in base.split('# %%'):
  if chunk.strip():
   if not chunk.startswith('"""'):chunk='# '+chunk
   cells.append(nb.v4.new_code_cell('if RUN_FULL_REPRODUCTION:\n'+textwrap.indent(chunk,'    ')))
 for title,code in [('Intervention paths',model),('Full optimizer and selection loop',trainer),('Source-pinned full-data preparation',prep)]:
  cells.append(nb.v4.new_markdown_cell('### PROVIDED · '+title))
  cells.append(nb.v4.new_code_cell('if RUN_FULL_REPRODUCTION:\n'+textwrap.indent(code,'    ')))
 c=nb.v4.new_code_cell('SOURCE_FILES='+repr(sourcefiles));c.metadata['tags']=['data-payload'];cells.append(c)
 cells.append(nb.v4.new_code_cell(full_execute))
 cells.append(nb.v4.new_markdown_cell('## EXIT · written defense\n\nWrite 150–250 words: exact intervention, what stayed fixed, paired result, confound, falsifier, interaction sign and next experiment. A smaller test MAE does not by itself isolate a cause. Submit your defense to the teaching agent.'))
 cells.append(nb.v4.new_code_cell("report=dict(status='PASS',verified_author_predictions=verified,removed_sampled_nodes=removed,interactions=interactions,full_training='COMPLETE' if RUN_FULL_REPRODUCTION else 'NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')\nPath('l148-report.json').write_text(json.dumps(report,indent=2))\nprint('Analysis complete; written defense pending.')"))
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python'}})
 for i,c in enumerate(cells):c.id=hashlib.sha256((str(i)+c.source).encode()).hexdigest()[:12]
 path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
  old=nb.read(path,4);notebook.metadata=old.metadata
  previous=[c for c in old.cells if c.cell_type=='code'];current=[c for c in cells if c.cell_type=='code']
  if [c.source for c in previous]==[c.source for c in current]:
   for before,after in zip(previous,current):
    after.outputs=before.outputs;after.execution_count=before.execution_count;after.metadata=before.metadata
 nb.write(notebook,path)
print('Built L148 lesson, reference, student and solution')
