"""Canonical Lesson149 builder; portable live contracts and complete training appendix."""
import ast,base64,copy,gzip,hashlib,json,re
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l149';S='0149-weakest-relbench-tasks';TITLE='Weakest RelBench tasks: stress-test the thesis'
C=json.loads((E/'catalog.json').read_text());A=json.loads((E/'errors.json').read_text());F=json.loads((E/'frozen.json').read_text())
def table(rows,header):return '| '+' | '.join(header)+' |\n|'+'|'.join(['---']*len(header))+'|\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
catalog=''
for metric,rs in C['rankings'].items():
 catalog+='### '+('Classification · AUROC points' if metric=='AUROC' else 'Regression · normalized MAE')+'\n\n'
 scale=100 if metric=='AUROC' else 1;prec=1 if metric=='AUROC' else 2
 catalog+=table([[r['task'],f"{r['rdl']*scale:.{prec}f}",f"{r['fe']*scale:.{prec}f}",f"{r['gap']*scale:+.{prec}f}"] for r in rs],['Task (figure label)','RDL','FE','RDL advantage'])+'\n'
rows=[]
for seed in range(5):
 v=A['splits']['val']['seeds'][seed];t=A['splits']['test']['seeds'][seed];rows.append([seed,f"{v['gnn']:.6f}",f"{v['fe']:.6f}",f"{t['gnn']:.6f}",f"{t['fe']:.6f}"])
results=table(rows,['Seed','GNN validation','FE validation','GNN test','FE test'])+'\n'
for split,x in A['splits'].items():results+=f"**{split.upper()} mean ± sample seed SD:** GNN {x['gnn_mean']:.6f} ± {x['gnn_sd']:.6f}; FE {x['fe_mean']:.6f} ± {x['fe_sd']:.6f} MAE.\n\n"
r=A['splits']['test']['slices']['all'];results+=f"Test GNN loss penalty **+{r['mean']:.6f}**, descriptive driver-cluster 95% interval **[{r['interval']['low']:.6f}, {r['interval']['high']:.6f}]**. The interval crosses zero; neither superiority nor equivalence is established. Five paired run labels do not imply common random numbers across model families.\n"
slices=table([[split,name,str(r['rows'])+'/'+str(r['drivers']), '—' if r['mean'] is None else f"{r['mean']:+.4f}",f"[{r['interval']['low']:+.4f}, {r['interval']['high']:+.4f}]" if r['interval'] else 'UNSUPPORTED'] for split,x in A['splits'].items() for name,r in x['slices'].items()],['Split','Slice','Queries/drivers','GNN loss penalty','95% conditional interval'])
slices+='\n**Frozen nomination: high_history.** Its validation penalty is +0.5544; test +0.0799 with an interval crossing zero. Low history is not the largest observed validation weakness. This undermines the simple cold-start story for this comparison without establishing the actual cause.\n'
captions={'contract':'Illustrative sign contract: metric advantage is positive when RDL wins; the loss penalty uses the opposite sign.','catalog':'Published Figure3 mean-bar extraction. Panels have different units and regression uses a boosted head; no original error bars or significance ranking recovered.','results':'Fresh F1 per-query mean losses across five fits. Intervals resample whole drivers and condition on the fitted models and split; highlighted slice was nominated using validation.'}
fallbacks={'WARMUP':'Recall: graph construction changes access to information; five seeds on one database do not test transfer.','PREDICT':'Predict: can Table7 basic GNN replace the Figure3 regression comparator? No. Figure3 uses a GNN+LightGBM head.','RANK_WIDGET':'Static ranking: driver-top3 has the largest negative classification point gap; item-sales has the largest negative normalized regression point gap. The complete tables remain above.','TEACHBACK':'Write a weakness profile with comparator, metric, provenance, uncertainty, intervention and falsifier. Ask the teaching agent to review it.'}
def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[CATALOG]]',catalog).replace('[[RESULTS]]',results).replace('[[SLICES]]',slices)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l149/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l149/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="weakness-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption} Scroll for detail on a narrow screen.</figcaption></figure>')
 for name,value in fallbacks.items():s=s.replace('[['+name+']]',value if portable else f'<div id="{name.lower()}"></div><noscript>{value}</noscript>')
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def document(title,body,interactive=False):
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','weakness-viz','l149-lesson'] if interactive else []
 html=re.sub(r'<table([^>]*)>',r'<div class="weakness-scroll" tabindex="0"><table\1>',render(body)).replace('</table>','</table></div>')
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/weakness.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../notebooks.html">Labs</a></nav><header><p class="eyebrow">Year 4 · Quarter 3 · Lesson 149</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join(f'<script src="../assets/{s}.js"></script>' for s in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
ref='''## Weakness-catalog contract

1. Name the task, split, metric, model variant, FE recipe and evidence source.
2. Orient advantage: RDL−FE for AUROC, FE−RDL for MAE. Negative means RDL loses.
3. Rank within compatible units and provenance. Normalized MAE is not raw MAE.
4. Mark missing evidence, plot-derived means and unresolved task labels.
5. Profile a supported validation slice before opening test slices; report every slice.
6. Distinguish a measured gap from its hypothesized cause. Write an intervention and falsifier.

## Published ranking
'''+catalog+'''
All entries PLOT_DERIVED from RelBench v1 Figure3. Regression uses GNN+LightGBM. The figure's user-votes label differs from Table7 post-votes; identity unresolved. No significance ordering.

## Fresh complete F1 replay
'''+results+'''
## Claim boundaries
Five full GNN fits and five ten-trial FE searches plus refits; complete selected released pipelines. Historical identity NOT_ESTABLISHED. Whole paper and H&M retraining NOT_RUN. Global and nominated test driver-cluster intervals cross zero. Actual feature-arrival legality NOT_ESTABLISHED. Test population reused; exploratory. Default notebook reanalyzes author results, not learner training or mastery.

[Lesson](../lessons/0149-weakest-relbench-tasks.html) · [Protocol](../labs/l149-reproduction.md) · [Catalog](../labs/evidence/l149/catalog.json) · [Primary paper](https://arxiv.org/html/2407.20060v1#S6). Ask the teaching agent to review your written falsifier.
'''
(R/'reference/weakest-relbench-tasks.html').write_text(document('Weakness catalog reference',ref))
js='WeaknessViz.mount(document.getElementById("rank_widget"),'+json.dumps(C)+');\n'
js+='''RetrievalBank.mount(document.getElementById('warmup'),{upTo:149,count:3});
Predict.mount(document.getElementById('predict'),{prompt:'Can a Table7 basic-GNN result stand in for Figure3 regression RDL?',options:[{label:'Yes, the pipelines are identical',value:'yes'},{label:'No, the output heads differ',value:'no'}],correct:'no',reveal:'Figure3 uses GNN features plus a LightGBM head for regression. Table7 basic RDL is a different comparator.'});
Teachback.mount(document.getElementById('teachback'),{prompt:'Defend one weakness claim. Name its comparator, metric and source; state what the interval permits; propose an intervention and a falsifying result.',points:['Name the exact model variant and evidence source.','Keep AUROC and MAE rankings separate.','Explain why crossing zero does not establish equivalence.','Separate cold start from low history and causal explanation from observation.','Give a falsifier and acknowledge the reused test population.'],model:'Figure3 suggests H&M item-sales is the largest normalized regression weakness in its user-study subset, but these are plot-derived means for boosted RDL. Our fresh basic-GNN F1 comparison has a positive mean loss penalty; its descriptive driver-cluster interval crosses zero. High history, not low history, was the largest validation weakness. I would test matched legal engineered summaries in the GNN; no validation improvement would weaken a missing-summary explanation. Neither comparison establishes a universal failure of relational learning.'});
'''
(R/'assets/l149-lesson.js').write_text(js)
def functions(path):
 s=path.read_text();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
fns=functions(P/'relkit/weakness_l149.py');err=functions(P/'relkit/error_reg_l149.py');checks=functions(P/'_check_l149.py');oldchecks=functions(P/'_check_l137.py')
bootstrap='''# @colab-bootstrap: portable CPU audit; no repository checkout required.
import importlib.util,subprocess,sys
missing=[p for p in ['numpy','pandas','IPython'] if importlib.util.find_spec(p) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install',*missing])
import base64,gzip,hashlib,io,json,math
from pathlib import Path
import numpy as np,pandas as pd
from IPython.display import display
'''+checks['rejects']
raw=(E/'portable.json').read_bytes();packed=base64.b64encode(gzip.compress(raw,mtime=0)).decode()
payload='''# PROVIDED: checksum-verified primary author predictions, not new fits.
raw=gzip.decompress(base64.b64decode('''+repr(packed)+'''))
assert hashlib.sha256(raw).hexdigest()=='''+repr(hashlib.sha256(raw).hexdigest())+'''
DATA=json.loads(raw)
CATALOG=json.loads('''+repr(json.dumps(C))+''')
EXPECTED=json.loads('''+repr(json.dumps(A))+''')
'''
audit='''# CHECK: your live ranking and nomination functions process the real evidence.
rankings={m:rank_catalog([r for r in CATALOG['rows'] if r['metric']==m]) for m in ['AUROC','MAE']}
assert rankings['AUROC'][0]['task']=='rel-f1/driver-top3'
assert rankings['MAE'][0]['task']=='rel-hm/item-sales'
for metric,rows in rankings.items():
    display(pd.DataFrame(rows)[['task','rdl','fe','gap','status']])
rows=[];count=0;selected=None
for split in ['val','test']:
    x=DATA[split];keys=x['keys'];ids=np.array([k[0] for k in keys]);deltas=[]
    for seed in range(5):
        order=np.random.default_rng(149+seed).permutation(len(keys))
        shuffled=[keys[i] for i in order]
        d=paired_errors(keys,x['target'],shuffled,np.array(x['gnn'][seed])[order],keys,x['fe'][seed])
        deltas.append(d);count+=len(keys)*2
    mean_delta=np.mean(deltas,axis=0)
    masks={name:np.array(mask,bool) for name,mask in x['masks'].items() if name!='all'}
    if split=='val':
        nomination=nominate_slice(mean_delta,ids,masks);selected=nomination['selected']
        assert selected=='high_history'
    for name,mask in x['masks'].items():
        mask=np.array(mask,bool);ref=EXPECTED['splits'][split]['slices'][name]
        value=float(mean_delta[mask].mean()) if mask.any() else None
        if value is not None:assert abs(value-ref['mean'])<1e-12
        interval=cluster_interval(mean_delta[mask],ids[mask]) if ref['supported'] else None
        if interval:assert interval==ref['interval']
        rows.append(dict(split=split,slice=name,queries=int(mask.sum()),drivers=len(np.unique(ids[mask])),loss_penalty=value,supported=ref['supported']))
assert count==12590
display(pd.DataFrame(rows))
Path('l149-weakness-catalog.json').write_text(json.dumps(dict(rankings=rankings,nomination=selected,slices=rows,hypothesis='PENDING_WRITTEN_DEFENSE',source='Author training and published plot extraction; no new training in this cell'),indent=2))
report=dict(status='PASS',individual_predictions=count,catalog_tasks=sum(map(len,rankings.values())),selected=selected,learner='PENDING_WRITTEN_DEFENSE')
Path('l149-report.json').write_text(json.dumps(report,indent=2));print(report)
'''
prior=nb.read(P/'solutions/0137-error-analysis-reg.ipynb',4)
start=next(i for i,c in enumerate(prior.cells) if c.cell_type=='markdown' and c.source.startswith('## Optional complete training lanes'))
appendix=[]
for c in prior.cells[start:]:
 cell=copy.deepcopy(c);cell.source=cell.source.replace('l137','l149').replace('L137','L149');cell.pop('outputs',None) if cell.cell_type=='markdown' else None
 if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
 appendix.append(cell)
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson149 · '+TITLE+'\n\n'+('Teacher reference' if solution else 'Student lab')+' · Tier B: full real RelBench query evidence. PROVIDED → three live TODO/CHECK tasks → catalog export → written EXIT. Default is a portable CPU analysis. Separate complete training gates use their pinned FE or GPU runtimes; do not mix their dependencies.'),nb.v4.new_code_cell(bootstrap)]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if not section.strip():continue
  cells.append(nb.v4.new_markdown_cell(section))
  task=None
  if section.startswith('## 2'):task=('oriented_gap',fns['oriented_gap'],checks['check_gap'],'check_gap','Compute a signed advantage with a single meaning across metrics. Reject invalid units and nonfinite scores.')
  if section.startswith('## 3'):task=('rank_catalog',fns['rank_catalog'],checks['check_rank'],'check_rank','Rank the weakest compatible evidence first. Reject duplicate tasks, missing measurements and mismatched provenance, split, variant or metric.')
  if section.startswith('## 5'):task=('nominate_slice',err['nominate_slice'],oldchecks['check_nominate'],'check_nominate','Nominate the largest supported positive validation loss penalty. Report all slices and return no nomination when no supported slice loses.')
  if task:
   name,src,check,checker,hint=task
   cells.append(nb.v4.new_markdown_cell('### TODO · '+name+'\n\n**Goal:** '+hint+'\n\n**Why:** an invalid ranking can manufacture a counterexample. Your function is called on the real evidence below.'))
   cells.append(nb.v4.new_code_cell(src if solution else src.splitlines()[0]+'\n    raise NotImplementedError("TODO: '+name+'")'))
   cells.append(nb.v4.new_code_cell(check+'\n'+checker+'('+name+')\nprint("PASS: '+name+'")'))
  if section.startswith('## 6'):
   cells.extend([nb.v4.new_markdown_cell('### PROVIDED · Complete key alignment and driver-cluster uncertainty\n\nThese inherited audited operators preserve whole-query identity and query weighting. Read the implementation before interpreting the result.'),nb.v4.new_code_cell(err['paired_errors']+'\n\n'+err['cluster_interval']),nb.v4.new_code_cell(oldchecks['check_pair']+'\n'+oldchecks['check_cluster']+'\ncheck_pair(paired_errors);check_cluster(cluster_interval)'),nb.v4.new_code_cell(payload),nb.v4.new_code_cell(audit)])
 cells.extend(appendix)
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}})
 dest=P/('solutions' if solution else '')/f'{S}.ipynb'
 if dest.exists():
  old=nb.read(dest,4);oc=[c for c in old.cells if c.cell_type=='code'];nc=[c for c in cells if c.cell_type=='code']
  if solution and [c.source for c in oc]==[c.source for c in nc]:
   notebook.metadata=old.metadata
   for a,b in zip(nc,oc):a.outputs=b.outputs;a.execution_count=b.execution_count;a.metadata=b.metadata
 for i,c in enumerate(notebook.cells):c.id=f'l149-{i:03d}'
 nb.write(notebook,dest)
print('Built Lesson149, reference, student and teacher notebooks')
