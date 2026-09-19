"""Build synchronized lesson HTML, student/solution notebooks and lab preview."""
import ast,base64,json,re,importlib.metadata as md
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0088-graph-classification';TITLE='Graph classification: what can GIN distinguish?'
FIGS={'POOL_FIG':('pooling','Synthetic categorical encodings expose counts, proportions and presence.'),'WL_FIG':('wl','With identical initial labels, path/star separate in one round; cycle/triangles remain indistinguishable.'),'ARCH':('architecture','Complete released GIN-0 path: four node updates, five graph readouts, per-depth heads, logit dropout and graph loss.'),'RESULT_FIG':('results','Fresh MUTAG reconstruction: fold scores at a common selected epoch for each configuration. Historical parity INCOMPARABLE.')}
def report():
 p=LAB/'_paper_l088_results.json'
 if not p.exists():return '**Author full-grid execution: RUNNING. No numerical reproduction claim yet.**'
 d=json.loads(p.read_text());s='**Author execution: '+d['status']+'.**\n\n| Width | Batch | Dropout | Common epoch | Mean ± sample SD (%) |\n|---|---|---|---|---|\n'
 for c in d['candidates']:
  a=c['config'];s+=f"| {a['hidden']} | {a['batch_size']} | {a['dropout']} | {c['epoch']} | {100*c['mean']:.2f} ± {100*c['sample_sd']:.2f} |\n"
 w=d.get('selected',d['candidates'][0]);s+=f"\nReported candidate mean **{100*w['mean']:.2f}%**, sample SD **{100*w['sample_sd']:.2f} percentage points** (population SD {100*w['population_sd']:.2f}). The published cell is 89.4 ± 5.6%. Fold SD is not a confidence interval; overlapping training sets make folds dependent. These are author results, separate from your kernel output.\n"
 if d['status']!='FULL_GRID_EXECUTED':s+='\n**Full hyperparameter search: INCOMPLETE.** Only the listed complete configurations were executed. The remaining grid is runnable but not completed; this is not full reproduction of the selected paper cell.\n'
 return s

def teaching():
 p=LAB/'_teaching_l088_results.json'
 if not p.exists():return 'Teaching measurements pending.'
 d=json.loads(p.read_text());s='| Readout | Common epoch | Mean ± sample SD (%) |\n|---|---|---|\n'
 for r in d['rows']:s+=f"| {r['readout']} | {r['selected_epoch']} | {100*r['mean']:.2f} ± {100*r['sample_sd']:.2f} |\n"
 winner=max(d['rows'],key=lambda r:r['mean']);s+=f"\n**{winner['readout'].capitalize()} has the highest selected mean in this declared teaching run.** This is descriptive; no significance or universal superiority claim follows.\n"
 return s

def prose(portable=False):
 s=(ROOT/'lessons/content'/f'{SLUG}.md').read_text()
 for tag,(name,caption) in FIGS.items():
  p=LAB/f'figures/l088/{name}.png'
  if not p.exists():s=s.replace('[['+tag+']]','');continue
  src='data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode() if portable else f'../labs/figures/l088/{name}.png'
  s=s.replace('[['+tag+']]',f'<figure class="mpnn-figure"><div class="figure-scroll" tabindex="0"><img src="{src}" alt="{caption}"></div><figcaption>{caption}</figcaption></figure>')
 for tag,id,text in [('WARMUP','warmup','Write the three retrieval answers below before reading.'),('PREDICT','prediction','Predict: sum, mean or max? Commit before reading the worked example.'),('WL_WIDGET','wl-refinement','Trace one joint WL round on path/star. Then prove why cycle/triangles cannot separate.'),('TEACHBACK','teachback','Explain the WL bound in your own words; include both graph pairs and the distinction between capacity and generalization.')]:
  s=s.replace('[['+tag+']]',text if portable else f'<div id="{id}"></div>')
 s=s.replace('[[RESULTS]]',report()).replace('[[TEACHING]]',teaching())
 if portable:
  s=s.replace('](0087-link-prediction.html)','](https://avistian.github.io/relational/lessons/0087-link-prediction.html)').replace('](../','](https://avistian.github.io/relational/')
 return s

head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{TITLE}</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','mpnn-lesson','wl-viz'])+'</head><body><article>'
head+=f'<nav><a href="../index.html">Course</a> · <a href="0087-link-prediction.html">Lesson 87</a> · <a href="../reference/graph-classification.html">Reference</a></nav><header><p>Year 3 · Quarter 1 · Lesson 088</p><h1>{TITLE}</h1></header><aside><a href="../labs/{SLUG}.ipynb">Download lab</a> · <a href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{SLUG}.ipynb">Open in Colab</a> · <a href="../labs/html/{SLUG}.html">Read lab</a> · <a href="../labs/solutions/{SLUG}.ipynb">Solution</a></aside>'
footer='</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','teachback','wl-viz','l088-lesson'])+'</body></html>'
(ROOT/'lessons'/f'{SLUG}.html').write_text(head+render(prose())+footer)
source=(LAB/'relkit/gin_l088.py').read_text();nodes=ast.parse(source).body
pins={n:md.version(n) for n in ['numpy','torch','scikit-learn','scipy','matplotlib','nbformat','nbclient','nbconvert','ipykernel','networkx','pandas']}
(LAB/'requirements-l088-runtime.txt').write_text('# Python 3.12; use CPU wheel index for torch as described in guide.\n'+'\n'.join(f'{k}=={v}' for k,v in pins.items())+'\n')
bootstrap=f"""# Complete inline implementation; no private module or repository checkout required.
import sys, subprocess
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--index-url', 'https://download.pytorch.org/whl/cpu', 'torch=={pins['torch']}'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy=={pins['numpy']}', 'scikit-learn=={pins['scikit-learn']}', 'scipy=={pins['scipy']}'])
import importlib.metadata as metadata
print({{n: metadata.version(n) for n in ['torch','numpy','scikit-learn']}})
"""
checks={
'neighbor_sum':"h=torch.tensor([[1.,2.],[3.,4.],[5.,6.]],requires_grad=True)\ne=torch.tensor([[0,1,1,2],[1,0,2,1]])\ny=neighbor_sum(h,e)\ntorch.testing.assert_close(y,torch.tensor([[3.,4.],[6.,8.],[3.,4.]]))\ny.sum().backward();torch.testing.assert_close(h.grad,torch.tensor([[1.,1.],[2.,2.],[1.,1.]]))\nprint('CHECK passed: incoming sums and gradients')",
'graph_readout':"h=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[2.,3.]])\nb=torch.tensor([0,0,0,1])\ntorch.testing.assert_close(graph_readout(h,b,2),torch.tensor([[2.,1.],[2.,3.]]))\ntorch.testing.assert_close(graph_readout(h,b,2,'mean'),torch.tensor([[2/3,1/3],[2.,3.]]))\ntorch.testing.assert_close(graph_readout(h,b,2,'max'),torch.tensor([[1.,1.],[2.,3.]]))\nprint('CHECK passed: unequal sizes and separate graphs')",
'select_epoch':"assert select_epoch([[.9,.8],[.5,.8]])==1\nassert select_epoch([[.8,.8],[.6,.6]])==0\nprint('CHECK passed: one common epoch, earliest tie')"}
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# '+TITLE+'\n\nTier B · three live TODOs. Author evidence is labeled separately. The default live run is a short demonstration; the complete paper search is explicitly available below.'),nbf.v4.new_code_cell(bootstrap)]
 cells.extend(nbf.v4.new_markdown_cell(x) for x in re.split(r'(?=^## )',prose(True),flags=re.M) if x.strip())
 cells.append(nbf.v4.new_markdown_cell('## PROVIDED / TODO / CHECK · complete visible implementation\n\nThe functions you implement feed the GIN forward pass and CV selection. No hidden solution imports. Read each docstring; replace each `NotImplementedError`, then run its immediate CHECK.'))
 imports='\n'.join(ast.get_source_segment(source,n) for n in nodes if isinstance(n,(ast.Import,ast.ImportFrom)))
 constants='\n'.join(ast.get_source_segment(source,n) for n in nodes if isinstance(n,ast.Assign))
 cells.append(nbf.v4.new_code_cell(imports+'\n'+constants+'\ntorch.set_num_threads(1)'))
 for n in nodes:
  if not isinstance(n,(ast.FunctionDef,ast.ClassDef)):continue
  code=ast.get_source_segment(source,n)
  if isinstance(n,ast.FunctionDef) and n.decorator_list:code='\n'.join('@'+ast.get_source_segment(source,d) for d in n.decorator_list)+'\n'+code
  if n.name in checks and not solution:
   # Preserve signature and explanatory contract, then require learner implementation.
   code=code[:code.index('\n')]+ '\n    '+repr(ast.get_docstring(n))+'\n    raise NotImplementedError("TODO: '+n.name+'")'
  cells.append(nbf.v4.new_markdown_cell(('TODO' if n.name in checks and not solution else 'PROVIDED')+' · `'+n.name+'`'))
  cells.append(nbf.v4.new_code_cell(code))
  if n.name in checks:cells.append(nbf.v4.new_code_cell(checks[n.name]))
 cells.append(nbf.v4.new_markdown_cell('## RUN · short live forward/training check\n\nAll 188 graphs are loaded. Train fold 0 for two epochs × 10 updates with each readout. These fresh numbers are a smoke check, not the 30-epoch/10-fold author comparison and not the paper result.'))
 cells.append(nbf.v4.new_code_cell("graphs=load_mutag()\nsmoke=[]\nfor mode in ['sum','mean','max']:\n    r,_=train_fold(graphs,fold=0,epochs=2,iters=10,readout=mode)\n    smoke.append({'readout':mode,'curve':r['curves']})\nprint(json.dumps(smoke,indent=2))"))
 cells.append(nbf.v4.new_markdown_cell('## RUN · reproduce the complete teaching comparison\n\nSet `RUN_TEACHING=True` to obtain ten-fold readout measurements from your own implementation. It uses the declared small budget; it does not reproduce Table 1.'))
 cells.append(nbf.v4.new_code_cell("RUN_TEACHING = False\nlearner_results=[]\nif RUN_TEACHING:\n    for mode in ['sum','mean','max']:\n        records=[train_fold(graphs,fold=f,epochs=30,iters=10,readout=mode)[0] for f in range(10)]\n        curves=np.array([[e['val_acc'] for e in r['curves']] for r in records])\n        epoch=select_epoch(curves);values=curves[:,epoch]\n        learner_results.append({'readout':mode,'selected_epoch':epoch+1,'mean':float(values.mean()),'sample_sd':float(values.std(ddof=1)),'fold_values':values.tolist()})\n    print(learner_results)"))
 cells.append(nbf.v4.new_markdown_cell('## FULL REPRODUCTION · MUTAG Table 1 GIN-0\n\nSet `RUN_FULL_PAPER=True` for all 80 trainings. This may take hours on a laptop. This notebook path is sequential and fresh. The repository CLI parallelizes folds and verifies fingerprints before reusing completed runs. No parameter is downscaled here.'))
 cells.append(nbf.v4.new_code_cell("RUN_FULL_PAPER = False\nif RUN_FULL_PAPER:\n    records=[]\n    out=Path('l088-paper');out.mkdir(exist_ok=True)\n    for ci,config in enumerate(paper_grid()):\n        for fold in range(10):\n            r,z=train_fold(graphs,fold=fold,**config)\n            records.append(r)\n            (out/f'config{ci}-fold{fold}.json').write_text(json.dumps(r,indent=2))\n            np.savez_compressed(out/f'config{ci}-fold{fold}.npz',logits=z,val_ids=r['val_ids'],labels=[graphs[i]['y'] for i in r['val_ids']])\n    result=summarize_grid(records)\n    Path('l088-paper-results.json').write_text(json.dumps(result,indent=2))\n    print(result['selected'])"))
 cells.append(nbf.v4.new_markdown_cell('## EXIT · submit measurements and explanation\n\nRun the teaching comparison, fill all three explanations, then save this artifact. A smoke output is not a completed EXIT. Compare to the author evidence only after your predictions are written.'))
 cells.append(nbf.v4.new_code_cell("explanations={'raw_sum_collision':'','wl_counterexample':'','selection_optimism':''}\nexit_artifact={'complete':bool(learner_results) and all(explanations.values()),'readout_results':learner_results,'smoke':smoke,'explanations':explanations}\nPath('l088-exit.json').write_text(json.dumps(exit_artifact,indent=2))\nprint('EXIT complete:',exit_artifact['complete'])"))
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 path=LAB/('solutions' if solution else '')/f'{SLUG}.ipynb';nbf.write(nb,path)
 if not solution:
  exporter=HTMLExporter(template_name='lab');body,_=exporter.from_notebook_node(nb);(LAB/'html'/f'{SLUG}.html').write_text(body)
print('Built L088 HTML, notebooks, preview and runtime pins')
