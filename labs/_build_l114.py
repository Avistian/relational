"""Build a standalone analysis notebook and lesson from shared canonical sources."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0114-ogb-error-analysis';TITLE='OGB error analysis: where does the GCN lose?'
summary=json.loads((P/'evidence/l114/summary.json').read_text());canonical=(P/'relkit/error_l114.py').read_text();gcn=(P/'relkit/ogb_l112.py').read_text()
CAP={'architecture':'Released MLP computation: two normalized hidden blocks, 40 output scores, train-only BN population. The GCN comparison changes more than adjacency.','neighbors':'Synthetic four-node trace: unique neighbors, true-label agreement, known training-neighbor fraction, and an isolate.','composition':'Synthetic denominator example: 91 correct of 110 nodes gives 82.7%; averaging two slice percentages gives 50%.','slices':'Complete validation and test homophily slices. Points are ten-run means; whiskers are sample seed SD. Empty isolate slices are omitted from this figure and retained in the explorer.'}
TASKS={'neighborhood_properties':'Construct unique non-self undirected neighbors. Return degree, true-label homophily, and train-neighbor fraction; preserve undefined ratios.','slice_metrics':'Compare both prediction matrices on the same IDs. Return support, per-seed accuracies, pp differences and four disagreement counts.','choose_failure':'From validation rows only, nominate the smallest mean gap among slices with enough nodes; handle empty eligibility and stable ties.'}
checks_source=(P/'_check_l114.py').read_text();checks={n.name:ast.get_source_segment(checks_source,n) for n in ast.parse(checks_source).body if isinstance(n,ast.FunctionDef)}
check_names={'neighborhood_properties':'check_neighborhood','slice_metrics':'check_metrics','choose_failure':'check_choice'}
def results():
 s='**Author-reference evidence, separate from your notebook execution.**\n\n| Model / population | Mean accuracy | Seed SD | Paper mean | Verdict |\n|---|---:|---:|---:|---|\n'
 for model in ['gcn','mlp']:
  for pop in ['valid','test']:
   a=summary['summary'][model][pop];s+=f"| {model.upper()} / {pop} | {a['mean_percent']:.4f}% | {a['sample_sd_pp']:.4f} pp | {a['target_percent']:.2f}% | {a['verdict']} |\n"
 s+=f"\nGCN training reused from L112; MLP training fresh in L114. Both models' **3,386,860 combined final node predictions** were replayed through their original classes with zero class mismatches. The MLP's recorded pilot-plus-ten-run resource estimate is **USD{summary['successful_resource_usd']:.4f}**, excluding unitemized build/startup/storage overhead; the conservative resource ceiling is USD1.372536 within the USD10 aggregate plan. [Complete evidence](../labs/evidence/l114/summary.json)."
 return s

def selected():
 s='**Validation-nominated rule: true-label homophily < 0.25.**\n\n| Population | Nodes | GCN mean ± seed SD | MLP mean ± seed SD | GCN − MLP |\n|---|---:|---:|---:|---:|\n'
 for name,row in [('Validation',summary['selected_validation_slice']),('Test, same rule',summary['same_rule_on_test'])]:
  s+=f"| {name} | {row['n']:,} | {row['gcn_mean_percent']:.2f}% ± {row['gcn_sd_pp']:.2f} pp | {row['mlp_mean_percent']:.2f}% ± {row['mlp_sd_pp']:.2f} pp | {row['mean_delta_pp']:+.2f} pp |\n"
 r=summary['same_rule_on_test'];dis=r['discordance'];a=sum(x[1] for x in dis)/10;b=sum(x[2] for x in dis)/10
 s+=f"\nAcross seeds, the test slice averages **{a:.1f} GCN-only correct** and **{b:.1f} MLP-only correct** nodes. Their difference, divided by 8,585, gives the {r['mean_delta_pp']:.2f} pp accuracy gap. These are mean counts over ten runs, so decimals are expected. Both models are poor here: MLP's relative advantage does not make 27.22% accuracy adequate."
 return s

def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results()).replace('[[SELECTED]]',selected())
 for name,cap in CAP.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l114/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l114/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="error-figure" tabindex="0"><img src="{src}" alt="{cap}"><figcaption>{cap}</figcaption></figure>')
 warm='**Recall before reading:** What does validation select? Which labels enter a transductive GCN loss? Why is seed SD different from uncertainty over datasets?'
 s=s.replace('[[WARMUP]]',warm if portable else '<div id="l114-warmup"></div><noscript><p>'+warm+'</p></noscript>')
 teach='**Teach back:** Explain the global-versus-slice reversal, the denominator, and why true-label homophily cannot route future predictions.'
 s=s.replace('[[TEACHBACK]]',teach if portable else '<div id="l114-teachback"></div><noscript><p>'+teach+'</p></noscript>')
 if portable:s=s.replace('[[EXPLORER]]','The static figure above preserves the key measured comparison. The RUN section below prints full slice tables from your live functions; use those tables to inspect class, degree and year.')
 else:
  payload=json.dumps({'rows':summary['slice_rows']},ensure_ascii=False).replace('<','\\u003c')
  s=s.replace('[[EXPLORER]]','<div class="error-explorer" data-error-slices><script type="application/json">'+payload+'</script></div><noscript><p>The static figure and validation-nominated table above remain available without JavaScript. Complete slice rows are in the linked evidence JSON.</p></noscript>')
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def document(title,body,interactive=False):
 html=render(body).replace('<table>','<div class="stream-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 scripts=''.join('<script src="../assets/'+n+'.js"></script>' for n in ['retrieval-pool','retrieval-bank','teachback','error-slices','l114-lesson']) if interactive else ''
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/event-snapshot.css"><link rel="stylesheet" href="../assets/reproduction.css"><link rel="stylesheet" href="../assets/error-slices.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0113-scaling-ogb.html">Lesson 113</a></nav><header><p class="stream-kicker">Year 3 · Quarter 4 · Lesson 114</p><h1>'+title+'</h1></header>'+html+'</article>'+scripts+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
bootstrap='''# @colab-bootstrap · missing-package installation only; full author pins are linked below.
import importlib.util, importlib.metadata, subprocess, sys
required={'torch':'torch==2.8.0','numpy':'numpy==2.2.6','pandas':'pandas==2.3.2','ogb':'ogb==1.3.6','sklearn':'scikit-learn==1.7.1'}
missing=[package for module,package in required.items() if importlib.util.find_spec(module) is None]
if missing: subprocess.check_call([sys.executable,'-m','pip','install',*missing])
print('Python',sys.version)
print({k:importlib.metadata.version(k) for k in ['torch','numpy','pandas','ogb']})
'''
run='''# RUN: every official validation/test node, using your three live functions.
# The bundle contains frozen author predictions and raw edge/label/year arrays.
import urllib.request
bundle_path=Path('l114-analysis-inputs.npz')
if not bundle_path.exists():
    local=Path('labs/evidence/l114/analysis-inputs.npz')
    if local.exists():bundle_path=local
    else:urllib.request.urlretrieve('https://raw.githubusercontent.com/avistian/relational/main/labs/evidence/l114/analysis-inputs.npz',bundle_path)
EXPECTED_BUNDLE_SHA = '__HASH__'
assert file_hash(bundle_path)==EXPECTED_BUNDLE_SHA,'Unexpected evidence bundle: stop and check version'
with np.load(bundle_path) as z:bundle={k:z[k] for k in z.files}
labels=bundle['labels'];props=neighborhood_properties(bundle['edge'],labels,bundle['train'])
for key in props:np.testing.assert_allclose(props[key],bundle[key],rtol=0,atol=0,equal_nan=True)
masks=slice_masks(props,labels,bundle['year']);rows=[]
for pop in ['valid','test']:
    ids=bundle[pop]
    for (family,name),mask in masks.items():
        result=slice_metrics(labels,bundle['gcn'],bundle['mlp'],ids[mask[ids]])
        rows.append({'population':pop,'family':family,'slice':name,**result})
chosen=choose_failure([r for r in rows if r['population']=='valid' and r['family']!='year'])
heldout=next(r for r in rows if r['population']=='test' and (r['family'],r['slice'])==(chosen['family'],chosen['slice']))
report={'status':'ANALYSIS_RECOMPUTED','training':'archived author GCN and MLP predictions, no new training in this cell','selected_validation_slice':chosen,'same_rule_on_test':heldout,'learner_status':'PENDING_WRITTEN_DEFENSE'}
Path('l114-error-report.json').write_text(json.dumps(report,indent=2))
table=[]
for r in rows:
    table.append({'population':r['population'],'family':r['family'],'slice':r['slice'],'n':r['n'],'GCN %':100*np.mean(r['gcn']) if r['n'] else np.nan,'MLP %':100*np.mean(r['mlp']) if r['n'] else np.nan,'gap pp':r['mean_delta_pp']})
frame=pd.DataFrame(table)
for family in ['degree','homophily','class','year']:
    print(family);display(frame[(frame.family==family)&(frame.population=='test')])
print('Validation nomination:',chosen['family'],chosen['slice'],'n=',chosen['n'])
print('Same rule on test: n=',heldout['n'],'gap pp=',heldout['mean_delta_pp'])
print('Provide your written explanation; passing this cell is not mastery.')
'''.replace('__HASH__',summary['analysis_inputs_sha256'])
gate='''# Full named experiment: fresh MLP fits, no hidden library model.
# Requires the original 80 MiB archive, preferably GPU, and an external budget/runtime cap.
# Default OFF: the analysis above is not fresh training.
RUN_FULL_REPRODUCTION = False
RUN_FRESH_GCN = False  # optional L112 replay from scratch, a separate compute allocation
if RUN_FULL_REPRODUCTION or RUN_FRESH_GCN:
    import tempfile
    torch.set_num_threads(1)
    x,edge,y,split,audit=load_arxiv(Path('l114-data'))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    root=Path(tempfile.mkdtemp(prefix='l114-full-'))
    if RUN_FULL_REPRODUCTION:
        records=[train_mlp(x,y,split,seed,500,root/'mlp'/f'seed-{seed}',device) for seed in range(10)]
        for pop,target in {'valid':57.65,'test':55.50}.items():
            values=np.array([r['scores'][pop]*100 for r in records])
            print(pop,'mean',values.mean(),'seed SD',values.std(ddof=1),'CLOSE' if abs(values.mean()-target)<=.5 else 'OUTSIDE_TOLERANCE')
    if RUN_FRESH_GCN:
        adj=normalized_adjacency(edge,len(y))
        gcn_records=[train_run(x,adj,y,split,seed,500,root/'gcn'/f'seed-{seed}',device) for seed in range(10)]
else:
    print('Fresh training NOT_RUN in this kernel; full author MLP experiment is separate evidence.')
'''
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 114 · '+TITLE+'\n\n'+('Teacher solution' if solution else 'Student lab')+' · Three live TODO/CHECK tasks. Full graph analysis uses archived predictions; full training is separately gated.'),nbf.v4.new_code_cell(bootstrap),nbf.v4.new_markdown_cell('## Runtime and evidence\n\nAllow about 2 GiB RAM for the default analysis. The 8 MiB evidence bundle is SHA-256 checked and contains every prediction, all raw edges, labels, years and split IDs. Before this lesson is published, use a local repository checkout or copy `labs/evidence/l114/analysis-inputs.npz` beside this notebook as `l114-analysis-inputs.npz`; the remote fallback only works after publication. Author runtime: Python 3.12, torch 2.8.0, NumPy 2.2.6, pandas 2.3.2, OGB 1.3.6, scikit-learn 1.7.1. Original GCN parity additionally uses PyG 2.6.1. Live Colab NOT_CHECKED. [Exact protocol](https://avistian.github.io/relational/labs/l114-reproduction.md).')]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if section.strip():cells.append(nbf.v4.new_markdown_cell(section))
 cells.append(nbf.v4.new_markdown_cell('## PROVIDED · inherited GCN reproduction\n\nThe complete L112 GCN, raw reader and trainer are visible below so the baseline can be regenerated. These definitions do not start training. [MIT source license](https://github.com/snap-stanford/ogb/blob/61e9784ca76edeaa6e259ba0f836099608ff0586/LICENSE). The three L114 tasks follow this provided reproduction code.'))
 for chunk in re.split(r'^# %% ',gcn,flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);cells.extend([nbf.v4.new_markdown_cell('### PROVIDED · '+heading.replace('Task 1: ','').replace('Task 2: ','').replace('Task 3: ','')),nbf.v4.new_code_cell(body.strip())])
 cells.append(nbf.v4.new_markdown_cell('## Visible MLP and analysis\n\nMLP blocks correspond to the released `MLP.forward`; training corresponds to its `train`/`test` functions. The analysis tasks are original course mechanisms, independently checked against graph joins and integer counts.'))
 for chunk in re.split(r'^# %% ',canonical,flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);task=next((k for k in TASKS if 'def '+k+'(' in body),None)
  cells.append(nbf.v4.new_markdown_cell('### '+heading+('\n\n**Goal:** '+TASKS[task] if task else ' · PROVIDED')))
  if task and not solution:
   node=next(n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name==task);body=body.splitlines()[node.lineno-1]+'\n    raise NotImplementedError("TODO: '+task+'")\n'
  cells.append(nbf.v4.new_code_cell(body.strip()))
  if task:
   check=check_names[task];cells.append(nbf.v4.new_code_cell(checks[check]+'\n'+check+'('+task+')\nprint("PASS: '+task+'")'))
 cells.extend([nbf.v4.new_markdown_cell('## RUN · full census from your live analysis functions'),nbf.v4.new_code_cell(run),nbf.v4.new_markdown_cell('## Full training gate · optional execution, complete implementation\n\nBoth code paths use the complete raw dataset and every declared epoch/seed. Changing the schedule changes the experiment. See the protocol for the bounded Modal author run; these notebook switches have no automatic cloud spending authority.'),nbf.v4.new_code_cell(gate),nbf.v4.new_markdown_cell('## EXIT TICKET\n\nSubmit your three functions, `l114-error-report.json`, and the five-sentence error report. Include both accuracies, seed SDs, n and the rule-selection population. Explain the information boundary and one next controlled experiment. Ask the teacher for feedback. **PENDING_WRITTEN_DEFENSE**.')])
 for i,c in enumerate(cells):c.id=f'l114-{i:03d}'
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}});path=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and path.exists():
  old=nbf.read(path,as_version=4);aa=[c for c in cells if c.cell_type=='code'];bb=[c for c in old.cells if c.cell_type=='code']
  if [c.source for c in aa]==[c.source for c in bb]:
   nb.metadata=old.metadata
   for a,b in zip(aa,bb):a.outputs=b.outputs;a.execution_count=b.execution_count;a.metadata=b.metadata
 nbf.write(nb,path)
 if solution and all(c.execution_count is not None for c in cells if c.cell_type=='code'):
  html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
ref='''## A defensible slice report

Population → fixed slice rule → support n → per-seed same-node comparison → selection boundary → interpretation → next experiment.

## Definitions

Degree: distinct undirected non-self neighbors. Homophily: fraction sharing the focal true label; undefined when degree is zero. Training-neighbor fraction: fraction of neighbors in the training split; does not compare labels. True class and homophily are retrospective, not serving-time inputs.

Accuracy(S) = correct(S) / n(S). Gap in percentage points = 100 × (GCN-only correct − MLP-only correct) / n. Empty slice: undefined. For disjoint exhaustive slices, global accuracy = sum of n × slice accuracy / total n. Overlapping families cannot be added together.

## Selection and uncertainty

Nominate lowest validation mean gap among degree/class/homophily slices with n≥200; stable tie-break by family then label. Evaluate the same rule on test. Year is excluded from nomination because validation/test years differ. Ten-seed sample SD describes training variation on a fixed graph, not an IID-node confidence interval or generalization across databases.

## Reproduction identity

MLP128→256→256→40, hidden BN/ReLU/dropout0.5, train-row-only forward/BN, Adam0.01, ten seeds,500epochs, first best validation checkpoint. GCN weights reused from L112, with fresh all-node replay. Both use original split IDs. Original source parity and close mean scores do not recover historical randomness or whole-paper identity.

## Five sentences

Name the split/rule/n and how chosen. Report both mean±seedSD and pp gap. Count discordant errors. State hidden-label and causal limits. Propose one controlled next experiment and protect evaluation data.

[Lesson114](../lessons/0114-ogb-error-analysis.html) · [Exact commands](../labs/l114-reproduction.md) · [Full evidence](../labs/evidence/l114/summary.json) · [OGB Table6](https://arxiv.org/html/2005.00687v6#S4.SS3).
'''
(R/'reference/ogb-error-analysis.html').write_text(document('OGB error analysis · quick reference',ref))
print('Built lesson, reference, student and solution notebooks')
