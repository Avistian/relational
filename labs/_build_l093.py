"""One source for lesson HTML and standalone portable HGT notebooks."""
import ast,base64,hashlib,json,re
from pathlib import Path
from _walkthrough_delivery import snapshot, finalize
snapshot(93)
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0093-hgt';TITLE='HGT: typed attention through time'
manifest=json.loads((LAB/'_sources_l093.json').read_text())
figs={'ARCH_FIG':('architecture','End-to-end HGT reconstruction: typed inputs, sampled graph, separate attention/message branches, residual update and field-ranking handoff.'),'ATTENTION_FIG':('attention','Synthetic worked trace: joint weights1/3 and2/3 produce14/3. Doubling the citation logit prior produces5.2.'),'TIME_FIG':('time','Same2012 source, different receiving contexts: release table indices123 and126 change source keys and values.'),'SAMPLING_FIG':('sampling','Illustrative per-type budget [.5,1,.5] becomes sampling probabilities [1/6,2/3,1/6].'),'RESULT_FIG':('results','Measured NN teaching comparison over3 model seeds; one dataset, paired query streams, not CS paper reproduction.')}
def result_text():
 p=json.loads((LAB/'_teaching_l093_results.json').read_text());b=json.loads((LAB/'_baseline_l093_results.json').read_text())
 s='**Author-reference evidence, not your current kernel output.** Nine complete neural fits on the NN graph. Mean ± sample SD across3 seeds.\n\n| Arm | Trainable parameters | NN NDCG | NN MRR |\n|---|---:|---:|---:|\n'
 for arm,label in [('hgt','HGT'),('rgcn','R-GCN'),('hgt_no_rte','HGT without RTE')]:
  r=p['summary'][arm];count=next(v['parameters'] for v in p['runs'] if v['arm']==arm);s+=f"|{label}|{count:,}|{r['ndcg']['mean']:.4f} ± {r['ndcg']['sample_sd']:.4f}|{r['mrr']['mean']:.4f} ± {r['mrr']['sample_sd']:.4f}|\n"
 r=b['summary'];s+=f"|Train-frequency baseline|0|{r['ndcg']['mean']:.4f} ± {r['ndcg']['sample_sd']:.4f}|{r['mrr']['mean']:.4f} ± {r['mrr']['sample_sd']:.4f}|\n"
 s+='\nThe most frequent training field is **Artificial neural network**, present in every training and test paper in this NN subset. A training-only frequency ranking reaches MRR 1.0 and higher NDCG than these small neural runs. That diagnosis matters more than declaring a winner among nearly tied neural arms. The baseline SD only reflects sampled queries; it has no model-initialization variability.\n\n[[RESULT_FIG]]'
 return s
status='''**Executed:** layer output/attention/gradient parity; two source-sampler comparisons; nine small NN fits; training-frequency baseline; one paper-width update on NN. **Not executed:** full CS five-run target, historical runtime and full-paper suite. CS loading exceeded a declared 10 GiB virtual-memory guard; Modal rejected the GPU invocation because a payment method was required. Full-paper parity remains **NOT_ESTABLISHED**, historical comparison **INCOMPARABLE**. These are actual blockers, not successful reproduction results.'''
def prose(portable=False):
 s=(ROOT/'lessons/content'/f'{SLUG}.md').read_text().replace('[[RESULTS]]',result_text()).replace('[[REPRO_STATUS]]',status)
 for tag,(name,caption) in figs.items():
  p=LAB/f'figures/l093/{name}.png';src='data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode() if portable else '../labs/figures/l093/'+name+'.png'
  s=s.replace('[['+tag+']]',f'<figure class="mpnn-figure"><div class="figure-scroll" tabindex="0"><img src="{src}" alt="{caption}"></div><figcaption>{caption}</figcaption></figure>')
 for tag,id_,fallback in [('WARMUP','warmup','**Retrieve first:** complete the three cold questions below before checking the answers.'),('PREDICT','prediction','**Predict first:** should author and citation edges share a denominator? Commit before reading the example.'),('ATTENTION_WIDGET','attention-widget','**Intervene:** set the citation prior to0,1,2. Compute weights and the first output coordinate. CHECK: outputs4,14/3,5.2.'),('TEACHBACK','teachback','**Teach back:** close the derivation and explain the full message, time and reproduction boundary in your own words.')]:s=s.replace('[['+tag+']]',fallback if portable else f'<div id="{id_}"></div>')
 if portable:
  s=re.sub(r'\]\((00\d\d-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s);s=s.replace('](../','](https://avistian.github.io/relational/')
 return s
head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 93 — {TITLE}</title>'+''.join(f'<link rel="stylesheet" href="../assets/{s}.css">' for s in ['lesson','mpnn-lesson','hgt-viz'])+'</head><body><article>'
head+='<nav><a href="../index.html">Course</a> · <a href="0092-meta-paths.html">Lesson92</a> · <a href="../reference/hgt.html">Reference</a></nav><header><p>Year 3 · Quarter 2 · Lesson 093</p><h1>'+TITLE+'</h1></header>'
footer='</article>'+''.join(f'<script src="../assets/{s}.js"></script>' for s in ['retrieval-pool','retrieval-bank','predict','teachback','hgt-viz','l093-lesson'])+'</body></html>'
(ROOT/'lessons'/f'{SLUG}.html').write_text(head+render(prose())+footer)
checks={
'receiver_softmax':"z=torch.tensor([[0.],[math.log(2)],[0.]],requires_grad=True)\nd=torch.tensor([2,2,3]);w=receiver_softmax(z,d,5)\ntorch.testing.assert_close(w,torch.tensor([[1/3],[2/3],[1.]]))\n(w*torch.tensor([[2.],[6.],[1.]])).sum().backward();assert z.grad.abs().sum()>0\nprint('PASS: receivers are independent; different relations compete; gradients flow')",
'relation_heads':"q=torch.tensor([[[1.,0.]],[[1.,0.]]]);k=torch.tensor([[[0.,1.]],[[math.sqrt(2)*math.log(2),0.]]],requires_grad=True)\nv=torch.tensor([[[2.,0.]],[[6.,0.]]]);w=torch.eye(2).reshape(1,1,2,2).expand(2,1,2,2);p=torch.ones(2,1)\ns,m=relation_heads(q,k,v,w,w,p);weights=receiver_softmax(s,torch.tensor([0,0]),1)\nvalue=(weights[:,:,None]*m).sum(0)\ntorch.testing.assert_close(value,torch.tensor([[14/3,0.]]));value.sum().backward();assert k.grad.abs().sum()>0\np[1]=2;s,m=relation_heads(q,k,v,w,w,p);changed=(receiver_softmax(s,torch.tensor([0,0]),1)[:,:,None]*m).sum(0)\ntorch.testing.assert_close(changed,torch.tensor([[5.2,0.]]))\nprint('PASS: relation transforms, head-width scaling and intervention')",
 'temporal_basis':"b=temporal_basis(torch.tensor([0.,1.]),4,'release');torch.testing.assert_close(b[0],torch.tensor([0.,.5,0.,.5]))\nexpected=torch.tensor([math.sin(1),math.cos(1),math.sin(.01),math.cos(.01)])/2\ntorch.testing.assert_close(b[1],expected)\np=temporal_basis(torch.tensor([1.]),4,'paper');torch.testing.assert_close(p[0],torch.tensor([math.sin(1),math.cos(.1),math.sin(.01),math.cos(.001)]))\nassert not torch.allclose(temporal_basis(torch.tensor([123.]),4),temporal_basis(torch.tensor([126.]),4))\nprint('PASS: frequency, scale, and changed relative time')"}
descriptions={'receiver_softmax':'Group by receiver and head. Normalize across every incoming relation. Preserve finite outputs and gradients; isolated nodes receive no edges.','relation_heads':'Apply different relation matrices to keys and values. Compute scaled compatibility with the query and prior. Return edge logits and messages.','temporal_basis':'Implement the stated release and printed-paper sinusoidal conventions. Preserve index order, even/odd coordinates and release scale.'}
annotations={'HGTLayer':'Paper§3, Figure2, Eqs3–5 versus the pinned OAG/pyHGT/conv.py. Trace source/target indexing, relation grouping, one joint softmax and the released gated update.','RelativeTime':'Paper§3.5 versus release. The table remains trainable to preserve the source bug. Index validation rejects unsupported gaps.','HGTModel':'Type adapters → repeated HGT layers → seed-paper gather → field log-probabilities. R-GCN uses the same adapters and classifier dimensions.','RGCNLayer':'Teaching comparator: relation-specific means, sum across relations and one self transform. This is not the original Table 2 baseline implementation.','OAGUnpickler':'Compatibility for stored data only: old serialized defaultdict factories are not executed. Original Python3.7 loading identity remains untested.','field_protocol':'Released L2 candidates and non-overlapping year boundaries. Candidate order and declared shuffle are part of the experiment.','sample_oag':'Paper Algorithms1–2 plus released capped expansion, inherited time, induced edges and bidirectional target-label masking. Read the entire function before training.','ranking_metrics':'Released DCG method0, not the commonly assumed rank+1 denominator.','train_oag':'Complete trainer: KL targets, AdamW, cosine step offset1500, validation-only selection and held-out ranking.','frequency_baseline':'Training-label frequencies reveal the universal NN field. Test labels are used only to score the fixed ranking.'}
bootstrap="""# @colab-bootstrap
import os,sys,subprocess
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable,'-m','pip','install','numpy==2.2.6','pandas==2.3.2','torch==2.8.0','dill==0.3.8'])
import importlib.metadata as metadata
print({k:metadata.version(k) for k in ['numpy','pandas','torch','dill']})
"""
source=(LAB/'relkit/hgt_l093.py').read_text();lines=source.splitlines()
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# '+TITLE+'\n\nA standalone paper-mirror lab. PROVIDED code is visible. Complete the three TODO functions, then run their immediate CHECK cells. EXIT requires an interpretation, not just successful execution. Default: one real NN diagnostic; longer runs use explicit switches.'),nbf.v4.new_code_cell(bootstrap)]
 cells +=[nbf.v4.new_markdown_cell(chunk) for chunk in re.split(r'(?=\n## )',prose(True)) if chunk.strip()]
 cells +=[nbf.v4.new_markdown_cell('## PROVIDED · Source and data provenance\nThe manifest pins both NN and CS downloads. Download hashes establish byte identity to our retrieved release, not to the unpublished historical experiment state.'),nbf.v4.new_code_cell('SOURCE_MANIFEST='+repr(manifest))]
 for node in ast.parse(source).body:
  code='\n'.join(lines[node.lineno-1:node.end_lineno]);name=getattr(node,'name','')
  if name in checks:
   cells.append(nbf.v4.new_markdown_cell('### TODO · '+name+'\n'+descriptions[name]))
   if not solution:code=code[:code.index(':')+1]+'\n    raise NotImplementedError("Implement '+name+' before running CHECK")'
   cells.append(nbf.v4.new_code_cell(code,metadata={'tags':['solution' if solution else 'todo'],'task':name}));cells.append(nbf.v4.new_code_cell('# CHECK\n'+checks[name],metadata={'tags':['check']}))
  else:
   if name:cells.append(nbf.v4.new_markdown_cell('### PROVIDED · '+name+'\n'+annotations.get(name,'Supporting code for the visible OAG implementation. Read the docstring and follow the data contract.')))
   cells.append(nbf.v4.new_code_cell(code))
 cells +=[nbf.v4.new_markdown_cell('## CHECK · Download and one real NN update\nThe656MB download is checked before loading. It needs several GB of free memory. This diagnostic is not a paper-table score. Every live TODO is called by the actual model.'),nbf.v4.new_code_cell("""torch.set_num_threads(2)
DATA_DIR=Path(os.environ.get('L093_DATA_DIR','l093-data'));DATA_DIR.mkdir(parents=True,exist_ok=True)
path=DATA_DIR/'graph_NN.pk'
if not path.exists():
    temporary=path.with_suffix('.pk.part');urllib.request.urlretrieve(SOURCE_MANIFEST['nn_data']['url'],temporary)
    assert file_sha256(temporary)==SOURCE_MANIFEST['nn_data']['sha256'];temporary.replace(path)
graph=load_oag(path,SOURCE_MANIFEST['nn_data']['sha256'])
assert HGTLayer.forward.__globals__['receiver_softmax'] is receiver_softmax
assert HGTLayer.forward.__globals__['relation_heads'] is relation_heads
assert RelativeTime.__init__.__globals__['temporal_basis'] is temporal_basis
smoke,smoke_model=train_oag(graph,0,protocol_config('smoke'))
print({'diagnostic':{'NDCG':smoke['test_ndcg'],'MRR':smoke['test_mrr']},'training_frequency':frequency_baseline(graph,smoke['test_queries'])})
"""),nbf.v4.new_markdown_cell('## CHECK · Fixed-weight temporal intervention\nUse the same sampled graph and weights, change only the time index. Then bypass RTE. These are inference interventions, not retrained ablations.'),nbf.v4.new_code_cell("""candidates,pairs=field_protocol(graph)
batch,audit=sample_oag(graph,pairs['valid'],candidates,123,8,2,8,2016)
smoke_model.eval()
with torch.no_grad():
    baseline=smoke_model(batch)
    changed=list(batch);changed[4]=(batch[4]+1).clamp(max=239);time_changed=smoke_model(tuple(changed))
    for layer in smoke_model.layers:layer.use_rte=False
    without_time=smoke_model(batch)
assert not torch.allclose(baseline,time_changed)
assert not torch.allclose(baseline,without_time)
print({'changed_time_max_logprob':float((baseline-time_changed).abs().max()),'disabled_RTE_max_logprob':float((baseline-without_time).abs().max())})
"""),nbf.v4.new_markdown_cell('## RUN · Matched local comparison\nThree seeds × three arms;40 updates per fit. Several minutes on CPU. NN evidence only. The full implementation above stays live.'),nbf.v4.new_code_cell("""RUN_TEACHING = os.environ.get('L093_RUN_TEACHING','0') == '1'
teaching_runs=[]
if RUN_TEACHING:
    for seed in [0,1,2]:
        for arm in ['hgt','rgcn','hgt_no_rte']:
            row,_=train_oag(graph,seed,protocol_config('teaching'),arm)
            teaching_runs.append(row)
            print({'seed':seed,'arm':arm,'NDCG':row['test_ndcg'],'MRR':row['test_mrr']})
    Path('l093-inline-teaching.json').write_text(json.dumps({'runs':teaching_runs},indent=2))
    display(pd.DataFrame([{'seed':r['seed'],'arm':r['arm'],'NDCG':r['test_ndcg'],'MRR':r['test_mrr']} for r in teaching_runs]))
else:
    print('Teaching comparison: NOT_RUN in this kernel')
"""),nbf.v4.new_markdown_cell('## EXIT · Written defense\nIn150–250 words derive the shared softmax, explain RTE and its source differences, diagnose the universal-field MRR, and separate NN evidence from CS reproduction. Submit your checks and per-seed table. Ask the teacher to challenge the weakest inference.'),nbf.v4.new_code_cell("""ticket={'lesson':93,'NN_fits_in_this_kernel':len(teaching_runs),'CS_full_target':'NOT_RUN','historical_parity':'INCOMPARABLE','full_paper_parity':'NOT_ESTABLISHED','learner_mastery':'PENDING_WRITTEN_DEFENSE'}
Path('l093-exit.json').write_text(json.dumps(ticket,indent=2));print(ticket)
"""),nbf.v4.new_markdown_cell('## NEXT STEP · Complete CS target reconstruction\nGated off. Requires8.61GB CS download plus substantially more host RAM after decoding and GPU memory for training. Local loader exceeded a 10 GiB address-space guard; GPU invocation was blocked by account billing. `paper` uses256/3 and validation loss; `release` uses400/4 and validation NDCG. Both retain modern released operators, not the archived2020 architecture. Original features, historical backend, seeds and table-producing revision remain unresolved. The same visible functions train all five seeds. Do not replace the CS file with NN.'),nbf.v4.new_code_cell("""RUN_FULL_REPRO = False
FULL_PRESET='paper'
if RUN_FULL_REPRO:
    cs_path=DATA_DIR/'graph_CS.pk'
    if not cs_path.exists():
        url='https://drive.usercontent.google.com/download?id='+SOURCE_MANIFEST['data_links']['CS']+'&export=download&confirm=t'
        tmp=cs_path.with_suffix('.pk.part');urllib.request.urlretrieve(url,tmp)
        assert file_sha256(tmp)==SOURCE_MANIFEST['cs_data']['sha256'];tmp.replace(cs_path)
    del graph
    import gc;gc.collect()
    cs_graph=load_oag(cs_path,SOURCE_MANIFEST['cs_data']['sha256'])
    config=protocol_config(FULL_PRESET);config['device']='cuda' if torch.cuda.is_available() else 'cpu'
    cs_runs=[]
    for seed in range(5):
        row,_=train_oag(cs_graph,seed,config,'hgt',progress=True);cs_runs.append(row)
        Path('l093-CS-'+FULL_PRESET+'.json').write_text(json.dumps({'runs':cs_runs,'full_paper_parity':'NOT_ESTABLISHED'},indent=2))
    print('Five reconstruction fits completed; historical parity still requires the deviation audit.')
else:
    print('Full CS reproduction: NOT_RUN in this kernel')
""")]
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python3','language':'python'}});path=LAB/('solutions' if solution else '')/(SLUG+'.ipynb')
 if solution and path.exists():
  old=nbf.read(path,as_version=4);before=[c for c in old.cells if c.cell_type=='code'];after=[c for c in nb.cells if c.cell_type=='code']
  if [c.source for c in before]==[c.source for c in after]:
   for a,b in zip(before,after):b.execution_count=a.execution_count;b.outputs=a.outputs
 for i,c in enumerate(nb.cells):c.id=f'l093-{i:03}-'+hashlib.sha256(c.source.encode()).hexdigest()[:8]
 nbf.validate(nb);nbf.write(nb,path)
 if not solution:
  html,_=HTMLExporter().from_notebook_node(nb);(LAB/'html'/f'{SLUG}.html').write_text(html)

finalize(93)
