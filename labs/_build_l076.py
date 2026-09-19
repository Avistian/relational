"""Build coherent lesson, student/solution, portable figures and reference."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
from bs4 import BeautifulSoup
from _colab import bootstrap_cells
from _figures_l076 import build as figures
LAB=Path(__file__).resolve().parent;ROOT=LAB.parent;SLUG='0076-encoder-predictor-stack';TITLE='Encoder → predictor: the relational stack'
SOURCE=(LAB/'relkit/stack_l076.py').read_text();TREE=ast.parse(SOURCE)
NODES={n.name:ast.get_source_segment(SOURCE,n) for n in TREE.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
IMPORTS='\n'.join(ast.get_source_segment(SOURCE,n) for n in TREE.body if isinstance(n,(ast.Import,ast.ImportFrom)))
CAPTIONS={'time':'Day10 cutoff: event3 happened on day4 but arrived on day11. Event4 is later still. Both are excluded; the first three events are eligible.',
'architecture':'Complete teaching variant: two independent Frame encoders, one eligible event-to-customer hop, mean, concatenation and binary head. d=D=4; keys are metadata.',
'routing':'Illustrative width-2 embeddings, distinct from learned width-4 rows. Customers42 and7 have equal means despite different histories. Event3 arrives late; event4 is in the future.',
'gradients':'Backward trace for customer7 with two eligible events. I denotes the identity matrix; each message receives one half of the arriving derivative. Excluded event embeddings have no loss path.'}
CHECKS={
'key_positions':'''assert key_positions([42,7,99,105],[7,42,7,99,42]).tolist()==[1,0,1,2,0], 'Map identities, not values or sort order'
assert key_positions([99,42,105,7],[7,42]).tolist()==[3,1]
for pk,fk in [([1,1],[1]),([1],[2])]:
    try: key_positions(pk,fk)
    except ValueError: pass
    else: raise AssertionError('Reject duplicate PKs and dangling FKs')
print('CHECK: key identity and integrity pass')''',
'eligible_events':'''actual = eligible_events(torch.tensor([1,2,3,4,12]),torch.tensor([1,2,3,11,12]),10)
assert actual.tolist()==[True,True,True,False,False], 'An old event can arrive late'
assert eligible_events(torch.tensor([10]),torch.tensor([10]),10).item(), 'Inclusive boundary'
print('CHECK: availability and inclusive cutoff pass')''',
'mean_messages':'''m = torch.tensor([[2.,1.],[4.,3.],[6.,5.]],requires_grad=True)
a,n = mean_messages(m,torch.tensor([1,0,1]),4)
assert torch.equal(a,torch.tensor([[4.,3.],[4.,3.],[0.,0.],[0.,0.]])), 'Repeated destinations must accumulate'
assert n.tolist()==[1,2,0,0], 'Count eligible events per receiver'
a[1].sum().backward()
assert torch.equal(m.grad,torch.tensor([[.5,.5],[0.,0.],[.5,.5]])), 'Do not detach messages'
empty,count = mean_messages(m[:0],torch.empty(0,dtype=torch.long),4)
assert torch.equal(empty,torch.zeros(4,2)) and not count.any()
print('CHECK: means, empty groups and gradients pass')'''}
GOALS={'key_positions':'Return a LongTensor of destination positions, preserving foreign-key order. Reject nonunique primary keys and unknown foreign keys. Why: valid shapes cannot prove correct identity.',
'eligible_events':'Return a boolean mask using event and availability days and the inclusive cutoff. Why: late-arriving history is not usable merely because its event time is old.',
'mean_messages':'Return differentiable per-destination means and integer counts. Handle repeated destinations and empty groups. Why: this operation determines both information access and gradient allocation.'}

def manuscript(notebook=False):
 t=(ROOT/'lessons/content'/f'{SLUG}.md').read_text();r=json.loads((LAB/'_verify_l076_results.json').read_text());d=r['diagnostic']
 evidence=f'''**Author-reference evidence — measured locally, not your current kernel output.**

| Measurement | Observed | Interpretation |
|---|---|---|
| First / final training loss | {d['first_loss']:.6f} / {d['final_loss']:.6f} | Four-row overfit diagnostic only |
| Counts in customer order [42,7,99,105] | {d['counts']} | Late and future events excluded |
| Customer / event row vectors | [4,4] / [5,4] | Actual PyTorch Frame path |
| Concatenated / logit shape | [4,8] / [4] | One prediction per customer |
| PyG mean primitive maximum error | {r['primitive_max_error']:.1e} | Primitive forward parity |
| Key/time/gradient/permutation checks | PASS | See saved verifier report |
| Historical RelBench training | NOT_RUN | No measured paper AUROC |
'''
 t=t.replace('<!--results-->',evidence)
 for name,caption in CAPTIONS.items():
  file=LAB/'figures/l076'/f'{name}.png';src='data:image/png;base64,'+base64.b64encode(file.read_bytes()).decode() if notebook else f'../labs/figures/l076/{name}.png'
  t=t.replace('<!--figure:'+name+'-->',f'<figure class="rdl-figure"><div class="rdl-figure-scroll" tabindex="0" role="region" aria-label="Scrollable {name} diagram"><img src="{src}" alt="{caption}"></div><figcaption>{caption}</figcaption></figure>')
 if notebook:
  t=re.sub(r'<div id="[^"]+"></div>','',t).replace('(0075-','(../lessons/0075-')
 return t

def notebook(solution=False):
 cells=[]
 def md(s):cells.append(nbf.v4.new_markdown_cell(s))
 def code(s):cells.append(nbf.v4.new_code_cell(s))
 md('# Lab076 · '+TITLE+'\n\n**Skill:** connect real typed row encoders to a manual one-hop aggregate and trainable head. TierC fixture; no benchmark claim. The optional historical RelBench experiment follows EXIT. Read the portable lesson, implement three TODOs, run CHECKs and export the trace.\n\n[Lesson](../lessons/'+SLUG+'.html) · [Reproduction contract](../labs/l076-reproduction.md)')
 cells.extend(nbf.from_dict(c) for c in bootstrap_cells())
 code("# PROVIDED — default lab is pinned to the L075 API. Historical replay uses an isolated interpreter.\nimport importlib.metadata\nassert importlib.metadata.version('pytorch-frame') == '0.3.0'\n")
 for part in re.split(r'(?=^## )',manuscript(True),flags=re.M):
  if part.strip():md(part)
 md('<a id="lab-exercises"></a>\n## Implement the live stack\n\nThe following cells contain the complete teaching implementation. No model is hidden behind a relkit import. In particular, materialize_tables and RelationalStack call your live functions. Run locally with the working directory set to labs/.')
 code(IMPORTS+'\nimport json\nfrom pathlib import Path\ntorch.set_num_threads(1)\ntorch.manual_seed(76)')
 md('### PROVIDED · exact fixture\n\nFour customer identities and five events. Customer99 has an event, but it is unavailable at day10. Customer105 has none. Labels are diagnostic targets only.')
 code(NODES['fixture'])
 for name in GOALS:
  md('### TODO · '+name+'\n\n'+GOALS[name]+'\n\nWrite your expected output before running the CHECK.')
  code(NODES[name] if solution else NODES[name].split('\n')[0]+'\n    raise NotImplementedError("Implement '+name+'")')
  code('# CHECK — do not edit\n'+CHECKS[name])
 for name,explanation in [('materialize_tables','Keep key/time columns outside input features. Fit event statistics only on allowed records, then reuse the converter.'),('TableEncoder','The library provides typed encoders. The projection/readout is visible. Separate instances give each table its own learned parameters.'),('RelationalStack','Trace encoder → eligible gather → your mean function → concatenate → head. The returned trace exposes every boundary.'),('run_diagnostic','Complete full-batch trainer. This is four-row overfitting, not held-out evaluation. The function resolves the live classes and TODOs above.')]:
  md('### PROVIDED · '+name+'\n\n'+explanation);code(NODES[name])
 md('### CHECK · future intervention and complete gradients\n\nPredict whether changing excluded amounts can alter outputs. Notice that this check also refits allowed materialization state: edge filtering alone is not sufficient.')
 code('''customers,events,labels = fixture()
cd,ed,et,destination,mask = materialize_tables(customers,events)
model = RelationalStack(cd,ed)
logits,trace = model(cd.tensor_frame,et,destination,mask)
assert logits.shape==(4,) and trace['combined'].shape==(4,8)
changed=events.copy();changed.loc[~mask.numpy(),'amount']=1e8
cd2,ed2,et2,d2,m2=materialize_tables(customers,changed)
assert ed.col_stats==ed2.col_stats
assert torch.equal(logits,model(cd2.tensor_frame,et2,d2,m2)[0])
trace['events'].retain_grad()
nn.functional.binary_cross_entropy_with_logits(logits,labels).backward()
assert torch.equal(trace['events'].grad[~mask],torch.zeros_like(trace['events'].grad[~mask]))
for block in [model.customer_encoder,model.event_encoder,model.head]:
    gradients=[p.grad for p in block.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients)>0
print('CHECK: future invariance and complete gradient path pass')''')
 md('### EXIT · export evidence and explain its limits\n\nRun the diagnostic, then save its result. In your own words: (1) why does customer99 receive zero neighbors; (2) where does each one-half gradient arise; (3) what else is needed before calling this a useful predictor? The saved file does not grade your written explanation.')
 code("result=run_diagnostic()\nassert result['final_loss'] < result['first_loss']/10\nPath('l076-student-results.json').write_text(json.dumps(result,indent=2)+'\\n')\nprint('Shapes:',result['shapes'])\nprint('Counts:',result['counts'])\nprint('Loss:',result['first_loss'],'→',result['final_loss'])")
 md('### EXIT · your explanation\n\nWrite your answer here. Then permute customer rows and rebuild the key mapping. Explain the relationship between the old and new predictions before running them. Send the trace and your explanation to the teaching agent for grading.')
 md('## NEXT STEP · full historical model and trainer\n\n**NOT_RUN locally.** Target: RelBench v1 Table6 rel-f1/driver-dnf. This is a separate model from the toy. The following complete archived files are shown as source references; the gated operator executes these same pinned files in an isolated environment. Do not execute fragments in the default Frame0.3.0 kernel. Read the MIT license at the end of this appendix.\n\n[Protocol, environment, commands and deviations](../labs/l076-reproduction.md)')
 for name,why in [('relbench/modeling/nn.py','Table-specific ResNet encoders, relative-time encoders and heterogeneous GraphSAGE layers.'),('examples/model.py','Complete forward composition, including seed-node readout and task head.'),('examples/text_embedder.py','The external GloVe adapter. Its checkpoint revision is a remaining reproducibility gap.'),('examples/gnn_node.py','Complete original data preparation, temporal loaders, objective, training loop, checkpoint selection and final evaluation.')]:
  md('### Archived '+name+'\n\n'+why+'\n\n```python\n'+(LAB/'sources/l076-relbench'/name).read_text()+'\n```')
 md('### MIT attribution\n\n```text\n'+(LAB/'sources/l076-relbench/LICENSE').read_text()+'\n```')
 md('### Gated historical replay on Colab\n\nChoose a supported x86-64 T4 runtime. Set the switch only when ready for downloads and training. This uses an isolated Python3.10 environment; it does not downgrade this notebook kernel. Environment installation, live Colab and cloud training are NOT_CHECKED here. The five chosen seeds are not known to be the paper\'s original seed IDs.')
 code('''RUN_PAPER_REPRO = False
if RUN_PAPER_REPRO:
    import subprocess,sys,platform
    assert platform.machine() in ('x86_64','AMD64'), 'Use a supported x86-64 runtime'
    subprocess.run([sys.executable,'-m','pip','install','uv==0.6.17'],check=True)
    subprocess.run([sys.executable,'-m','uv','python','install','3.10'],check=True)
    subprocess.run([sys.executable,'-m','uv','venv','--python','3.10','/content/l076-paper-env'],check=True)
    py='/content/l076-paper-env/bin/python'
    prefix=[sys.executable,'-m','uv','pip','install','--python',py]
    subprocess.run(prefix+['torch==2.3.0','--index-url','https://download.pytorch.org/whl/cu121'],check=True)
    subprocess.run(prefix+['pyg-lib==0.4.0','--no-index','--find-links','https://data.pyg.org/whl/torch-2.3.0+cu121.html'],check=True)
    subprocess.run(prefix+['pip','-r','requirements-l076-replay.txt'],check=True)
    subprocess.run([py,'_replay_l076.py','--run','--seeds','0','1','2','3','4','--output','/content/l076-paper-results'],check=True)
else:
    print('Full RelBench release replay NOT_RUN; see the reproduction contract.')''')
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}});return nb

def build():
 figures();head='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson076 — '+TITLE+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/rdl-stack-viz.css"></head><body><article>'
 nav=f'<nav><a href="../index.html">Course</a> · <a href="0075-pytorch-frame-row-encoder.html">Lesson75</a></nav><header><p>Year2 · Quarter4 · Lesson076</p><h1>{TITLE}</h1><p>Encode records → route eligible messages → predict → differentiate.</p></header><aside><a href="../labs/html/{SLUG}.html">Read lab</a> · <a href="../labs/{SLUG}.ipynb" download>Download notebook</a> · <a href="../reference/{SLUG}.html">Reference</a> · <a href="../labs/l076-reproduction.md">Reproduction contract</a> · <a href="../labs/_verify_l076_results.json">Measured evidence</a></aside>'
 body=render(manuscript()).replace('<table>','<div class="result-scroll"><table>').replace('</table>','</table></div>')
 scripts=''.join(f'<script src="../assets/{s}.js"></script>' for s in ['retrieval-pool','retrieval-bank','predict','teachback','rdl-stack-viz','l076-lesson'])
 (ROOT/'lessons'/f'{SLUG}.html').write_text(head+nav+body+'<div id="teachback"></div></article>'+scripts+'</body></html>')
 for solution in [False,True]:
  nb=notebook(solution);p=LAB/('solutions' if solution else '')/f'{SLUG}.ipynb'
  if solution and p.exists():
   old=nbf.read(p,as_version=4);codes=[c for c in old.cells if c.cell_type=='code'];new=[c for c in nb.cells if c.cell_type=='code']
   if [c.source for c in codes]==[c.source for c in new]:
    for a,b in zip(new,codes):a.outputs=b.outputs;a.execution_count=b.execution_count
  nbf.write(nb,p)
 preview,_=HTMLExporter().from_notebook_node(notebook());soup=BeautifulSoup(preview,'html.parser')
 for a in soup.find_all('a',href=True):
  if a['href'].startswith('../'):a['href']='../'+a['href']
 (LAB/'html'/f'{SLUG}.html').write_text(str(soup)+'\n')
 ref='''# Encoder → predictor: trace card

**Identity.** PK identifies records. FK points to a PK; map it to the current tensor position. Reject dangling FKs and duplicate PKs.

**Time.** Our fixture permits event_day≤cutoff AND available_day≤cutoff. Also protect fitted statistics. Per-query times require per-query eligibility, not one global mask.

**Shapes.** Customer tokens[4,2,4]→rows[4,4]; event tokens[5,2,4]→rows[5,4]; eligible events[3,4]→mean neighbors[4,4]; concatenate[4,8]→head logits[4].

**Reduction.** Sum messages by destination, count eligible messages, divide by max(count,1). Empty groups become zero. Mean loses count; a real zero-vector neighbor collides with no neighbors.

**Gradient.** With n eligible events, each gets 1/n of the incoming mean derivative. Excluded embeddings get zero. Shared weights still learn from eligible rows.

**Verify.** Independent incidence-matrix oracle; repeated destinations; empty groups; PK permutation with remapping; event permutation; excluded-value interventions; finite nonzero gradients into both encoders.

**Evidence.** Four-row training loss is a wiring diagnostic. Historical RelBench replay is a separate two-layer sum-GraphSAGE model. Benchmark NOT_RUN; no paper-result parity.

[Source: PyTorch Frame §3](https://arxiv.org/html/2404.00776v2) · [RelBench Appendix B](https://arxiv.org/html/2407.20060v1)
'''
 (ROOT/'reference'/f'{SLUG}.html').write_text(head+render(ref+f'\n[Full lesson](../lessons/{SLUG}.html) · [Lab](../labs/html/{SLUG}.html)')+'</article></body></html>')
 print('Built L076 lesson, student/solution, preview, reference and four portable figures')
if __name__=='__main__':build()
