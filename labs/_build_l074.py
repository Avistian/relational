"""Build the CARTE lesson, portable student/solution labs and reference."""
import ast,base64,json,re
import numpy as np
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
from bs4 import BeautifulSoup
from _colab import bootstrap_cells
from _figures_l074 import build as figures
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent;SLUG='0074-carte-cross-table-transfer';TITLE='CARTE: transfer across table schemas'
SOURCE=(ROOT/'relkit/carte_l074.py').read_text();TREE=ast.parse(SOURCE)
NODES={n.name:ast.get_source_segment(SOURCE,n) for n in TREE.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
IMPORTS='\n'.join(ast.get_source_segment(SOURCE,n) for n in TREE.body if isinstance(n,(ast.Import,ast.ImportFrom)))
CAPTIONS={'audit':'Paired initialization gains with descriptive t95 intervals over three repetitions; separate dataset-balanced rank/CD view. Repeated splits overlap and the datasets share a domain. These are exploratory summaries, not broad population intervals.', 'graph':'Illustrative two-dimensional row encoding. Trace the value-edge products before averaging. The real cache uses 300 coordinates.', 'architecture':'Complete pretraining-to-prediction route. Upper lane: paper pretraining overview. Lower lane: the exact single-readout local encoder and its two target adaptation choices.', 'attention':'Illustrative projected query/key/value fixture. Weights and contributions are calculated, then rounded to four decimals. A second receiver has a separate denominator.', 'results':'Author-reference measurements, three paired split/model repetitions per table. Dots show each run; bars show mean plus or minus sample SD. Poland uses a symmetric-log vertical scale (linear within ±0.5) to keep the large negative baseline score visible; the other panels are linear. These related wine datasets do not establish general cross-domain superiority.'}
GOALS={'make_graph':'Build the release-specific row graph from a mapping of column names to values and a fixed string-vector dictionary. Return node features, receiver/sender indices and edge attributes. Omit missing cells; reject an all-missing row. Preserve value-column pairing.', 'grouped_attention':'Compute scaled dot-product attention and weighted outputs, normalizing independently for each receiver. Use a stable softmax. Return the output tensor and one weight per edge. The result must remain differentiable.', 'ridge_predict':'Fit StandardScaler on training embeddings only. Try Ridge penalties 1, 10 and 100. Choose the smallest validation MSE (first candidate on a tie), and return test predictions plus the selected penalty. Do not refit using validation or test rows.'}
CHECKS={
'make_graph':"vec={'a':np.array([1.,2.]),'b':np.array([3.,1.]),'red':np.array([2.,-1.])}\ng=make_graph({'a':'red','b':2.,'missing':None},vec)\nassert torch.allclose(g[0][0],torch.tensor([10.,0.]))\nassert torch.equal(g[1],torch.tensor([[0,0,1,2],[1,2,1,2]]))\nassert torch.equal(g[2][-2:],torch.ones(2,2))\ntry:make_graph({'a':None},vec)\nexcept ValueError:pass\nelse:raise AssertionError('Reject an undefined all-missing center')",
'grouped_attention':"ei=torch.tensor([[0,0,1],[0,1,0]])\nq=torch.tensor([[1.,0.],[0.,1.]],requires_grad=True)\nk=torch.tensor([[1.,0.],[0.,1.],[3.,2.]])\nv=torch.tensor([[2.,0.],[0.,4.],[7.,8.]])\no,a=grouped_attention(ei,q,k,v)\nassert torch.allclose(a[:2],torch.tensor([.6697615,.3302385]),atol=1e-6)\nassert a[2]==1 and torch.equal(o[1],v[2])\no.sum().backward();assert q.grad is not None and torch.isfinite(q.grad).all()\no,a=grouped_attention(ei,q.detach()*10000,k,v)\nassert torch.isfinite(o).all() and torch.isfinite(a).all()",
'ridge_predict':"x=np.arange(18,dtype=float).reshape(9,2);y=np.arange(9,dtype=float)\np,alpha=ridge_predict(x[:5],x[5:7],x[7:],y[:5],y[5:7])\nchanged=x[7:].copy();changed[1]=1e9\np2,alpha2=ridge_predict(x[:5],x[5:7],changed,y[:5],y[5:7])\nassert alpha in [1.,10.,100.] and alpha==alpha2\nassert np.allclose(p[0],p2[0]),'Test features must not fit the scaler'\nassert p.shape==(2,)"}

def evidence():
 r=json.loads((ROOT/'_verify_l074_results.json').read_text());t='**Author-reference R², mean ± sample SD across three seeds.** These saved values are separate from your fresh notebook run.\n\n| Method | Wine Poland | Wine.com | Vivino |\n|---|---|---|---|\n'
 for arm in r['config']['arms']:
  t+='| '+arm+' | '+' | '.join(f"{s['mean']:.3f} ± {s['sd']:.3f}" for d in r['config']['datasets'] for s in r['summary'] if s['dataset']==d and s['arm']==arm)+' |\n'
 means={(z['dataset'],z['arm']):z['mean'] for z in r['summary']}
 ft=sum(means[d,'pretrained_ft']>means[d,'scratch'] for d in r['config']['datasets'])
 probe=sum(means[d,'pretrained_probe']>means[d,'random_probe'] for d in r['config']['datasets'])
 t+=f'\n**Observed pattern.** Pretrained fine-tuning has higher mean R² than scratch on {ft} of 3 targets. The frozen pretrained probe exceeds the frozen random probe on {probe} of 3. These comparisons isolate different adaptation settings; neither count establishes broad superiority.\n'
 t+='\n**Paired fine-tuned pretrained minus scratch gains:**\n\n'
 for d in r['config']['datasets']:
  vals={a:{z['seed']:z['r2'] for z in r['records'] if z['dataset']==d and z['arm']==a} for a in ['pretrained_ft','scratch']}
  g=[vals['pretrained_ft'][s]-vals['scratch'][s] for s in r['config']['seeds']]
  t+='- '+d+': '+', '.join(f'{v:+.3f}' for v in g)+f'; mean {sum(g)/len(g):+.3f}, descriptive t95 [{np.mean(g)-4.3026527299*np.std(g,ddof=1)/np.sqrt(3):+.3f}, {np.mean(g)+4.3026527299*np.std(g,ddof=1)/np.sqrt(3):+.3f}].\n'
 t+='\n**Exploratory dataset-balanced mean ranks** (lower is better): '+', '.join(f'{k} {v:.2f}' for k,v in r['ranks']['mean'].items())+f". Friedman p={r['ranks']['friedman_p']:.3f}; Nemenyi critical difference={r['ranks']['nemenyi_cd']:.2f}. Only three related datasets contribute.\n"
 return t

def manuscript(notebook=False):
 t=(REPO/'lessons/content'/f'{SLUG}.md').read_text().replace('<!--results-->',evidence())
 for name,caption in CAPTIONS.items():
  p=ROOT/'figures/l074'/f'{name}.png';src='data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode() if notebook else f'../labs/figures/l074/{name}.png'
  t=t.replace('<!--figure:'+name+'-->',f'<figure><div class="figure-scroll" tabindex="0" role="region" aria-label="Scrollable {name} figure"><img src="{src}" alt="{caption}"></div><figcaption>{caption} On narrow screens, scroll horizontally.</figcaption></figure>')
 if notebook:
  t=re.sub(r'<div id="[^"]+"[^>]*></div>','',t)
  t=t.replace('(0073-', '(../lessons/0073-').replace('(0072-','(../lessons/0072-')
  t=t.replace('then use the control','then calculate the new center on paper').replace('then complete the spaced warm-up','then retain your answers for tutor review')
 return t

def notebook(solution=False):
 cells=[]
 def md(x):cells.append(nbf.v4.new_markdown_cell(x))
 def code(x):cells.append(nbf.v4.new_code_cell(x))
 md('# Lab 074 · CARTE cross-table transfer\n\n**Skill:** follow schema-variable rows through a source-checked CARTE encoder and evaluate real transferred embeddings against matched scratch controls. **Scope:** exact single-readout released architecture, selected YAGO weights, real cached FastText vectors; different downstream head/training from the paper. Three Tier-A wine datasets × three seeds × six arms. Full benchmark INCOMPARABLE.\n\nPROVIDED exposes the architecture and trainer; TODO holds three live functions; CHECK gives immediate feedback; EXIT requires measurements and explanation.\n\n[Lesson](../lessons/'+SLUG+'.html) · [Reproduction contract](../labs/l074-reproduction.md)')
 for c in bootstrap_cells():cells.append(nbf.from_dict(c))
 for part in re.split(r'(?=^## )',manuscript(True),flags=re.M):
  if part.strip():md(part)
 md('<a id="lab-exercises"></a>\n## Implement and measure\n\nThe real string cache and selected pretrained parameters are provided under `data/l074`. No full FastText installation is required for the default lab. The source audit uses an optional PyG dependency outside this notebook. Run from the course `labs/` directory; the bootstrap does this on Colab. Local CPU execution takes several minutes.')
 code(IMPORTS+'\nimport matplotlib.pyplot as plt\ntorch.set_num_threads(1)\nassert Path("data/l074/manifest.json").exists(), "Run this notebook from the course labs directory"\nmanifest=json.loads(Path("data/l074/manifest.json").read_text())\nfor name,digest in manifest["files"].items():\n    assert hashlib.sha256((Path("data/l074")/name).read_bytes()).hexdigest()==digest\nprint("Data, vector and checkpoint hashes verified")')
 for name,goal in GOALS.items():
  md('### TODO · '+name+'\n\n**Goal.** '+goal+'\n\n**Why it matters.** This function controls the representation or the evidence boundary of the actual run below.\n\n**Hint boundary.** Return the documented shapes and preserve the training/validation/test roles. Predict the CHECK result before running it.')
  code(NODES[name] if solution else NODES[name].split('\n')[0]+'\n    raise NotImplementedError("'+name+'")')
  code('# CHECK\n'+CHECKS[name]+'\nprint("'+name+': passed")')
 for names,title in [(['Attention','Readout','Encoder'],'Model architecture: column-conditioned attention, normalization and center readout'),(['load_encoder','batch_graphs','embed'],'Strict checkpoint loading and disconnected graph batches'),(['tune'],'Matched target training and validation checkpoint selection'),(['normalize_numeric','run_transfer'],'Same target IDs, multiple controls, saved predictions and dataset-balanced ranks')]:
  md('### PROVIDED · '+title+'\n\nTrace the exact tensors and label access through these visible definitions. Each TODO remains bound in this notebook kernel; no hidden model import replaces it.')
  for name in names:code(NODES[name])
 md('### RUN · commit predictions\n\nWrite the sign you expect for pretrained fine-tuning minus scratch on each table. State which comparison isolates graph pretraining beyond the text vectors.')
 code("assert run_transfer.__globals__['make_graph'] is make_graph\nassert Attention.forward.__globals__['grouped_attention'] is grouped_attention\nassert run_transfer.__globals__['ridge_predict'] is ridge_predict\nresult=run_transfer()\nPath('l074-student-results.json').write_text(json.dumps(result,indent=2))\ndisplay(pd.DataFrame(result['summary']).pivot(index='arm',columns='dataset',values='mean'))")
 code("# Fresh paired gains: join seed identities before subtracting\nframe=pd.DataFrame(result['records'])\npair=frame.pivot(index=['dataset','seed'],columns='arm',values='r2')\npair['gain']=pair['pretrained_ft']-pair['scratch']\ndisplay(pair[['pretrained_ft','scratch','gain']])\nfig,ax=plt.subplots(figsize=(8,4))\nfor i,d in enumerate(result['config']['datasets']):\n    vals=pair.loc[d,'gain'];ax.scatter([i]*len(vals),vals);ax.plot(i,vals.mean(),'k_',markersize=18)\nax.axhline(0,color='gray');ax.set_xticks(range(3),result['config']['datasets'],rotation=15);ax.set_ylabel('Pretrained FT − scratch R²');plt.tight_layout();plt.show()")
 md('### EXIT · submit the evidence and explanation\n\nSubmit the mean/SD table, per-seed paired gains and a 150–250-word explanation of one failed prediction. Name the pretraining source, input semantics, development-label cost, and one inference boundary. Describe a follow-up that changes exactly one intended factor. A loss for transfer is an admissible result.')
 code("assert len(result['records'])==54\nfor s in result['splits']:\n    tr,va,te=map(set,[s['train'],s['validation'],s['test']])\n    assert len(tr)==64 and len(va)==64 and len(te)==256\n    assert not tr&va and not tr&te and not va&te\nfor r in result['records']:\n    assert abs(r2_score(r['target'],r['prediction'])-r['r2'])<1e-12\nprint('Structural EXIT passed. Written interpretation requires tutor review.')")
 md('**Write your interpretation here.** Return tomorrow and reconstruct the graph without the figure.')
 md('### NEXT STEP · required larger-run plan\n\nPredeclare the 128-row target budget and 100 supervised epochs before running. This local extension uses the same graph/model functions, 64 validation rows and the remaining 192 test rows. It varies two resources together; use separate one-factor follow-ups to diagnose the cause of any difference. Its test population differs from the default, so do not interpret score changes as a pure training-budget effect. This is closer local training, still INCOMPARABLE to paper pretraining, joint transfer and bagged benchmarks. CPU is sufficient; no Modal dependency is needed.')
 code("RUN_PAPER_REPRO=False\nif RUN_PAPER_REPRO:\n    larger=run_transfer(train_size=128,epochs=100)\n    Path('l074-closer-results.json').write_text(json.dumps(larger,indent=2))\n    print('Verified here: larger local target adaptation; paper benchmark: INCOMPARABLE')\nelse:\n    print('Verified here: default local run; paper claim: cited; larger run: NOT_RUN; live Colab: NOT_CHECKED')")
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
 for i,c in enumerate(nb.cells):c.id=f'l074-{i:03d}'
 return nb

def build():
 figures()
 head='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 074 — '+TITLE+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/contrastive-views.css"></head><body><article>'
 nav=f'<nav><a href="../index.html">Course</a> · <a href="0073-when-ssl-helps.html">Previous lesson</a></nav><header><p>Year 2 · Quarter 4 · Lesson 074</p><h1>{TITLE}</h1><p>Encode the row → trace the message → transfer the parameters → measure the gain.</p></header><aside class="lab-access"><nav><a href="../labs/html/{SLUG}.html">Read lab preview</a><a href="../labs/{SLUG}.ipynb" download>Download student notebook</a><a href="../labs/html/{SLUG}.html#lab-exercises">Jump to exercises</a><a href="../reference/{SLUG}.html">Reference</a><a href="../labs/_verify_l074_results.json">Measured evidence</a><a href="../labs/l074-reproduction.md">Reproduction contract</a></nav><p>Run locally in the course labs directory, or upload the notebook to Colab after the course assets are published. Local package; live Colab and deployment NOT_CHECKED.</p></aside>'
 body=render(manuscript()).replace('<table>','<div class="result-scroll"><table>').replace('</table>','</table></div>')
 scripts=''.join(f'<script src="../assets/{n}.js"></script>' for n in ['retrieval-pool','retrieval-bank','predict','teachback','cell-graph-viz','l074-lesson'])
 (REPO/'lessons'/f'{SLUG}.html').write_text(head+nav+body+'<div id="teachback"></div></article>'+scripts+'</body></html>')
 for sol in [False,True]:
  p=ROOT/('solutions' if sol else '')/f'{SLUG}.ipynb';nb=notebook(sol)
  if sol and p.exists():
   old=nbf.read(p,as_version=4)
   if [c.source for c in old.cells if c.cell_type=='code']==[c.source for c in nb.cells if c.cell_type=='code']:
    codes=iter(c for c in old.cells if c.cell_type=='code')
    for c in nb.cells:
     if c.cell_type=='code':o=next(codes);c.outputs=o.outputs;c.execution_count=o.execution_count
  nbf.write(nb,p)
 preview,_=HTMLExporter().from_notebook_node(notebook());soup=BeautifulSoup(preview,'html.parser')
 for a in soup.find_all('a',href=True):
  if a['href'].startswith('../'):a['href']='../'+a['href']
 (ROOT/'html'/f'{SLUG}.html').write_text(str(soup))
 ref='''# CARTE: a transfer audit card

**Represent.** Each observed cell becomes a leaf; its column supplies the edge vector. Text uses a string embedding. Numeric values multiply the column embedding after train-fitted scaling. Missing values remove leaves.

**Release center.** Average the value-edge products. The paper describes the plain leaf mean; keep that discrepancy visible. With leaves [2,−1], [6,2] and edges [1,2], [3,1], the release center is [10,0].

**Attend.** q = Wq(receiver); k = Wk(sender ⊙ edge); v = Wv(sender ⊙ edge). Divide q·k by √head_width. Normalize over the same receiver's neighbors. Average their values; concatenate heads. A neighbor permutation preserving column pairing leaves the readout invariant.

**Read.** The local 300-wide encoder has initial node/edge maps and one 12-head readout block. It strictly loads selected YAGO weights, including the initial node map. The release estimator renames that map's checkpoint keys; the lab deliberately differs. Frozen ridge and trainable linear-head variants have distinct adaptation protocols.

**Compare.** Pretrained frozen versus random frozen isolates graph pretraining beyond architecture and text input; fine-tuned pretrained versus scratch isolates initialization under a matched training recipe. Raw language-conditioned ridge and CatBoost are practical controls.

**Audit.** Fit preprocessing on train, select alpha/epoch on validation, score test once. Count validation labels. Save source row IDs, hashes, selected hyperparameters and predictions. Average seed scores before ranking datasets. Three wine sources are related, not broad domain coverage.

**Transfer settings.** Background graph → target is measured here. Joint source-table training additionally needs compatible outcomes and a source/target mixing protocol. Within-row graphs do not supply foreign-key or temporal guarantees.

**Evidence.** Published benchmark reproduction INCOMPARABLE; fresh YAGO pretraining and supervised source-table mixing NOT_RUN. Local measurements have their own contract.

**Primary reading:** [paper Figures 1–3](https://arxiv.org/html/2402.16785v2), [pinned release](https://github.com/soda-inria/carte/tree/f54690da4cddbedd1e1a9113a312f85783d2c125).
'''
 (REPO/'reference'/f'{SLUG}.html').write_text(head+render(ref+f'\n[Full lesson](../lessons/{SLUG}.html) · [Lab](../labs/html/{SLUG}.html)')+'</article></body></html>')
 print('Built L074 lesson, reference, notebooks, preview and five figures')
if __name__=='__main__':build()
