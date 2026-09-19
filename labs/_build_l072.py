"""Canonical L072 lesson/notebook builder; live TODOs share trainer globals."""
import ast,base64,html,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune
from bs4 import BeautifulSoup
from _colab import bootstrap_cells
from _figures_l072 import build as figures
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent
SLUG='0072-scarf-subtab-contrastive-views';TITLE='SCARF / SubTab: learning from different views'
SOURCE=(ROOT/'relkit/contrastive_l072.py').read_text()
TREE=ast.parse(SOURCE)
NODES={n.name:ast.get_source_segment(SOURCE,n) for n in TREE.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
IMPORTS='\n'.join(ast.get_source_segment(SOURCE,n) for n in TREE.body if isinstance(n,(ast.Import,ast.ImportFrom)))
CAPTIONS={
'scarf-architecture':'Complete local SCARF path. The two branches share weights; row identity supplies the diagonal targets. At transfer, keep the encoder and discard the projector. The diagram distinguishes the frozen lab probe from paper fine-tuning. Values in the five-column row are an illustrative corruption trace.',
'subtab-architecture':'Complete local SubTab path. The coverage map shows which measurements each view can use. All views share E, G and D; reconstruction and pair losses train them together. Inference averages latent vectors for the same row before the frozen probe. The twelve-column layout and latent mean are illustrative, not measured model outputs.',
'results':'Author-run measurements: three real datasets, three paired split/initialization seeds, fixed frozen-probe protocol. Orange points are individual seeds; bars are sample SD, not confidence intervals over datasets. R = reconstruction, C = contrastive recognition, D = projection distance.',
'ranks':'Dataset-balanced rank audit of these six local recipes. Seeds are averaged before ranking. The orange segment is the Nemenyi critical difference; comparisons exceeding it would pass that exploratory threshold. Three selected datasets provide weak evidence for general rankings.'}
TASKS={
'corrupt':('def corrupt(x, bank, mask, donors):','Produce a new [B,d] tensor. For selected cells use bank[donor row, original column]; preserve other cells and do not mutate x.','Draw a two-row, three-column example first. A donor row index applies to one cell, not its whole row.',"x0=torch.tensor([[1.,10.,100.],[2.,20.,200.]])\nb0=torch.tensor([[3.,30.,300.],[4.,40.,400.]])\nm0=torch.tensor([[True,False,True],[False,True,False]])\ni0=torch.tensor([[1,0,0],[0,1,1]])\nassert torch.equal(corrupt(x0,b0,m0,i0),torch.tensor([[4.,10.,300.],[2.,40.,200.]]))\nassert x0[0,0]==1, 'Clean input was mutated'\nassert torch.equal(corrupt(x0,x0,torch.zeros_like(m0),i0),x0)"),
'scarf_loss':('def scarf_loss(clean, changed, tau=1.):','Return the mean N-way cross-entropy for cosine scores / tau. Target i is changed row i. Preserve autograd and include the positive in the denominator.','Distinguish the candidate axis from the embedding axis. Work out the two orthogonal vectors before coding.',"z0=torch.eye(2,requires_grad=True)\nl0=scarf_loss(z0,z0,1.)\nassert abs(l0.item()-np.log1p(np.exp(-1)))<1e-6, 'Expected N-way loss 0.3133, not 2N-way loss'\nl0.backward();assert z0.grad.abs().sum()>0\nassert abs(scarf_loss(torch.ones(3,2),torch.ones(3,2)).item()-np.log(3))<1e-6\nassert torch.allclose(scarf_loss(3*z0,z0),scarf_loss(z0,z0)), 'Cosine removes positive rescaling'"),
'aggregate_views':('def aggregate_views(h):','Input [views, rows, hidden]; average representations across views while preserving each row. Return [rows, hidden].','Label all three axes on paper. Use different values for the two rows so the wrong axis cannot silently pass.',"h0=torch.tensor([[[1.,3.],[2.,4.]],[[5.,7.],[6.,8.]]])\nassert torch.equal(aggregate_views(h0),torch.tensor([[3.,5.],[4.,6.]])), 'Average views, never rows'\nassert torch.equal(aggregate_views(h0[:1]),h0[0])")}

def figure(name,notebook=False):
 p=ROOT/'figures/l072'/f'{name}.png'
 src='data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode() if notebook else f'../labs/figures/l072/{name}.png'
 return f'<figure><div class="figure-scroll" tabindex="0" role="region" aria-label="Scrollable {name} diagram"><img src="{src}" alt="{html.escape(CAPTIONS[name])}"></div><figcaption>{CAPTIONS[name]} On small screens, scroll the diagram horizontally.</figcaption></figure>'

def evidence():
 r=json.loads((ROOT/'_verify_l072_results.json').read_text())
 s='**Author-reference accuracy (%), mean ± sample SD. These are saved author results, not your current kernel output.**\n\n| Dataset | Raw | Random | SCARF | SCARF c=0 | SubTab R | SubTab R+C+D |\n|---|---|---|---|---|---|---|\n'
 for d in ['wine','breast_cancer','digits']:
  values=[]
  for a in ['raw','random','scarf','scarf_c0','subtab_recon','subtab_joint']:
   z=next(z for z in r['summary'] if z['dataset']==d and z['arm']==a);values.append(f"{z['mean']*100:.2f} ± {z['sd']*100:.2f}")
  s+='| '+d+' | '+' | '.join(values)+' |\n'
 s+='\n**Paired SCARF minus random-encoder gains (percentage points):**\n\n'
 for z in r['summary']:
  if z['arm']=='scarf':s+=f"- {z['dataset']}: "+', '.join(f'{100*g:+.2f}' for g in z['gain_vs_random'])+'.\n'
 s+='\n**Observed in this run:** SCARF improves the random encoder at the mean on all three datasets, but the raw-feature probe has higher mean accuracy on all three. Joint SubTab does not consistently beat reconstruction-only SubTab. These outcomes separate learning a representation from choosing the best prediction recipe.\n'
 s+='\nRead these gains alongside the raw-feature arm. A positive gain over random features need not beat the original inputs. Differences between SubTab and SCARF also include different architectures and objectives. This experiment cannot attribute them solely to the choice of views.\n'
 return s

def manuscript(notebook=False):
 t=(REPO/'lessons/content'/f'{SLUG}.md').read_text()
 for n in CAPTIONS:t=t.replace('<!--figure:'+n+'-->',figure(n,notebook))
 t=t.replace('<!--results-->',evidence())
 if notebook:
  t=re.sub(r'<div id="(?:warmup|loss-trace|subset-trace|prediction|teachback)"></div>','',t)
  t=t.replace('Use the temperature control to predict the change before reading the output. Then switch the candidate layout.', 'Notebook prediction: recompute the example at τ=0.5 before reading further. Then change the candidate layout.')
  t=t.replace('That is the layout used by the SubTab contrastive operator we audit, not SCARF’s N-way layout.', 'That is the layout used by the SubTab contrastive operator we audit, not SCARF’s N-way layout.')
  t=t.replace('This is the matrix layout you toggled above.', 'This is the second candidate layout in the worked example above.')
  t=t.replace('Hide a view and predict the new mean.', 'Omit the third view and predict the new mean: check against (3,5).')
  t=t.replace('(0071-vime-masked-tabular-ssl.html)','(../lessons/0071-vime-masked-tabular-ssl.html)').replace('(0059-validation-set-overfitting.html)','(../lessons/0059-validation-set-overfitting.html)')
 return t

def launcher():
 return f'''<aside class="lab-access"><strong>Lesson 072 package</strong><nav><a href="../labs/html/{SLUG}.html">Lab preview</a><a href="../labs/{SLUG}.ipynb">Student notebook</a><a href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{SLUG}.ipynb">Colab after publication</a><a href="../reference/{SLUG}.html">Reference</a><a href="../labs/_verify_l072_results.json">Measured evidence</a><a href="../labs/l072-reproduction.md">Reproduction contract</a></nav><p>Prepared locally. Live Colab and deployment NOT_CHECKED.</p></aside>'''

def notebook(solution=False):
 cells=[]
 def md(t):cells.append(nbf.v4.new_markdown_cell(t))
 def code(t):cells.append(nbf.v4.new_code_cell(t))
 md(f'# Lab 072 · {TITLE}\n\n**Skill:** build two kinds of tabular views and test the frozen representations. Three TODOs feed the real experiment. PROVIDED shows the entire model and trainer; CHECK tests mechanisms; EXIT requires both measurements and an explanation.\n\n[Lesson](../lessons/{SLUG}.html) · [Contract](../labs/l072-reproduction.md)\n\n**Environment:** course Python environment, PyTorch CPU, sklearn, SciPy, NumPy, pandas, matplotlib. Run locally from labs/. Offline real data ships with sklearn. Full run: three datasets × three seeds × six arms, with 40 pretraining epochs. This is a local numeric mechanism experiment, not original-paper table reproduction.\n\n**Cold retrieval:** write VIME’s two targets and explain why unlabeled test features are not free training data. Predict whether useful row recognition guarantees useful class prediction.')
 for i,c in enumerate(bootstrap_cells()):
  c['id']=f'l072-bootstrap-{i}';cells.append(nbf.from_dict(c))
 for part in re.split(r'(?=^## )',manuscript(True),flags=re.M):
  if part.strip():md(part)
 md('<a id="lab-exercises"></a>\n## Lab exercises · make the computation yours\n\nThe figures and table above are author reference material. The following experiment will call your definitions in this kernel. Do not replace TODOs with imports from the solution module.')
 code('# PROVIDED: dependencies\n'+IMPORTS+'\nimport json\nfrom pathlib import Path\nimport pandas as pd\nimport matplotlib.pyplot as plt\ntorch.set_num_threads(1)')
 for name,(sig,goal,hint,check) in TASKS.items():
  md(f'### TODO · {name}\n\n**Goal:** {goal}\n\n**Predict first:** {hint}\n\n**Connection:** '+{'corrupt':'This creates SCARF’s positive companion without exposing labels.','scarf_loss':'This converts companion identity into the training signal drawn in the SCARF diagram.','aggregate_views':'This is the handoff from SubTab pretraining to one representation per prediction row.'}[name])
  code(NODES[name] if solution else sig+'\n    """'+goal+'"""\n    raise NotImplementedError("'+name+'")')
  code('# CHECK: independent arithmetic\n'+check+'\nprint("'+name+': passed")')
 for names,title,why in [
 (['subtab_loss','subsets','draw_view'],'Two view rules and two candidate matrices','SCARF counts selected positions exactly. SubTab keeps fixed windows; its symmetric contrastive loss masks only self. Compare these definitions to the two architecture diagrams.'),
 (['SCARF','SubTab'],'Visible model architecture','Trace h, z and reconstruction. Find shared weights, discarded heads, training noise and the call to your aggregation function.'),
 (['pretrain'],'The unlabeled training loop','No y argument exists. Locate the calls to your loss and corruption. Reconstruction sums feature errors before averaging rows, preserving the audited reduction.'),
 (['load_split','probe'],'Boundaries and the frozen probe','Find where each scaler is fitted, which labels choose regularization and when test scoring first occurs. Validation labels cost annotations.'),
 (['summarize','run_experiment'],'Paired experiment and dataset-balanced summary','The same seed and split are used across arms. The SCARF and random encoders start from identical weights. SubTab is a different architecture, so their difference is a recipe comparison.')]:
  md('### PROVIDED · '+title+'\n\n'+why)
  for name in names:code('# PROVIDED: '+name+'\n'+NODES[name])
 md('### RUN · commit your predictions\n\nBefore running, predict the SCARF-minus-random sign on each dataset and whether raw features will win anywhere. Explain what changing c to zero removes and what it leaves intact. Keep the protocol fixed after seeing test scores.')
 code("assert draw_view.__globals__['corrupt'] is corrupt\nassert pretrain.__globals__['scarf_loss'] is scarf_loss\nassert SubTab.represent.__globals__['aggregate_views'] is aggregate_views\nresult=run_experiment()\nsummary=pd.DataFrame(result['summary'])\ndisplay(summary[['dataset','arm','mean','sd']])\nPath('l072-student-results.json').write_text(json.dumps(result,indent=2))")
 code("# PROVIDED: plot only fresh kernel results\nfig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)\nfor ax,d in zip(axes,result['config']['datasets']):\n    rows=summary[summary.dataset==d]\n    for i,(_,r) in enumerate(rows.iterrows()):\n        ax.errorbar(i,100*r['mean'],yerr=100*r['sd'],fmt='o',capsize=4)\n        ax.scatter(i+np.linspace(-.12,.12,len(r.seed_values)),100*np.array(r.seed_values),s=12)\n    ax.set(title=d,xticks=range(len(rows)),xticklabels=rows.arm,ylim=(30,102))\n    ax.tick_params(axis='x',rotation=60)\naxes[0].set_ylabel('Fresh frozen-probe test accuracy (%)')\nplt.tight_layout();plt.show()")
 md('### EXIT · the result is an argument, not just a table\n\nThe structural checks verify completeness and boundaries. Submit the table, per-seed SCARF-minus-random gains, total labeled budgets, and 150–250 words interpreting one success or failure of your prediction. Explain why these frozen-probe scores cannot reproduce SCARF’s fine-tuning benchmark. Include a concrete false-negative example and propose one ablation for Lesson 73.')
 code("assert len(result['records'])==54\nassert len({(r['dataset'],r['seed'],r['arm']) for r in result['records']})==54\nfor s in result['splits']:\n    tr,va,te=map(set,[s['train'],s['validation'],s['test']])\n    assert not tr&va and not tr&te and not va&te\n    assert set(s['labeled'])<=tr\nfor r in result['records']:\n    assert r['frozen_delta']==0, 'The probe must not update the representation'\n    assert abs(np.mean(np.array(r['prediction'])==r['target'])-r['accuracy'])<1e-12\n    assert len(r['trials'])==3\nfor r in result['summary']:\n    if r['arm']=='scarf':print(r['dataset'], 'paired gain pp:',100*np.array(r['gain_vs_random']))\nprint('Structural EXIT passed; written interpretation still requires tutor review.')")
 md('**Write your interpretation here.** After a day, reconstruct both architectures without looking. Ask the tutor to check what you preserved or omitted.')
 md('### NEXT STEP · longer convergence and seed audit\n\nThis reuses YOUR functions for 200 epochs and five seeds. It is not a paper-fidelity preset: the suite, compact models and frozen probes still differ. Set a GPU runtime before enabling if available. The unattended alternative is `modal run --detach modal/l072_paper_repro.py --preset closer`; see the contract for retrieval of the persisted evidence.')
 code("RUN_PAPER_REPRO=False\nif RUN_PAPER_REPRO:\n    followup=run_experiment(seeds=tuple(range(5)),epochs=200,device='cuda' if torch.cuda.is_available() else 'cpu')\n    Path('l072-closer-results.json').write_text(json.dumps(followup,indent=2))\n    display(pd.DataFrame(followup['summary']))\n    print('Longer local protocol measured. Original paper tables: INCOMPARABLE.')\nelse:\n    print('Larger run NOT_RUN. Original paper benchmarks INCOMPARABLE. Live Colab NOT_CHECKED.')")
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
 for i,c in enumerate(nb.cells):c.id=f'l072-{i:03d}'
 return nb

def build():
 figures()
 body=markdown2html_mistune(manuscript()).replace('<table>','<div class="result-scroll"><table>').replace('</table>','</table></div>')
 head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 072 — {TITLE}</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/contrastive-views.css"></head><body><article>'
 page=head+f'<nav><a href="../index.html">Course</a> · <a href="0071-vime-masked-tabular-ssl.html">Previous: VIME</a></nav><header><p>Year 2 · Quarter 4 · Lesson 072</p><h1>{TITLE}</h1><p class="lesson-route">Repair a row → recognize its companion → reconcile partial observations → test the representation.</p></header>'+launcher()+body+'</article>'
 page+=''.join(f'<script src="../assets/{s}.js"></script>' for s in ['retrieval-pool','retrieval-bank','predict','teachback','contrastive-views-viz','l072-lesson'])+'</body></html>'
 (REPO/'lessons'/f'{SLUG}.html').write_text(page)
 for sol in [False,True]:
  dest=ROOT/('solutions' if sol else '')/f'{SLUG}.ipynb';fresh=notebook(sol)
  if sol and dest.exists():
   old=nbf.read(dest,as_version=4)
   if [c.source for c in old.cells if c.cell_type=='code']==[c.source for c in fresh.cells if c.cell_type=='code']:
    oldcodes=iter(c for c in old.cells if c.cell_type=='code')
    for c in fresh.cells:
     if c.cell_type=='code':prev=next(oldcodes);c.outputs=prev.outputs;c.execution_count=prev.execution_count
  nbf.write(fresh,dest)
 preview,_=HTMLExporter().from_notebook_node(notebook(False));soup=BeautifulSoup(preview,'html.parser')
 for a in soup.find_all('a',href=True):
  if a['href'].startswith('../'):a['href']='../'+a['href']
 (ROOT/'html'/f'{SLUG}.html').write_text(str(soup))
 ref='''# Reference · two ways to learn from views

Follow the training-to-prediction boundary: a head used to train the representation may disappear before classification.

| Step | SCARF | SubTab |
|---|---|---|
| View | Clean row plus marginally corrupted companion | Fixed overlapping feature subsets |
| Encoder | Shared across clean/corrupt branches | Shared across all subsets |
| Signal | N-way companion recognition | Full-row reconstruction, optional pair recognition and distance |
| Transfer representation | Clean-row encoder output h | Mean of same-row subset latents h |
| Local downstream | Frozen linear probe | Frozen linear probe |

N-way loss: mean negative log softmax diagonal of normalized clean × corrupted vectors / temperature. Keep the diagonal. SubTab pair loss: concatenate two views, remove self-similarity, retain the other-view positive. Do not interchange these candidate sets.

**Arithmetic:** two orthogonal companions at tau=1 give SCARF CE 0.3133; two-view NT-Xent 0.5514. Identical vectors give N-way CE log(N). The paper's SCARF convention subtracts log(N), without changing gradients.

**Axes:** views × rows × hidden → average over views → rows × hidden. For one row with (1,3), (5,7), (3,2), the mean is (3,4).

**Audit:** train-only donor pool and scaler; no test rows in pretraining; count validation labels; freeze the representation for a probe; compare paired seeds; state the raw-feature result. Contrastive positives encode augmentation assumptions, not guaranteed class equivalence.

**Failure diagnosis:** wrong column donors change feature meaning; a row-mean mixes examples; positive removal changes the denominator; low pretext loss does not establish a downstream gain. SubTab's released floor division can omit trailing input columns.

**Evidence boundary:** operator parity is not training parity. These compact numeric models and frozen probes do not reproduce either original benchmark table.
'''
 ref+=f'\n[Full lesson](../lessons/{SLUG}.html) · [Reproduction contract](../labs/l072-reproduction.md)\n'
 (REPO/'reference'/f'{SLUG}.html').write_text(head+markdown2html_mistune(ref)+'</article></body></html>')
 print('Built lesson, reference, student/solution notebooks, preview and four figures')
if __name__=='__main__':build()
