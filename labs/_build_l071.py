"""Canonical generator: L071 lesson, reference, notebooks, portable figures and preview."""
import ast,base64,html,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune
from _colab import bootstrap_cells
from _figures_l071 import figures
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent
SLUG='0071-vime-masked-tabular-ssl';TITLE='VIME: learn from unlabeled tabular rows'
SOURCE=(ROOT/'relkit/vime_l071.py').read_text()
NODES={n.name:ast.get_source_segment(SOURCE,n) for n in ast.parse(SOURCE).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
CAPTIONS={
'architecture':'Numeric released-example architecture. Pretraining updates the encoder and two heads. At prediction time, clean rows pass through the retained encoder and a task head; local fine-tuning is explicitly separate from frozen transfer.',
'corruption':'Computed illustrative trace: sampled mask [1,1,0] produces actual-change target [1,0,0]. The second donor repeats the original value.',
'loss':'Computed illustrative loss contributions. BCE mean 0.2284 + 2 × full-coordinate MSE mean 0.0200 = 0.2684. These are hand-chosen predictions, not model scores.',
'split':'Local digits protocol per seed. Test rows are excluded from pretraining, validation, and donor pools. Validation labels count toward each labeled budget.',
'results':'Author-run evidence on offline digits. Every point is a fresh fit; bars show sample SD over three paired split/initialization seeds. The lower plot preserves each paired fine-tuning gain. No original-paper score is plotted.'}
TASKS={
'corrupt':('def corrupt(x, mask, donors):','Return (changed, xt), both [B,d]. donors[i,j] is a row index for column j. Preserve x; define changed by actual value inequality, including donor collisions.','Eq. 3 and released vime_utils.pretext_generator. Think about independent donor indices per column; do not replace a complete row.',"x0=torch.tensor([[0.,1.],[1.,1.]])\ndonors0=torch.tensor([[1,1],[0,0]])\nchanged0,xt0=corrupt(x0,torch.ones_like(x0),donors0)\nassert torch.equal(xt0,torch.tensor([[1.,1.],[0.,1.]]))\nassert torch.equal(changed0,torch.tensor([[1.,0.],[1.,0.]])), 'Collision must have target zero'\nassert torch.equal(corrupt(x0,torch.zeros_like(x0),donors0)[1],x0)"),
'pretext_loss':('def pretext_loss(mask_logits, reconstruction, x, changed, alpha=2., mask_weight=1.):','Return (total, lm, lr): mean BCE from mask logits, mean squared error on every coordinate, and mask_weight*lm + alpha*lr. Preserve gradients.','Eqs. 4–6. The reconstruction denominator includes unchanged cells. F.binary_cross_entropy_with_logits accepts raw scores.',"z0=torch.zeros(2,2,requires_grad=True)\nx0=torch.tensor([[0.,1.],[1.,1.]])\ntotal0,lm0,lr0=pretext_loss(z0,torch.zeros_like(x0),x0,torch.zeros_like(x0),2.)\nassert abs(lm0.item()-np.log(2))<1e-6\nassert abs(lr0.item()-.75)<1e-6, 'Use all coordinates, even when changed is zero'\nassert abs(total0.item()-(np.log(2)+1.5))<1e-6\ntotal0.backward(); assert z0.grad.abs().sum()>0"),
'consistency_loss':('def consistency_loss(logits):','Input [K,B,C]. Return population variance over augmented views, then average over batch and classes. K=1 must give zero.','Released vime_semi.py. Identify the view axis before reducing. Avoid the default unbiased sample variance.',"a0=torch.tensor([[[0.,2.]],[[2.,4.]]],requires_grad=True)\nassert consistency_loss(a0).item()==1.\nassert consistency_loss(a0[:1]).item()==0.\nassert consistency_loss(torch.ones(3,2,4)).item()==0.\nconsistency_loss(a0).backward(); assert a0.grad.abs().sum()>0")}

def figure(name,notebook=False):
 p=ROOT/'figures/l071'/f'{name}.png'
 src='data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode() if notebook else '../labs/figures/l071/'+name+'.png'
 return '<figure><div class="figure-scroll" style="max-width:100%;overflow-x:auto" tabindex="0" role="region" aria-label="Scrollable computation figure"><img src="'+src+'" alt="'+html.escape(CAPTIONS[name])+'" style="width:100%;min-width:600px;height:auto"></div><figcaption>'+CAPTIONS[name]+' On narrow screens, scroll the figure horizontally.</figcaption></figure>'

def evidence():
 r=json.loads((ROOT/'_verify_l071_results.json').read_text())
 s='**Measured test accuracy, mean ± sample SD over three seeds (%).** These are author-reference outputs, not your current notebook run.\n\n| Total labels | Scratch | VIME fine-tune | VIME frozen | Reconstruction fine-tune | VIME semi |\n|---|---|---|---|---|---|\n'
 for b in r['config']['budgets']:
  values=[]
  for a in ['scratch','vime_finetune','vime_frozen','recon_finetune','vime_semi']:
   row=next(z for z in r['summary'] if z['budget']==b and z['arm']==a)
   values.append(f"{row['accuracy']*100:.2f} ± {row['sd']*100:.2f}")
  s+='| '+str(b)+' | '+' | '.join(values)+' |\n'
 s+='\n**Paired fine-tuning gains (percentage points).**\n\n'
 for z in r['summary']:
  if z['arm']=='vime_finetune':s+=f"- {z['budget']} total labels: "+', '.join(f'{v*100:+.2f}' for v in z['seed_gains'])+f"; mean {z['paired_gain']*100:+.2f}.\n"
 s+='\n**Observed conclusion.** Under this fixed digits recipe, fine-tuned VIME has lower mean accuracy than scratch at all three budgets. The paired gains above show the size and seed variability. This does not establish that VIME cannot help other datasets or protocols.\n'
 return s

def launcher(prefix='../'):
 return f'''<aside class="lab-access"><strong>Lesson package · implement three live operations</strong><nav>
 <a href="{prefix}labs/html/{SLUG}.html">Read-only lab preview</a>
 <a href="{prefix}labs/{SLUG}.ipynb">Download student notebook</a>
 <a href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{SLUG}.ipynb">Run in Colab after publication</a>
 <a href="{prefix}labs/html/{SLUG}.html#lab-exercises">Jump to exercises</a>
 <a href="{prefix}reference/{SLUG}.html">Printable reference</a>
 <a href="{prefix}labs/_verify_l071_results.json">Measured evidence</a>
 <a href="{prefix}labs/l071-reproduction.md">Reproduction instructions</a></nav>
 <p>The preview is read-only. Run the downloaded notebook in Jupyter from labs/. Colab requires this revision to be published; live Colab is NOT_CHECKED.</p></aside>'''

def manuscript(notebook=False):
 text=(REPO/'lessons/content'/f'{SLUG}.md').read_text()
 for name in CAPTIONS:text=text.replace('<!--figure:'+name+'-->',figure(name,notebook))
 text=text.replace('<!--results-->',evidence())
 if notebook:
  text=re.sub(r'<div id="(?:warmup|mask-trace|consistency-trace|prediction|teachback)"[^>]*></div>','',text)
  # Notebook recap retains the lesson's full mechanism explanations and static traces.
  text=text.replace('(0070-foundation-model-checkpoint.html)','(../lessons/0070-foundation-model-checkpoint.html)').replace('(0045-tabtransformer.html)','(../lessons/0045-tabtransformer.html)')
 return text

def make_notebook(solution=False):
 cells=[]
 def md(s):cells.append(nbf.v4.new_markdown_cell(s))
 def code(s):cells.append(nbf.v4.new_code_cell(s))
 md(f'# Lab 071 · {TITLE}\n\n[Lesson](../lessons/{SLUG}.html) · [Reference](../reference/{SLUG}.html)\n\n**Skill:** implement VIME pretraining and measure label efficiency. PROVIDED cells expose the full architecture and training; TODO cells are yours; CHECK cells diagnose specific errors; EXIT requires a complete curve and a written conclusion.\n\n**Environment:** PyTorch CPU, NumPy, scikit-learn, pandas and matplotlib. Run locally from `labs/`. Default data ships with sklearn; no dataset download. Full reference run takes several minutes on this machine. Budget a separate sitting for interpretation.\n\n**Cold retrieval:** without notes, explain a validation boundary and define a paired comparison. Predict whether pretraining must improve accuracy.')
 for i,c in enumerate(bootstrap_cells()):
  c['id']=f'bootstrap-{i}';cells.append(nbf.from_dict(c))
 # Chunk the recap by major headings for readable notebook navigation.
 for part in re.split(r'(?=^## )',manuscript(True),flags=re.M):
  if part.strip():md(part)
 md('<a id="lab-exercises"></a>\n## Lab exercises\n\nThe author evidence above is a reference. The following cells run your live implementations. Completing these cells must change the functions that the actual training loop calls.')
 code('# PROVIDED: dependencies\nimport copy, hashlib, json, platform, time\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\nfrom sklearn.datasets import load_digits\nfrom sklearn.metrics import accuracy_score, log_loss\nfrom sklearn.model_selection import train_test_split\nimport matplotlib.pyplot as plt\ntorch.set_num_threads(1)')
 for name,(signature,contract,hint,check) in TASKS.items():
  md(f'### TODO · {name}\n\n**Goal:** {contract}\n\n**Why:** this operation is part of the live VIME objective. A plausible wrong answer changes the learning problem.\n\n**Hint boundary:** {hint}\n\nPredict the CHECK outcome before running it.')
  code(NODES[name] if solution else signature+'\n    """'+contract+'"""\n    # TODO: implement the operation; keep the function signature.\n    raise NotImplementedError("'+name+'")')
  code('# CHECK: independent numeric fixture\n'+check+'\nprint("'+name+': CHECK passed")')
 groups=[(['draw_corruption','VIME','Predictor'],'PROVIDED · architecture and corruption draws','The pretext encoder is d→d ReLU. Its two heads have d outputs. The predictor copies the encoder; each arm receives a newly initialized, paired task head.'),
 (['pretrain'],'PROVIDED · pretraining loop','The corruption bank is created once, as in the release. Follow the calls to your corrupt and pretext_loss functions. The reconstruction-only arm sets mask_weight to zero.'),
 (['make_split','load_data'],'PROVIDED · data and information boundaries','Data labels are used only by the benchmark split designer, and by designated supervised fits. Pixel normalization uses the known measurement range; it estimates no statistic from held-out rows.'),
 (['fit_predictor'],'PROVIDED · downstream training and validation','Find the requires_grad switch, the call to your consistency_loss, and the best-validation-state restoration. Test features and labels cannot enter this function.'),
 (['summarize','run_experiment'],'PROVIDED · matched experiment','The loop holds splits and head seeds fixed across arms. Inspect each arm tuple. A model is selected before its test predictions are computed.')]
 for names,title,explanation in groups:
  md('### '+title+'\n\n'+explanation)
  for name in names:code('# PROVIDED: '+name+'\n'+NODES[name])
 md('### RUN · your label-efficiency curve\n\nHold p=0.3, alpha=2, epochs and seeds fixed. Do not change them after looking at test scores. Predict the sign of the fine-tuning gain at 50 and 500 total labels. A lower pretext loss alone cannot settle that prediction.')
 code("# Verify the trainer resolves YOUR functions, then run fresh fits.\nassert draw_corruption.__globals__['corrupt'] is corrupt\nassert pretrain.__globals__['pretext_loss'] is pretext_loss\nassert fit_predictor.__globals__['consistency_loss'] is consistency_loss\nresult = run_experiment()\nsummary = pd.DataFrame(result['summary'])\ndisplay(summary[['budget','arm','accuracy','sd','paired_gain']])\nPath('l071-student-results.json').write_text(json.dumps(result,indent=2))")
 code("# PROVIDED: plot fresh results from this kernel\nfig, ax = plt.subplots(figsize=(8,5))\nfor arm, rows in summary.groupby('arm'):\n    line=ax.errorbar(rows.budget,100*rows.accuracy,yerr=100*rows.sd,marker='o',capsize=3,label=arm)\n    for r in result['records']:\n        if r['arm']==arm: ax.scatter(r['budget'],100*r['accuracy'],s=12,alpha=.4,color=line[0].get_color())\nax.set(xscale='log',xticks=[50,150,500],xticklabels=[50,150,500],ylim=(0,100),xlabel='Total labeled rows including validation',ylabel='Test accuracy (%)')\nax.legend(); ax.set_title('Fresh kernel output: points and sample SD across seeds'); plt.show()")
 md('### EXIT · defend the measured claim\n\nSubmit all 45 fitted results, the three paired fine-tuning gains at each budget, and a written interpretation. State what your comparison holds fixed and what it cannot establish. Include one donor-collision example and one paper/code discrepancy. Paste your conclusion to the tutor for review; the structural CHECK below does not grade understanding.')
 code("# EXIT: completeness and information-boundary checks\nassert len(result['records'])==45\nassert len({(r['seed'],r['budget'],r['arm']) for r in result['records']})==45\nassert all(r['train_labels']+r['validation_labels']==r['budget'] for r in result['records'])\nassert all(r['encoder_delta']==0 for r in result['records'] if r['arm'] in ['vime_frozen','vime_semi'])\nassert all(r['encoder_delta']>0 for r in result['records'] if r['arm'] in ['scratch','vime_finetune','recon_finetune'])\nfor s in result['splits']:\n    u=set(s['unlabeled']); test=set(s['test'])\n    for b in s['budgets'].values():\n        tr=set(b['train']); va=set(b['validation'])\n        assert not tr&va and not (tr|va)&u and not (tr|va|u)&test\nfor row in result['summary']:\n    if row['arm']=='vime_finetune': print(row['budget'], 'paired gains (pp):', np.round(100*np.array(row['seed_gains']),2))\nprint('Structural EXIT passed. Written interpretation still requires review.')")
 md('**Your conclusion:** write 150–250 words addressing direction and variability of the gains, validation-label cost, fine-tuning versus frozen transfer, and why these results are INCOMPARABLE with the paper tables. Propose one useful ablation before generalizing to relational data.')
 md('### Optional teaching extension · public MNIST follow-up\n\nThe same visible training functions can run on 28×28 MNIST. Select a GPU runtime before enabling this cell. It downloads and checks the public archive. The larger run still differs from the original protocol: fixed architecture, 60/40 unlabeled/reserve split, limited budgets and different framework. Full paper reproduction remains unestablished. For unattended execution use `modal run --detach modal/l071_paper_repro.py --preset closer` from the repository root. See [the contract](../labs/l071-reproduction.md).')
 code("RUN_TEACHING_EXTENSION = False\nif RUN_TEACHING_EXTENSION:\n    device='cuda' if torch.cuda.is_available() else 'cpu'\n    followup=run_experiment(dataset='mnist',seeds=(0,1,2),budgets=(100,1000,6000),pre_epochs=30,epochs=60,device=device)\n    Path('l071-mnist-followup.json').write_text(json.dumps(followup,indent=2))\n    display(pd.DataFrame(followup['summary']))\n    print('Verified here: public MNIST controlled fits. Paper claim: cited. Verdict: INCOMPARABLE.')\nelse:\n    print('MNIST scale-up: NOT_RUN; original paper-table reproduction: INCOMPARABLE.')")
 from _paper_tracks_071_074 import paper_cells
 cells.extend(paper_cells(71))
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
 # Stable IDs make regeneration reviewable.
 for i,c in enumerate(nb.cells):c.id=f'l071-{i:03d}'
 return nb

def build():
 figures()
 body=markdown2html_mistune(manuscript()).replace('<table>', '<div class="result-scroll"><table>').replace('</table>', '</table></div>')
 scripts=''.join(f'<script src="../assets/{name}.js"></script>' for name in ['retrieval-pool','retrieval-bank','predict','teachback','mask-pretrain-viz'])
 page=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>071 · {TITLE}</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/vime-lesson.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="0070-foundation-model-checkpoint.html">Previous lesson</a></nav><header><p>Year 2 · Quarter 4 · Lesson 071</p><h1>{TITLE}</h1></header>{launcher()}{body}</article>{scripts}</body></html>'
 (REPO/'lessons'/f'{SLUG}.html').write_text(page)
 for solution in [False,True]:
  dest=ROOT/('solutions' if solution else '')/f'{SLUG}.ipynb';dest.parent.mkdir(exist_ok=True)
  fresh=make_notebook(solution)
  if solution and dest.exists():
   previous=nbf.read(dest,as_version=4)
   if [c.source for c in previous.cells if c.cell_type=='code']==[c.source for c in fresh.cells if c.cell_type=='code']:
    if 'l071_loader_correction' in previous.metadata:fresh.metadata['l071_loader_correction']=previous.metadata['l071_loader_correction']
    old_code=iter(c for c in previous.cells if c.cell_type=='code')
    for c in fresh.cells:
     if c.cell_type=='code':
      old=next(old_code);c.outputs=old.outputs;c.execution_count=old.execution_count
  nbf.write(fresh,dest)
 preview,_=HTMLExporter().from_notebook_node(make_notebook(False))
 # Notebook links are authored relative to labs; preview lives one level deeper.
 from bs4 import BeautifulSoup
 soup=BeautifulSoup(preview,'html.parser')
 for anchor in soup.find_all('a',href=True):
  href=anchor['href']
  if not href.startswith(('http:','https:','#','data:','mailto:')):anchor['href']='../'+href
 (ROOT/'html'/f'{SLUG}.html').write_text('\n'.join(line.rstrip() for line in str(soup).splitlines())+'\n')
 ref='''# VIME reference · Lesson 071

- **Corruption:** x̃ = (1−m)x + m x̄. Each donor coordinate comes from its own column permutation. The encoder sees only x̃.
- **Released mask target:** 1[x̃ ≠ x]. Selection can leave a value unchanged. Example: x=[.2,.8,.8], donor=[.9,.8,.1], m=[1,1,0] → target=[1,0,0].
- **Encoder:** d→d ReLU. Mask head: d logits. Value head: d sigmoid outputs. Prediction uses a new head on clean-row encodings.
- **Pretext loss:** mean BCE(mask logits, actual changes) + α mean((reconstruction−original)²). Reconstruct all coordinates; α=2 in the lab.
- **Frozen transfer:** train only the new predictor. **Fine-tuning:** update the retained encoder too; an extension to the released pipeline.
- **Released semi-supervision:** task cross-entropy + β mean population variance of K augmented-view logits. Variance axis is K, not rows or classes.
- **Clean-anchor identity:** mean((a−c)²)=variance(a)+(mean(a)−c)². The paper equation and released variance differ in general.
- **Budget:** training labels + validation labels. Hold paired rows, head initialization and supervised budget fixed. Exclude test rows from all fitting and donors.
- **Local evidence:** digits, five arms, three budgets, three seeds. SD is seed variability, not a cross-dataset interval. Original paper comparison: INCOMPARABLE.

''' + evidence()+f'\n\n[Lesson](../lessons/{SLUG}.html) · [Notebook](../labs/{SLUG}.ipynb) · [Reproduction contract](../labs/l071-reproduction.md) · [Pinned source audit](../labs/_sources_l071.json).'
 (REPO/'reference'/f'{SLUG}.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VIME reference</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/vime-lesson.css"></head><body><article>'+markdown2html_mistune(ref).replace('<table>', '<div style="overflow-x:auto"><table>').replace('</table>', '</table></div>')+'</article></body></html>')
 print('Built L071 lesson, reference, student/solution notebooks and read-only preview')
if __name__=='__main__':build()
