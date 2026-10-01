"""Deterministic lesson/reference/notebook builder; never trains or downloads."""
import ast,base64,html,json,re
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from relkit.foundation_preview_l159 import summarize
P=Path(__file__).resolve().parent;R=P.parent;F=P/'figures/l159'
S='0159-foundation-model-preview';TITLE='Foundation models: what pre-training teaches'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'svg.hashsalt':'l159'})

def canvas(height):
    fig,ax=plt.subplots(figsize=(6.4,height));fig.subplots_adjust(left=.03,right=.97,bottom=.02,top=.98)
    fig.patch.set_facecolor('#f4f7f2');ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');return fig,ax

def save(fig,name):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    bounds=[(text.get_text(),text.get_window_extent(renderer)) for ax in fig.axes for text in ax.texts if text.get_text()]
    outer=fig.bbox
    for name_text,box in bounds:
        assert outer.contains(box.x0,box.y0) and outer.contains(box.x1,box.y1), ('Figure text outside canvas',name,name_text)
    for i,(left,a) in enumerate(bounds):
        for right,b in bounds[i+1:]:
            assert not a.overlaps(b), ('Overlapping figure labels',name,left,right)
    for ext in ['svg','png']:
        fig.savefig(F/(name+'.'+ext),dpi=160,metadata={'Date':None} if ext=='svg' else {'Software':'L159'})
    plt.close(fig)
    p=F/(name+'.svg');p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')

def card(ax,y,title,body,fill='#ffffff',height=.115):
    ax.add_patch(FancyBboxPatch((.045,y),.91,height,boxstyle='round,pad=.008',facecolor=fill,edgecolor='#9bbab0',lw=1))
    ax.text(.075,y+height-.014,title,fontsize=14,weight='bold',va='top',color='#214e43')
    ax.text(.075,y+.012,body,fontsize=12,va='bottom',color='#283d37',linespacing=1.5)

def arrow(ax,y):
    ax.annotate('',xy=(.5,y-.038),xytext=(.5,y),arrowprops={'arrowstyle':'->','lw':1.7,'color':'#247568'})

fig,ax=canvas(9)
ax.text(.045,.979,'WORDS WITHIN ROWS\nCONTEXT BETWEEN ROWS',va='top',fontsize=17,weight='bold',color='#214e43',linespacing=1.4)
ax.text(.045,.868,'Paper data flow · symbolic dimensions',fontsize=12,color='#52645d')
card(ax,.738,'1  Hide the selected answer','Schema + row tokens, already corrupted\nExample: [MASK] | planet | Earth')
arrow(ax,.729)
card(ax,.583,'2  BART row encoder','Adapt to rows first; freeze for graph training\nRow tokens → node representations H: N × d',fill='#e4efea')
arrow(ax,.574)
card(ax,.428,'3  Graph convolution','Mix permitted neighboring representations\nH → contextual H′: N × d',fill='#f7e8d4')
arrow(ax,.419)
card(ax,.273,'4  BART text decoder','Frozen parameters; differentiable input\nTarget representation → token logits T × V',fill='#e4efea')
arrow(ax,.264)
card(ax,.118,'5  Reconstruct the masked target','Training: text target supplies the loss\nPrediction: the answer is not an input')
ax.text(.05,.043,'Green: frozen in stage 2. Amber: trained in stage 2.\nExact pooling / decoder interface: not specified in source.',fontsize=11,color='#44594f',linespacing=1.5)
save(fig,'architecture')
fig,ax=canvas(5.8)
ax.text(.05,.95,'HIDE AN IDENTITY, NOT A STRING',fontsize=16,weight='bold',color='#214e43')
ax.text(.05,.87,'Target: table name (shared ID 10)',fontsize=13)
labels=[['Moons','planet','Earth'],['Moons','planet','Mars'],['Moons','planet','Earth'],['Moons','planet','Venus']]
for i,row in enumerate(labels):
    y=.71-i*.105
    ax.text(.055,y,'row '+str(i),fontsize=12)
    for j,value in enumerate(row):
        x=.20+j*.255
        ax.add_patch(FancyBboxPatch((x,y-.025),.22,.078,boxstyle='round,pad=.007',fc='#f5ddbe' if j==0 else 'white',ec='#b5c8bd'))
        ax.text(x+.11,y+.013,'[MASK]' if j==0 else value,ha='center',va='center',fontsize=13)
ax.text(.06,.245,'Four copies removed before encoding.\nTwo Earth cells remain distinct identities.',fontsize=14,color='#214e43',linespacing=1.6)
ax.text(.06,.105,'Root-only masking would leave 3 answer copies.\nA clean embedding cache would preserve all 4.',fontsize=12,color='#66543e',linespacing=1.5)
save(fig,'masking')
fig,ax=canvas(5.9)
ax.text(.05,.95,'FREEZE ≠ DETACH',fontsize=19,weight='bold',color='#214e43')
card(ax,.72,'Graph supplies h = 0.4','Learned representation; can change',height=.12,fill='#f7e8d4')
arrow(ax,.71)
card(ax,.51,'Frozen decoder w = 2','z = w h = 0.8  →  p(target 1) = 0.690',height=.14,fill='#e4efea')
arrow(ax,.50)
card(ax,.30,'Target 1 gives loss ≈ 0.371','−log p; decoder weight stays fixed',height=.13)
ax.text(.06,.18,'BACKWARD: dL/dh = 2 × (0.690 − 1)',fontsize=13,weight='bold',color='#9b5f28')
ax.text(.06,.045,'≈ −0.620 reaches the graph.\nDetaching h severs that path.',fontsize=14,color='#214e43',linespacing=1.5)
save(fig,'gradient')

source=(P/'relkit/foundation_preview_l159.py').read_text()
nodes=ast.parse(source).body
code_by_name={n.name:ast.get_source_segment(source,n) for n in nodes if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
report=json.loads((P/'evidence/l159/mechanism.json').read_text())
captions={
'architecture':'Stated paper architecture, not the local binary codec. N is node count, d width, T output length, V vocabulary. Frozen decoder weights still transmit input gradients. The paper leaves the exact representation interface unspecified.',
'masking':'A table name is repeated in four row serializations. Global semantic masking removes all copies; equal-valued cells are not the same identity.',
'gradient':'One scalar worked example, not a fitted result. A frozen decoder keeps w fixed while passing a nonzero gradient into the graph.'}

def prose(portable=False):
    text=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',summarize(report))
    for name,caption in captions.items():
        url='data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l159/'+name+'.svg'
        text=text.replace('[[FIG:'+name+']]',f'<figure class="fm-figure"><img src="{url}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>')
    fragments={
     'WARMUP':('Recall: what separates a reusable training pipeline from evidence of generalization? Which split selects checkpoints? What information can a prediction legally access?','<div id="warmup"></div>'),
     'PREDICT':('Predict first: does better single-table reconstruction prove transfer to an unseen database? Write the missing evaluation.','<div id="predict"></div>'),
     'MASK_WIDGET':('Try on paper: table or column target, root-only policy leaves three copies; global policy leaves zero; clean-cache policy retains four. A root-cell target has one identity, so root-only and global both remove it.','<div id="mask-objective"></div><noscript>For table/column targets, mask every row copy before encoding. Root-only masking leaves three copies; cached clean vectors retain all four. A root cell has one identity.</noscript>'),
     'GRADIENT_WIDGET':('Change h to 0: p=0.5 and the graph gradient is −1 with w=2. Freeze w: the graph gradient remains. Detach h: the graph gradient is absent.','<div id="gradient-objective"></div><noscript>With h=0.4 and w=2, p≈0.690 and the graph receives gradient≈−0.620. Frozen weights preserve this path; detaching its input severs it.</noscript>'),
     'TEACHBACK':('Explain frozen versus detached and the gap between reconstruction and transfer. Ask the teaching agent to review your own explanation.','<div id="teachback"></div>')}
    for key,(plain,widget) in fragments.items():text=text.replace('[['+key+']]',plain if portable else widget)
    if portable:
        text=text.replace('](../','](https://avistian.github.io/relational/')
        text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
    return text

def doc(title,text,interactive=False):
    body=render(text).replace('<table>','<div class="route-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
    css=['lesson','atomic-route','checkpoint','pretraining-objective']
    scripts=['retrieval-pool','retrieval-bank','predict','teachback','pretraining-objective','l159-lesson'] if interactive else []
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in css)+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0158-year-4-synthesis.html">Lesson 158</a></nav><header><p class="fm-kicker">Year 4 · Lesson 159 · Foundation-model preview</p><h1>'+title+'</h1></header>'+body+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in scripts)+'</body></html>'

visible='\n## Visible mechanism implementation\n\nThe full trainer and generator are inline in the notebook. These are the local binary mechanism components; they do not implement BART.\n\n<details><summary>Read the masking, loss, freezing and neural layers</summary>\n\n```python\n'+'\n\n'.join(code_by_name[n] for n in ['mask_serializations','masked_cross_entropy','freeze_codec','RowCodec','TableGraph'])+'\n```\n\n</details>\n'
(R/'lessons'/f'{S}.html').write_text(doc(TITLE,prose()+visible,True))
ref='''## The objective contract

Name **target identity → permitted context → corruption before encoding → prediction → selected loss → evaluation split**.

| Decision | Good contract | Failure |
|---|---|---|
| Schema target | Mask every serialization of that identity | Root-only mask leaves copies |
| Cell target | Mask that cell, retain other identities | Remove every matching string |
| Cached features | Recompute from corrupted input | Clean cache leaks the answer |
| Loss | Average selected targets only | Reward unmasked fields or accept empty target set |
| Frozen decoder | Fixed weights; differentiate its input | Detach the decoder path |
| Transfer | Hold out databases; declare adaptation | Rename reconstruction as transfer |

## Two stages, two learning boundaries

Stage1 learns row encoder and decoder from reconstruction. Stage2 fixes those weights and learns contextual graph representations. A frozen decoder is still a differentiable function. For target1, z=wh, p=sigmoid(z): dL/dh=w(p−1). At h=.4,w=2, this is approximately−.620.

## Scope ladder

**Paper:** [Vogel, Hilprecht and Binnig 2023](https://arxiv.org/html/2305.15321v1), BART/GCN masked text reconstruction. Selected Table1 wikiTables experiment NOT_RUN locally; exact protocol/source gaps recorded.

**Executed lab:** synthetic four-row tables; mean-pooled 16-wide codec, complete-graph aggregation and three binary heads. Frozen-weight and masking checks pass; binary accuracy is INCOMPARABLE to paper text accuracy.

**Desired transfer:** new databases, matched scratch baseline, declared label/adaptation budgets and validation-only selection. NOT_TESTED here.

[Lesson](../lessons/0159-foundation-model-preview.html) · [Student notebook](../labs/0159-foundation-model-preview.ipynb) · [Protocol](../labs/l159-reproduction.md) · [Vision brief](../labs/l159-vision-brief-template.md). Ask the teaching agent to review your explanation; author execution does not prove learner mastery.
'''
(R/'reference/foundation-model-preview.html').write_text(doc('Pre-training objectives — quick reference',ref))

imports='''# PROVIDED · Python 3.10+, PyTorch. Validated with torch 2.13.0+cpu.
# No downloads or cloud dispatch. In a new environment install a CPU PyTorch
# build first; the recorded result is tied to the validated version.
import copy,hashlib,json,math,time
from pathlib import Path
import torch
from torch import nn
torch.set_num_threads(1)
print('PyTorch',torch.__version__)
'''
tasks={
'mask_serializations':('Remove all copies of one semantic identity before encoding.','Return a new tensor; MASK=0. Require equal 2D shapes, a nonnegative target and at least one match. Do not remove unrelated equal values.',"x=torch.tensor([[2,4,6],[2,4,7]]);ids=torch.tensor([[10,20,30],[10,20,31]])\nassert mask_serializations(x,ids,20).tolist()==[[2,0,6],[2,0,7]]\nassert x[0,1].item()==4\nprint('CHECK: schema copies removed; clean input preserved')"),
'masked_cross_entropy':('Score only the selected prediction targets.','Logits shape [N,3,2]; labels and boolean mask [N,3]. Reject shape mismatch or an empty selection. Return mean negative log-probability of selected true classes.',"z=torch.tensor([[[0.,math.log(3.)],[10.,-10.]]],requires_grad=True)\ny=torch.tensor([[1,1]]);m=torch.tensor([[True,False]])\nloss=masked_cross_entropy(z,y,m);assert abs(float(loss.detach())+math.log(.75))<1e-6\nloss.backward();assert z.grad[0,1].abs().sum()==0\nprint('CHECK: selected loss≈0.288; unselected gradient0')"),
'freeze_codec':('Fix the codec weights while keeping its input differentiable.','Operate on every parameter, clearing stale gradients; return value is not used. Your function starts stage 2 of the actual training below.',"codec_check=nn.Linear(2,2);freeze_codec(codec_check)\nh=torch.ones(1,2,requires_grad=True);codec_check(h).sum().backward()\nassert h.grad is not None and all(not p.requires_grad for p in codec_check.parameters())\nprint('CHECK: frozen parameters preserve input differentiation')")}
check_source=(P/'_check_l159.py').read_text();check_node=next(n for n in ast.parse(check_source).body if isinstance(n,ast.FunctionDef) and n.name=='check');checks=ast.get_source_segment(check_source,check_node)
for solution in [False,True]:
    cells=[nb.v4.new_markdown_cell('# Lesson 159 · Pre-training objectives\n\nStandalone **synthetic mechanism** lab, not a BART or paper reproduction. PROVIDED cells contain the complete codec, graph and trainer; three TODOs affect actual training; CHECK cells reject incorrect behavior; EXIT asks for a vision brief. One CPU thread, 120-second training cap, $0 cloud. Paper target NOT_RUN. Course links may be unavailable before publication; the experiment itself needs no repository files.'),nb.v4.new_markdown_cell(prose(True)),nb.v4.new_code_cell(imports)]
    for name,(goal,contract,checkcode) in tasks.items():
        cells.append(nb.v4.new_markdown_cell('## TODO · '+name+'\n\n**Goal:** '+goal+'\n\n**Why:** this function controls the real experiment, so a mistake changes what the model learns.\n\n**Contract:** '+contract))
        complete=code_by_name[name];code=complete if solution else complete.split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'
        cells.append(nb.v4.new_code_cell(code));cells.append(nb.v4.new_code_cell('# CHECK\n'+checkcode))
    for names,heading,description in [
      (['RowCodec','TableGraph'],'PROVIDED · All neural layers','The paper uses BART and a graph convolution. Here the deliberately smaller binary codec uses token-embedding means; the graph mixes the four rows with normalized adjacency 1/4. Read both forward paths.'),
      (['make_examples','batch'],'PROVIDED · Complete synthetic data and split','The generator creates independent tables and splits entire tables. Cell values repeat by construction; context cues are noisy. The masked semantic identity, not string equality, defines the information policy.'),
      (['accuracy','fingerprint','run_experiment','summarize'],'PROVIDED · Full two-stage trainer','Stage 1 trains row codec; stage 2 trains graph using frozen codec. Both select by validation accuracy; ties keep earliest epoch. Test is read only after both selections. Saved weights, logits and every validation score are returned. The graph receives extra compute; this does not isolate architecture alone.')]:
        cells.append(nb.v4.new_markdown_cell('## '+heading+'\n\n'+description))
        for name in names:cells.append(nb.v4.new_code_cell(code_by_name[name]))
    cells.append(nb.v4.new_markdown_cell('## CHECK · Adversarial contracts\nBefore fitting, reject partial masking, empty objectives and a severed frozen-decoder path. The target counterfactual below must not alter corrupted inputs.'))
    cells.append(nb.v4.new_code_cell(checks))
    cells.append(nb.v4.new_code_cell("print(check(mask_serializations,masked_cross_entropy,freeze_codec))\nclean=torch.tensor([[2,4,6],[2,4,7]]);ids=torch.tensor([[10,20,30],[10,20,31]])\nchanged=clean.clone();changed[ids==20]=5\nassert torch.equal(mask_serializations(clean,ids,20),mask_serializations(changed,ids,20))"))
    cells.append(nb.v4.new_markdown_cell('## Run the complete declared mechanism\nThree paired seeds × two 80-update stages. No benchmark download or paid launch. Predict which target benefits from neighboring unmasked cell values, then run. A timeout stops with INCOMPLETE rather than changing scope.'))
    cells.append(nb.v4.new_code_cell("report=run_experiment()\nprint(summarize(report))\nassert len(report['runs'])==3\nassert sum(len(r['predictions']) for r in report['runs'])==216\nassert all(r['frozen_before']==r['frozen_after'] for r in report['runs'])\nPath('l159-report.json').write_text(json.dumps(report,indent=2)+'\\n')"))
    cells.append(nb.v4.new_markdown_cell('## NEXT STEP · Historical reproduction gate\nTable 1 wikiTables BART-table versus graph, three targets and three runs: **NOT_RUN**. Exact tables/splits, release, model/training/masking/decoding recipe were not located. This notebook contains no historical trainer. Recover artifacts, audit the recipe and estimate all compute before paid execution. Increasing these synthetic epochs does not reproduce the paper.'))
    cells.append(nb.v4.new_code_cell("paper_gate=dict(target='Vogel et al. 2023 Table1 wikiTables',status='NOT_RUN',fidelity='NOT_ESTABLISHED',runnable_historical_trainer=False,cloud_spend_usd=0)\nprint(json.dumps(paper_gate,indent=2))"))
    cells.append(nb.v4.new_markdown_cell('## EXIT · Write your 400–600-word vision brief\nUse the five prompts in the lesson. State what is hidden, what context survives, how gradients reach the graph, what was actually measured, and the held-out-database experiment that could disconfirm your transfer claim. Ask the teaching agent to score the brief; code execution does not score prose.'))
    example='A frozen decoder preserves a gradient with respect to the graph representation while its own weights remain fixed. We must mask every serialization of a schema identity before encoding. The synthetic graph result tests redundancy under one generator, not the historical BART model or transfer to another database. My next experiment would hold databases out of pre-training and compare adaptation with a scratch baseline under a declared label and tuning budget. Missing paper data and configuration must be recovered before claiming full reproduction.' if solution else ''
    cells.append(nb.v4.new_code_cell("# Replace the author example (solution) or blank (student) with your brief.\nwritten_defense="+repr(example)+"\nPath('l159-brief.json').write_text(json.dumps(dict(written_defense=written_defense,learner='PENDING_WRITTEN_DEFENSE',paper=paper_gate),indent=2)+'\\n')\nprint('Evidence exported; written defense pending teacher review')"))
    notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    for i,c in enumerate(notebook.cells):c.id=f'l159-{i:03d}'
    path=P/('solutions' if solution else '')/(S+'.ipynb')
    if solution and path.exists():
        old=nb.read(path,4)
        if [(c.cell_type,c.source) for c in old.cells]==[(c.cell_type,c.source) for c in notebook.cells]:
            notebook.metadata=old.metadata
            for c,prior in zip(notebook.cells,old.cells):
                if c.cell_type=='code':c.outputs=prior.outputs;c.execution_count=prior.execution_count;c.metadata=prior.metadata
    nb.write(notebook,path)
print('Built L159 lesson, reference, three figures and portable student/solution notebooks')
