"""Canonical prose → lesson/reference, portable figures and student/solution labs."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l164';D=P/'figures/l164';D.mkdir(parents=True,exist_ok=True);S='0164-griffin-graph-centric-rdb-fm'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l164'})
TEAL='#176b68';AMBER='#a46b25';INK='#1e383b';BG='#f5f8f5'

def canvas(title,size):
    fig,ax=plt.subplots(figsize=size);fig.subplots_adjust(left=.02,right=.98,bottom=.025,top=.89);fig.patch.set_facecolor(BG);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');fig.suptitle(title,x=.035,y=.965,ha='left',fontsize=18,color=INK,weight='bold');return fig,ax

def box(ax,x,y,w,h,title,body,color='#e2efec',fs=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.006',facecolor=color,edgecolor='#abc2bb'))
    ax.text(x+.014,y+h-.02,title,fontsize=12,weight='bold',va='top',color=INK)
    ax.text(x+.014,y+h-.061,body,fontsize=fs,va='top',linespacing=1.6,color=INK)

def arrow(ax,a,b,color=TEAL):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=2,color=color))

def save(fig,name):
    fig.canvas.draw();rr=fig.canvas.get_renderer()
    for ax in fig.axes:
        for t in ax.texts:
            extent=t.get_window_extent(rr)
            assert extent.x0>=0 and extent.y0>=0 and extent.x1<=fig.bbox.width and extent.y1<=fig.bbox.height,(name,t.get_text())
    for ext in ['svg','png']:fig.savefig(D/(name+'.'+ext),dpi=150,metadata={'Date':None} if ext=='svg' else {'Software':'L164'})
    plt.close(fig)

fig,ax=canvas('Griffin · cells stay available as graph context evolves',(12,9.5))
box(ax,.015,.815,.30,.16,'QUERY + DATABASE','(driver, cutoff, task)\nPK–FK edges → owned trees\n2 hops · fanout 20 per relation',color='#f8eddc')
box(ax,.365,.815,.62,.16,'FROZEN INPUT SPACE · D = 512','Text/category cells + column names → text vectors\nNormalized numbers → float encoder · task / relation names → vectors\nPer row: values C×512 and metadata C×512; C varies by table')
arrow(ax,(.32,.885),(.355,.885));arrow(ax,(.65,.81),(.65,.758))
box(ax,.015,.60,.30,.145,'KEEP BOTH ARRAYS','Columns:   [form]  [experience]\nKeys K:       k₁              k₂\nValues V:    v₁              v₂')
box(ax,.365,.60,.62,.16,'LAYER 0 · INITIAL CELL READ','Metadata queries Q = K → 8-head attention(Q,K,V)\nLinear projection → mean across column queries → row state x\nHidden cell keys are masked; their values cannot contribute')
arrow(ax,(.32,.665),(.355,.665));arrow(ax,(.67,.596),(.67,.54))
box(ax,.015,.32,.30,.225,'LAYERS 1–3 · CELL READS','Task state t → normalize → Q\nMetadata K + cell values V\nQKᵀ / √64 → softmax → readout\nOutput × learned query projection\nAdd cell readout to row state x',color='#deefeb')
box(ax,.365,.32,.62,.22,'EACH LAYER · RELATIONAL UPDATE','Normalize x → transformed neighbour messages\nMEAN within relation → × relation vector → MAX across relations\nSeparate forward / reverse weights; add both + feedforward residual\nLayers 0–2: task state += projected normalized row state\nRepeat later cell read + update for layers 1–3; graph gates OFF')
arrow(ax,(.32,.435),(.355,.435))
arrow(ax,(.67,.31),(.67,.255))
box(ax,.015,.03,.30,.215,'TRAINED / FROZEN','Train: attention, node / relation\ntransforms, task updates\nFrozen: text + float encoders,\nlabel embeddings / float decoder\nSampling defines legal context',color='#f8eddc')
box(ax,.365,.145,.62,.11,'ROOT OUTPUT · GATHER AFTER FINAL NORMALIZATION','All nodes N×512 → requested roots B×512 = z')
arrow(ax,(.52,.14),(.52,.108));arrow(ax,(.84,.14),(.84,.108))
box(ax,.365,.015,.30,.095,'CLASSIFICATION','z · candidate labels → softmax',fs=10.5)
box(ax,.695,.015,.29,.095,'REGRESSION','float decoder → inverse scale',fs=10.5)
save(fig,'architecture')

fig,ax=canvas('One output can take coordinates from different relations',(12,5.2))
box(ax,.02,.56,.235,.36,'RESULTS · two neighbours','[2,4] and [4,2]\nmean = [3,3]\n× relation vector [1,1]\nmessage = [3,3]')
box(ax,.02,.10,.235,.36,'TEAMMATE · one row','[8,1]\nmean = [8,1]\n× relation vector [0.5,2]\nmessage = [4,2]',color='#f8eddc')
arrow(ax,(.27,.71),(.38,.60));arrow(ax,(.27,.25),(.38,.44),AMBER)
ax.text(.39,.78,'STACK RELATION MESSAGES',fontsize=12,weight='bold',color=INK)
for i,vals in enumerate([[3,3],[4,2]]):
    for j,value in enumerate(vals):
        x=.405+.11*j;y=.51-.18*i
        ax.add_patch(FancyBboxPatch((x,y),.095,.13,boxstyle='round,pad=.005',facecolor='#f1d9ae' if (i,j)==(1,0) else '#cce5df' if (i,j)==(0,1) else 'white',edgecolor='#a8bdb4'))
        ax.text(x+.047,y+.065,str(value),ha='center',va='center',fontsize=22,color=INK)
ax.text(.40,.20,'max ↓    max ↓',fontsize=12,color=INK)
arrow(ax,(.65,.49),(.72,.49))
box(ax,.74,.33,.235,.36,'OUTPUT [4,3]','Coordinate 1: teammate\nCoordinate 2: results\nMax is not one chosen row.\nNegative maxima stay negative.')
ax.text(.30,.015,'Repeat the whole results set: its mean stays [3,3]. Flat neighbour averaging changes.',fontsize=10.5,color=INK)
save(fig,'relations')

fig,ax=canvas('Three stages · downstream reproduction is budget-gated',(12,4.6))
for x,title,body,col in [(.015,'1 · COMPLETION','Single-table rows\n[known] [MASK] [known]\nPredict hidden cell embedding\nLoss: cosine distance\nTrain shared cell machinery','#e2efec'),(.355,'2 · JOINT SFT','Other labeled databases\nKnown task → target value\nClassification: cross-entropy\nRegression: squared error\nTrain shared relational model','#e2efec'),(.695,'3 · DOWNSTREAM','F1 · 512 or 4096 train queries\nFine-tune on driver-dnf\nSelect with validation\nEvaluate final test once for claim\n5 subset seeds per arm','#f8eddc')]:
    box(ax,x,.27,.285,.65,title,body,col,fs=11)
arrow(ax,(.308,.62),(.345,.62));arrow(ax,(.648,.62),(.685,.62))
ax.text(.02,.11,'RELEASED CHECKPOINT replaces stages 1–2 here; fresh pretraining is NOT_RUN.',fontsize=12,weight='bold',color=TEAL)
ax.text(.02,.005,'Compare transferred versus random initialization. Full 20-fit downstream evidence remains budget-gated.',fontsize=11,color=INK)
save(fig,'training')

captions={'architecture':'Released transfer variant: four message layers, eight attention heads, forward/reverse updates and graph gates off. C varies by table; shared width is512. Frozen inputs and decoder are separate from trainable shared updates.','relations':'Synthetic already-transformed neighbour vectors at cutoff5. Multiply each relation mean by its vector, then maximize each coordinate. The output [4,3] combines two relations.','training':'Conceptual training lifecycle. Our checkpoint-based downstream replay would reuse stages 1–2; it does not repeat pretraining. All20full fits remain unrun.'}
prose_text=(R/'lessons/content'/(S+'.md')).read_text();decision=json.loads((E/'cost-decision.json').read_text())
evidence=f'''<div class="evidence-note"><strong>INCOMPLETE_BUDGET_GATE.</strong> The L4 probe completed 512 training queries (two optimizer steps) and all 566 validation queries, with 12.72 GB peak allocated GPU memory. It produced no final test score. A conservative 200-epoch scenario for all 20 fits projects USD{decision['raw_compute_usd']:.2f} compute, or USD{decision['safety_adjusted_compute_usd']:.2f} with 25% margin, above the USD 7 fit/check allowance. Early stopping could reduce cost but is unmeasured. Paid work stopped. The retained reservation plus overhead allowance is USD0.86816; the measured worker-body estimate alone is USD0.006683, not an itemized invoice.</div>'''

def prose(portable=False):
    t=prose_text
    for name,caption in captions.items():
        if portable:fig='![Griffin '+name+'](data:image/png;base64,'+base64.b64encode((D/(name+'.png')).read_bytes()).decode()+')\n\n'+caption
        else:fig='<figure class="route-figure"><img src="../labs/figures/l164/'+name+'.svg" alt="'+caption+'"><figcaption>'+caption+' On narrow screens, scroll the figure horizontally.</figcaption></figure>'
        t=t.replace('[[FIG:'+name+']]',fig)
    replacements={'WARMUP':('Recall without notes: what is lost by a fixed row pooling step? What must remain frozen before test evaluation?','<div id="warmup"></div>'),
      'PREDICT':('Before calculating: predict what repeating every neighbour in one relation does to its mean.','<div id="predict"></div>'),
      'CELLS':('Notebook task: reverse paired keys/values, switch task queries, and mask a cell; predict each readout first.','<div id="cell-explorer"></div><noscript>Interactive boards require JavaScript. Worked arithmetic, figures and reproduction status remain available below and in the notebook.</noscript>'),
      'RELATIONS':('Notebook task: compare cutoff5 with cutoff7; repeat every results neighbour and compare relation-balanced versus flat aggregation.','<div id="relation-explorer"></div>'),
      'TEACHBACK':('Explain the complete prediction path and the evidence boundary in your own words before comparing with the source.','<div id="teachback"></div>')}
    for key,(plain,html) in replacements.items():t=t.replace('[['+key+']]',plain if portable else html)
    t=t.replace('[[EVIDENCE]]',evidence)
    if portable:
        t=t.replace('](../','](https://avistian.github.io/relational/').replace('href="../','href="https://avistian.github.io/relational/')
        t=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',t)
    return t

def doc(title,body,interactive=False):
    body=re.sub(r'<table([^>]*)>',r'<div class="route-scroll" tabindex="0"><table\1>',render(body)).replace('</table>','</table></div>')
    scripts=['retrieval-pool','retrieval-bank','predict','teachback','griffin','l164-lesson'] if interactive else []
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+n+'.css">' for n in ['lesson','atomic-route','checkpoint','lab-access','griffin'])+'</head><body class="checkpoint griffin-lesson"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0163-lm-encoders-for-rows.html">Lesson163</a></nav><header><p class="route-kicker">Year5 · Quarter1 · Lesson164</p><h1>'+title+'</h1></header>'+body+'</article>'+''.join('<script src="../assets/'+n+'.js"></script>' for n in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Griffin — a graph-centric relational foundation model',prose(),True))
ref='''## Forward-pass checklist

1. Query=(entity,cutoff,task). Sample owner-specific temporal trees; keep metadata paired with cells.
2. Frozen cell/column/task/relation embeddings use shared width512; table cell counts may differ.
3. Released first layer: metadata Q/K,cell V,self-attention then average. Later: evolving task Q,metadata K,cell V; output multiplied by a query projection.
4. Normalize row state; transform neighbours; mean within each relation, multiply by learned relation vector, coordinate-wise max across relations. Add forward/reverse messages and feedforward residual.
5. Update task state between layers. Four layers, graph gates off in the selected transfer script.
6. Normalize,gather roots; classification dot products with candidate labels→softmax. Regression uses frozen float decoder and inverse scale.

## Arithmetic you should recover

Q=[√2,0],K=identity,V=[[2,0],[0,4]] → scores[1,0] → weights[.731,.269] → readout[1.462,1.076]. Joint key/value permutation preserves it; changing metadata may not.

Results mean[3,3]×[1,1]=[3,3]; teammate mean[8,1]×[.5,2]=[4,2]; max=[4,3]. Repeat an entire relation's neighbour set: unchanged mean. One unequal neighbour duplicated: may change mean. Empty receiver: zero; nonempty negative messages: keep negative max.

## Temporal and transfer boundaries

Strict owner-specific timestamp<cutoff at every hop. Row-index order alone is insufficient; Griffin's separate few-shot helper ignores timestamps. The selected F1 static-root case passed a narrow audit; historical availability remains unresolved.

No-pretrain fits Griffin from random initialization. Others-2 SFT starts from a released jointly pretrained model and then fine-tunes on F1. This is weight adaptation, not zero-shot inference. Published Table12 scores are external means. Source/gradient parity and a timed pilot do not establish a transfer gain.

'''+evidence+'''

[Lesson](../lessons/0164-griffin-graph-centric-rdb-fm.html) · [Notebook](../labs/0164-griffin-graph-centric-rdb-fm.ipynb) · [Full protocol](../labs/l164-reproduction.md) · [Paper](https://arxiv.org/html/2505.05568v1) · [Pinned source](https://github.com/yanxwb/Griffin/tree/b9d0e1fa8d89dfb1cd8bd5976b71de8a3b515427).
'''
(R/'reference/griffin.html').write_text(doc('Griffin — quick reference',ref))

# Readable canonical definitions, never hide a learner function behind an import.
def definitions(path):
    text=path.read_text();return [(n.name,ast.get_source_segment(text,n)) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]

checks=dict(definitions(P/'_check_l164.py'))
contracts={
 'cell_attention':('Read the permitted cells','Implement scaled dot-product attention for[...,Q,d]queries,[...,C,d]keys and[...,C,v]values. blocked is a broadcastable boolean mask(True excludes). Reject an all-masked query. Normalize over cells; dropout applies to weights only when training=True. Preserve dtype,device and gradients.','check_attention'),
 'relation_pool':('Balance relations before combining them','Receive[N,D]neighbour states,edge_index[2,E]with receiver then sender,Erelation IDs,and[R,D]already-transformed relation vectors. Mean by(receiver,relation),multiply by that relation vector,then maximize over present relations per receiver. Empty receiver→zeros; negative messages remain negative. Preserve gradients.','check_relations'),
 'eligible_edges':('Carry the owner cutoff','Receive[B,F]timestamps and[B]owner cutoffs. Return a boolean strict-time mask. -1is padding in this teaching fixture; other negative times can be valid. Require exactly one cutoff per owner,without broadcasting a single global time.','check_cutoffs')}
release_files={str(f.relative_to(P)):f.read_bytes() for f in sorted((P/'sources/l164/upstream').glob('*')) if f.is_file()}
for name in ['_prepare_l164.py','_run_l164.py','l164-requirements.txt','sources/l164/data-api.json','sources/l164/source-ledger.json']:release_files[name]=(P/name).read_bytes()
trace=(E/'attention-trace.json').read_bytes();report=(E/'report.json').read_bytes()
for solution in [False,True]:
    cells=[]
    def md(text):cells.append(nb.v4.new_markdown_cell(text))
    def code(text,hidden=False):
        c=nb.v4.new_code_cell(text)
        if hidden:c.metadata['tags']=['data-payload']
        cells.append(c)
    md('# Lesson164 · Griffin\n\nStandalone lab: three live mechanisms, a complete visible model, cached real checkpoint attention and the pinned full release experiment. Default lane needs CPU PyTorch/NumPy only. No paid service is contacted. Author preparation is not learner mastery. Website links become available after publication.')
    md(prose(True))
    md('## PROVIDED · Environment\nThe local author checks used PyTorch2.13.0 CPU. The cloud source probe used2.5.1/cu124. This default mechanism lane avoids PyG/datasets/Accelerate; the optional full lane has separate direct pins.')
    code("# @colab-bootstrap\nimport os,sys,subprocess,importlib.util\nos.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')\nif any(importlib.util.find_spec(n) is None for n in ['torch','numpy']):\n    subprocess.check_call([sys.executable,'-m','pip','install','torch','numpy'])\nimport math,json,base64,hashlib,copy\nfrom pathlib import Path\nimport numpy as np\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\ntorch.set_num_threads(1)\nprint('PyTorch',torch.__version__)")
    md('## PROVIDED · Cached real projected attention\nStorage only: projected Q/K/V from the first validation root in the released Others-2 model’s second layer. The expected readout uses an independent PyTorch attention operator. Replaying these tensors does not rerun the full pretrained model.')
    code('trace_blob=base64.b64decode('+repr(base64.b64encode(trace).decode())+')\nassert hashlib.sha256(trace_blob).hexdigest()=='+repr(hashlib.sha256(trace).hexdigest())+'\nreal_trace=json.loads(trace_blob)\nprint(real_trace["note"])',True)
    for name,source in definitions(P/'_check_l164.py'):code(source)
    for name,source in definitions(P/'relkit/griffin_l164.py'):
        if name in contracts:
            title,contract,check=contracts[name];md('## TODO · '+title+'\n\n'+contract+'\n\nPredict a counterexample before coding; use the equations above to derive the update.')
            code(source if solution else source.split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")')
            code('# CHECK\n'+check+'('+name+')\nprint("'+title+': PASS")')
        else:
            md('## PROVIDED · '+name+'\nThis is the checkpoint-compatible visible computation. CellAttention calls your cell_attention; RelationMessage calls your relation_pool. Study the forward pass,not just the class name.')
            code(source)
    md('## CHECK · Complete model forward and backward\nSmall untrained four-layer model,D16,8heads,4nodes. This checks live integration and finite gradients; its logits are not an F1performance result. The exact512-wide pretrained-model parity is recorded separately in author evidence.')
    code("torch.manual_seed(164)\nmodel=GriffinMod(hiddim=16,num_mp=4).double()\nnode=[(torch.randn(3,16,dtype=torch.float64),torch.randn(4,3,16,dtype=torch.float64))]\nargs=[node,[None],[torch.randn(16,dtype=torch.float64)],torch.tensor([[0,0,1,2],[1,2,2,3]]),torch.tensor([0,1,0,1]),torch.randn(2,16,dtype=torch.float64)]\noutput=model(*args)\nassert output.shape==(4,16)\noutput.square().mean().backward()\nassert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())\nprint('Four-layer live forward/backward PASS; no accuracy claim')")
    md('## PROVIDED · Recompute the worked interventions\nBoth the synthetic and real cached attention paths call your function. The cutoff mask directly selects the worked graph edges before your relation_pool runs.')
    for _,source in definitions(P/'_lesson_run_l164.py'):code(source)
    code("report=worked_report(cell_attention,relation_pool,eligible_edges,real_trace)\nprint(json.dumps(report,indent=2))\nPath('l164-report.json').write_text(json.dumps(report,indent=2)+'\\n')")
    md('## EXIT · Write the defense\nUse the five-part template in the lesson. Include a complete prediction trace,the[4,3]derivation,a temporal counterexample,and exactly what the unrun20fitcomparison prevents you from claiming.200–300words;rubric computation/architecture/legality/evidence/next test,0–2each,≥8with no zero after review.')
    defense='The model preserves cell vectors while graph and task states evolve. Relation-wise mean followed by coordinate-wise max prevents a complete duplication of one relation from reweighting the others. A source-matching forward pass and a resource probe do not show that pretraining transfers. That requires all20matched downstream fits and final keyed test predictions. The separate index-based few-shot helper also needs a database-specific temporal audit.' if solution else ''
    code('defense='+repr(defense)+'\nPath("l164-submission.json").write_text(json.dumps(dict(defense=defense,learner="PENDING_WRITTEN_DEFENSE",review=None),indent=2))\nprint("Defense exported; learner PENDING_WRITTEN_DEFENSE")')
    md('## NEXT STEP · Complete selected release reproduction (OFF)\nAuthor decision: INCOMPLETE_BUDGET_GATE. The following executable lane reproduces the complete source protocol,not a larger toy. All20fits may take many hours; the author cap does not authorize additional cloud spend. The manual learner gate requires explicit runtime acknowledgment. It uses the original released model after the visible port was independently checked. One compatibility property repairs model.device; the trainer and data sampler otherwise retain their release behavior. Full wrapper execution and live Colab remain unverified.')
    md('### Read the complete released trainer\nThe main notebook already exposes the model. The following source appendix exposes the original loss,evaluation,checkpoint selection and training loop. These are source listings,not executed notebook commands. Supporting samplers/model source are pinned in the materialization payload and linked from the protocol.')
    for name,source in definitions(P/'sources/l164/upstream/hmaintask_downsample_absolute_eval_sample.py'):
        md('#### Released '+name+'\n```python\n'+source+'\n```')
    md('### Read the replay instrumentation\nThe wrapper below changes device access and saves independent keyed prediction evidence. No algorithm is hidden in the encoded storage cell.')
    for name,source in definitions(P/'_run_l164.py'):
        if name in ['setup','run_source']:md('#### Wrapper '+name+'\n```python\n'+source+'\n```')
    payload={k:base64.b64encode(v).decode() for k,v in release_files.items()};hashes={k:hashlib.sha256(v).hexdigest() for k,v in release_files.items()}
    code('release_payload='+repr(payload)+'\nrelease_hashes='+repr(hashes)+'\nprint("Pinned release payload stored; no downloads or training")',True)
    code("RUN_FULL_REPRO=False\nACKNOWLEDGE_RUNTIME_AND_BUDGET=False\nif RUN_FULL_REPRO:\n    assert ACKNOWLEDGE_RUNTIME_AND_BUDGET, 'Read the budget/protocol and acknowledge a separate learner run'\n    assert torch.cuda.is_available(), 'GPU required for full20fitrelease replay'\n    workspace=Path('l164-runtime').resolve();labs=workspace/'labs';labs.mkdir(parents=True,exist_ok=True)\n    for name,encoded in release_payload.items():\n        raw=base64.b64decode(encoded);assert hashlib.sha256(raw).hexdigest()==release_hashes[name]\n        path=labs/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)\n    subprocess.check_call([sys.executable,'-m','pip','install','-r',str(labs/'l164-requirements.txt')])\n    data=workspace/'release'\n    subprocess.check_call([sys.executable,str(labs/'_prepare_l164.py'),str(data)])\n    for arm in ['no-pretrain','others-2']:\n        for size in [512,4096]:\n            for seed in range(42,47):\n                subprocess.check_call([sys.executable,str(labs/'_run_l164.py'),'--phase','fit','--root',str(data),'--out',str(workspace/f'{arm}-{size}-{seed}'),'--arm',arm,'--size',str(size),'--seed',str(seed)])\nelse:\n    print('Full20fitreproduction NOT_RUN; author stop retained')")
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    for i,c in enumerate(book.cells):c.id=f'l164-{i:03d}'
    path=P/('solutions' if solution else '')/(S+'.ipynb');path.parent.mkdir(exist_ok=True)
    if solution and path.exists():
        old=nb.read(path,4);book.metadata=old.metadata
        previous=[c for c in old.cells if c.cell_type=='code'];current=[c for c in book.cells if c.cell_type=='code']
        if [c.source for c in previous]==[c.source for c in current]:
            for o,c in zip(previous,current):
                c.outputs=o.outputs;c.execution_count=o.execution_count;c.metadata=o.metadata
    nb.write(book,path)
print('Built L164 lesson,reference,three portable figures and both notebooks')
