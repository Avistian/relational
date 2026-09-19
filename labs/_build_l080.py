"""Build the exam from canonical prose, complete visible source and pinned inputs."""
import ast,base64,gzip,hashlib,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
from _colab import BOOTSTRAP_MD,BOOTSTRAP_CODE
from _figures_l080 import build as figures
LAB=Path(__file__).resolve().parent;ROOT=LAB.parent
SLUG='0080-year-2-exit-exam';TITLE='Year 2 exit exam: defend a reproducible comparison'
CHECKS={
'binary_loss':"""assert abs(binary_loss([0,1],[.2,.8])+np.log(.8))<1e-12
for y,p in [([0,1],[.2]),([0,2],[.2,.3]),([0,1],[-.1,.9])]:
    try: binary_loss(y,p)
    except ValueError: pass
    else: raise AssertionError('Reject invalid labels, shapes or probabilities')
print('CHECK: independently calculated loss and invalid inputs pass')""",
'choose_validation':"""assert choose_validation([.31,.33])==0
assert choose_validation([.4,.2,.2])==1
for values in [[],[float('nan')],[float('inf')]]:
    try: choose_validation(values)
    except ValueError: pass
    else: raise AssertionError('Reject empty or nonfinite validation evidence')
print('CHECK: first minimum, ties and invalid evidence pass')""",
 'temperature_probability':"""np.testing.assert_allclose(temperature_probability([[0,np.log(3)]],1),[.75])
a=temperature_probability([[0,2]],1)[0];b=temperature_probability([[0,2]],2)[0]
assert .5<b<a<1
np.testing.assert_allclose(temperature_probability([[100,102]],2),[b])
print('CHECK: odds, temperature and additive-logit invariance pass')"""}

def source_cells(solution):
    cells=[]
    explanations={'ft_l080.py':'Pinned released FT architecture: tokenizer → multi-head attention → ReGLU residual blocks → CLS readout. Peripheral activation dispatch is explicit.',
        'tabm_v2.py':'Corrected numeric TabM: member adapters around shared matrices, independent heads, member training loss and probability averaging.',
        'tabpfn_l064_v2.py':'Complete historical v2 architecture, checkpoint mapping and simple numeric inference. Every checkpoint tensor is loaded. No new pretraining.',
        'exit_l080.py':'The exam protocol, full trainer and evidence auditor. The three TODO functions are called by this live driver.'}
    for filename in explanations:
        source=(LAB/'relkit'/filename).read_text();tree=ast.parse(source)
        cells.append(nbf.v4.new_markdown_cell('## PROVIDED · '+filename+'\n\n'+explanations[filename]))
        for node in tree.body:
            if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant):continue
            if filename=='tabpfn_l064_v2.py' and isinstance(node,ast.FunctionDef) and node.name in ['load_dataset','run_experiment']:continue
            if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('relkit.'):continue
            code=ast.get_source_segment(source,node)
            if filename=='exit_l080.py' and isinstance(node,ast.FunctionDef) and node.name in CHECKS:
                name=node.name;cells.append(nbf.v4.new_markdown_cell('### TODO · '+name+'\n\nImplement the contract in the function signature. Predict the CHECK outcome before running it.'))
                if not solution:
                    sig=code.split('\n',1)[0];code=sig+'\n    # TODO: implement the mathematical/selection contract.\n    raise NotImplementedError("'+name+'")'
                cells.append(nbf.v4.new_code_cell(code));cells.append(nbf.v4.new_markdown_cell('### CHECK · '+name));cells.append(nbf.v4.new_code_cell(CHECKS[name]))
            else:
                if isinstance(node,(ast.ClassDef,ast.FunctionDef)):
                    cells.append(nbf.v4.new_markdown_cell('### PROVIDED · '+node.name+'\n\n'+(ast.get_docstring(node) or 'Read the inputs and trace the returned values before continuing.')))
                # Accumulate adjacent imports/constants into one coherent cell.
                if isinstance(node,(ast.Import,ast.ImportFrom,ast.Assign)) and cells and cells[-1].cell_type=='code' and cells[-1].metadata.get('preamble'):
                    cells[-1].source+='\n'+code
                else:cells.append(nbf.v4.new_code_cell(code,metadata={'preamble':True} if isinstance(node,(ast.Import,ast.ImportFrom,ast.Assign)) else {}))
    return cells

def image_markup(name,notebook=False):
    captions={'protocol':'Illustrative validation selection: .31 beats .33, so the selected test loss is .40 even if another candidate has .25.','results':'Author-reference evidence: fresh local runs, three downstream seeds per task/regime. Dots show seeds; black bars show means.'}
    path=LAB/'figures/l080'/f'{name}.png';src='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode() if notebook else f'../labs/figures/l080/{name}.png'
    if notebook:
        return f'<figure class="exam-figure"><img src="{src}" alt="{captions[name]}"><figcaption>{captions[name]}</figcaption></figure>'
    return f'<figure class="exam-figure dg-figure"><span class="dg-scroll-note">Scroll horizontally to inspect the full-size figure.</span><div class="dg-image" tabindex="0" role="region" aria-label="Scrollable exam figure"><img src="{src}" alt="{captions[name]}"></div><figcaption>{captions[name]}</figcaption></figure>'

def results_table():
    r=json.loads((LAB/'_verify_l080_results.json').read_text());s='**Fresh author training / frozen-v2 inference; not student output or paper-result reproduction.**\n\n| Task / regime | XGBoost | FT-Transformer | TabM | TabPFN v2 |\n|---|---:|---:|---:|---:|\n'
    for d,name in enumerate(r['tasks']):
        for regime in ['random','temporal']:
            z=r['summary'][regime];s+='| '+name+' / '+regime+' | '+' | '.join(f'{m:.3f} ± {sd:.3f}' for m,sd in zip(z['means'][d],z['sample_sd'][d]))+' |\n'
    return s+'\nMean ± sample SD of test log loss across three seeds. Lower is better.\n'

def notebook(solution):
    cells=[nbf.v4.new_markdown_cell('# '+TITLE+'\n\n**PROVIDED / TODO / CHECK / EXIT.** Your deliverable is a fresh comparison plus a written defense. This capstone reuses three visible model implementations; it introduces no new paper. Read the protocol before running. Author-reference evidence is separate from your output.'),nbf.v4.new_markdown_cell(BOOTSTRAP_MD),nbf.v4.new_code_cell(BOOTSTRAP_CODE)]
    cells.append(nbf.v4.new_markdown_cell('## Cold retrieval\n\nExplain the three tree biases, TabM training/inference, TabPFN context and TabICL stages before looking below. Predict which family will change most across regimes, and why.\n\n## Concept recap\n\nLog loss is −log of the probability of the observed label, averaged over rows. With y=1 and p=.8 it is .223 nats. Validation chooses recipes; test evaluates a frozen choice. Mean seed losses are ranked within each dataset. Seeds do not create independent datasets. Train-only imputation/scaling and a query context containing training labels only preserve the comparison boundary.\n\n'+image_markup('protocol',True)))
    cells.append(nbf.v4.new_markdown_cell('## Exact protocol and reproduction\n\n'+(LAB/'l080-reproduction.md').read_text()))
    cells.append(nbf.v4.new_markdown_cell('## PROVIDED · author-reference evidence\n\n'+results_table()+image_markup('results',True)))
    # Embed exact portable inputs and all predictions for detached analysis; models stay visible.
    payload={}
    for relative in ['data/l080/exam.npz','data/l080/exam.json','data/l080/smoke.npz','data/l080/smoke.json','_verify_l080_results.json']:
        b=(LAB/relative).read_bytes();payload[relative]={'sha256':hashlib.sha256(b).hexdigest(),'gzip_base64':base64.b64encode(gzip.compress(b,mtime=0)).decode()}
    cells.append(nbf.v4.new_markdown_cell('## PROVIDED · exact input archive\n\nThe following cell restores hashed raw subsets and author predictions. It does not manufacture a fresh result. If a local file differs, stop and inspect it.'))
    cells.append(nbf.v4.new_code_cell('import base64,gzip,hashlib,json\nfrom pathlib import Path\nPACKAGED='+repr(payload)+'''\nROOT=Path.cwd()
if not (ROOT/'relkit').is_dir() and (ROOT/'labs/relkit').is_dir(): ROOT=ROOT/'labs'
for name,item in PACKAGED.items():
    data=gzip.decompress(base64.b64decode(item['gzip_base64']))
    assert hashlib.sha256(data).hexdigest()==item['sha256']
    dest=ROOT/name;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists(): assert dest.read_bytes()==data, 'Local input differs: '+name
    else: dest.write_bytes(data)
print('Exact raw inputs and author archive available')'''))
    cells.extend(source_cells(solution))
    cells.append(nbf.v4.new_markdown_cell('## CHECK · complete author evidence and failure injection\n\nThis is a fresh analysis of saved predictions, not fresh fitting. The audit calls your live loss and selection functions.'))
    cells.append(nbf.v4.new_code_cell('''author=json.loads((ROOT/'_verify_l080_results.json').read_text())
summary=audit_result(author)
assert summary==author['summary']
for kind in ['missing','prediction','overlap']:
    broken=copy.deepcopy(author)
    if kind=='missing': broken['records'].pop()
    elif kind=='prediction': broken['records'][0]['predictions'][0]=1-broken['records'][0]['predictions'][0]
    else:
        ids=next(iter(broken['data']['panels'].values()))['ids'];ids['test'][0]=ids['train'][0]
    try: audit_result(broken)
    except ValueError: print('Correctly rejected:',kind)
    else: raise AssertionError('Audit accepted '+kind)
print('Audited',len(author['records']),'records')'''))
    cells.append(nbf.v4.new_markdown_cell('## RUN · fresh complete smoke, then full exam\n\nThe smoke runs all four methods on both tasks and regimes, one seed. It is a different smaller experiment. Choose `exam` for the assessed full 48-record run. The operator consumes the functions above, including your TODOs; it does not reload completed solutions. A new path is created each time. Download the output JSON and live-source file.'))
    cells.append(nbf.v4.new_code_cell(r'''import uuid
RUN_FRESH_COMPARISON=True
PRESET='smoke'  # Change to 'exam' for your assessed submission.
if RUN_FRESH_COMPARISON:
    output=ROOT/'data/cache'/('l080-'+PRESET+'-'+uuid.uuid4().hex+'.json')
    fresh=run_experiment(ROOT,PRESET,output,namespace=globals())
    kernel_source='\n\n'.join(get_ipython().history_manager.input_hist_raw)
    source_path=output.with_suffix('.kernel.py');source_path.write_text(kernel_source)
    fresh['live_kernel_source_sha256']=hashlib.sha256(kernel_source.encode()).hexdigest()
    fresh['live_kernel_source_file']=source_path.name
    fresh['reference_source_hashes_note']='Disk hashes identify course references; live source above identifies edited notebook code.'
    output.write_text(json.dumps(fresh,indent=2)+'\n')
    print('Fresh evidence:',output)
    print('Local protocol:',PRESET,'records:',len(fresh['records']))
    for regime,panel in fresh['summary'].items():print(regime,panel['mean_ranks'])
else:
    print('Fresh comparison NOT_RUN; analysis-only is not an exit pass.')'''))
    cells.append(nbf.v4.new_markdown_cell('## EXIT · written submission\n\n'+(LAB/'l080-submission.md').read_text()+'\n\nAsk the tutor to score your cold explanations and artifacts against the lesson rubric. Do not mark yourself complete solely because the checks pass.'))
    for i,c in enumerate(cells):c.id=f'l080-{i:03d}'
    return nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}})

def build():
    from _walkthrough_delivery import snapshot, finalize
    snapshot(80)
    figures();body=(ROOT/'lessons/content'/f'{SLUG}.md').read_text().replace('{{RESULTS}}',results_table())
    body=re.sub(r'\{\{FIG:(\w+)\}\}',lambda m:image_markup(m[1]),body)
    head='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+TITLE+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/decision-guide.css"><link rel="stylesheet" href="../assets/exit-exam.css"></head><body><article>'
    nav=f'<nav><a href="../index.html">Course</a> · <a href="0079-neural-tabular-decision-guide.html">Lesson 79</a></nav><header><p>Year 2 · Quarter 4 · Lesson 080</p><h1>{TITLE}</h1></header><aside><a href="../labs/{SLUG}.ipynb">Download lab</a> · <a href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{SLUG}.ipynb">Open in Colab</a> · <a href="../labs/html/{SLUG}.html">Read lab</a> · <a href="../labs/l080-submission.md">Submission template</a> · <a href="../labs/l080-reproduction.md">Reproduce</a></aside>'
    html=render(body).replace('<table>','<div class="exam-table"><table>').replace('</table>','</table></div>')
    scripts=''.join(f'<script src="../assets/{name}.js"></script>' for name in ['retrieval-pool','retrieval-bank','teachback','l080-lesson'])
    (ROOT/'lessons'/f'{SLUG}.html').write_text(head+nav+html+'</article>'+scripts+'</body></html>')
    ref='''# Year 2 exit · compact reference

Freeze **information → split → metric → candidate budget → validation selection → test**.

| Mechanism | Explain or check |
|---|---|
| Irregular targets | Tree partitions can fit sharp changes; smooth-function assumptions may hurt. Test target smoothing. |
| Meaningful feature axes | Axis-aligned splits can exploit original coordinates; rotate numeric features to probe this. |
| Irrelevant columns | Compare nuisance-feature additions under matched tuning; selection/attention can still overfit. |
| TabM | Shared W with member adapters; average member losses during training, probabilities during inference. |
| TabPFN v2 | Prior-trained weights + labeled context + unlabeled queries; check context and query boundaries. |
| TabICL 2025 | Column embedding → row interaction → ICL prediction; examine cost and distribution fit. |
| Information ceiling | Identical inputs with different required labels cannot be separated without more information. |

Worked selection: validation [.31,.33] selects index0 even when test [.40,.25] favors index1. Report .40.

Log loss = mean −[y log(p)+(1−y) log(1−p)]. At y=1,p=.8: .223 nats. Rank seed means **within** a dataset; do not count seeds as datasets.

Pass requires ≥13/16, no zero, full protocol and evidence credit. No winning-model requirement. A smoke, archive audit or author result is not a student exam pass.

[Exam](../lessons/0080-year-2-exit-exam.html) · [Submission](../labs/l080-submission.md) · [Full reproducibility contract](../labs/l080-reproduction.md)
'''
    (ROOT/'reference'/f'{SLUG}.html').write_text(head+render(ref)+'</article></body></html>')
    for solution in [False,True]:
        nb=notebook(solution);path=LAB/('solutions' if solution else '')/f'{SLUG}.ipynb';path.parent.mkdir(exist_ok=True)
        if solution and path.exists():
            old=nbf.read(path,as_version=4);a=[c for c in nb.cells if c.cell_type=='code'];b=[c for c in old.cells if c.cell_type=='code']
            if [c.source for c in a]==[c.source for c in b]:
                for x,y in zip(a,b):x.outputs=y.outputs;x.execution_count=y.execution_count
        nbf.write(nb,path)
        if not solution:
            preview,_=HTMLExporter().from_notebook_node(nb)
            # Notebook prose follows lesson-relative links; render those from labs/html.
            preview=preview.replace('href="../labs/','href="../').replace('href="../lessons/','href="../../lessons/')
            (LAB/'html'/f'{SLUG}.html').write_text(preview)
    print('Built L080')
    finalize(80)
if __name__=='__main__':build()
