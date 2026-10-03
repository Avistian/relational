"""Build printable lesson, architecture figures and portable source-visible labs."""
import ast,base64,hashlib,io,json,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;R=P.parent;S='b11-supervised-relational-baselines';E=P/'evidence/b11';F=P/'figures/b11'
F.mkdir(parents=True,exist_ok=True)
def box(ax,x,y,w,h,title,body,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.025',facecolor=color,edgecolor='#bad0d5',lw=1))
    ax.text(x+.025,y+h-.025,title,fontsize=12,fontweight='bold',color='#173a50',va='top')
    ax.text(x+.025,y+h-.075,body,fontsize=10,color='#173a50',va='top',linespacing=1.45)
def arrow(ax,x,y1,y2):ax.annotate('',xy=(x,y2),xytext=(x,y1),arrowprops=dict(arrowstyle='->',color='#537687',lw=1.6))
fig,ax=plt.subplots(figsize=(13,10));fig.patch.set_facecolor('#fbfcfd');ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.text(.03,.97,'One task, two computational paths',fontsize=23,weight='bold',color='#173a50')
ax.text(.03,.925,'Supervised training: eligible relational evidence → representation → task loss',fontsize=12,color='#52616b')
left=[('RelGNN · row encoders','Typed attributes + relative time\n→ vector for each row'),('Ordered atomic routes','Product → purchase → customer\nRoute-specific intermediate, then attention'),('Combine and predict','Sum route outputs → norm / activation\n→ target-row head → prediction'),('Optimize on task labels','L1 for driver-position regression\nValidation chooses the saved checkpoint')]
right=[('RelGT · five token ingredients','Attributes + type + time + hop + structure\n→ learned mixer → row tokens'),('Local and global branches','Local token self-attention + target readout\nGlobal attention to occupied centroids'),('Combine and predict','Normalize / combine branches\n→ task head → prediction'),('Optimize and update state','Task loss trains model weights\nTraining updates codebook via EMA')]
for col,items in enumerate([left,right]):
    x=.03+col*.50
    for i,(title,body) in enumerate(items):
        y=.69-i*.19;box(ax,x,y,.44,.15,title,body,'#eaf6f2' if col==0 else '#eef0fb')
        if i<3:arrow(ax,x+.22,y-.01,y-.035)
ax.text(.03,.045,'Fairness audit: align task keys, evidence, selection and resource budgets; declare global-state provenance.',fontsize=11,color='#173a50')
fig.savefig(F/'architecture.png',dpi=150,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(11,5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');fig.patch.set_facecolor('#fbfcfd')
ax.text(.03,.93,'Day 11 · two admitted purchases',fontsize=21,weight='bold',color='#173a50')
box(ax,.03,.56,.26,.22,'Products','P0 = 2      P1 = 6','#eaf6f2');box(ax,.37,.56,.28,.22,'Purchases','B0 = 1      B1 = 3\nB0→P0, B1→P1; both → C','#eef0fb');box(ax,.74,.56,.22,.22,'Customer','C = 0','#fff3e5')
ax.annotate('',xy=(.36,.68),xytext=(.3,.68),arrowprops=dict(arrowstyle='->',color='#537687'));ax.annotate('',xy=(.73,.68),xytext=(.66,.68),arrowprops=dict(arrowstyle='->',color='#537687'))
ax.text(.03,.40,'Route update',fontsize=13,weight='bold',color='#16877c');ax.text(.30,.40,'B0: 2 + 1 = 3     B1: 6 + 3 = 9     →     mean = 6',fontsize=12)
ax.text(.03,.26,'Token update',fontsize=13,weight='bold',color='#56569b');ax.text(.30,.26,'[C, B0, B1, P0, P1] = [0, 1, 3, 2, 6]   →   mean = 2.4',fontsize=12)
ax.text(.03,.10,'Identity scalar transforms and equal attention. These illustrations omit full-model encoders and heads.',fontsize=10,color='#52616b')
fig.savefig(F/'trace.png',dpi=150,bbox_inches='tight');plt.close(fig)

mechanism=json.loads((E/'mechanism.json').read_text());replay=json.loads((E/'replay.json').read_text())
def figure(name,caption,cls=''):return f'<figure class="b11-figure {cls}"><img src="../labs/figures/b11/{name}.png" alt="{caption}"><figcaption>{caption}</figcaption></figure>'
widget='''<div class="b11-board" data-b11-trace><h3>When can P1 change the answer?</h3><label>Prediction cutoff: day <span data-day>11</span><input name="cutoff" type="range" min="9" max="12" value="11" step="1"></label><label>P1 feature: <span data-product>6</span><input name="product" type="range" min="0" max="12" value="6" step="1"></label><button type="button">Reset example</button><output aria-live="polite">Day 11: B0 and B1 admitted. Scalar route update = 6; scalar target attention update = 2.4. At day 10 only B0 is admitted: 3 and 1.</output><noscript><p>The controls require JavaScript. Static worked values above remain available.</p></noscript></div>'''
results='| Checked block | Cases | Maximum output error | Maximum gradient error |\n|---|---:|---:|---:|\n'
for model in ['RelGNN composite','RelGT local+global']:
    rr=[r for r in mechanism['cases'] if r['model']==model];results+=f'| {model} | {len(rr)} | {max(r["output_error"] for r in rr):.2e} | {max(r["gradient_error"] for r in rr):.2e} |\n'
table='| Archived experiment | Seeds | Validation MAE | Test MAE |\n|---|---:|---:|---:|\n'
for name,label,seeds in [('RelGNN reconstructed','L143 · reconstructed RelGNN',5),('gnn','L146 · typed-mean GNN',3),('relgt','L146 · reduced corrected RelGT',3)]:
    r=replay['summaries'][name];table+=f'| {label} | {seeds} | {r["val"]["mean"]:.4f} ± {r["val"]["sample_sd"]:.4f} | {r["test"]["mean"]:.4f} ± {r["test"]["sample_sd"]:.4f} |\n'
table+='\n± denotes sample standard deviation across seeds, not a confidence interval. The blocks are separate experiments.'
def document(title,body,js=False):return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/relational-baselines.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../reference/curriculum.html#research-bridge">Research bridge</a></nav>'+render(body)+'</article>'+('<script src="../assets/relational-baselines.js"></script>' if js else '')+'</body></html>'
body=(R/'lessons/content'/f'{S}.md').read_text()
mobile='''<div class="b11-mobile"><h3>RelGNN path</h3><ol><li>Typed row attributes and relative time → row vectors.</li><li>Ordered product→purchase→customer route: transform the purchase for this direction, then attend into the customer.</li><li>Combine route outputs → task head → prediction.</li><li>Train on task loss; validation selects checkpoint.</li></ol><h3>RelGT path</h3><ol><li>Attributes, type, time, hop and structural position → mixed row tokens.</li><li>Local token attention and target readout; global attention to centroids.</li><li>Normalize/combine branches → task head → prediction.</li><li>Task loss trains weights; training updates centroid state via EMA.</li></ol></div>'''
for key,value in dict(ARCHITECTURE=figure('architecture','Full model paths. The B11 numerical experiment checks reduced pre-encoded blocks, not every stage.','b11-architecture')+mobile,TRACE_FIGURE=figure('trace','Same admitted facts, different grouping and normalization. Scalar arithmetic is an illustration.'),WIDGET=widget,RESULTS=results,REPLAY_TABLE=table).items():body=body.replace('{{'+key+'}}',value)
(R/'lessons'/f'{S}.html').write_text(document('B11 · Supervised relational baselines',body,True))
reference='''# B11 · Relational baseline comparison card

**Atomic route:** an ordered foreign-key-derived relational computation. Product→purchase→customer first constructs a direction-specific purchase representation, then attends into the customer. One composite block can contain multiple operations.

**RelGNN:** typed row/time encoders → ordered route aggregation and attention → combine routes → task head. **RelGT:** attribute/type/time/hop/structure encodings → mixed row tokens → local attention/readout + global centroid attention → task head. Both selected baselines train on task labels.

**Worked scalar example:** product features2,6; purchase features1,3; customer0. Identity route transforms give mean(2+1,6+3)=6. Equal target attention gives mean(0,1,3,2,6)=2.4. Neither is a full-model prediction.

**Temporal admission:** event≤cutoff AND arrival≤cutoff; then expand neighbors. Unknown arrival is not proof of historical availability. Targets stay out of the input.

**Fairness contract:** same complete task keys, labels, raw features, fit population, eligible contexts and induced edges; declare global-state training population and freeze it at evaluation. Fix validation selection, search allowance, seeds and query order. Report actual compute, parameters and memory. Same width/epochs/local rows do not suffice.

**Selection:** B11/L143/L146 course uses first strict validation minimum. Released RelGT uses last tied minimum. Preserve each lane's tie rule. Never use test MAE to select a configuration.

**Evidence:** three-seed reduced block checks PASS;11 saved runs/13,849 predictions rescored; fresh B11 benchmark fits NOT_RUN. L146 typed-mean GNN is not RelGNN. L143 versus L146 is not a matched comparison. Full RelGT search remains temporally and financially blocked. Whole-paper reproduction NOT_RUN; learner PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/b11-supervised-relational-baselines.html) · [Student lab](../labs/b11-supervised-relational-baselines.ipynb) · [Full protocol](../labs/b11-reproduction.md) · [RelGNN §3](https://arxiv.org/html/2502.06784v2) · [RelGT §3](https://arxiv.org/html/2505.10960v1)
'''
(R/'reference'/f'{S}.html').write_text(document('B11 · Comparison card',reference))

# Self-contained packet. Source/evidence paths follow the same layout as the repository labs directory.
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
    for f in (P/'sources/b11').rglob('*'):
        if f.is_file() and '__pycache__' not in str(f):z.write(f,str(f.relative_to(P)))
packet=buf.getvalue();(E/'portable-source-evidence.zip').write_bytes(packet)
module=(P/'relkit/baselines_b11.py').read_text();segments={n.name:ast.get_source_segment(module,n) for n in ast.parse(module).body if isinstance(n,ast.FunctionDef)}
test=(P/'_test_b11.py').read_text();checks={n.name:ast.get_source_segment(test,n) for n in ast.parse(test).body if isinstance(n,ast.FunctionDef)}
def png(name):return '!['+name+'](data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode()+')'
for solution in [False,True]:
    cells=[]
    def md(s):cells.append(nb.v4.new_markdown_cell(s))
    def code(s):cells.append(nb.v4.new_code_cell(s))
    md('# B11 · Same facts, different computation\n\n**PROVIDED:** source blocks and sealed saved evidence. **TODO:** implement three live operations. **CHECK:** immediate behavioral feedback. **EXIT:** defend a fair comparison.\n\nFresh execution is CPU mechanism verification and saved-evidence replay, not new benchmark training. Primary readings: [RelGNN §3](https://arxiv.org/html/2502.06784v2), [RelGT §3/Table6](https://arxiv.org/html/2505.10960v1).')
    code('''# @colab-bootstrap — no repository imports or downloads of private data.
import importlib.util, subprocess, sys, os, tempfile, types, math, json
from pathlib import Path
for package, pip_name in [('torch','torch'),('torch_geometric','torch-geometric'),('einops','einops'),('numpy','numpy')]:
    if importlib.util.find_spec(package) is None:
        subprocess.check_call([sys.executable,'-m','pip','install',pip_name])
import torch
torch.set_num_threads(1)
workspace=Path(tempfile.mkdtemp(prefix='b11-lab-'));os.chdir(workspace);sys.path.insert(0,str(workspace))
Path('evidence/b11').mkdir(parents=True)
print('Author runtime: torch 2.13.0+cpu; current runtime:',torch.__version__)
''')
    md('## Recall and worked trace\n\nA purchase row points to a customer and product. A query combines entity identity with a cutoff. Product features2,6 and purchase features1,3 yield route-specific purchase values3,9 under identity scalar transforms. Equal customer attention gives6. Equal attention over customer0 and all four other tokens gives2.4. These simplified updates are not benchmark predictions.\n\n'+png('trace')+'\n\n'+png('architecture'))
    md('## Inspectable source packet\n\nThe packet includes pinned original sources, licenses, full prior models/trainers, protocols and all11 prediction populations. Its digest protects transport; source and L143 evidence have additional inherited hash checks. The L146 prediction seal is newly made in B11. No historical checkpoint identity follows from this seal.')
    code('import base64, hashlib, io, zipfile\npacket=base64.b64decode('+repr(base64.b64encode(packet).decode())+')\nassert hashlib.sha256(packet).hexdigest()=='+repr(hashlib.sha256(packet).hexdigest())+'\nwith zipfile.ZipFile(io.BytesIO(packet)) as z:\n    assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in z.namelist())\n    z.extractall(workspace)\nprint("Verified source/evidence packet")')
    prompts={'eligible':'Return a boolean mask requiring finite event AND arrival times, each no later than cutoff. Reject unknown times. This fixture policy is stricter than merely checking event time.','explicit_attention':'Compute QKᵀ/√d, apply an allowed boolean mask (or additive float bias), softmax, then multiply V. Empty allowed sets return zero. Reject nonzero attention dropout and causal mode in this declared lab.','first_validation_min':'Return the zero-based first occurrence of minimum validation MAE. Reject empty or nonfinite histories. Do not accept test scores.'}
    for name,check in [('eligible','check_eligible'),('explicit_attention','check_attention'),('first_validation_min','check_selection')]:
        md('## TODO · '+name+'\n\n'+prompts[name])
        code(segments[name] if solution else segments[name].split('\n')[0]+f'\n    raise NotImplementedError("Complete {name}")')
        code(checks[check]+'\n'+check+'('+name+')\nprint("CHECK passed: '+name+'")')
    md('## PROVIDED · Wire your functions into the model and audit\n\nThe following module exports your live functions. The source blocks and saved-run audit import these exact objects; TODOs are not decorative exercises.')
    code('pkg=types.ModuleType("relkit");pkg.__path__=[];sys.modules["relkit"]=pkg\nlive=types.ModuleType("relkit.baselines_b11")\nfor name in ["eligible","explicit_attention","first_validation_min"]:\n    setattr(live,name,globals()[name])\nsys.modules["relkit.baselines_b11"]=live')
    for name in ['relgnn_visible','relgt_visible']:
        md('## PROVIDED · Visible '+('RelGNN composite arithmetic' if 'relgnn' in name else 'RelGT local/global blocks')+'\n\nMIT upstream license is in the extracted archive. B11 supplies encoded vectors and freezes global buffers in evaluation. It does not claim full row-encoder or EMA-update parity.')
        code('%%writefile sources/b11/'+name+'.py\n'+(P/'sources/b11'/f'{name}.py').read_text())
    code('''# IPython writefile normalizes blank-line whitespace. Restore only those bytes;
# refuse any substantive change to the displayed source before authenticating it.
with zipfile.ZipFile(io.BytesIO(packet)) as z:
    for name in ['relgnn_visible','relgt_visible']:
        p=Path('sources/b11')/(name+'.py');original=z.read(str(p))
        assert [s.rstrip() for s in p.read_bytes().splitlines()]==[s.rstrip() for s in original.splitlines()]
        p.write_bytes(original)
''')
    for name in ['_run_b11','_audit_b11']:
        md('## CHECK · '+('Three-seed source comparison and interventions' if 'run' in name else 'Complete saved-prediction replay')+'\n\nRead the visible checker before running it. No benchmark fits are dispatched.')
        code('%%writefile '+name+'.py\n'+(P/(name+'.py')).read_text())
        code('from '+name+' import '+('run' if 'run' in name else 'audit')+'\n'+('mechanism=run()' if 'run' in name else 'replay=audit()'))
    md('## Measured evidence\n\n'+results+'\n\n'+table+'\n\nThe current mechanism errors can vary slightly by runtime within1e-9. Saved MAEs must match. L143 retained nonfinite source gradients. L146 compares a typed-mean GNN with reduced corrected RelGT; it does not compare RelGNN with RelGT. Full RelGT search remains INCOMPLETE_TEMPORAL_AND_BUDGET_GATE. Fresh B11 benchmark fits and whole-paper reproduction are NOT_RUN.')
    code('assert replay["verified_predictions"]==13849 and replay["checkpoint_selections"]==11\nassert mechanism["status"]=="PASS"\nPath("b11-report.json").write_text(json.dumps({"mechanism":mechanism,"replay":replay},indent=2))\nprint("Author checks complete; learner defense still required")')
    md('## EXIT · Your written defense\n\nExplain the two paths, the global-centroid information boundary, and why these saved scores are not a matched RelGNN–RelGT comparison. Specify complete query keys, a validation-only selection rule and a shared information/resource contract. Name a falsifier for your preferred explanation. Revisit tomorrow, in7days and in30days. Ask the teaching agent for feedback.\n\n**Learner: PENDING_WRITTEN_DEFENSE. Live Colab: NOT_CHECKED.** Full reproduction contracts/operators remain in `sources/b11/archive/`; read them before planning any full run. Default execution here allocates no paid compute.')
    n=nb.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),language_info=dict(name='python')))
    dest=P/('solutions' if solution else '')/(S+'.ipynb');dest.parent.mkdir(exist_ok=True);nb.write(n,dest)
print('Built lesson, reference, two figures and two portable notebooks')
