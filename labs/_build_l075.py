"""Build the source-grounded API lesson and independent, portable notebook pair."""
import ast, base64, json, re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
from bs4 import BeautifulSoup
from _colab import bootstrap_cells
from _figures_l075 import build as figures
ROOT=Path(__file__).resolve().parent; REPO=ROOT.parent
SLUG='0075-pytorch-frame-row-encoder'; TITLE='PyTorch Frame: the row encoder'
SOURCE=(ROOT/'relkit/frame_l075.py').read_text(); TREE=ast.parse(SOURCE)
NODES={n.name:ast.get_source_segment(SOURCE,n) for n in TREE.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
IMPORTS='\n'.join(ast.get_source_segment(SOURCE,n) for n in TREE.body if isinstance(n,(ast.Import,ast.ImportFrom)))
CAPTIONS={'types':'Synthetic five-column fixture: text width4 and stored-vector width3 share the embedding parent type, then each receives a width8 token. B denotes rows.',
'scope':'Exact fit-scope intervention: missing values are ignored for this mean. Holding training values fixed, adding held-out1000 changes the all-row mean from20 to265.',
'numeric':'Illustrative parameters, not the measured fixture standard deviation. Trace normalization, multiplication and bias. Mean imputation makes missing tokens equal the bias.',
'architecture':'Complete local teaching composition. Column mixing occurs inside the flattened MLP. No cross-row message passing or benchmark training is pictured.'}
GOALS={
'fit_materializer':'Return (training Dataset, query TensorFrame). Configure the supplied text_features adapter through TextEmbedderConfig. Fit all column state on training rows and reuse that fitted converter. Do not concatenate or materialize query rows independently.',
'encoder_config':'Return a dictionary keyed by the four parent semantic types needed by the five-column fixture. Choose LinearEncoder with numeric mean imputation, default EmbeddingEncoder, TimestampEncoder and LinearEmbeddingEncoder. Do not use a child stype as a key.',
'numeric_tokens':'Implement the mean-imputed normalized affine map. raw is [B,C]; mean and scale are [C]; weight and bias are [C,d]. The supplied scale already contains the release epsilon. Preserve gradients and return [B,C,d].'}
CHECKS={
'fit_materializer':'''from torch_frame.data.stats import StatType
train, query = fixture()
ds, held = fit_materializer(train, query, schema())
assert ds.col_stats['amount'][StatType.MEAN] == 20, 'Query rows leaked into fitting'
changed = query.copy(); changed['amount'] = [1e8, -1e8]
again, _ = fit_materializer(train, changed, schema())
assert again.col_stats['amount'][StatType.MEAN] == 20
assert held.feat_dict[stype.categorical][:,0].tolist() == [-1, -1]
assert ds.tensor_frame.num_cols == 5
print('Training-only fit and unknown-category checks passed')''',
'encoder_config':'''torch.manual_seed(75)
model = RowEncoder(ds, width=8, row_width=6)
tokens, column_names = model.columns(held)
assert tokens.shape == (2,5,8), 'Need one width8 token per feature'
assert torch.isfinite(tokens).all()
assert torch.equal(tokens[:,column_names.index('region')],torch.zeros(2,8))
assert model(held).shape == (2,6)
print('Column order:', column_names)
print('Tokens:', tuple(tokens.shape), 'Rows:', tuple(model(held).shape))''',
'numeric_tokens':'''manual = numeric_tokens(torch.tensor([[30.]]),torch.tensor([20.]),torch.tensor([10.]),torch.tensor([[2.,-1.]]),torch.tensor([[.5,.5]]))
assert torch.equal(manual,torch.tensor([[[2.5,-.5]]])), 'Trace the two coordinates by hand'
enc = model.columns.encoder_dict['numerical']
raw = ds.tensor_frame.feat_dict[stype.numerical]
actual = numeric_tokens(raw, enc.mean, enc.std, enc.weight, enc.bias)
assert torch.allclose(actual, enc(raw), atol=1e-7), 'Inspect missing policy, scale and shape'
actual.sum().backward()
assert torch.isfinite(enc.weight.grad).all(), 'Missing inputs must not poison weight gradients'
model.zero_grad()
print('Numeric primitive matches configured release')'''}

def evidence():
 r=json.loads((ROOT/'_verify_l075_results.json').read_text()); real=r['real_table']
 return f'''**Author-reference evidence, measured locally.** These values are saved audit results, not outputs from your current notebook kernel.

| Check | Observed | What it establishes |
|---|---|---|
| Five-type query tokens | `{r['tokens']}` | One width8 token per column |
| Five-type query rows | `{r['rows']}` | Readout returns one width6 vector per row |
| Fitted numeric standard deviation | {r['train_std']:.6f} | Training population SD; encoder adds1e-6 |
| Numeric primitive maximum absolute error | {r['numeric_parity_max_error']:.1e} | Forward parity on this fixture |
| Training mean: train-only / all rows | 20 / 265 | Split-column metadata does not enforce fit scope |
| Unknown and missing region IDs | −1 / −1 | Both enter default category padding |
| Real credit_g tokens / row vectors | `{real['token_shape']}` / `{real['row_shape']}` | Actual mixed-schema encoding, no accuracy claim |
| Gradient flow, row locality, column reorder | PASS | Tested configured composition behaves as described |

**Evidence limits:** no trained benchmark, no pretrained language model, no relational message passing. Browser and notebook execution have their own delivery reports. Paper-result parity is INCOMPARABLE.
'''

def manuscript(notebook=False):
 t=(REPO/'lessons/content'/f'{SLUG}.md').read_text().replace('<!--results-->',evidence())
 for name,caption in CAPTIONS.items():
  p=ROOT/'figures/l075'/f'{name}.png';src='data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode() if notebook else f'../labs/figures/l075/{name}.png'
  t=t.replace('<!--figure:'+name+'-->',f'<figure><div class="figure-scroll" tabindex="0" role="region" aria-label="Scrollable {name} diagram"><img src="{src}" alt="{caption}"></div><figcaption>{caption} Scroll the diagram horizontally on narrow screens.</figcaption></figure>')
 if notebook:
  t=re.sub(r'<div id="[^"]+"></div>','',t)
  t=t.replace('(0074-', '(../lessons/0074-')
 return t

def notebook(solution=False):
 cells=[]
 def md(s):cells.append(nbf.v4.new_markdown_cell(s))
 def code(s):cells.append(nbf.v4.new_code_cell(s))
 md('# Lab075 · '+TITLE+'\n\n**Skill:** use the real PyTorch Frame API to produce typed column tokens and row vectors. This is the tool/API exception: a framework composition lab, not a paper benchmark reproduction. TierC five-type fixture isolates mechanisms; TierA credit_g uses actual numeric/categorical columns. No accuracy claim. Complete the TODOs, pass CHECKs and export row vectors with their schema.\n\n[Lesson](../lessons/'+SLUG+'.html) · [Contract](../labs/l075-reproduction.md)')
 cells.extend(nbf.from_dict(c) for c in bootstrap_cells())
 code("# PROVIDED — version is part of the behavior contract.\nimport importlib.metadata\nassert importlib.metadata.version('pytorch-frame') == '0.3.0', 'Install pytorch-frame==0.3.0 in this kernel'\n")
 for part in re.split(r'(?=^## )',manuscript(True),flags=re.M):
  if part.strip():md(part)
 md('<a id="lab-exercises"></a>\n## Implement and trace\n\nRun from `labs/` locally. The shared Colab bootstrap configures this directory after publication. No cloud model download is needed; the text adapter is deliberately transparent. Provided package encoders are the API being learned. The readout and all composition code are visible below.')
 code(IMPORTS+'\ntorch.set_num_threads(1)\ntorch.manual_seed(75)')
 md('### PROVIDED · exact data and fixed text adapter\n\nFour training rows, two query rows. The fixed vocabulary has never fitted query data. Predict which query string produces no vocabulary matches.')
 for name in ['fixture','schema','text_features']:code(NODES[name])
 for name in GOALS:
  md('### TODO · '+name+'\n\n'+GOALS[name]+'\n\nWrite your predicted CHECK output before running it. Use the API documentation for signatures; keep the fitting boundary explicit.')
  signature=NODES[name].split('\n')[0]
  code(NODES[name] if solution else signature+'\n    raise NotImplementedError("Implement '+name+'")')
  if name=='encoder_config':
   md('### PROVIDED · complete local row-encoder architecture\n\nThis class calls your live encoder_config. The first hidden layer reads all column coordinates, so it performs within-row column mixing. It does not exchange rows.')
   code(NODES['RowEncoder'])
  code('# CHECK\n'+CHECKS[name])
 md('### CHECK · identities, gradients and a task head\n\nPredict whether a second query row can change the first row vector. Then permute input DataFrame columns. These interventions test different kinds of order. The two-class head below checks differentiability only; the two synthetic labels are not a real task.')
 code("_, permuted = fit_materializer(train, query[list(reversed(query.columns))], schema())\nassert torch.allclose(model(held), model(permuted))\nassert torch.allclose(model(held)[:1], model(held[:1]), atol=1e-6)\nhead = nn.Linear(6, 2)\nloss = nn.functional.cross_entropy(head(model(held)), torch.tensor([0,1]))\nloss.backward()\nassert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())\nprint('Row locality, schema alignment and finite end-to-end gradients passed')")
 md('### PROVIDED · real-data encoding and artifact export\n\nThe labels from OpenML31 are deliberately unused. Seed75 selects row IDs without inspecting the labels. Your fit_materializer and encoder_config remain bound in this notebook; the reconstruction TODO is a checked primitive, not a replacement silently installed inside the library.')
 code(NODES['real_table_demo'])
 code("assert RowEncoder.__init__.__globals__['encoder_config'] is encoder_config\nassert real_table_demo.__globals__['fit_materializer'] is fit_materializer\nreal = real_table_demo(export=True)\nassert real['row_shape'] == [32,16]\nassert real['token_shape'] == [32,20,8]\nprint('Real-data token shape:', real['token_shape'])\nprint('Real-data row shape:', real['row_shape'])\nprint('Saved l075-row-encoder.pt and l075-schema.json')\nPath('l075-student-results.json').write_text(json.dumps(real,indent=2))")
 md('## EXIT TICKET\n\nSubmit both artifact files and the five answers from the lesson EXIT. Include the printed shapes, actual token order and package version. Explain what would be lost by retaining only row vectors without row IDs or fitted schema. Do not infer accuracy from the magnitude of a random vector.\n\n## Paper-results boundary\n\nThe named local target is the release-specific Figure1-style API composition, not a published benchmark metric. Paper §5.1/5.2/5.3 benchmark experiments are NOT_RUN; local outputs are INCOMPARABLE to their accuracy tables. This tool/API lesson has no misleading paper preset. The [contract](../labs/l075-reproduction.md) lists the evidence and missing experiment components.\n\nTomorrow: reconstruct the fit/conversion/encoding/readout sequence from memory. Optional extension: replace the MLP readout while preserving the [B,D] interface and repeat locality/gradient checks; that is an extension, not paper-result reproduction.')
 nb=nbf.v4.new_notebook(cells=cells);nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}}
 return nb

def build():
 figures()
 head='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson075 — '+TITLE+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/contrastive-views.css"></head><body><article>'
 nav=f'<nav><a href="../index.html">Course</a> · <a href="0074-carte-cross-table-transfer.html">Previous lesson</a></nav><header><p>Year2 · Quarter4 · Lesson075</p><h1>{TITLE}</h1><p>Choose types → fit the converter → encode columns → read out a row.</p></header><aside class="lab-access"><nav><a href="../labs/html/{SLUG}.html">Read lab</a><a href="../labs/{SLUG}.ipynb" download>Download notebook</a><a href="../labs/html/{SLUG}.html#lab-exercises">Exercises</a><a href="../reference/{SLUG}.html">Reference</a><a href="../labs/_verify_l075_results.json">Measured evidence</a><a href="../labs/l075-reproduction.md">Reproduction contract</a></nav><p>Local package; live Colab and deployment NOT_CHECKED.</p></aside>'
 body=render(manuscript()).replace('<table>','<div class="result-scroll"><table>').replace('</table>','</table></div>')
 scripts=''.join(f'<script src="../assets/{n}.js"></script>' for n in ['retrieval-pool','retrieval-bank','predict','teachback','frame-encoder-viz','l075-lesson'])
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
 (ROOT/'html'/f'{SLUG}.html').write_text(str(soup)+'\n')
 ref='''# PyTorch Frame: row-encoder audit card

**Type by meaning.** A numeric dtype does not make a column continuous. IDs, labels and timestamps require distinct decisions. Pin the release: this lesson uses pytorch-frame0.3.0.

**Fit before apply.** Dataset(training).materialize() fits state. Reuse its convert_to_tensor_frame callable for query rows. Materializing all rows with a split column still fits all-row statistics in this release.

**Pack.** TensorFrame groups columns by parent type. text_embedded joins embedding; varying original vector widths use MultiEmbeddingTensor. Read the column names returned by the feature encoder.

**Encode.** Numeric mean-imputation then ((x−mean)/scale)w+b; default categorical negative IDs select zero padding; timestamps use calendar features and default median imputation; embedding columns receive separate learned projections. All become [B,C,d].

**Read out.** The local MLP flattens [B,C,d] to [B,Cd], mixes columns in a hidden layer, and returns [B,D]. It is schema-dependent and row-local. A GNN additionally needs IDs, edges and temporal eligibility.

**Diagnose.** Changing query values must not change fitted statistics. Permuting DataFrame column order must not reassign feature roles. Batched versus singleton row output should agree for this encoder. Check finite backward gradients, not only finite forward outputs.

**Save.** Row vectors + query row IDs + schema + column order + encoder weights + fitted statistics + versions. Random vectors verify the interface; they do not demonstrate useful predictive learning.

**Evidence.** Primitive forward parity and API checks PASS; paper benchmark NOT_RUN; paper metric comparison INCOMPARABLE; live Colab/deployment NOT_CHECKED.

[Primary paper](https://arxiv.org/html/2404.00776v2) · [Pinned source](https://github.com/pyg-team/pytorch-frame/tree/d998aae368db6a4e36139ccc56bd54579a70874b)
'''
 (REPO/'reference'/f'{SLUG}.html').write_text(head+render(ref+f'\n[Full lesson](../lessons/{SLUG}.html) · [Lab](../labs/html/{SLUG}.html)')+'</article></body></html>')
 print('Built L075 lesson, notebook pair, preview, reference and four portable figures')
if __name__=='__main__':build()
