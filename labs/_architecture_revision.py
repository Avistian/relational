"""Authored model overviews with a visible operator, shared across HTML/notebooks.

The native HTML reflows on phones. Browser-exported PNGs are portable notebook
snapshots of the same panel, rather than a separately maintained drawing.
Numbers below are pedagogical fixtures, never measured model predictions.
"""
import base64,hashlib,html,math,re
from pathlib import Path
import nbformat
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[1]
E=html.escape


def matrix(columns,rows,corner='stage ↓'):
    out=f'<table class="aa-matrix"><thead><tr><th>{E(corner)}</th>'+''.join(f'<th>{E(c)}</th>' for c in columns)+'</tr></thead><tbody>'
    for label,values in rows:
        out+=f'<tr><th>{E(label)}</th>'
        for v in values:
            cls='aa-blocked' if v in ('×','held out','hidden') else 'aa-query' if v=='?' else ''
            out+=f'<td class="{cls}">{E(str(v))}</td>'
        out+='</tr>'
    return out+'</tbody></table>'


def vec(values):return '<span class="aa-vector">'+''.join(f'<b>{E(str(v))}</b>' for v in values)+'</span>'
def op(*parts):return '<div class="aa-op">'+'<span class="aa-arrow" aria-label="then">→</span>'.join(parts)+'</div>'
def eq(text):return f'<div class="aa-equation">{E(text)}</div>'
def pair(a,x,b,y):return f'<div class="aa-pair"><div class="aa-tile"><strong>{E(a)}</strong><b>{E(str(x))}</b></div><div class="aa-tile aa-change"><strong>{E(b)}</strong><b>{E(str(y))}</b></div></div>'
def bars(values,maximum):return ''.join(f'<div class="aa-bar-row"><span>{E(k)}</span><div class="aa-track"><div class="aa-bar" style="width:{100*v/maximum:.4f}%"></div></div><b>{v:.3g}</b></div>' for k,v in values)


def topology(key,title,nodes,edges,height=250):
    """Small authored routing graph; edges are behind labels, with explicit ports."""
    out=f'<svg class="aa-topology" viewBox="0 0 320 {height}" role="img" aria-labelledby="topology-{key}"><title id="topology-{key}">{E(title)}</title><defs><marker id="arrow-{key}" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0L6 3L0 6Z" fill="#087e83"/></marker></defs>'
    for points,dashed in edges:
        out+=f'<polyline points="{points}" fill="none" stroke="#087e83" stroke-width="1.7" '+('stroke-dasharray="4 3" ' if dashed else '')+f'marker-end="url(#arrow-{key})"/>'
    for x,y,w,label,role in nodes:
        fill,stroke=('#fff1df','#dfbd91') if role=='query' else ('#eaf5f2','#a9cdc8') if role=='memory' else ('#f3f6f8','#b5c8d1')
        out+=f'<g data-architecture-node="true"><rect x="{x}" y="{y}" width="{w}" height="38" rx="5" fill="{fill}" stroke="{stroke}"/><text x="{x+w/2}" y="{y+24}" text-anchor="middle" fill="#203d4b" font-family="system-ui,sans-serif" font-size="14" font-weight="600">{E(label)}</text></g>'
    return out+'</svg>'


PANELS={}
def panel(n,key,title,anchor,question,route,focus,graphic,caption,answer,scope):
    PANELS.setdefault(n,[]).append(dict(key=key,title=title,anchor=anchor,question=question,route=route,focus=focus,graphic=graphic,caption=caption,answer=answer,scope=scope))


panel(47,'saint','SAINT · two axes of attention','Model architecture',
      'Where can another row first change the query prediction?',
      [('Encode the table','Numeric networks, category lookups and one CLS token per row.','B × T × d'),
       ('Mix features','Attention + feed-forward inside each row; then pack whole rows.','B × T × d → 1 × B × Td'),
       ('Mix rows · repeat stage','Attention + feed-forward across the batch; unpack. Repeat L stages.','1 × B × Td → B × T × d'),
       ('Read CLS','Final CLS → Linear → ReLU → Linear → softmax.','B × d → B × 1000 → B × 2')],
      'One query, two companion values',
      matrix(['self','companion'],[('scores',['0','ln 3']),('weights',['¼','¾']),('values',['2','6'])])+eq('¼ × 2 + ¾ × 6 = 5')+pair('Baseline value 6','output 5','Change value to 10','output 8'),
      'Illustrative projected head: hold query, keys and weights fixed; change only the companion value. T includes CLS.',
      'The row-attention stage opens the cross-row path. Packing changes which axis attention treats as a sequence.',
      'Supervised colrow lab: L=1, d=8. Released residual is u+f(u), u=LayerNorm(x). Pretraining augmentation and auxiliary heads are separate, absent from this inference path.')

panel(48,'dcnv2','DCNv2 · keep the original input alive','Model architecture',
      'Which vector anchors every cross update, and what does rank one actually compute?',
      [('Embed one row','Concatenate numeric features and categorical embeddings. Save x₀.','B × d'),
       ('Cross repeatedly','xₗ₊₁ = xₗ + x₀ ⊙ (Wₗxₗ + bₗ). The anchor stays x₀.','B × d → B × d'),
       ('Route the deep branch','Parallel: MLP(x₀). Stacked: MLP(xL). Parallel concatenates both branches.','cross + deep → head input'),
       ('Predict','Linear head produces a logit; sigmoid gives class probability.','B × 1')],
      'Inside a rank-one cross layer',
      '<p class="aa-small">Column-vector fixture: x₀=xₗ=(2,3), V=(1,2), U=(1,3), bias=0.</p>'+op(vec([2,3]),'<span><b>Vᵀx = 8</b><br><small>2 → 1</small></span>','<span><small>U · 8 &nbsp; (1 → 2)</small><br>'+vec([8,24])+'</span>')+eq('x₀ ⊙ U(Vᵀxₗ) = (16, 72)')+eq('xₗ + update = (18, 75)'),
      'The bottleneck compresses d→r, then expands r→d. The original input multiplies the expanded update coordinate by coordinate.',
      'Low rank constrains the mixed directions; it does not remove the anchor or residual. Parallel and stacked deep branches read different inputs.',
      'Dense and low-rank DCNv2 cross paths. The nonlinear mixture extension adds gates and expert nonlinearities; its function is not the same polynomial cross stack.')

panel(49,'excelformer','ExcelFormer · direct the feature graph','Model architecture — ExcelFormer',
      'Can the highest-ranked feature token read the weakest feature token?',
      [('Order and encode','Rank columns by training-only importance; apply per-feature gated tokenizers.','B × F → B × F × d'),
       ('Restrict feature attention','Mask senders with lower importance than the receiving token.','per head: B × F × F'),
       ('Transform and repeat','Residual attention and gated feed-forward stages update feature states.','B × F × d'),
       ('Pool and predict','Learned feature-axis pooling → normalization → activation → head.','B × d → class logits')],
      'The mask is a directed graph',
      matrix(['strong','middle','weak'],[('strong',['1','×','×']),('middle',['½','½','×']),('weak',['⅓','⅓','⅓'])],'reads ↓')+op(vec([2,4,9]),vec([2,3,5])),
      'Uniform allowed scores, one value coordinate. × means forbidden. Rows receive; columns send. Outputs are weighted sums, not mask counts.',
      'The strong token cannot read the weak token through this attention mask. The final pooled prediction can still use the weak token directly.',
      'Local ExcelFormer mechanism. Importance is fitted on training labels. Feat-Mix changes training examples/targets and is absent from this prediction path.')
panel(49,'trompt','Trompt · values refine prompt routing','Model architecture — Trompt',
      'Why can cell 2 route two rows differently when cell 1 cannot?',
      [('Encode each original row','Each cell: per-column affine → ReLU → LayerNorm. Separate learned column/prompt identities also use LayerNorm.','x: B × C → values: B × C × d'),
       ('Fuse and route','Start O⁰=0; concatenate normalized prompts with Oprev, map 2d→d, add both residuals; score column identities.','softmax over C: B × P × C'),
       ('Expand the value path','Learned scalar→P scaling → ReLU → GroupNorm (2 groups) + broadcast original values.','B × C × d → B × P × C × d'),
       ('Gather and pass state','Multiply routing weights by expanded values and sum C; feed Oℓ to the next distinct cell, which rereads x.','Oℓ: B × P × d; repeat L cells'),
       ('Shared head for every cell','Dense prompt scores → softmax over P → weighted sum → Dense → ReLU → LayerNorm → Dense.','B × P × d → B × d → B × T'),
       ('Train and infer','Training sums L cross-entropies. Local inference averages L logits, then class softmax.','cell logits B × L × T → probabilities B × T')],
      'State makes later routing sample-specific',
      topology('trompt-state','Original row enters both cells; prior state connects them; prediction head is shared',
          [(15,8,90,'row x','query'),(150,8,150,'O⁰ = zero','memory'),(120,78,180,'cell 1 → O¹','memory'),
           (120,160,180,'cell 2 → O²','memory'),(45,245,230,'shared head → logits','output')],
          [('105,27 110,27 110,97 120,97',False),('60,46 60,179 120,179',False),('225,46 225,78',False),
           ('210,116 210,160',False),('300,97 314,97 314,264 275,264',True),('210,198 210,245',True)],290)
      +matrix(['col 1','col 2','col 3'],[('one prompt M',['0.2','0.3','0.5']),('value coordinate',['1','0','4'])])+eq('sum over C: 0.2×1 + 0.3×0 + 0.5×4 = 2.2'),
      'Solid paths carry x or recurrent state; dashed paths invoke the same head. M¹ is shared across rows, but O¹ differs when x differs. Fixture arithmetic is synthetic.',
      'Cell 1 sees zero previous state and shared identities, so its M¹ cannot depend on x. Its value aggregation O¹ does depend on x; cell 2 can use O¹ to change M².',
      'Complete local numeric mirror: d=16, P=8, L=2, T=2. Zero initial state follows paper §5.1; expansion/group count follow pinned independent PyTorch Frame components. Original benchmark defaults d=P=128, L=6 are not the local training recipe.')

panel(50,'ftt','FT-Transformer · numeric checkpoint','Model architecture',
      'Where do row values enter, and how does a residual carry information around each update?',
      [('Fit and transform','Fit median, mean and SD on train; apply to every partition. Numeric features only.','B × C → B × C'),
       ('Tokenize and prepend CLS','Each scalar xⱼ becomes xⱼwⱼ+bⱼ. A learned CLS is prepended.','B × C → B × T × 32; T=C+1'),
       ('Attention residual','Four heads, eight coordinates per head. Skip attention LayerNorm in block 1; normalize in block 2.','u = x + Attention(N(x))'),
       ('Gated residual; repeat twice','LayerNorm → Linear 32→84 → ReGLU → dropout → Linear 42→32. Two separate learned blocks.','v = u + FFN(LayerNorm(u))'),
       ('Read final CLS','LayerNorm → ReLU → Linear 32→1; sigmoid for probability. Train with binary cross-entropy on logits.','B × T × 32 → B × 32 → B')],
      'One residual branch, then a numerical token',
      topology('ftt-residual','The input goes through the update and directly to the addition',
          [(75,8,165,'input x','query'),(75,77,165,'Attention(N(x))','memory'),
           (75,155,165,'ADD → u','output')],
          [('157,46 157,77',False),('157,115 157,155',False),('75,27 30,27 30,174 75,174',True)],207)
      +matrix(['xⱼ','wⱼ','bⱼ'],[('fixture',['2','(.5,−1)','(.1,.3)'])])
      +eq('2 × (.5, −1) + (.1, .3) = (1.1, −1.7)'),
      'Dashed path is the identity skip, not dropout. The FFN has its own identical residual pattern. The tokenizer fixture uses d=2 for arithmetic only.',
      'Both residuals preserve a direct path. Attention mixes tokens in one row; no other row is a sender. Only the final CLS reaches the head.',
      'Local numeric variant: d=32, L=2, H=4, ReGLU h=42; attention/FFN dropout 0.1 during training and off in eval, residual dropout 0. The paper fixes H=8. No categorical lookup, compression or pretraining. Copied-weight parity is scoped to evaluation; the benchmark recipe differs.')

weights=[1/(1+math.exp(-1)),1/(1+math.exp(1))]
panel(52,'tabr','TabR-S · search, correct, combine','Model architecture',
      'How can the same two neighbors contribute more than an average of their labels?',
      [('Encode query and memory','Shared numeric encoder makes row representations; learned K makes keys.','query B × d; memory N × d'),
       ('Select context','Exact squared distances; exclude own record ID; choose m neighbors.','B × N → B × m IDs'),
       ('Correct neighbor values','E(yᵢ)+T(k−kᵢ), weighted by softmax of negative squared distances.','B × m × d → B × d'),
       ('Add and predict','Query representation + context → residual predictor → head.','B × d → B × 1')],
      'Selection is not aggregation',
      bars([('a: dist²',1),('b: dist²',4),('c: dist²',2)],4)+matrix(['a selected','c selected'],[('weight',[f'{weights[0]:.3f}',f'{weights[1]:.3f}']),('E(y)+T',['2+1','6−2'])])+eq(f'context ≈ {weights[0]*3+weights[1]*4:.3f}'),
      'Query key (1,1); memory keys a=(1,0), b=(3,1), c=(0,0). Illustrative scalar values; displayed weights rounded.',
      'The discrete search chooses a and c. Their learned relative-key corrections change what the continuous weighted sum carries.',
      'Numeric TabR-S local exact-search path. Memory labels must be eligible; equal features do not imply equal record identity. Source forward parity does not establish training or approximate-search parity.')

panel(53,'realmlp','RealMLP-TD-S · the recipe is in the path','Model architecture',
      'Which transformations control input scale before the first hidden activation?',
      [('Fit robust preprocessing','Training median and IQR; range fallback; smooth clipping.','B × F'),
       ('Learn feature scale','Multiply each coordinate by its learned scale; width is unchanged.','B × F → B × F'),
       ('Repeat hidden layers','Three NTP linear → activation stages; SELU classification, Mish regression.','F → d → d → d; local d=64, paper d=256'),
       ('Predict and select','Zero-initialized output head; validation selects a scheduled training checkpoint.','B × output width')],
      'Smooth clipping preserves order',
      matrix(['z=0','z=3','z=6'],[('c(z)',[0,f'{3/math.sqrt(2):.3f}',f'{6/math.sqrt(5):.3f}'])])+eq('c(z) = z / √(1 + (z/3)²)')+eq('linear(x) = xW / √dᵢₙ + b')+pair('Input scale','learned per feature','Large magnitudes','approach ±3'),
      'Numeric transform values are computed from the displayed formula. Constant and zero-IQR features have explicit fallback policies.',
      'The architecture alone does not specify training: parameter groups, initialization and coslog4 scheduling change how this path is learned.',
      'Local numeric TD-S variant. Robust preprocessing is fitted only on training rows. A reduced-width local run is distinct from the published recipe-size experiment.')

panel(54,'tabm','TabM · separate members, shared matrices','Model architecture',
      'How do identical input rows take different paths through the same W?',
      [('Broadcast the row','Train-only imputation/z-score, then k views; R starts ±1 and learns real values.','B × F → B × k × F'),
       ('Share expensive weights','Mini: first R only; N shared W and shared biases, ReLU, training-only dropout.','B × k × hidden width'),
       ('Use member heads','Each member has an independent output projection.','B × k × output width'),
       ('Aggregate probabilities','Classification: mean probabilities. Regression: mean scalars. Train: mean member losses.','B × output width')],
      'Two members, one shared matrix',
      '<p class="aa-small">x=(2,3), W=[[1,2],[3,4]], S=1, bias=0.</p>'+matrix(['member A','member B'],[('R',['(1,1)','(1,−1)']),('x ⊙ R',['(2,3)','(2,−3)']),('× shared W',['(11,16)','(−7,−8)'])])+pair('p₁=0.9, p₂=0.6','mean p = 0.750','Mean logits first','p ≈ 0.786'),
      'Pre-activation adapter fixture. The probability example is a separate head-output fixture showing why operation order matters.',
      'Adapters create different effective functions while retaining one W. Softmax and member averaging do not commute.',
      'Corrected v2 numeric mini: only first R; shared backbone weights/biases; separate heads. Full TabM adds R,S and member biases in each block. Input labels appear only in the training loss. Shared batches pictured; the closer trainer also supports distinct rows per member.')

panel(57,'stack','OOF stack · predictions become features','Model architecture',
      'Which fitted base model is allowed to produce the training feature for row i?',
      [('Partition outer training rows','Assign folds and preserve stable row IDs; keep outer test untouched.','N rows → K folds'),
       ('Predict held-out folds','For each family, fit on other folds and predict the excluded fold.','fold outputs → aligned N × M'),
       ('Train the combiner','Use OOF probabilities Z and training targets y; select under the declared validation rule.','Z: N × M; y: N'),
       ('Serve the stack','Declared refit/fold-model policy → aligned base probabilities → frozen combiner.','new rows × M → probabilities')],
      'Every diagonal is held out',
      matrix(['fit B+C','fit A+C','fit A+B'],[('predict A',['OOF','×','×']),('predict B',['×','OOF','×']),('predict C',['×','×','OOF'])])+eq('Z[i, :] = base predictions made without yᵢ'),
      'M counts model probability features in this binary illustration. The diagonal maps each row fold to its eligible training complement.',
      'The combiner may train on yᵢ. The base fit producing Zᵢ must not have trained on that target.',
      'Cross-family stacking procedure, not a new base network. Group/time structure must also be respected by folds. Class order, row IDs and serving refit policy are part of the saved artifact.')

panel(61,'gppfn','Row PFN · infer from a labeled set','Model architecture',
      'Can a query read itself without reading its hidden target?',
      [('Sample and split a GP task','One joint function draw; reveal n context labels, hide m query labels.','x: B × N × F; yc: B × n'),
       ('Encode whole rows','Context: Ex(x)+Ey(y). Query: Ex(x) only. No positional encoding.','B × N × d; lab d=64'),
       ('Masked attention residual','All context columns plus identity. H=4 heads; concatenate → output projection.','u = LayerNorm(h + Attention(h))'),
       ('GELU residual; repeat three blocks','Linear 64→128 → GELU → Linear 128→64; independent postnorm blocks.','h′ = LayerNorm(u + FFN(u))'),
       ('Predict query densities','Query states → Linear 64→128 → GELU → Linear 128→64; softmax masses.','B × m × 64 → full-support Riemann density'),
       ('Train / infer','Pretraining query NLL updates weights. A new task only changes context inputs.','no query labels enter forward(x,yc)')],
      'One head: equal allowed scores',
      matrix(['c1','c2','q1','q2'],[('c1 reads',['½','½','×','×']),('c2 reads',['½','½','×','×']),('q1 reads',['⅓','⅓','⅓','×']),('q2 reads',['⅓','⅓','×','⅓']),('values',[2,6,10,90])],'receives ↓')+eq('q1 output = (2 + 6 + 10) / 3 = 6')+pair('q2 value = 90','q1 remains 6','q2 value = 900','q1 remains 6'),
      'Synthetic one-coordinate attention fixture, not trained weights. Context outputs equal 4; q2 initially equals 98/3. Every block preserves this information boundary.',
      'Self attention uses the query feature token only. Context tokens cannot read queries; extra query rows cannot leak through a later context state.',
      'Original released row PFN mask includes query self edges. Postnorm GELU blocks, zero initial attention/FFN output projections, dropout 0. Local width 64/layers 3/bins 64 differs from original large GP training. Full-support tails preserve density outside sampled borders.')

mask=matrix(['ctx 1','ctx 2','query 1','query 2'],[(r,['✓','✓','×','×']) for r in ['ctx 1','ctx 2','query 1','query 2']],'reads ↓')
panel(62,'rowpfn','TabPFN v1 · the actual released predictor','Model architecture',
      'Where do labels enter, and how does a query keep its own information?',
      [('Prepare each view','Context-only moments → constants/power/outliers → scale 100/k and zero pad.','B × (C+Q) × 100'),
       ('Encode whole rows','Linear feature 100→512; add Linear label 1→512 only to context. No unknown-label or position embedding.','context: Ex(x)+Ey(y); query: Ex(x)'),
       ('Read context memory','Four shared-projection heads; all receivers read only context keys/values. Query self edges removed.','head width 128; scores (C+Q) × C'),
       ('Postnorm residuals · repeat 12','Attention → add → LayerNorm; Linear 512→1024 → GELU → Linear 1024→512 → add → LayerNorm.','B × (C+Q) × 512'),
       ('Read query states','Linear 512→1024 → GELU → Linear 1024→10; keep first K class logits.','B × Q × K'),
       ('Align and combine views','Undo class rotations → mean logits → divide by .8 → softmax. Frozen weights at inference.','V × Q × K → Q × K')],
      'One connected query path',
      topology('tabpfn-v1-encode','Context and query rows are encoded separately before the repeated blocks',
       [(5,8,145,'context x,y','memory'),(170,8,145,'query x','query'),(5,78,145,'Ex(x) + Ey(y)','memory'),(170,78,145,'Ex(x) only','query')],
       [('77,46 77,78',False),('242,46 242,78',False)],128)
      +'<div style="border:2px solid #087e83;border-radius:8px;padding:10px"><strong>↓ Repeat this block 12 times ↓</strong><p>Current context and query states enter each block. Context states update at every layer.</p>'
      +topology('tabpfn-v1-block','Shared projections; context self attention and query cross attention; both residual paths then feed forward',
       [(5,8,145,'context h → U,K,V','memory'),(170,8,145,'query h → U','query'),(5,90,145,'context ← context','memory'),(170,90,145,'query ← context','query'),(45,174,230,'add own h → LayerNorm','output'),(45,246,230,'FFN → add → LayerNorm','output')],
       [('77,46 77,90',False),('242,46 242,90',False),('105,46 105,65 220,65 220,90',False),('77,128 77,152 105,152 105,174',False),('242,128 242,152 215,152 215,174',False),('160,212 160,246',False)],290)
      +'<p>Feed both updated row sets into the next block. After block 12, read only query states.</p></div>'
      +topology('tabpfn-v1-head','Query states enter a classification head; class mapping is reversed before logit averaging and softmax',
       [(20,8,280,'query head 512→1024→10','output'),(20,78,280,'class align → mean logits / .8','output'),(20,148,280,'softmax → probabilities','output')],
       [('160,46 160,78',False),('160,116 160,148',False)],194)
      +matrix(['c1','c2','q1','q2'],[('q1 weights',['.269','.731','×','×']),('value a',[1,0,8,90]),('value b',[0,2,6,70])],'receives ↓')+eq('q1 attention = (.269, 1.462)'),
      'Synthetic width-2 head fixture: scores(0,1), softmax(.269,.731). Query values are blocked. Residual h preserves the receiver; the final head is not this illustrative two-coordinate vector.',
      'Context labels enter Ey before all 12 blocks. Another query supplies no key/value; each query keeps its own projected features and residual state.',
      'Full released historical v1: d512, H4, FFN1024, L12, output10, dropout0, postnorm, GELU, numeric wrapper and actual copied pretrained weights. Prior fitting used synthetic held-out labels; this lab measures frozen inference, not new training.')

# Lesson 064 owns its complete historical-v2 architecture in _build_l064.

panel(65,'crossfit','Query embeddings · hide the right labels','Model architecture',
      'How can a supervised head train on labels that the encoder was forbidden to see?',
      [('Partition training rows','Preserve row IDs and assign K folds.','N rows → fold IDs'),
       ('Encode each held-out fold','Frozen pretrained encoder gets other folds as labeled context; held-out rows are unlabeled queries.','fold queries → fold × D'),
       ('Restore order and train head','Scatter embeddings to original rows; fit scaler and logistic head on (Z,y).','Z: N × D'),
       ('Evaluate with frozen head','Average test embeddings over declared fold contexts; test labels score only.','test × D → probabilities')],
      'One row has two distinct label roles',
      matrix(['encoder context','encoder query','head target'],[('row in A',['absent','hidden','yᵢ allowed']),('rows B+C',['labels','not queried','their own OOF y'])])+eq('zᵢ = encoder(D outside fold A, xᵢ)')+eq('head training pair = (zᵢ, yᵢ)'),
      'The row’s target is hidden while constructing its representation, then used as ordinary supervised head-training data.',
      'Cross-fitting protects the encoder input route. It does not remove the need for independent head selection and test evaluation.',
      'Frozen historical v2 encoder plus local cross-fitted linear head. The final-layer local protocol differs from the paper’s intermediate-layer selection experiment.')

panel(66,'tabicl','TabICL · column, row, dataset','Model architecture',
      'Which stage compresses the context, and which later stage can still be quadratic?',
      [('Build column context','Inducing vectors read training-column tokens; every cell then reads the summaries.','C × d → m × d → N × d'),
       ('Create cell and row representations','Context-conditioned W·x+B; row Transformer gathers feature tokens into four summaries.','4 × 128 → row width 512'),
       ('Run dataset-level ICL','Add context labels; a separate Transformer learns from row representations.','N × 512; context-only memory'),
       ('Predict query labels','Query states → output MLP → class probabilities.','Q × classes')],
      'Compression changes one cost term',
      op('<span><b>C=500</b><br>context rows</span>','<span><b>m=16</b><br>summaries</span>','<span><b>N=500</b><br>cell readers</span>')+pair('Full column attention','250,000','Two inducing stages','16,000')+eq('mC + Nm = 8,000 + 8,000')+eq('cell x=2: W=(.5,−1), B=(.1,.3)')+op('<span>W·x+B</span>',vec([1.1,-1.7])),
      'Score elements per column/head; projection, row and dataset costs are excluded. W and B are cell/context-conditioned, not global constants.',
      'Inducing attention reduces the column-memory term. It does not remove the final dataset learner’s context-attention cost.',
      'Original paper dimensions in the overview; local code isolates inducing and affine operators. Actual timing/quality use pinned v1.1; larger v2 comparisons belong to the later checkpoint.')

panel(67,'localpfn','Local PFN · select context, then adapt','Model architecture',
      'Which operation changes record IDs, and which changes pretrained tensors?',
      [('Define a memory geometry','Fit scaling on training rows; preserve IDs and label eligibility.','N × F'),
       ('Retrieve a local context','Squared distances → stable eligible top-k; exclude anchor/query identity.','query → k labeled rows'),
       ('Optional adaptation','Build disjoint local context/query episodes; query loss updates a copied checkpoint.','θ → θ′ by optimizer steps'),
       ('Select and predict','Validation selects adaptation step; frozen chosen weights predict untouched test queries.','PFN(context, query) → probability')],
      'Two interventions, two state changes',
      matrix(['context IDs','weights'],[('retrieval',['change','fixed θ']),('fine-tuning',['episode-specific','change θ→θ′'])])+eq('episode context IDs ∩ query IDs = ∅')+pair('Retrieval evidence','saved neighbor IDs','Adaptation evidence','nonzero tensor delta'),
      'This is a state-change audit. A different prediction alone cannot tell you whether retrieval or gradient adaptation caused it.',
      'Retrieval changes the model input. Fine-tuning changes the fitted function. Compare them as separate arms before interpreting their combination.',
      'Actual historical v1 checkpoint in the local comparison. Exact retrieval, six-step adaptation and tiny test panels differ from the full paper protocol; approximate-neighborhood scale claims remain separate.')

panel(68,'driftpfn','Temporal PFN · train on changing tasks','Model architecture',
      'What generates drift during pretraining, and what history is legal at inference?',
      [('Sample a changing mechanism','A task-specific function of time produces SCM edge weights.','time t → W(t)'),
       ('Generate an episode','Earlier context and later queries share that changing mechanism and declared noise model.','SCM → [features, time], labels'),
       ('Pretrain a PFN','Query-label loss trains the predictor across episodes; generating graph is not supplied at inference.','episodes → fixed θ'),
       ('Predict later rows','Eligible historical labels + query features/time → fixed pretrained predictor.','context before cutoff → future probability')],
      'Past event does not imply known label',
      matrix(['event','label arrives','cutoff'],[('record',[3,5,4])])+pair('Event test: 3 ≤ 4','passes','Label test: 5 ≤ 4','blocked')+eq('eligible = (event ≤ cutoff) AND (label ≤ cutoff)'),
      'A concrete availability fixture. Time changes the task generator during pretraining; at deployment the model receives observed features/time, not the true SCM.',
      'Two boundaries matter: what dynamics pretraining teaches, and which labels were actually available when a prediction was made.',
      'Local changing-edge RowPFN ablation, not the full published temporal architecture or benchmark. Earlier/later evaluation is paired across the stationary and drifting-prior arms.')

# Routing diagrams expose branches that a numbered forward-path list cannot.
PANELS[48][0]['graphic']=topology('dcn-route','Parallel DCNv2: original input branches into cross and deep networks, then concatenates before the head.',
    [(115,5,90,'input x₀','query'),(10,75,135,'cross × L','memory'),(185,75,125,'MLP(x₀)',''),(90,145,140,'concatenate',''),(100,210,120,'linear head','')],
    [('160,43 160,57 77,57 77,73',False),('160,57 247,57 247,73',False),('77,113 77,133 125,133 125,143',False),('247,113 247,133 195,133 195,143',False),('160,183 160,208',False)],255)+'<p class="aa-small">Parallel route pictured. Stacked route sends the cross output into the MLP instead.</p>'+PANELS[48][0]['graphic']
PANELS[52][0]['graphic']=topology('tabr-route','Query representation takes a direct residual route while learned keys retrieve corrected labeled values.',
    [(5,5,135,'query encoder','query'),(180,5,135,'memory enc.','memory'),(92,77,136,'key search',''),(92,142,136,'weighted values','memory'),(92,212,136,'+ query → head','')],
    [('73,43 73,60 130,60 130,75',False),('247,43 247,60 191,60 191,75',False),('160,115 160,140',False),('160,180 160,210',False),('25,43 25,231 90,231',True)],258)+PANELS[52][0]['graphic']

PANELS[61][0]['graphic']=topology('gp-row-route','Context labels enter context tokens; query labels enter pretraining loss only.',
    [(3,5,150,'context x,y → Ec','memory'),(170,5,147,'query x → Eq','query'),(48,74,224,'row blocks × 3',''),(48,142,224,'query head → density',''),(3,220,140,'held-out y','query'),(177,220,140,'query NLL','')],
    [('78,43 78,72',False),('244,43 244,72',False),('160,112 160,140',False),('244,180 244,218',False),('143,239 175,239',False),('245,258 245,281 24,281 24,94 46,94',True)],296)+PANELS[61][0]['graphic']
PANELS[61][0]['caption']+=' Ec=Ex(x)+Ey(y), Eq=Ex(x); the feature encoder Ex is shared. Dashed path is the pretraining gradient; it is absent at inference.'


def render(p,n,standalone=False):
    route=''.join(f'<li><strong>{E(a)}</strong><span>{E(b)}</span><code class="aa-shape">{E(c)}</code></li>' for a,b,c in p['route'])
    href='./index.html' if standalone else f'../labs/html/architecture-review/{n:04}-{p["key"]}.html'
    link=f'<a href="{href}">'+('All architecture studies' if standalone else 'Open the responsive diagram')+'</a>'
    return f'''<figure class="arch-atlas" id="architecture-{p['key']}" data-architecture-revision="{p['key']}" aria-labelledby="aa-title-{p['key']}">
<div class="aa-header"><p class="aa-kicker">Lesson {n:03} / architecture study</p><h3 id="aa-title-{p['key']}">{E(p['title'])}</h3><p class="aa-question"><strong>Trace this:</strong> {E(p['question'])}</p></div>
<div class="aa-body"><div class="aa-route"><p class="aa-section-label">The forward path</p><ol>{route}</ol></div><div class="aa-focus"><p class="aa-section-label">Inside the key operation</p><h4>{E(p['focus'])}</h4>{p['graphic']}<p class="aa-caption">{E(p['caption'])}</p></div></div>
<figcaption class="aa-answer"><strong>Read the path:</strong> {E(p['answer'])}</figcaption><div class="aa-scope"><p><strong>Variant &amp; evidence.</strong> {E(p['scope'])}</p><p>{link}</p></div></figure>'''


def enrich_html(soup,n):
    panels=PANELS.get(n,[])
    if not panels:return
    for old in soup.select('[data-architecture-revision]'):old.decompose()
    for p in panels:
        headings=[h for h in soup.find_all('h2') if p['anchor'].casefold() in h.get_text().casefold()]
        assert len(headings)==1,(n,p['key'],'architecture heading',len(headings))
        heading=headings[0]
        # Replace the old overview, preserving distinct interactive mechanism figures.
        section=heading.find_parent('section')
        if section:
            for figure in list(section.find_all('figure')):
                if any('architecture' in img.get('src','') for img in figure.find_all('img')):figure.decompose()
            for old in section.select('.foundation-route'):old.decompose()
        else:
            for sibling in list(heading.next_siblings):
                if getattr(sibling,'name',None)=='h2':break
                if getattr(sibling,'name',None)=='ol' and 'foundation-route' in sibling.get('class',[]):sibling.decompose()
                elif getattr(sibling,'name',None)=='figure' and any('architecture' in i.get('src','') for i in sibling.find_all('img')):sibling.decompose()
        heading.insert_after(BeautifulSoup(render(p,n),'html.parser'))
    if not soup.select_one('link[href="../assets/architecture-atlas.css"]'):
        soup.head.append(soup.new_tag('link',rel='stylesheet',href='../assets/architecture-atlas.css'))


def enrich_notebook(nb,n):
    panels=PANELS.get(n,[])
    if not panels:return nb
    # Prior overview images are superseded; operator/result images stay intact.
    old_images={hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'labs/figures'/f'l{n:03}').glob('*architecture*.png')}
    kept=[]
    for c in nb.cells:
        if c.metadata.get('architecture_revision'):continue
        if c.cell_type=='markdown' and 'data:image' in c.source:
            payloads=re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',c.source)
            if any(hashlib.sha256(base64.b64decode(v)).hexdigest() in old_images for v in payloads):continue
        kept.append(c)
    nb.cells=kept
    for p in reversed(panels):
        matches=[i for i,c in enumerate(nb.cells) if c.cell_type=='markdown' and not c.metadata.get('lesson_depth') and
                 any(p['anchor'].casefold() in line.casefold() for line in c.source.splitlines() if line.startswith('#'))]
        assert len(matches)==1,(n,p['key'],'notebook architecture anchor',matches)
        path=ROOT/'labs/figures/architecture-revision'/f'{n:04}-{p["key"]}.png'
        assert path.exists(),('Export browser snapshot before notebook build',path)
        encoded=base64.b64encode(path.read_bytes()).decode()
        alt=E(p['title']+': '+p['question'])
        source=f'<div style="overflow-x:auto;max-width:100%"><img alt="{alt}" src="data:image/png;base64,{encoded}" style="width:100%;min-width:640px;max-width:760px;height:auto"></div>\n\n*Portable architecture figure. On a narrow notebook screen, scroll horizontally to retain readable labels.* [Open the responsive diagram](https://avistian.github.io/relational/labs/html/architecture-review/{n:04}-{p["key"]}.html).\n\n**Read the path:** {p["answer"]}\n\n**Variant and evidence:** {p["scope"]}'
        cell=nbformat.v4.new_markdown_cell(source,metadata={'architecture_revision':p['key']});cell.id=f'architecture-{n}-{p["key"]}'
        nb.cells.insert(matches[0]+1,cell)
    return nb


def preview():
    folder=ROOT/'labs/html/architecture-review';folder.mkdir(exist_ok=True)
    for n,panels in PANELS.items():
        for p in panels:
            (folder/f'{n:04}-{p["key"]}.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'+E(p['title'])+'</title><link rel="stylesheet" href="../../../assets/architecture-atlas.css"><style>body{margin:0;padding:20px;background:#e9eff1}main{max-width:760px;margin:auto}.arch-atlas{margin:0}@media(max-width:600px){body{padding:8px}}</style><main>'+render(p,n,True)+'</main></html>')
    cards=[]
    for n,panels in PANELS.items():
        lesson=next((ROOT/'lessons').glob(f'{n:04}-*.html')).name
        for p in panels:
            cards.append(f'<article><small>LESSON {n:03}</small><h2><a href="{n:04}-{p["key"]}.html">{E(p["title"])}</a></h2><p>{E(p["question"])}</p><a href="../../../lessons/{lesson}">Read the lesson →</a></article>')
    (folder/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Model architecture studies · Relational learning</title><style>body{margin:0;background:#f3f7f7;color:#172e40;font:16px/1.6 system-ui,sans-serif}main{max-width:1000px;margin:50px auto;padding:0 24px}h1{font:42px/1.15 Georgia,serif;max-width:700px}header p{max-width:700px;color:#48606a}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:18px;margin-top:32px}article{padding:24px;background:white;border:1px solid #d5e1e4;border-radius:10px}h2{font:23px/1.25 Georgia,serif}small{letter-spacing:.1em;color:#48606a;font-size:11px}a{color:#086e75;text-underline-offset:3px}article p{font-size:15px}a:focus-visible{outline:3px solid #ba651e;outline-offset:4px}</style><main><header><a href="../../../index.html">← Course</a><h1>Trace the model.<br>Understand the operation.</h1><p>Sixteen architecture studies from lessons 47–70. Each connects the full prediction path with a worked view of its key mechanism. Open a study for a layout that adapts to your screen; follow the lesson link for derivations, code and the companion lab.</p></header><section>'''+''.join(cards)+'</section></main></html>')


if __name__=='__main__':preview()
