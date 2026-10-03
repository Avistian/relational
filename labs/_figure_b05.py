"""Portable model-specific SVG: separate paper episode order from frozen inference."""
from pathlib import Path
import html
P=Path(__file__).resolve().parent/'figures/b05';P.mkdir(exist_ok=True)
parts=['''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1260" viewBox="0 0 1000 1260" role="img" aria-labelledby="title desc"><title id="title">TabDPT: an episode becomes a prediction</title><desc id="desc">Real table, target-free retrieval, disjoint context and query,100-wide row projection,768-wide transformer with support-only keys and values, two task heads. Training updates weights; inference freezes them. Released training sampler order differs.</desc><defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="none" stroke="#267566" stroke-width="1.5"/></marker></defs><rect width="1000" height="1260" rx="20" fill="#f8faf6"/><style>text{font-family:system-ui,sans-serif;fill:#173a35}.head{font-size:28px;font-weight:700}.sub{font-size:19px}.label{font-size:23px;font-weight:650}.small{font-size:18px}.box{fill:white;stroke:#b3cec4;stroke-width:2}.arrow{stroke:#267566;stroke-width:3;fill:none;marker-end:url(#a)}</style><text x="45" y="53" class="head">TabDPT · learn from columns, predict from context</text><text x="45" y="84" class="sub">Original architecture · 16 layers · width 768 · 4 attention heads</text>''']
def box(x,y,w,h,title,lines,fill='white'):
 parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" class="box" style="fill:{fill}"/><text x="{x+18}" y="{y+32}" class="label">{html.escape(title)}</text>')
 for i,line in enumerate(lines):parts.append(f'<text x="{x+18}" y="{y+62+i*26}" class="small">{html.escape(line)}</text>')
def arrow(x,y,a,b):parts.append(f'<path d="M{x},{y} L{a},{b}" class="arrow"/>')
box(40,115,440,145,'PRETRAIN · real table T ∈ Rⁿˣᶠ',['Choose column c as the prediction target.','Paper order: remove c, then retrieve rows.','Split selected rows without replacement.'])
box(520,115,440,145,'INFER · new task, fixed weights',['Eligible training rows provide Xₛ and yₛ.','Retrieve using features of query xq.','Query labels never enter this path.'])
arrow(255,260,255,305);arrow(745,260,745,305)
box(40,305,440,150,'Support: Xₛ [S,F], yₛ [S,1]',['Fit eligible-context preprocessing.','Pad / reduce feature width to 100.','Row projection: 100 → 768; RMS normalize.'])
box(520,305,440,150,'Queries: Xq [Q,F]',['Apply inference support statistics.','Same feature view and row projection.','No query target embedding.'])
arrow(255,455,255,490);arrow(745,455,745,490)
box(40,490,440,125,'Condition support rows',['φₓ(Xₛ) + φᵧ(yₛ) → [S,768]','Regression y normalization uses support.'],'#e4f1e9')
box(520,490,440,125,'Keep query targets outside',['φₓ(Xq) → [Q,768]','Concatenate support then queries.'],'#e4f1e9')
arrow(255,615,255,657);arrow(745,615,745,657)
box(40,657,920,193,'ROW TRANSFORMER ×16 · [S+Q,768]',['Pre-norm → attention → residual → pre-norm → MLP → residual','Q projections: all S+Q rows.  K and V projections: S support rows only.','Per-head width: 768 / 4 = 192. Attention matrix: [4,S+Q,S].','Queries read support; support does not read queries. No query-to-query attention.'],'#dcece5')
arrow(255,850,255,889);arrow(745,850,745,889)
box(40,889,440,118,'Classification head',['Query states → active-class logits','Cross-entropy during pretraining.'])
box(520,889,440,118,'Regression head',['Query states → scalar target','Undo support target normalization.'])
box(40,1040,920,86,'Training boundary',['Query targets score the loss and update weights; inference only changes context.'],'#e8f0f5')
box(40,1153,920,78,'SOURCE AUDIT · do not silently merge these paths',['Released training use_knn retrieves before target removal. See the counterexample.'],'#fff0d9')
parts.append('</svg>');(P/'architecture.svg').write_text(''.join(parts))
# Raster only for portable notebook rendering; SVG remains repository-native source.
import cairosvg
cairosvg.svg2png(url=str(P/'architecture.svg'),write_to=str(P/'architecture.png'),output_width=1200)
print(P/'architecture.svg')
