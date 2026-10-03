from pathlib import Path
import html
parts=['<svg xmlns="http://www.w3.org/2000/svg" width="560" height="1030" viewBox="0 0 560 1030" role="img" aria-labelledby="t d"><title id="t">Numerical representation, shared paths, and packed members</title><desc id="d">Training-only preprocessing feeds either raw or embedded numbers. TabM shares a weight matrix across adapted paths; TabPack gives each member independent weights and hyperparameters, selects using validation and averages at inference.</desc><rect width="560" height="1030" rx="14" fill="#eef5f2"/><style>text{font-family:Arial,sans-serif;fill:#153b30;font-size:18px}.h{font-size:23px;font-weight:bold}.small{font-size:16px}.box{fill:white;stroke:#adc7b9;stroke-width:1.5}.arrow{fill:#44755f;font-size:26px}</style>']
def text(x,y,s,cls=''):parts.append(f'<text x="{x}" y="{y}" class="{cls}">{html.escape(s)}</text>')
def box(y,h):parts.append(f'<rect x="16" y="{y}" width="528" height="{h}" rx="10" class="box"/>')
text(25,35,'Two axes, three mechanisms','h');text(25,64,'Trace training information through to a prediction.','small')
box(85,120);text(32,116,'01 · Representation','h');text(32,148,'Training-only transforms → each numeric feature');text(32,180,'Raw scalar x   OR   embedding e(x) → concatenate')
text(270,238,'↓','arrow')
box(255,180);text(32,287,'02a · TabM: shared backbone weights','h');text(32,322,'Path 1:  x ⊙ r₁  →  shared W  →  ⊙ s₁ + b₁');text(32,355,'Path 2:  x ⊙ r₂  →  shared W  →  ⊙ s₂ + b₂');text(32,390,'Nonlinear layers → independent output heads');text(32,418,'Mean member losses train all paths.','small')
box(455,195);text(32,488,'02b · TabPack: independent weights','h');text(32,523,'Member 1: e₁(x) → W₁ → prediction 1');text(32,556,'Member 2: e₂(x) → W₂ → prediction 2');text(32,590,'Different depth / dropout / optimizer settings');text(32,622,'Packed operations + shape masks; no shared W.','small')
text(270,683,'↓','arrow');box(700,145);text(32,734,'03 · Validation selects','h');text(32,770,'TabM: checkpoint / hyperparameter decisions');text(32,805,'TabPack: online member ensemble + stopping');text(32,831,'Count selection even when it occurs inside one run.','small')
text(270,878,'↓','arrow');box(890,115);text(32,922,'04 · Freeze → predict → score','h');text(32,957,'Average retained predictions; test labels score only.');text(32,986,'Changed representations need controlled ablations.','small')
parts.append('</svg>');Path('labs/figures/b02/architecture.svg').write_text(''.join(parts))
