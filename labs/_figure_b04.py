"""Portable, model-specific architecture and evidence plots; deterministic outputs."""
from pathlib import Path
import html,json
import cairosvg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent;D=P/'figures/b04';D.mkdir(exist_ok=True)
parts=['<svg xmlns="http://www.w3.org/2000/svg" width="560" height="1330" viewBox="0 0 560 1330"><defs><marker id="a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#47766f"/></marker></defs><rect width="560" height="1330" rx="18" fill="#f2f6f3"/>']
def text(y,s,size=17,weight='400'):
    parts.append(f'<text x="280" y="{y}" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="{size}" font-weight="{weight}" fill="#173e39">{html.escape(s)}</text>')
def box(y,h,title,lines,fill='#fff'):
    parts.append(f'<rect x="22" y="{y}" width="516" height="{h}" rx="12" fill="{fill}" stroke="#9eb9ab"/>');text(y+30,title,20,'600')
    for i,line in enumerate(lines):text(y+58+23*i,line)
def arrow(y):parts.append(f'<path d="M280 {y}v24" stroke="#47766f" stroke-width="2" marker-end="url(#a)"/>')
text(37,'TabICLv2 · one prediction, end to end',23,'600')
box(60,102,'Before this task: learn an inference rule',['Synthetic tasks → query loss → update θ','Prior, architecture and optimizer all matter.'],'#e2eee8');arrow(166)
box(201,110,'At inference: freeze θ',['Support X[S,F], y[S] · Query X[Q,F]','Query targets stay in the scorer only.','Fit wrapper transforms on support.']);arrow(315)
box(349,110,'Repeated groups + target-aware embedding',['N = S + Q · three values per feature position','N × F × 128; support labels added early','“same” grouping preserves F positions.']);arrow(463)
box(497,112,'Column stage · 3 induced-attention blocks',['128 inducing vectors per feature position','Aggregate support → broadcast to rows','QASSMax on inducing-query aggregation.'],'#e2eee8');arrow(613)
box(647,112,'Row stage · 3 transformer blocks',['Within each row: F tokens + 4 CLS tokens','4 × 128 CLS outputs → one 512-wide row','Representation compression happens here.'],'#e2eee8');arrow(763)
box(797,112,'Dataset ICL · 12 transformer blocks',['N × 512 row vectors; add support targets','8 heads, QASSMax; queries read support','Longer context still means more attention work.'],'#e2eee8');arrow(913)
box(947,112,'Prediction head + wrapper',['512 → 1024 → up to 10 class logits','Temperature → class alignment → view average','B04: one view, Q × 2 output probabilities.']);arrow(1063)
box(1097,115,'Score and audit outside the model',['Query IDs + probabilities + held-out labels','Accuracy · log loss · Brier · time · peak RSS','384 fresh course predictions; paper lane blocked.'],'#f2e8d3')
text(1250,'Classification checkpoint: v2-20260212',17,'600');text(1278,'Full source + shapes are inspectable in the notebook.',16)
parts.append('</svg>');svg=''.join(parts);(D/'architecture.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(D/'architecture.png'),scale=1.5)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
n=np.array([2,8,64,256,1024,15001]);fixed=np.exp(2)/(np.exp(2)+n-1);scaled=n*n/(n*n+n-1)
fig,ax=plt.subplots(figsize=(7,3.8),layout='constrained');ax.semilogx(n,fixed,'o-',color='#a46232',label='Fixed logit gap 2');ax.semilogx(n,scaled,'o-',color='#216b60',label='Gap 2 × log(n)');ax.set(xlabel='Support keys n (one anchor)',ylabel='Anchor attention weight',ylim=(0,1.04),title='Analytical attention toy · not trained TabICLv2');ax.legend();fig.savefig(D/'attention.png',dpi=160);plt.close(fig)
r=json.loads((P/'evidence/b04/diagnostic-audit.json').read_text())['rows'];clean=r[:3]
fig,axs=plt.subplots(1,2,figsize=(8,3.8),layout='constrained');axs[0].plot([x['support_n'] for x in clean],[x['log_loss'] for x in clean],'o-',color='#216b60');axs[0].set(xlabel='Support rows',ylabel='Log loss on 64 fixed queries',title='One checkpoint, nested support');axs[1].bar(['Mean','Mean + flags'],[r[4]['predict_seconds'],r[5]['predict_seconds']],color=['#a46232','#216b60']);axs[1].set(ylabel='Prediction seconds · CPU',title='30 vs 60 input features');fig.suptitle('B04 course diagnostic · one split and timing per configuration',fontsize=12);fig.savefig(D/'results.png',dpi=160);plt.close(fig)
print('Created three portable figures')
