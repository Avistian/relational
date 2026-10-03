"""Repository-native figures; portable rasters generated for the notebook."""
from pathlib import Path
import html,json
import cairosvg
P=Path(__file__).resolve().parent/'figures/b06';P.mkdir(exist_ok=True)
parts=['''<svg xmlns="http://www.w3.org/2000/svg" width="480" height="1510" viewBox="0 0 480 1510" role="img"><title>Mitra concept and explicit course forward pass</title><rect width="480" height="1510" rx="16" fill="#f4f7f2"/><style>text{font-family:system-ui;fill:#183c37;font-size:20px}.title{font-size:23px;font-weight:bold}.small{font-size:18px}rect.box{stroke:#a7c8bc;fill:white;stroke-width:2}.arr{stroke:#287b67;stroke-width:3}</style><text x="24" y="36" class="title">The generator shapes the learner</text>''']
def box(y,h,title,lines,fill='white'):
 parts.append(f'<rect class="box" x="18" y="{y}" width="444" height="{h}" rx="10" style="fill:{fill}"/><text x="32" y="{y+30}" class="title">{html.escape(title)}</text>')
 for i,line in enumerate(lines):parts.append(f'<text x="32" y="{y+59+i*27}">{html.escape(line)}</text>')
def arrow(y):parts.append(f'<path d="M240,{y} v24 m-7,-8 l7,8 l7,-8" class="arr" fill="none"/>')
box(58,145,'1 · Choose one task generator',['u < p → SCM task','u ≥ p → tree task','Choice applies to the whole table.'],'#e1f1e8');arrow(203)
box(233,171,'2 · Split the generated table',['24 support rows: X and known y','8 query rows: X; y hidden from model','Fit transforms on support only.','Query y enters the loss afterward.']);arrow(404)
box(434,171,'3 · Embed individual cells',['Course: 32 rows × 5 tokens × 32','4 scalar features → linear projection','Target → class0 / class1 / missing','Query target token is “missing”.']);arrow(605)
box(635,171,'4 · Column attention',['Each row mixes its five tokens.','Scores: 4 heads × 5 × 5','Weighted values carry feature','information into its target token.'],'#e6edf8');arrow(806)
box(836,171,'5 · Row attention',['For each column, read support rows.','Scores: 4 heads × 32 × 24','No query rows in keys or values.','Residual + MLP; repeat blocks ×2.'],'#e6edf8');arrow(1007)
box(1037,144,'6 · Read query target tokens',['8 query tokens → 2 logits each','Softmax → class probabilities','Cross-entropy → update in training']);arrow(1181)
box(1211,144,'Inference boundary',['Weights stay fixed on a new task.','Supply fresh support X,y and query X.','Fine-tuning would update weights.'],'#fff0d7')
box(1383,107,'Original Mitra scope',['12 blocks; width512; 4 heads.','Course sizes and generators differ.'])
parts.append('</svg>');(P/'architecture.svg').write_text(''.join(parts));cairosvg.svg2png(url=str(P/'architecture.svg'),write_to=str(P/'architecture.png'))
# Small explicit attention matrix. Query labels never enter the model.
parts=['''<svg xmlns="http://www.w3.org/2000/svg" width="480" height="400" viewBox="0 0 480 400"><title>Allowed row-attention reads, support4 and queries2</title><rect width="480" height="400" fill="#f4f7f2"/><style>text{font-family:system-ui;fill:#193d36;font-size:19px}</style><text x="20" y="32">Row-attention toy: S=4, Q=2</text><text x="135" y="65">Keys / values →</text>''']
for i,l in enumerate(['s0','s1','s2','s3','q0','q1']):
 parts.append(f'<text x="{133+49*i}" y="94">{l}</text><text x="57" y="{128+35*i}">{l}</text>')
 for j in range(6):parts.append(f'<rect x="{124+49*j}" y="{105+35*i}" width="43" height="29" fill="{"#227962" if j<4 else "#ede5da"}"/><text x="{137+49*j}" y="{127+35*i}" style="fill:{"white" if j<4 else "#66574c"}">{"✓" if j<4 else "×"}</text>')
parts.append('<text x="20" y="350">All rows read support; none read queries.</text><text x="20" y="379">Repeat independently for each column.</text></svg>');(P/'mask.svg').write_text(''.join(parts));cairosvg.svg2png(url=str(P/'mask.svg'),write_to=str(P/'mask.png'))
# Measured per-seed results with analytical uniform baseline.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=json.loads((P.parents[1]/'evidence/b06/course-audit.json').read_text())
plt.rcParams.update({'font.size':11})
fig,axs=plt.subplots(3,1,figsize=(4.8,8.4),sharey=True)
for ax,f in zip(axs,['scm','tree','hybrid']):
 for i,a in enumerate(['scm','tree','mixed']):
  y=[x['cross_entropy'] for x in r['rows'] if x['family']==f and x['arm']==a];ax.scatter([i-.1,i,i+.1],y,s=48,color=['#257966','#456da8','#be7836'][i]);ax.plot([i-.2,i+.2],[sum(y)/3]*2,color='black',lw=1.5)
 ax.axhline(__import__('math').log(2),ls='--',color='#666',label='Uniform p=0.5');ax.set_xticks(range(3),['SCM','Tree','Mixed']);ax.set_title(f.upper()+' evaluation tasks');ax.grid(axis='y',alpha=.15);ax.set_ylabel('Cross-entropy')
axs[0].legend(fontsize=9,loc='upper right');fig.suptitle('Nine fresh course fits\nDetail view: lower loss is better',fontsize=13);fig.tight_layout();fig.savefig(P/'results.png',dpi=160);plt.close(fig)
print('Created architecture, attention mask and measured result figures')
