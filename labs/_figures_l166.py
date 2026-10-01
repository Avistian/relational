"""Model-specific diagrams: prior, two-stage training, axis-specific inference."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent/'figures/l166';P.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l166'})
def canvas(title,height):
    fig,ax=plt.subplots(figsize=(12,height));fig.subplots_adjust(left=.03,right=.97,bottom=.03,top=.88)
    fig.patch.set_facecolor('#f5f7fb');ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    fig.suptitle(title,x=.04,y=.97,ha='left',fontsize=18,color='#22385f');return fig,ax
def box(ax,x,y,w,h,title,body,color='#e4edf9'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008',facecolor=color,edgecolor='#9aacc9'))
    ax.text(x+.012,y+h-.025,title,va='top',weight='bold',fontsize=12,color='#22385f')
    ax.text(x+.012,y+h-.092,body,va='top',fontsize=10.5,linespacing=1.5,color='#23344b')
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=2,color='#4874ad'))
def save(fig,name):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for ax in fig.axes:
        for text in ax.texts:
            bbox=text.get_window_extent(renderer)
            assert bbox.x0>=0 and bbox.y0>=0 and bbox.x1<=fig.bbox.width and bbox.y1<=fig.bbox.height,(name,text.get_text())
    for ext in ['svg','png']:fig.savefig(P/(name+'.'+ext),dpi=140,metadata={'Date':None} if ext=='svg' else {'Software':'L166'})
    plt.close(fig)
    p=P/(name+'.svg');p.write_text('\n'.join(l.rstrip() for l in p.read_text().splitlines())+'\n')
fig,ax=canvas('RDB-PFN: the relational knowledge starts in the training data',7.3)
box(ax,.01,.65,.28,.31,'1 · Schema','Sample an acyclic table graph\nUsers ← Orders → Products\nChoose table dimensions')
box(ax,.36,.65,.28,.31,'2 · Keys and latent state','Candidate parent rows\nSelective SCM + random noise\nValid keys, nonrandom dependence')
box(ax,.71,.65,.28,.31,'3 · Complete the cells','Generate numeric / category cells\nOptional row-GNN propagation\nChoose a prediction target')
arrow(ax,(.30,.81),(.35,.81));arrow(ax,(.65,.81),(.70,.81))
box(ax,.01,.22,.45,.29,'Linearize with DFS','Follow parent attributes; aggregate child records.\nPer target row: COUNT, MEAN, MAX, MIN, ...\nSame feature construction for every comparator.','#e5f1eb')
box(ax,.55,.22,.44,.29,'Turn rows into an ICL episode','Split rows into support S and queries Q.\nProvide S labels; hide Q labels.\nPredict Q labels and update shared weights.','#fff0d9')
ax.plot([.85,.85,.24],[.63,.56,.56],color='#4874ad',lw=2)
arrow(ax,(.24,.56),(.24,.52));arrow(ax,(.47,.36),(.54,.36))
ax.text(.015,.07,'Executed generator sample: 14 tables · 13 FK relations · 14 tasks; schema pool reused, no LayerDAG retraining.',fontsize=10.5)
ax.text(.015,.01,'The course two-table SCM isolates one dependency; it does not replace the released relational prior.',fontsize=10.5)
save(fig,'prior')
fig,ax=canvas('Model architecture: two training stages, one frozen inference path',8.6)
box(ax,.01,.77,.45,.21,'Pretrain stage 1','Synthetic single-table episodes → query cross-entropy\nUpdate shared weights; learn statistical regularities.')
box(ax,.55,.77,.44,.21,'Pretrain stage 2','Synthetic RDB + single-table mixture → query loss\nInitialize from stage 1; learn DFS feature patterns.')
arrow(ax,(.47,.87),(.54,.87))
ax.text(.015,.69,'INFERENCE · pretrained weights frozen · no optimizer steps on the real task',weight='bold',fontsize=11,color='#84602c')
box(ax,.01,.36,.28,.27,'DFS task matrix','S support rows + Q queries\nF numeric-coded columns\nSupport-only median + scale')
box(ax,.36,.36,.28,.27,'Cell + target tokens','Each scalar → 96 channels\nAppend one label column\nQuery labels = support mean')
box(ax,.71,.36,.28,.27,'Six bi-attention blocks','Features within each row\nRows within each column\n4 heads; residuals + MLP192')
arrow(ax,(.30,.50),(.35,.50));arrow(ax,(.65,.50),(.70,.50))
box(ax,.01,.025,.61,.21,'Tensor path','[B, S+Q, F] → [B, S+Q, F+1, 96]\nKeep query target tokens → [B, Q, 96] → [B, Q, 2] logits','#e5f1eb')
box(ax,.71,.025,.28,.21,'Binary prediction','MLP decoder + softmax\nQ class-1 probabilities\n692,738 parameters','#e5f1eb')
arrow(ax,(.85,.34),(.85,.25));arrow(ax,(.63,.13),(.70,.13))
save(fig,'architecture')
fig,ax=canvas('Inside each block: feature attention, then support-only row attention',6.0)
# A literal key-visibility matrix; shape is the mechanism.
labels=['support A','support B','query C','query D'];x0=.54;y0=.72;cell=.078
ax.text(.54,.94,'KEY / VALUE ROW',weight='bold',fontsize=11)
for j,label in enumerate(['A','B','C','D']):ax.text(x0+j*cell+cell/2,.84,label,ha='center',fontsize=12)
for i,label in enumerate(labels):
    ax.text(x0-.02,y0-i*cell+cell/2,label,ha='right',va='center',fontsize=11)
    for j in range(4):
        ax.add_patch(plt.Rectangle((x0+j*cell,y0-i*cell),cell*.95,cell*.95,facecolor='#3975aa' if j<2 else '#e3e8ef'))
        ax.text(x0+j*cell+cell/2,y0-i*cell+cell/2,'read' if j<2 else '—',ha='center',va='center',fontsize=10,color='white' if j<2 else '#738195')
box(ax,.01,.51,.30,.39,'Across features','At each row independently:\nF feature tokens + label token\nexchange information.\nNo cross-row mixing here.')
box(ax,.01,.055,.98,.30,'Across rows · the same column only','Support tokens attend to support tokens. Each query token also attends only to support tokens.\nQuery D cannot affect query C through key/value attention. The support statistics stay fixed.\nAfter residual + layer norm: GELU MLP, another residual + layer norm. Repeat six times.','#e5f1eb')
save(fig,'attention')
print('Built three portable model-specific figures')
