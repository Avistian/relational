"""Computation-specific REG, identity and architecture diagrams."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;D=P/'figures/l117';D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none','svg.hashsalt':'l117','savefig.facecolor':'#fcfbf8'})
INK='#233846';TEAL='#176c68';GOLD='#a85c21';RED='#a5383d'
def save(fig,name):
 fig.savefig(D/f'{name}.svg',bbox_inches='tight',metadata={'Date':None});fig.savefig(D/f'{name}.png',bbox_inches='tight',dpi=150,metadata={'Software':'L117'});plt.close(fig)
 path=D/f'{name}.svg';path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
def box(ax,x,y,w,h,title,detail,color=TEAL):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',facecolor='#edf5f2',edgecolor=color,lw=1.4));ax.text(x+w/2,y+h*.73,title,ha='center',va='center',weight='bold',color=INK,fontsize=12);ax.text(x+w/2,y+h*.28,detail,ha='center',va='center',fontsize=10,color=INK)
def arrow(ax,x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':INK,'lw':1.5})
fig,ax=plt.subplots(figsize=(9,10));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title('One schema → many entities → one legal computation',loc='left',weight='bold',pad=18)
box(ax,.04,.80,.92,.13,'SCHEMA · one node per table','Drivers ← Results → Races → Circuits\nForeign-key column names define typed relationships')
arrow(ax,.5,.79,.5,.73)
box(ax,.04,.48,.92,.23,'REG · one node per row','Driver key 90   ←   Result R1 (time 5)   →   Race 1\nDriver key 90   ←   Result R2 (time 11) →   Race 2\nAdd separately typed reverse edges; keep row attributes')
arrow(ax,.5,.47,.5,.40)
box(ax,.04,.14,.92,.24,'QUERY · driver 90 at time 7','R1 is eligible: 5 ≤ 7. R2 is excluded: 11 > 7.\nAt a later query time, the same driver can have more context.\nThe future target belongs to the query, not to input nodes.',GOLD)
ax.text(.04,.04,'Sampling bounds message computation; it does not change what the keys mean.',fontsize=11,color=INK);save(fig,'graphs')
fig,ax=plt.subplots(figsize=(9,7.5));ax.axis('off');ax.set_title('Three index spaces: query, entity, sampled copy',loc='left',weight='bold',pad=22)
rows=[['0','90','day 7','3'],['1','10','day 7','7'],['2','90','day 12','11']]
t=ax.table(cellText=rows,colLabels=['Query row','Driver key','Prediction time','Future target'],cellLoc='center',bbox=[.02,.59,.96,.31]);t.auto_set_font_size(False);t.set_fontsize(12)
for (row,col),c in t.get_celld().items():c.set_facecolor('#edf5f2' if row in [1,3] else '#faf9f6');c.set_edgecolor('#bdcece')
ax.text(.03,.48,'Shuffled input_id = [2, 0, 1] → target = [11, 3, 7]',fontsize=14,color=TEAL,weight='bold')
ax.text(.03,.33,'input_id  identifies a training-table query row\nn_id        identifies a database entity-row position\nbatch      identifies the query owning a sampled copy',fontsize=13,linespacing=1.8,color=INK)
ax.text(.03,.08,'Driver 90 appears twice. Its node identity is shared;\nits prediction time, legal neighborhood and target are not.',fontsize=13,color=GOLD);save(fig,'queries')
fig,ax=plt.subplots(figsize=(9,12));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title('Model architecture · released RelBench node regression',loc='left',weight='bold',pad=18)
boxes=[(.86,.11,'Query rows + complete database','B ≤ 512 (driver, time) pairs; PK/FK REG; sample [128,64]'),(.68,.12,'Per-table row encoder + relative age','Column encodings → four-block ResNet → [nₜ,128]\nAdd Linear(PositionalEncoding((query time − row time)/86400))'),(.48,.13,'Two typed GraphSAGE layers','Sum neighbor vectors per relation + relation-specific root map\nSum relations → node-wise LayerNorm → ReLU; width 128'),(.30,.10,'Read seed driver embeddings → linear head','First B driver copies: [B,128] → [B,1] prediction'),(.12,.11,'Train and select','Mean absolute error → Adam .005 → 10 epochs\nFirst minimum validation MAE; restore selected state')]
for i,(y,h,title,detail) in enumerate(boxes):
 box(ax,.04,y,.92,h,title,detail,GOLD if i==4 else TEAL)
 if i:arrow(ax,.5,boxes[i-1][0],.5,y+h+.015)
ax.text(.04,.02,'Inference: clip to training-target percentiles (2,98).\nFrozen GloVe text inputs; trainable row encoders, messages and head.',fontsize=11,color=INK);save(fig,'architecture')
print('Built three portable computation figures')
