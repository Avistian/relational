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
fig,ax=plt.subplots(figsize=(11,12));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.set_title('RelBench regression · trace a query through rows, messages and loss',loc='left',weight='bold',pad=20)
ax.text(.02,.97,'1  ASK · illustrative query: driver 90 at day 7. Keep its future label separate.',fontsize=12,color=INK)
# A concrete input graph makes the cutoff and message direction visible.
for x,y,label,col in [(.17,.88,'Driver 90',TEAL),(.50,.88,'Result R1\nday 5',TEAL),(.83,.88,'Race 1',TEAL),(.50,.78,'Result R2\nday 11',RED)]:
 ax.add_patch(FancyBboxPatch((x-.12,y-.035),.24,.07,boxstyle='round,pad=.008',facecolor='#edf5f2' if col==TEAL else '#fceeed',edgecolor=col,lw=1.5));ax.text(x,y,label,ha='center',va='center',color=col,fontsize=12)
arrow(ax,.37,.88,.30,.88);arrow(ax,.63,.88,.70,.88)
ax.text(.03,.79,'Keep R1: 5 ≤ 7\nReverse edges carry context back',fontsize=11,color=TEAL)
ax.text(.66,.78,'Exclude R2: 11 > 7\nAt every sampled hop',fontsize=11,color=RED)
ax.text(.02,.71,'2  ENCODE · separate parameters per table (three of nine shown), width 128',weight='bold',color=INK)
for x,label in [(.02,'Driver columns'),(.35,'Result columns'),(.68,'Race columns')]:
 box(ax,x,.58,.29,.10,label,'Typed columns → ResNet\n'+('row vector; undated table' if x==.02 else 'row vector + relative-time vector'))
ax.text(.02,.545,'Dated occurrence age = (its query cutoff − its row time) / 86400 days.',fontsize=11,color=INK)
ax.text(.02,.495,'3  PASS MESSAGES · open one relation before summing relations',weight='bold',color=INK)
box(ax,.02,.355,.45,.115,'Neighbors on relation r','mᵣ(v) = Wₙ,ᵣ Σ hᵤ + bᵣ\nSum over eligible sampled senders')
box(ax,.53,.355,.45,.115,'Destination on relation r','rootᵣ(v) = Wᵣ,ᵣ hᵥ\nEach incoming relation has its own map')
arrow(ax,.24,.35,.40,.31);arrow(ax,.76,.35,.60,.31)
box(ax,.18,.235,.64,.075,'Combine, normalize, activate','Σᵣ (mᵣ + rootᵣ) → LayerNorm → ReLU')
ax.text(.02,.205,'Repeat with a second set of learned weights. Width remains 128; two-hop context can reach the root.',fontsize=11,color=INK)

box(ax,.02,.08,.45,.09,'4  PREDICT · query roots only','First B driver occurrences: [B,128]\nLinear head → [B,1] positions',GOLD)
box(ax,.53,.08,.45,.09,'5  TRAIN · future query labels','Mean |prediction − target| → Adam\nGradients reach encoders and both layers',GOLD)
arrow(ax,.48,.125,.52,.125)
ax.text(.02,.025,'Select: first minimum validation MAE. Inference: restore selected state; clip to train-label percentiles 2 and 98.',fontsize=10.5,color=INK)
save(fig,'architecture')
print('Built three portable computation figures')
