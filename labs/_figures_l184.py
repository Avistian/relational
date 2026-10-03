"""Portable, code-native GelGT diagrams; synthetic arithmetic explicitly labeled."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from relkit.gelgt_l184 import attention_trace
P=Path(__file__).resolve().parent/'figures/l184';P.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
def save(fig,name):
 fig.savefig(P/(name+'.svg'),bbox_inches='tight');fig.savefig(P/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(9,11));ax.set(xlim=(0,10),ylim=(0,12));ax.axis('off')
ax.text(.3,11.7,'GelGT · a query becomes a prediction',size=21,weight='bold')
ax.text(.3,11.15,'Paper / released architecture overview · no full model trained here',size=11)
def box(x,y,w,h,title,body,color='#e9f2f5'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',facecolor=color,edgecolor='#41616c'))
 ax.text(x+.16,y+h-.28,title,weight='bold',va='top',size=12)
 ax.text(x+.16,y+h-.68,body,va='top',size=11,linespacing=1.5)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#345','lw':1.8})
box(.5,9.4,9,1.2,'1  Query + database','(driver, cutoff) → PK/FK graph → legal two-hop candidates\nPublished regression budget: 500 candidates → 300 retained tokens')
arrow((5,9.25),(5,8.95))
box(.5,7.1,9,1.7,'2  Encode, then choose','Type + hop + age + table fields + structural position\nFive normalized channels → concatenate → MLP → H [B, K, 512]\nPreserve nearby nodes; score distant nodes with H[j] · H[seed]')
arrow((3,6.97),(2.7,6.83));arrow((7,6.97),(7.3,6.83))
box(.5,3.7,4.25,3,'3a  Local topology','Induced edges + node states\n3 × GraphSAGE\nNormalize → GELU → residual\nRead query node\n\nh_GNN [B, 512]')
box(5.25,3.7,4.25,3,'3b  Temporal attention','Q, K, V [B, heads, K, 128]\nlag → exp(-((lag-μ)/σ)²)\nLearned projection → bias\nsoftmax(QKᵀ/√128 + bias)V\nQuery + weighted neighbors\nh_attention [B, 512]','#fff0dd')
arrow((2.7,3.55),(4,2.95));arrow((7.3,3.55),(6,2.95))
box(.5,1.4,9,1.35,'4  Learn the mixture; predict finishing position','Release: sigmoid(w) h_GNN + (1 - sigmoid(w)) h_attention\nTwo-layer prediction head → one value per complete query key')
ax.text(.5,.65,'Release: one outer hybrid block; configurable attention depth.\nPaper lists four model layers. That discrepancy remains unresolved.',size=11)
save(fig,'architecture')
fig,axs=plt.subplots(2,1,figsize=(9,7));fig.subplots_adjust(hspace=.62,bottom=.23)
for ax,cutoff,title in zip(axs,[10,20],['Early query (driver 0, day 10)','Later query (driver 0, day 20)']):
 ax.axvline(cutoff,color='#994526',lw=2,label='query cutoff')
 for t in [5,15]:
  legal=t<cutoff;ax.scatter(t,0,s=350,marker='o' if legal else 'x',color='#237d87' if legal else '#bb4b39');ax.text(t,.3,f'Race day {t}\n'+('admitted' if legal else 'future: excluded'),ha='center')
 ax.set(xlim=(0,23),ylim=(-.5,1),yticks=[],xlabel='Raw event day',title=title);ax.legend(loc='lower left')
fig.suptitle('A cache key must keep the cutoff',fontsize=20,weight='bold')
fig.text(.12,.045,'Original source probe: both queries retrieve the day-20 context from cache[driver 0].\nA Gaussian bias cannot make the day-15 observation legal at day 10.',fontsize=12)
save(fig,'cache')
fig,axs=plt.subplots(1,2,figsize=(10,4.5));fig.subplots_adjust(bottom=.25,wspace=.35)
x=np.linspace(0,6,150)
for center in [0,2,4]:axs[0].plot(x,np.exp(-((x-center)/2)**2),label=f'center {center}')
axs[0].set(xlabel='Lag (days)',ylabel='Gaussian feature',title='Choose a temporal band');axs[0].legend()
t=attention_trace([0,2,4]);pos=np.arange(3)
axs[1].bar(pos-.17,[1/3]*3,.34,label='No bias',color='#adb9c0');axs[1].bar(pos+.17,t['weights'],.34,label='center2, width2',color='#237d87');axs[1].set(xticks=pos,xticklabels=['0 days','2 days','4 days'],ylim=(0,.65),ylabel='Attention weight',title='Equal content scores');axs[1].legend()
fig.text(.08,.03,f'Synthetic one-head trace: kernel [0.368, 1.000, 0.368], identity projection.\nValues [1, 3, 9]: uniform output 4.333 → biased output {t["output"]:.3f}. This is not test MAE.',size=12)
save(fig,'attention')
