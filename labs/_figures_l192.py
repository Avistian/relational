"""Three portable operation diagrams; all numerical failure values are measured."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D=Path(__file__).resolve().parent/'figures/l192';D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.hashsalt':'l192'})
def save(fig,name):
 fig.tight_layout()
 for ext in ['svg','png']:fig.savefig(D/(name+'.'+ext),dpi=160,metadata={'Date':None} if ext=='svg' else None)
 plt.close(fig)
fig,ax=plt.subplots(figsize=(7,9));ax.set(xlim=(0,7),ylim=(0,11));ax.axis('off')
items=[('RELATIONAL INPUT','Study keys + prediction cutoffs\nRelated records before each cutoff'),('DFS: FIXED AGGREGATION','Join along foreign keys, then summarize\nIllustration: MEAN(3, 5) = 4; COUNT = 2'),('FIT / TRANSFORM','Support learns preprocessing state\nQuery features must keep its meaning'),('FROZEN TABULAR FOUNDATION MODEL','Support: Xₛ [nₛ × D], labels yₛ [nₛ]\nQuery: Xq [nq × D] → class probabilities'),('VALIDATE → SELECT → TEST','3 depths × 3 backends × 3 new seeds\n27 validation runs; 3 selected test runs')]
for i,(title,body) in enumerate(items):
 y=8.7-i*2.05
 ax.add_patch(FancyBboxPatch((.2,y),6.6,1.65,boxstyle='round,pad=.06',facecolor='#fff2df' if i==2 else '#eaf4f1',edgecolor='#9fbcb5'))
 ax.text(.45,y+1.23,title,fontsize=12,color='#176760',weight='bold');ax.text(.45,y+.62,body,fontsize=12,va='center',linespacing=1.55)
 if i<4:ax.annotate('',xy=(3.5,y-.32),xytext=(3.5,y-.07),arrowprops={'arrowstyle':'->','color':'#176760','lw':2})
ax.text(.25,10.6,'RDBLearn · inference pipeline',fontsize=17,weight='bold')
save(fig,'architecture')
fig,ax=plt.subplots(figsize=(7,5));ax.axis('off');ax.set(xlim=(0,7),ylim=(0,5))
ax.text(.2,4.6,'One new category changes old codes',fontsize=17,weight='bold')
for x,title,labels,values in [(.2,'Fitted support',['b','c','d'],[0,1,2]),(3.8,'After unseen a',['b','c','d'],[1,2,3])]:
 ax.text(x,3.95,title,weight='bold',color='#176760')
 for i,(c,v) in enumerate(zip(labels,values)):
  y=3.1-i*.72
  ax.add_patch(FancyBboxPatch((x,y),2.9,.5,boxstyle='round,pad=.04',facecolor='#eaf4f1' if x<1 else '#fff2df',edgecolor='#a7b8b2'))
  ax.text(x+.25,y+.25,c,va='center',weight='bold');ax.text(x+2.45,y+.25,str(v),va='center',ha='right')
 ax.text(x,1.2,'Vocabulary: '+('b, c, d' if x<1 else 'a, b, c, d'),fontsize=11)
ax.annotate('insert a\nand sort',xy=(3.7,2.65),xytext=(3.35,2.65),ha='center',va='center',fontsize=9)
ax.text(.2,.45,'Measured with the original full preprocessor.\nSynthetic inputs; no study-outcome AUROC was measured.',fontsize=11,linespacing=1.6)
save(fig,'encoding')
fig,ax=plt.subplots(figsize=(7,4.2));ax.axis('off');ax.set(xlim=(-20,420),ylim=(0,4))
ax.text(0,3.55,'Earlier prediction ≠ available label',fontsize=17,weight='bold')
ax.plot([0,365],[2.5,2.5],color='#9eb5b3',lw=9,solid_capstyle='round');ax.scatter([0,180,365],[2.5]*3,s=[70,120,70],c=['#176760','#b76d1d','#176760'],zorder=3)
for t,label in [(0,'Day 0\nHistorical query'),(180,'Day 180\nNew query'),(365,'Day 365\nWindow ends')]:ax.text(t,2.05,label,ha='left' if t==0 else 'center',va='top',fontsize=11)
ax.text(0,.7,'At day 180: timestamp test passes; availability test fails.\nActual L192 annual schedule: zero unfinished past windows.',fontsize=11,linespacing=1.6)
save(fig,'clocks')
print('Three portable figures')
# Phone layouts reflow the same operations instead of shrinking desktop labels.
fig,ax=plt.subplots(figsize=(4.1,9));ax.set(xlim=(0,4.1),ylim=(0,10.7));ax.axis('off')
mobile=[('INPUT','Study + cutoff\nHistorical related records'),('DFS: FIXED SUMMARIES','Join foreign-key paths\nMEAN(3, 5) = 4; COUNT = 2'),('PREPROCESS SUPPORT / QUERY','Fit state on support\nKeep query meanings fixed'),('FROZEN TABULAR MODEL','Support Xₛ[nₛ,D], yₛ[nₛ]\nQuery Xq[nq,D] → probability'),('SELECT ON VALIDATION','3 depths × 3 backends\n× 3 seeds = 27 candidates')]
for i,(title,body) in enumerate(mobile):
 y=8.55-i*2.02
 ax.add_patch(FancyBboxPatch((.1,y),3.9,1.55,boxstyle='round,pad=.04',facecolor='#fff2df' if i==2 else '#eaf4f1',edgecolor='#9fbcb5'))
 ax.text(.26,y+1.17,title,fontsize=10,weight='bold',color='#176760');ax.text(.26,y+.57,body,fontsize=12,va='center',linespacing=1.6)
 if i<4:ax.annotate('',xy=(2.05,y-.36),xytext=(2.05,y-.06),arrowprops={'arrowstyle':'->','color':'#176760','lw':2})
ax.text(.1,10.4,'RDBLearn · inference route',fontsize=14,weight='bold');save(fig,'architecture-mobile')
fig,ax=plt.subplots(figsize=(4.1,5.4));ax.set(xlim=(0,4.1),ylim=(0,5.4));ax.axis('off')
ax.text(.15,4.95,'After the unseen category a',fontsize=14,weight='bold')
ax.text(.2,4.35,'Known value',fontsize=11);ax.text(1.7,4.35,'Support',fontsize=11);ax.text(3,4.35,'Query',fontsize=11)
for i,name in enumerate(['b','c','d']):
 y=3.6-i*.75;ax.axhspan(y-.22,y+.32,color='#eaf4f1');ax.text(.4,y,name,fontsize=16);ax.text(2,y,str(i),fontsize=16);ax.text(3.3,y,str(i+1),fontsize=16,color='#985512')
ax.text(.2,1.25,'Before: b, c, d → 0, 1, 2\nInsert a, sort: a, b, c, d\nKnown query codes shift.',fontsize=12,linespacing=1.65)
ax.text(.2,.25,'Measured synthetic source diagnostic.\nNo model score measured.',fontsize=10,linespacing=1.5);save(fig,'encoding-mobile')
fig,ax=plt.subplots(figsize=(4.1,5.5));ax.set(xlim=(0,4.1),ylim=(0,5.5));ax.axis('off')
ax.text(.15,5.1,'Two clocks for one label',fontsize=15,weight='bold');ax.plot([.4,.4],[1.9,4.5],color='#9eb5b3',lw=5)
for y,title,body in [(4.5,'Day 0','Historical prediction'),(3.2,'Day 180','New query: label unfinished'),(1.9,'Day 365','Label window ends')]:
 ax.scatter([.4],[y],s=65,color='#176760');ax.text(.75,y+.1,title,fontsize=12,weight='bold');ax.text(.75,y-.3,body,fontsize=11)
ax.text(.15,.9,'Day 180 passes event time,\nbut fails label availability.\nActual annual task cuts: zero\nunfinished past windows.',fontsize=11,linespacing=1.5);save(fig,'clocks-mobile')
