"""Portable computation diagrams, designed for ~720px notebook display width."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parent/'figures/l076'
plt.rcParams.update({'font.size':12,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
def canvas(title,height=6):
 fig,ax=plt.subplots(figsize=(9,height));ax.set_xlim(0,9);ax.set_ylim(0,height);ax.axis('off');ax.set_title(title,loc='left',fontweight='bold',pad=18);return fig,ax
def box(ax,x,y,w,h,text,color='#e8f1f8'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.05',facecolor=color,edgecolor='#69839a'))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
def arrow(ax,start,end):ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#315e7b','lw':2})
def save(fig,name):fig.savefig(ROOT/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)
def build():
 ROOT.mkdir(parents=True,exist_ok=True)
 fig,ax=canvas('Two-table teaching model · follow the row axis',6.5)
 box(ax,.2,5.1,3.7,.9,'Customers: age + region\n4 rows × 2 columns')
 box(ax,5.1,5.1,3.7,.9,'Events: amount + kind\n5 rows × 2 columns')
 box(ax,.2,3.7,3.7,.9,'Frame tokens [4,2,4]\nflatten → Linear(8,4) → tanh')
 box(ax,5.1,3.7,3.7,.9,'Frame tokens [5,2,4]\nflatten → Linear(8,4) → tanh')
 arrow(ax,(2,5.05),(2,4.65));arrow(ax,(7,5.05),(7,4.65))
 box(ax,5.1,2.1,3.7,1,'[5,4] → keep 3 eligible rows\nroute FK → customer position\nmean → [4,4]', '#e6f3e9')
 arrow(ax,(7,3.65),(7,3.15));box(ax,.2,2.1,3.7,1,'Own customer vectors [4,4]\nKeys/time remain metadata')
 box(ax,1.1,.2,6.8,1.1,'Concatenate own + neighbor → [4,8]\nLinear(8,8) → tanh → Linear(8,1)\n4 logits → binary cross-entropy', '#fff0d6')
 arrow(ax,(2,3.65),(2,3.15));arrow(ax,(2,2.05),(2,1.35));arrow(ax,(7,2.05),(7,1.35));save(fig,'architecture')
 fig,ax=canvas('Route identity, then reduce · hand-chosen 2D embeddings',5.6)
 rows=[('event0: FK7 → position1','[2,1]'),('event1: FK42 → position0','[4,3]'),('event2: FK7 → position1','[6,5]')]
 for i,(a,b) in enumerate(rows):box(ax,.1,4-i*1.35,3.7,.85,a+'\n'+b)
 box(ax,5.1,4,3.7,.85,'Customer42 (position0)\n[4,3] / 1 = [4,3]','#e6f3e9')
 box(ax,5.1,2.4,3.7,1.05,'Customer7 (position1)\n([2,1]+[6,5]) / 2\n= [4,3]','#e6f3e9')
 arrow(ax,(3.9,4.4),(5.05,3.1));arrow(ax,(3.9,3.05),(5.05,4.4));arrow(ax,(3.9,1.7),(5.05,2.8))
 ax.text(.2,.4,'Customer99: late arrival excluded → [0,0]\nCustomer105: no events → [0,0]. Same vector, different histories.',fontsize=12);save(fig,'routing')
 fig,ax=canvas('Gradient support · who can influence the current loss?',5)
 box(ax,2.8,3.65,3.5,.8,'Customer7 loss → head\narriving gradient at mean: g','#fff0d6')
 box(ax,2.8,2.15,3.5,.8,'Mean = (z₀ + z₂) / 2\neach derivative = ½ I','#e6f3e9');arrow(ax,(4.5,3.6),(4.5,3))
 box(ax,.1,.7,2.8,.8,'Event0 embedding\ngradient = g / 2');box(ax,3.1,.7,2.8,.8,'Event2 embedding\ngradient = g / 2');box(ax,6.1,.7,2.8,.8,'Excluded embeddings\ngradient = 0','#eeeeee')
 arrow(ax,(3.2,2.1),(1.5,1.55));arrow(ax,(4.5,2.1),(4.5,1.55));ax.text(.15,.05,'Shared event-encoder weights still learn from eligible events. No detach().',fontsize=11);save(fig,'gradients')
 fig,ax=plt.subplots(figsize=(9,4.2))
 ax.set_title('An old event can arrive too late · cutoff = day10',loc='left',fontweight='bold',pad=18)
 for row,(event,available) in enumerate([(1,1),(2,2),(3,3),(4,11),(12,12)]):
  y=4-row;ax.plot([event,available],[y+.10,y-.10],color='#69839a')
  ax.scatter(event,y+.10,marker='o',s=65,color='#315e7b',label='Event time' if row==0 else None)
  ax.scatter(available,y-.10,marker='s',s=65,color='#b46618',label='Availability time' if row==0 else None)
  ax.text(12.5,y,'included' if max(event,available)<=10 else 'EXCLUDED',va='center',fontsize=11)
 ax.axvline(10,color='#922b21',linestyle='--');ax.set_yticks(range(5),['event4 → C42','event3 → C99','event2 → C7','event1 → C42','event0 → C7'])
 ax.set_xlim(0,15.5);ax.set_xticks(range(1,13));ax.set_xlabel('Day (inclusive cutoff)');ax.set_ylim(-.6,4.6);ax.legend(loc='upper left',bbox_to_anchor=(0,-.22),ncol=2,fontsize=10)
 save(fig,'time')
if __name__=='__main__':build()
