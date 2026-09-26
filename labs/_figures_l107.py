"""Model-specific state lattices, matrix arithmetic and measured experiments."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
P=Path(__file__).resolve().parent;O=P/'figures/l107';O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none','svg.hashsalt':'l107','axes.spines.top':False,'axes.spines.right':False})
INK='#253b48';TEAL='#087e82';ORANGE='#b65b2c';GRAY='#61747b';PAPER='#fbfcf9'
def canvas(title,sub,h=6):
 f,a=plt.subplots(figsize=(11,h));f.patch.set_facecolor(PAPER);a.set(xlim=(0,11),ylim=(0,h));a.axis('off');a.text(.15,h-.35,title,fontsize=19,weight='bold',color=INK);a.text(.15,h-.73,sub,fontsize=11,color=GRAY);return f,a
def txt(a,x,y,s,**kw):return a.text(x,y,s,**({"color":INK}|kw))
def arrow(a,x,y,u,v,color=TEAL):a.annotate('',xy=(u,v),xytext=(x,y),arrowprops={'arrowstyle':'->','color':color,'lw':1.8})
def mat(a,x,y,values,label,color=TEAL,w=.55,h=.44):
 values=np.array(values);nr,nc=values.shape
 for i in range(nr):
  for j in range(nc):
   xx=x+j*w;yy=y-i*h;a.add_patch(Rectangle((xx,yy-h),w,h,facecolor='#e4f0eb' if (i+j)%2==0 else 'white',edgecolor=color,lw=.7));txt(a,xx+w/2,yy-h/2,str(values[i,j]),ha='center',va='center',fontsize=10)
 txt(a,x,y+.16,label,fontsize=11,color_=color) if False else a.text(x,y+.16,label,fontsize=11,color=color)
 return nc*w,nr*h

def save(f,name):
 f.canvas.draw();renderer=f.canvas.get_renderer()
 for a in f.axes:
  if not a.axison:
   bounds=a.get_window_extent()
   boxes=[p for p in a.patches if isinstance(p,Rectangle)]
   for i,b in enumerate(boxes):
    x,y=b.get_xy();w,h=b.get_width(),b.get_height();assert x>=0 and y>=0 and x+w<=11 and y+h<=a.get_ylim()[1],(name,'box outside canvas')
    for other in boxes[i+1:]:
     xx,yy=other.get_xy();ww,hh=other.get_width(),other.get_height()
     assert not (min(x+w,xx+ww)-max(x,xx)>1e-8 and min(y+h,yy+hh)-max(y,yy)>1e-8),(name,'overlapping filled boxes')
   for t in a.texts:
    b=t.get_window_extent(renderer)
    assert b.x0>=bounds.x0-2 and b.x1<=bounds.x1+2 and b.y0>=bounds.y0-2 and b.y1<=bounds.y1+2,(name,t.get_text())
 f.savefig(O/f'{name}.svg',bbox_inches='tight',metadata={'Date':None});f.savefig(O/f'{name}.png',dpi=160,bbox_inches='tight');plt.close(f)

f,a=canvas('Two temporal states, two different axes of identity','Architecture map · k indexes completed snapshots · supervised pair prediction follows the last usable graph',8.6)
txt(a,.2,7.45,'GCN → node-state GRU',fontsize=14,weight='bold');txt(a,.2,7.08,'Aₖ [N,N], Xₖ [N,F] → Zₖ = GCN(Aₖ,Xₖ) [N,d] → Sₖ = GRU(Zₖ,Sₖ₋₁)',fontsize=12)
for i,x in enumerate([.55,3.05,5.55]):
 txt(a,x,6.62,'snapshot '+str(i),fontsize=11,color=GRAY)
 mat(a,x,6.2,[['•','•'],['•','•'],['•','•']],'S'+str(i)+' [N,d]',w=.57,h=.38)
 for j,label in enumerate(['u','v','w']):txt(a,x-.17,6.01-j*.38,label,ha='right',va='center',fontsize=10)
 if i<2:arrow(a,x+1.5,5.65,x+2.35,5.65)
 txt(a,x,4.88,'rows retain node IDs',fontsize=10)
arrow(a,7.1,5.65,7.7,5.65);txt(a,7.85,6.1,'gather u and v',fontsize=12);txt(a,7.85,5.68,'[S[u] ‖ S[v]] [B,2d]',fontsize=11);txt(a,7.85,5.25,'MLP → logits [B,2]',fontsize=11);txt(a,7.85,4.82,'softmax → score',fontsize=11)
a.plot([.2,10.7],[4.35,4.35],color='#c5d6d4',lw=1)
txt(a,.2,3.95,'EvolveGCN → weight-state recurrence',fontsize=14,weight='bold');txt(a,.2,3.57,'H: summarize node rows → Qₖ [F,d] → Wₖ = matrixGRU(Qₖ,Wₖ₋₁)',fontsize=12)
for i,x in enumerate([.55,3.05,5.55]):
 mat(a,x,2.97,[['•','•'],['•','•']],'W'+str(i)+' [F,d]',color=ORANGE,w=.57,h=.38)
 if i<2:arrow(a,x+1.3,2.62,x+2.34,2.62,ORANGE)
 arrow(a,x+.56,2.12,x+.56,1.75,ORANGE);txt(a,x-.02,1.47,'Âₖ Xₖ Wₖ → Zₖ',fontsize=11)
arrow(a,7.1,1.55,7.7,1.55);txt(a,7.85,2.02,'[Z[u] ‖ Z[v]]',fontsize=12);txt(a,7.85,1.6,'same pair-head pattern',fontsize=10);txt(a,7.85,1.18,'forecast next snapshot',fontsize=10)
txt(a,.2,.65,'Training: cross-entropy updates GCN, recurrence and pair head. Inference: freeze parameters; evolve states.',fontsize=11)
txt(a,.2,.23,'Course GCN-GRU: two graph layers, d=32. EvolveGCN: repeat the weight recurrence separately for each layer.',fontsize=10,color=GRAY);save(f,'architecture')

f,a=canvas('One spatial update: follow node B','Synthetic path A—B—C · self-loops included · X = [1,0,2]ᵀ · W = 1',5.9)
for i,(label,val) in enumerate([('A',1),('B',0),('C',2)]):
 x=.7+i*1.25
 if i<2:a.plot([x+.3,x+.95],[4.2,4.2],color=TEAL,lw=2)
 a.add_patch(Circle((x,4.2),.3,facecolor='white',edgecolor=TEAL,lw=2));txt(a,x,4.2,label,ha='center',va='center');txt(a,x,3.62,'x='+str(val),ha='center',fontsize=11)
mat(a,4.4,4.75,[['1/2','1/√6','0'],['1/√6','1/3','1/√6'],['0','1/√6','1/2']],'normalized Â [3,3]',w=.82,h=.57)
arrow(a,7.1,3.9,7.8,3.9);mat(a,8.1,4.65,[['0.5000'],['1.2247'],['1.0000']],'ÂX [3,1]',w=1.3,h=.57)
txt(a,.4,2.7,'B coordinate',weight='bold',fontsize=14);txt(a,.4,2.13,'(1/√6 × 1)  +  (1/3 × 0)  +  (1/√6 × 2)  =  1.2247',fontsize=16,color=TEAL)
txt(a,.4,1.36,'Degrees after self-loops: [2, 3, 2]. Row B sums to 1.1498, not one.',fontsize=12)
txt(a,.4,.7,'A row-mean implementation gives B = 1.0000: a different operator.',fontsize=12,color=ORANGE)
txt(a,.4,.24,'Rounded to four decimals. Spatial propagation alone carries no recurrent state.',fontsize=11,color=GRAY);save(f,'normalization')


f,a=canvas('Make the node summary fit the weight matrix','Synthetic H summary · N=4, F=2, d=2 · scorer p=[1,0]ᵀ, so its norm is 1',6)
mat(a,.5,4.72,[['0.2','1.0'],['1.0','2.0'],['−0.5','3.0'],['0.7','4.0']],'X [4,2]',w=.75,h=.52)
for i,label in enumerate(['A','B','C','D']):txt(a,.35,4.46-i*.52,label,ha='right',va='center',fontsize=11)
arrow(a,2.2,3.7,2.85,3.7)
mat(a,3.05,4.72,[['0.2'],['1.0 ← B'],['−0.5'],['0.7 ← D']],'Xp / ‖p‖ [4,1]',w=1.35,h=.52)
arrow(a,4.62,3.7,5.23,3.7)
mat(a,5.45,4.4,[['0.7616','1.5232'],['0.4231','2.4175']],'top-2 weighted rows',w=.94,h=.62)
arrow(a,7.52,3.7,8.11,3.7)
mat(a,8.37,4.4,[['0.7616','0.4231'],['1.5232','2.4175']],'Q = transpose [2,2]',w=1.01,h=.62)
txt(a,.5,1.96,'B: [1,2] × tanh(1) = [0.7616, 1.5232]',fontsize=13,color=TEAL)
txt(a,.5,1.39,'D: [0.7,4] × tanh(0.7) = [0.4231, 2.4175]',fontsize=13,color=TEAL)
txt(a,.5,.79,'Select by score, preserve ranked order, then transpose: Q now matches W [F,d].',fontsize=12)
txt(a,.5,.25,'All four nodes are active in this example. Mask inactive nodes before top-k selection.',fontsize=11,color=GRAY);save(f,'summary')

f,a=canvas('One weight update: gate the proposed matrix','Illustrative supplied gate/candidate values · not a trained model · ⊙ means entrywise multiplication',5.5)
mat(a,.4,4.12,[['0.2','0.4'],['0.6','0.8']],'old W [2,2]',w=.8,h=.6)
txt(a,2.35,3.52,'× 0.75',fontsize=16);txt(a,4,3.52,'+',fontsize=23)
mat(a,4.7,4.12,[['0.8','0.8'],['0.8','0.8']],'candidate C [2,2]',w=.8,h=.6)
txt(a,6.63,3.52,'× 0.25',fontsize=16);txt(a,8.17,3.52,'=',fontsize=23)
mat(a,8.8,4.12,[['0.35','0.50'],['0.65','0.80']],'new W [2,2]',w=.8,h=.6)
txt(a,.4,2.23,'W_new = (1 − z) ⊙ W_old + z ⊙ C',fontsize=18,color=TEAL)
txt(a,.4,1.5,'Here z = 0.25 everywhere. The top-left weight moves from 0.20 to 0.35.',fontsize=12)
txt(a,.4,.87,'The new matrix acts inside ÂXW_new; it is not a node embedding table.',fontsize=12)
txt(a,.4,.27,'H learns z and C from Q and W_old. Released O substitutes W_old for Q.',fontsize=11,color=GRAY);save(f,'recurrence')

w=json.loads((P/'_history_witness_l107_results.json').read_text());f,a=canvas('Does the early snapshot reach the final output?','Measured deterministic mechanism witness · same final graph/features · fixed mean-slope activation',5.3)
for i,x in enumerate([.55,2.05,3.55,5.05,6.55,8.05]):
 a.add_patch(Rectangle((x,3.45),1.04,.75,edgecolor=ORANGE if i==0 else TEAL,facecolor='#f5e7df' if i==0 else '#e4f0eb'));txt(a,x+.52,3.82,'t−'+str(5-i) if i<5 else 't',ha='center',va='center');
 if i<5:arrow(a,x+1.07,3.82,x+1.43,3.82)
txt(a,.55,3.03,'change X only here',fontsize=10,color=ORANGE);txt(a,7.7,3.03,'hold final X and A fixed',fontsize=10)
txt(a,.55,2.3,'H: max output change = '+f"{w['max_output_change']['H']:.6f}",fontsize=16,color=TEAL)
txt(a,.55,1.66,'Released O: max output change = 0',fontsize=16,color=ORANGE)
txt(a,.55,.94,'O reinitializes W for every query. Six observation-free steps give the same effective final weights.',fontsize=11)
txt(a,.55,.34,'This is a structural witness, not an accuracy comparison or the stochastic paper replay.',fontsize=11,color=GRAY);save(f,'history')

path=P/'_analysis_l107_results.json'
if path.exists():
 r=json.loads(path.read_text());f,axes=plt.subplots(1,2,figsize=(11,4.8));f.patch.set_facecolor(PAPER)
 for i,(arm,color) in enumerate([('3600',TEAL),('86400',ORANGE),('tgn',INK)]):
  values=[row['test']['ap'] for row in r['wiki'][arm]['runs']];axes[0].scatter([i-.1,i,i+.1],values,color=color);axes[0].errorbar(i,np.mean(values),yerr=np.std(values,ddof=1),color=color,capsize=5)
 axes[0].set(xticks=[0,1,2],xticklabels=['Hourly','Daily','TGN'],ylabel='Pooled test AP',ylim=(0,1),title='Wikipedia · three independent seeds');axes[0].grid(axis='y',alpha=.2)
 for i,(v,color) in enumerate([('H',TEAL),('O',ORANGE)]):
  row=r['sbm'].get(v)
  if row is None:continue
  axes[1].scatter(i,row['test']['map'],color=color,label=v+' release replay');axes[1].scatter(i,row['target']['map'],marker='x',color=color,s=70)
 axes[1].set(xticks=[0,1],xticklabels=['H','Released O'],ylabel='Mean snapshot AP',xlim=(-.4,1.4),ylim=(.16,.22),title='SBM · detail scale 0.16–0.22');axes[1].grid(axis='y',alpha=.2)
 f.text(.07,.03,'Left: points = seeds, bars = ±1 sample SD (not CI). Right: crosses = paper targets; protocols differ.',fontsize=10);f.tight_layout(rect=(0,.08,1,1));save(f,'results')
print('L107 figures built')
