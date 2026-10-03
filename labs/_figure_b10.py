"""Portable mechanism diagrams; no invented benchmark bars."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from _test_b10 import fixture,oracle
P=Path(__file__).resolve().parent;D=P/'figures/b10';D.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(3,2,figsize=(12,10));fig.patch.set_facecolor('#f6f9fb')
steps=[('1  Build the question','Customer 10 ← task row 100\nTarget y = hidden; time = day 10\nPast task row 101: y = 1 at day 2\nTask rows become an ordinary table.'),('2  Admit the evidence','Traverse row links; impose context cap.\nReject later events before tokenization.\nMask query target; keep eligible past labels.\nIDs define links, not ordered magnitudes.'),('3  Encode each cell','Visible: normalized type projection\nHidden: learned type-specific mask vector\n+ projected name embedding\nS cells × D coordinates (paper D = 256)'),('4  Mix inside each block','Column → feature → neighbor → full\nEach: RMSNorm → masked attention → add\nThen RMSNorm → SwiGLU → add\nRepeat 12 blocks in the paper model.'),('5  Read the masked position','Final RMSNorm → datatype decoder\nBoolean logit z → sigmoid(z) = p\nNumeric/datetime head → scalar\nOnly the target position is evaluated.'),('6  Train or evaluate','Training: masked BCE / Huber mean\nBackward → optimizer → new weights\nEvaluation: fixed weights, keyed AUROC\nB10 checks mechanisms; no paper inference.')]
for ax,(title,body) in zip(axes.flat,steps):
 ax.set_facecolor('white');ax.set_xticks([]);ax.set_yticks([])
 for spine in ax.spines.values():spine.set_color('#c8d6de')
 ax.text(.05,.87,title,transform=ax.transAxes,weight='bold',size=14,color='#174761')
 ax.text(.05,.67,body,transform=ax.transAxes,va='top',linespacing=1.8,size=12)
fig.suptitle('Relational Transformer · from database cells to a prediction',size=19,weight='bold',y=.98)
fig.tight_layout(rect=(0,0,1,.95));fig.savefig(D/'architecture.png',dpi=150);plt.close(fig)
names=['T0.y','T0.t','C0.age','C0.name','O0.amt','O0.t','T1.y','T1.t','C1.age']
fig,axes=plt.subplots(2,2,figsize=(10,10));m=oracle(fixture())
for ax,k,title in zip(axes.flat,['col','feat','nbr','full'],['Column: same table + column','Feature: same row + parents','Neighbor: children only','Full: all admitted cells']):
 ax.imshow(m[k][0,:9,:9],cmap='Blues',vmin=0,vmax=1);ax.set_title(title,size=13,pad=10)
 ax.set_xticks(range(9),names,rotation=55,ha='right');ax.set_yticks(range(9),names);ax.set_xlabel('Source cell');ax.set_ylabel('Reader cell')
 ax.set_xticks(np.arange(-.5,9,1),minor=True);ax.set_yticks(np.arange(-.5,9,1),minor=True);ax.grid(which='minor',color='white');ax.tick_params(which='minor',bottom=False,left=False)
fig.suptitle('One context, four permissions · blue means allowed',size=17);fig.tight_layout(rect=(0,0,1,.96));fig.savefig(D/'masks.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,4.8));ax.axvline(10,color='#b25043',lw=2,label='Prediction cutoff: day 10')
for y,(name,event,arrival) in enumerate([('Order',7,7),('Old task label (3-day window)',8,11),('Scheduled race',12,9)]):
 ax.plot([arrival,event],[y,y],color='#718497',lw=3);ax.scatter([event],[y],s=110,marker='o',color='#174761');ax.scatter([arrival],[y],s=100,marker='D',color='#2c8a71');ax.text(6.05,y+.2,name,size=12)
ax.scatter([],[],marker='o',color='#174761',label='Event / task timestamp');ax.scatter([],[],marker='D',color='#2c8a71',label='Recorded availability')
ax.set_xlim(6,14);ax.set_ylim(-.5,3);ax.set_yticks([]);ax.set_xlabel('Illustrative day (not the F1 audit dates)');ax.legend(loc='upper right',ncol=1,fontsize=10);ax.set_title('A timestamp is not an arrival log',size=17,pad=12)
fig.tight_layout();fig.savefig(D/'visibility.png',dpi=150);plt.close(fig)
print('Wrote three portable figures')
