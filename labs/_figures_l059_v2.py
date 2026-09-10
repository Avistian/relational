"""Portable computational traces, historical control and fresh protocol evidence."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures/l059';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fbfaf6','axes.facecolor':'#fbfaf6'})
def finish(fig,name):fig.savefig(OUT/name,dpi=170,bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(9,3.6));ax.axis('off')
t=ax.table(cellText=[['.4','.4','.4','.25'],['.4','.6','.4','.25'],['.6','.4','.4','.25'],['.6','.6','.6','.25']],colLabels=['Candidate A','Candidate B','Selected min','Probability'],cellLoc='center',bbox=[0,.22,1,.68]);t.auto_set_font_size(False);t.set_fontsize(14)
ax.set_title('Predict the expected minimum before averaging',loc='left',pad=15)
ax.text(.02,.03,'E[min] = (.4 + .4 + .4 + .6) / 4 = .45    |    Fresh risk = .50',transform=ax.transAxes,fontsize=14)
finish(fig,'null-minimum.png')

r=json.loads((ROOT/'_verify_l059_v2_results.json').read_text());s=r['historical_null'];fig,ax=plt.subplots(figsize=(9,4))
x=[v['candidates'] for v in s]
ax.plot(x,[v['validation_mean'] for v in s],'o-',label='Historical selected validation')
ax.plot(x,[v['exact_validation'] for v in s],'x--',label='Exact independent-binomial expectation')
ax.plot(x,[v['test_mean'] for v in s],'o-',label='Historical independent test')
ax.axhline(.5,color='gray',ls=':');ax.set(xscale='log',ylim=(.30,.53),xlabel='Candidate budget (paired prefixes)',ylabel='Classification error',title='80 validation rows · 2,000 test rows · 200 independent repeats')
ax.set_xticks(x,labels=x);ax.legend(fontsize=10,loc='lower left');finish(fig,'null-results-v2.png')

fig,ax=plt.subplots(figsize=(10,4.6));ax.axis('off')
lines=[('Input pair','x=(.4,.7), z=(−.3,.7); η=(2,.25)'),('Similarity','2×(.7)² + .25×0² = .98 → exp(−.98) = .3753'),('Fit on n rows','C=[K+λI, 1; 1ᵀ, 0]   (n+1)×(n+1)'),('Coefficients','C[α; b]=[y;0] → α[n], b[1]; sum(α)=0'),('Select λ and η','rᵢ = αᵢ / (C⁻¹)ᵢᵢ → PRESS/n = mean(r²) → argmin'),('New rows','X*[m,2] → K(X*,X)[m,n] × α[n] + b → scores[m]'),('Classify','Score ≥0 → +1; score <0 → −1 (not probabilities)')]
t=ax.table(cellText=lines,colLabels=['Operation','Numbers, shapes and information'],cellLoc='left',colWidths=[.23,.77],bbox=[0,0,1,.95]);t.auto_set_font_size(False);t.set_fontsize(12)
ax.set_title('The paper’s KRR + unpenalized intercept + LOO instrument',loc='left',pad=10);finish(fig,'kernel-v2.png')

fig,ax=plt.subplots(figsize=(11,5.6));ax.axis('off')
rows=[['1 · candidate choice','Only 48 development labels','All 64 labels (includes held 16)'],['2 · inner fit size','LOO: 47 rows, score deleted row','LOO: 63 rows, score deleted row'],['3 · freeze choice','New selected index in each fold','One global index for all folds'],['4 · outer coefficient fit','48 rows → α and b','Same 48 rows → α and b'],['5 · outer MSE','Predict held 16; average 4 folds','Predict 16 used in selection'],['6 · fresh target','Each 48-row fit → fresh 4,096 rows','Same 48-row fit → same fresh rows'],['7 · paired optimism','Fresh MSE − internal outer MSE','Fresh MSE − external outer MSE']]
t=ax.table(cellText=rows,colLabels=['Step','Internal selection','Contaminated external selection'],cellLoc='left',colWidths=[.24,.37,.39],bbox=[0,.14,1,.83]);t.auto_set_font_size(False);t.set_fontsize(11)
ax.set_title('Trace a 64-row repetition: the selector is inside the evaluated procedure',loc='left',pad=15)
ax.text(0,.01,'Four folds overlap in training → average them first. Repeat-level differences are the independent units.\nChanging held 16 labels must not change their internal choice/predictions. Fresh rows are drawn after choices freeze.',transform=ax.transAxes,fontsize=11)
finish(fig,'protocol-v2.png')

closer=json.loads((ROOT/'_verify_l059_closer_results.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4.6))
keys=['selected_loo_optimism','internal_optimism','external_optimism'];labels=['Selected LOO\n(63 vs 64 rows)','Internal outer\n(matched 48)','External outer\n(matched 48)']
for ax,data,title in zip(axes,[r,closer],['30-repeat teaching run','1,000-repeat precision track']):
 for i,k in enumerate(keys):
  v=data['summary'][k];ax.errorbar(i,v['mean'],yerr=[[v['mean']-v['t95'][0]],[v['t95'][1]-v['mean']]],fmt='o',capsize=5)
 ax.axhline(0,color='gray',ls=':');ax.set_xticks(range(3),labels,fontsize=10);ax.set_title(title);ax.set_ylim(-.04,.13);ax.set_ylabel('Fresh MSE − reported MSE')
fig.suptitle('Paired t95 intervals over independent synthetic dataset draws',fontsize=14);fig.tight_layout();finish(fig,'nested-results-v2.png')

# Connected architecture, retaining the numerical operation table separately.
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
fig,ax=plt.subplots(figsize=(11,7.5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.text(.02,.97,'Fit and select on development rows only · repeat for 27 declared candidates',fontsize=13,weight='bold',va='top')
boxes={
 'data':(.02,.71,.27,.16,'Development data\nX[n,2], y[n]\nCandidate λ>0, η[2]>0'),
 'matrix':(.36,.71,.28,.16,'K[n,n] = ARD similarities\nC = [K+λI, 1; 1ᵀ, 0]\nC[n+1,n+1]'),
 'solve':(.71,.71,.27,.16,'Solve C[α;b]=[y;0]\nα[n], b; sum(α)=0\nRead diag(C⁻¹)[:n]'),
 'loo':(.71,.43,.27,.16,'Deleted residuals r[n]\nrᵢ=αᵢ/(C⁻¹)ᵢᵢ\nMean PRESS = mean(r²)'),
 'choose':(.36,.43,.28,.16,'Compare all 27 PRESS means\nChoose first minimum\nFreeze selected model'),
 'frozen':(.02,.43,.27,.16,'Selected fitted state\nTraining X, η, α and b\nNo query labels used'),
 'query':(.02,.09,.27,.17,'New features X*[m,2]\nwith frozen training X, η\nCross-kernel K*[m,n]'),
 'scores':(.36,.09,.28,.17,'K*α + b → scores[m]\nReal values, not probabilities\nHeld-label evaluation → MSE'),
 'head':(.71,.09,.27,.17,'Classification head\nscore ≥0 → +1\nscore <0 → −1')}
for i,a in enumerate(boxes.values()):
 for b in list(boxes.values())[i+1:]:
  assert a[0]+a[2]<=b[0] or b[0]+b[2]<=a[0] or a[1]+a[3]<=b[1] or b[1]+b[3]<=a[1], 'Architecture boxes overlap'
for name,(x,y,w,h,label) in boxes.items():
 color='#e4edf4' if name in ['query','scores','head'] else '#eef1e4'
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.007',facecolor=color,edgecolor='#536571'))
 ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=11)

def arrow(a,b,side):
 x,y,w,h,_=boxes[a];xx,yy,ww,hh,_=boxes[b]
 if side=='right':start=(x+w+.01,y+h/2);end=(xx-.012,yy+hh/2)
 elif side=='left':start=(x-.01,y+h/2);end=(xx+ww+.012,yy+hh/2)
 else:start=(x+w/2,y-.01);end=(xx+ww/2,yy+hh+.012)
 ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=18,color='#536571',lw=2))
for a,b,side in [('data','matrix','right'),('matrix','solve','right'),('solve','loo','down'),('loo','choose','left'),('choose','frozen','left'),('frozen','query','down'),('query','scores','right'),('scores','head','right')]:arrow(a,b,side)
ax.axhline(.345,color='#a2a2a2',ls='--',lw=1)
ax.text(.34,.32,'Inference · frozen candidate and coefficients',fontsize=13,weight='bold',va='top')
ax.text(.02,.015,'Arrow order: fit → deleted-row score → select → freeze → cross-kernel query → score → optional sign head.',fontsize=11)
fig.canvas.draw()
# Each label remains within the drawing; all boxes are disjoint by construction.
renderer=fig.canvas.get_renderer();canvas=ax.get_window_extent(renderer)
for label in ax.texts:
 b=label.get_window_extent(renderer)
 assert b.x0>=canvas.x0-1 and b.x1<=canvas.x1+1 and b.y0>=canvas.y0-1 and b.y1<=canvas.y1+1, label.get_text()
finish(fig,'architecture-v2.png')
