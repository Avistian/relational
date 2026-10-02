"""Four portable, operation-specific figures; synthetic and measured states explicit."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;F=P/'figures/l181';F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l181'})
navy='#17324d';teal='#087f83';red='#a93d36';gold='#a66412'
def save(fig,name):
 fig.savefig(F/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});fig.savefig(F/(name+'.png'),dpi=145,bbox_inches='tight',metadata={'Software':'L181'});plt.close(fig)
def box(ax,x,y,w,h,label,color=teal):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',facecolor='#f2f6f6',edgecolor=color));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=11,color=navy)
def arrow(ax,a,b,color=teal):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=color,lw=1.7))
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for ax,title,rows in zip(axes,['Released RelBench: remove across table','RelGT-AC paper: mask query row only'],[[['query / 10','3','hidden','hidden'],['past / 9','5','hidden','hidden'],['future / 11','excluded','excluded','excluded']],[['query / 10','3','hidden','hidden'],['past / 9','5','2','18'],['future / 11','excluded','excluded','excluded']]]):
 ax.axis('off');ax.set_title(title,color=navy,fontsize=13,pad=22)
 t=ax.table(cellText=rows,colLabels=['row / time','grid','position','points'],loc='center',cellLoc='center',colWidths=[.28,.2,.25,.25]);t.auto_set_font_size(False);t.set_fontsize(11);t.scale(1,2.5)
 for (i,j),cell in t.get_celld().items():
  cell.set_edgecolor('#c9d5db');cell.set_facecolor('#dceeee' if i==0 else '#f7f9fa');cell.get_text().set_color(red if i>0 and j>1 and rows[i-1][j] in ['2','18'] else navy)
fig.suptitle('The masking location changes available information',fontsize=18,color=navy,y=1.03)
fig.text(.04,.02,'Synthetic cells; query cutoff = 10. Position is the target, points a proxy. Future rows are excluded in both policies.',fontsize=10,color=navy);save(fig,'masking')
fig,ax=plt.subplots(figsize=(12,6.5));ax.set(xlim=(0,12),ylim=(0,6.5));ax.axis('off')
ax.text(.1,6.05,'Released GNN: row features → relational prediction',fontsize=19,weight='bold',color=navy)
box(ax,.15,4.55,3.4,1.05,'F1 tables + (row ID, time)\nRemove target + named proxies\nAll rows materialized')
box(ax,4.3,4.55,3.35,1.05,'Temporal FK neighborhood\nFanouts 128 then 64\nDisjoint graph per query')
box(ax,8.4,4.55,3.35,1.05,'Typed encoders + ResNet\n4 row-network layers\nN sampled rows × 128 values')
arrow(ax,(3.65,5.08),(4.2,5.08));arrow(ax,(7.75,5.08),(8.3,5.08))
ax.text(.2,3.8,'For each relation: sum neighbor messages, combine with self, then aggregate relation outputs.',color=navy)
box(ax,.15,2.2,3.4,1.05,'Add relative-time encoding\nTwo HeteroGraphSAGE layers\n[N, 128] → [N, 128]')
box(ax,4.3,2.2,3.35,1.05,'Select B query embeddings\nLinear head: [B, 128] → [B, 1]\nPredict one position per row')
box(ax,8.4,2.2,3.35,1.05,'L1 loss → Adam updates\nValidation MAE selects epoch\nTrain-percentile clipping',gold)
ax.plot([10.1,10.1,1.85],[4.43,3.5,3.5],color=teal);arrow(ax,(1.85,3.5),(1.85,3.33));arrow(ax,(3.65,2.72),(4.2,2.72));arrow(ax,(7.75,2.72),(8.3,2.72))
ax.text(.2,1.35,'Worked sum: two neighbor messages [1, 2] + [3, 0] = [4, 2]. This is an illustrative 2-wide slice.',color=navy)
ax.text(.2,.77,'Source behavior: feature statistics use the whole graph before temporal sampling. That fit scope needs separate scrutiny.',color=red,fontsize=10)
ax.text(.2,.25,'Observed stop: numerical encoder gradients fail on missing F1 values. No full sampled GNN update or paid run was performed.',color=red,fontsize=10)
save(fig,'gnn')
fig,ax=plt.subplots(figsize=(12,7));ax.set(xlim=(0,12),ylim=(0,7));ax.axis('off')
ax.text(.1,6.6,'RelGT-AC: the paper-described prediction path',fontsize=19,weight='bold',color=navy)
ax.text(.1,6.13,'Architecture walkthrough only · authenticated implementation/checkpoints not located in bounded search',color=red,fontsize=10)
box(ax,.15,4.7,3.4,1,'Query target/proxies masked\nPast context features retained\nTask + foreign-key neighborhood')
box(ax,4.3,4.7,3.35,1,'Numeric / categorical / text\nText TF-IDF: 64 features\nText projection: 64 → 32')
box(ax,8.4,4.7,3.35,1,'Add five 128-wide terms\nFeature + table type + hop\n+ relative time + degree')
arrow(ax,(3.65,5.2),(4.2,5.2));arrow(ax,(7.75,5.2),(8.3,5.2))
box(ax,.15,2.6,3.4,1,'Local: 2 GraphSAGE layers\nSum FK-neighbor messages\n[N, 128] representations')
box(ax,4.3,2.6,3.35,1,'Global: 2 Transformer layers\n8 heads; QKᵀ / √d → softmax\nWeighted sum of value vectors')
box(ax,8.4,2.6,3.35,1,'Query embedding → task head\nRegression: scalar / MSE\nBinary: logit / BCE; multi: CE',gold)
ax.plot([10.1,10.1,1.85],[4.58,4.05,4.05],color=teal);arrow(ax,(1.85,4.05),(1.85,3.7));arrow(ax,(3.65,3.1),(4.2,3.1));arrow(ax,(7.75,3.1),(8.3,3.1))
ax.text(.2,1.79,'Worked attention slice: weights [0.2, 0.8] × values [[1, 0], [0, 1]] → [0.2, 0.8]. Synthetic, not measured.',fontsize=10.7,color=navy)
ax.text(.2,1.17,'Five terms are added into a row representation; they are not five new rows in this equation. N = sampled rows.',fontsize=10.7,color=navy)
ax.text(.2,.55,'All task-model parameters are trained. This is supervised autocomplete; fresh foundation-model pretraining is not implied.',fontsize=10.7,color=navy)
save(fig,'relgt-ac')
r=json.loads((P/'evidence/l181/report.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11.5,4.8))
for ax,(task,data) in zip(axes,r['tasks'].items()):
 vals=[data['scores']['test'][k]['mae'] for k in ['global_zero','global_mean','global_median','entity_mean','entity_median']]
 ax.barh(['Zero','Mean','Median','Entity mean','Entity median'],vals,color=[gold,teal,teal,gold,gold]);ax.invert_yaxis();ax.set_xlim(0,13);ax.set_title(task,fontsize=13,color=navy);ax.set_xlabel('Complete test MAE · lower is better')
 for i,v in enumerate(vals):ax.text(v+.15,i,f'{v:.3f}',va='center',fontsize=10)
 ax.spines[['top','right']].set_visible(False)
fig.suptitle('Measured baselines: new row identities trigger the zero fallback',fontsize=15,color=navy,y=1.01)
fig.tight_layout(rect=[0,.06,1,.95]);fig.text(.04,.01,'Deterministic recipes, no seed uncertainty. Test fitting uses train + validation. Fresh GNN results are absent.',fontsize=10,color=red);save(fig,'baselines')
print('Four portable SVG/PNG figures built')
