"""L060 computation and evidence figures, only from current corrected artifacts."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from relkit.comparison_l060 import audit_records,paired_effect
from relkit.rank_audit_l060 import rank_permutation
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures/l060';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'#fafcfb'})
colors=['#2668a4','#267a52','#8b5296','#ba681b','#bb4055']

def save(fig,name):fig.savefig(OUT/name,dpi=125,bbox_inches='tight');plt.close(fig)

def protocol():
 fig,ax=plt.subplots(figsize=(9,7));ax.set(xlim=(0,10),ylim=(0,10));ax.axis('off')
 def box(x,y,w,h,title,detail,c):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor=c,edgecolor='#546675'))
  ax.text(x+w/2,y+h*.72,title,ha='center',va='center',weight='bold',fontsize=11.8)
  ax.text(x+w/2,y+h*.33,detail,ha='center',va='center',fontsize=11)
 def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.6,'color':'#384d60'})
 box(.3,8,4.1,1.3,'TRAIN IDs → frozen encoder','fit medians + one-hot vocabulary','#e2f1e9')
 box(5.5,8,4.1,1.3,'VALIDATION IDs','transform with training state','#e6eef8')
 box(.3,5.5,4.1,1.6,'Two candidate procedures','fit on train; restore validation epoch\nno test argument','#e2f1e9')
 box(5.5,5.5,4.1,1.6,'Validation errors [.43, .39]','first argmin → candidate 1\nfreeze predictor + selected epoch','#e6eef8')
 arrow((2.35,8),(2.35,7.1));arrow((7.55,8),(7.55,7.1));arrow((4.5,6.3),(5.4,6.3))
 ax.plot([0,10],[4.75,4.75],ls='--',color='#ba681b');ax.text(.3,4.87,'CHOICE FROZEN: evaluation labels become eligible only below',fontsize=11,color='#8d4c16')
 box(.3,2.5,4.1,1.6,'TEST features → probabilities','frozen encoder + candidate 1\nP(y=1) = [.8, .3]','#f7efd9')
 box(5.5,2.5,4.1,1.6,'TEST labels [1, 0] → metric','row IDs join labels to probabilities\nmean(−ln .8, −ln .7) = .28991','#f7efd9')
 ax.plot([7.55,7.55,2.35],[5.4,4.35,4.35],color='#384d60',lw=1.6);arrow((2.35,4.35),(2.35,4.1));arrow((4.5,3.3),(5.4,3.3))
 box(1.2,.25,7.6,1.3,'Saved artifact → independent reconstruction','IDs + targets + predictions + candidate errors + costs\n3 seeds → task mean → 1 dataset rank block','#ece8f2');arrow((7.55,2.4),(6.6,1.65))
 fig.suptitle('Trace the allowed labels and one metric contribution',fontsize=15,weight='bold');save(fig,'protocol-v2.png')
 fig,ax=plt.subplots(figsize=(8,3));ax.axis('off');table=ax.table(cellText=[['A','0.00','1.00','0.50','2','1.5'],['B','0.40','0.40','0.40','1','1.5']],colLabels=['Arm','Seed 0','Seed 1','Mean loss','Rank of mean','Mean seed rank'],loc='center');table.auto_set_font_size(False);table.set_fontsize(11);table.scale(1,2)
 ax.set_title('Same errors, different order of operations',weight='bold',pad=20);ax.text(.5,.05,'Average errors → B wins. Average seed ranks → a tie.\nTwo seeds still represent one task.',ha='center',transform=ax.transAxes);save(fig,'aggregation-v2.png')

def results():
 result=json.loads((ROOT/'_verify_l060_v2_results.json').read_text());assert result['status']=='COMPLETE'
 summary=audit_records(result);arms=result['design']['arms'];cal={reg:rank_permutation(list(s['dataset_ranks'].values())) for reg,s in summary.items()}
 pairs=[dict(dataset=d,arm=a,**paired_effect(result['records'],d,a,'XGBoost')) for d in result['datasets'] for a in arms if a!='XGBoost']
 for regime in summary:
  datasets=result['design']['panels'][regime];fig,axs=plt.subplots(len(datasets),1,figsize=(8,2.4*len(datasets)),squeeze=False)
  for ax,d in zip(axs[:,0],datasets):
   for i,a in enumerate(arms):
    values=np.array([r['error'] for r in result['records'] if r['dataset']==d and r['arm']==a]);ax.scatter(values,np.full(3,i)+[-.12,0,.12],color=colors[i],s=28,alpha=.75);ax.scatter(values.mean(),i,marker='|',s=230,color='#152b3b',zorder=3)
   ax.set_yticks(range(5),arms);ax.invert_yaxis();ax.set_title(d,loc='left',fontsize=13,weight='bold');ax.set_xlabel('RMSE of released log target ↓' if 'sberbank' in d else 'Binary log loss ↓');ax.grid(axis='x',alpha=.2)
  fig.suptitle(f'Corrected L060 v2 · {regime} · 3 seed points + mean bar',fontsize=15,y=1);fig.tight_layout();save(fig,f'scores-{regime}-v2.png')
 fig,axs=plt.subplots(2,1,figsize=(9,7))
 for ax,(reg,s) in zip(axs,summary.items()):
  order=sorted(arms,key=s['mean_ranks'].get)
  for i,a in enumerate(order):ax.scatter(s['mean_ranks'][a],i,s=70,color=colors[arms.index(a)]);ax.text(s['mean_ranks'][a]+.1,i,f"{s['mean_ranks'][a]:.3f}",va='center')
  ax.set_yticks(range(5),order);ax.invert_yaxis();ax.set_xlim(.8,5.8);ax.set_xlabel('Mean dataset rank ↓');ax.set_title(f"{reg}: N={s['datasets']}; permutation p={cal[reg]['p']:.4f}",loc='left',pad=22);ax.grid(axis='x',alpha=.2)
  ax.plot([1,1+s['nemenyi_cd']],[-.85,-.85],color='#384d60',lw=2);ax.text(1,-1.02,f"Exploratory CD = {s['nemenyi_cd']:.3f}",fontsize=10)
 fig.suptitle('Rank differences are not raw loss effects',weight='bold');fig.tight_layout();save(fig,'ranks-v2.png')
 texts=[]
 for reg,s in summary.items():
  winner=min(s['mean_ranks'],key=s['mean_ranks'].get);texts.append(f"{reg.capitalize()}: {s['datasets']} blocks; {winner} has the smallest observed mean rank ({s['mean_ranks'][winner]:.3f}). Exploratory chi-square p={s['friedman_p']:.4f}; conditional permutation p={cal[reg]['p']:.4f}.")
 interpretation=' '.join(texts)+' These are corrected local measurements, not reproduced paper rankings. The random calibration uses 20,000 draws; the temporal calibration is exact under the stated method-label exchangeability null. Neither panel establishes universal superiority.'
 report=dict(status='PASS',evidence_sha256=hashlib.sha256((ROOT/'_verify_l060_v2_results.json').read_bytes()).hexdigest(),summary=summary,paired_effects=pairs,permutation=cal,interpretation=interpretation,record_count=len(result['records']),analysis_source_sha256=hashlib.sha256((ROOT/'relkit/rank_audit_l060.py').read_bytes()).hexdigest())
 (ROOT/'_analysis_l060_v2_results.json').write_text(json.dumps(report,indent=2)+'\n');print(interpretation)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--protocol-only',action='store_true');a=p.parse_args();protocol()
 if not a.protocol_only:results()
