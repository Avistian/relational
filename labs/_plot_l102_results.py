"""Plot only the complete selected experiment, with shared axes and explicit detail scale."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;r=json.loads((P/'_paper_l102_results.json').read_text());assert r['status']=='COMPLETE'
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','svg.hashsalt':'l102-results','font.size':11})
fig,axes=plt.subplots(1,2,figsize=(10,4.4),sharey=True,layout='constrained');fig.patch.set_facecolor('#fcfbf7')
all_values=[x[key]['ap']*100 for x in r['records'] for key in ['test','new_test']]
lo=np.floor((min(all_values)-.25)*2)/2;hi=np.ceil((max(all_values)+.25)*2)/2
for ax,(lane,key,label,color) in zip(axes,[('all','test','All test events','#196b70'),('new','new_test','New-node event subset','#a96b20')]):
 x=np.array([z['seed'] for z in r['records']]);y=np.array([z[key]['ap']*100 for z in r['records']]);s=r['summary'][lane]
 ax.set_facecolor('#fcfbf7');ax.scatter(x,y,s=45,color=color,zorder=3,label='Measured seed')
 ax.axhline(s['paper_ap_percent'],color=color,ls='--',lw=1.5,label=f"Paper {s['paper_ap_percent']:.2f}%")
 ax.set_title(f"{label}\nMean {s['mean_ap_percent']:.3f}% · SD {s['sample_sd_pp']:.3f} pp",fontsize=12)
 ax.set_xticks(range(10));ax.set_xlabel('Initialization seed');ax.set_ylim(lo,hi);ax.grid(axis='y',alpha=.2)
 ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=9,loc='lower left')
axes[0].set_ylabel('AP (%) — detail scale')
fig.suptitle('Complete Wikipedia TGN-attn replay / ten GPU runs',fontsize=15,fontweight='bold')
fig.savefig(P/'figures/l102/results.svg',metadata={'Date':None});fig.savefig(P/'figures/l102/results.png',dpi=160)
print('Complete-run figure generated')
