"""Three portable views: signed metric contract, published catalog, fresh weakness."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent;D=P/'figures/l149';D.mkdir(parents=True,exist_ok=True)
C=json.loads((P/'evidence/l149/catalog.json').read_text());E=json.loads((P/'evidence/l149/errors.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l149','axes.spines.top':False,'axes.spines.right':False})
def save(f,name):
 f.savefig(D/f'{name}.svg',bbox_inches='tight',metadata={'Date':None});f.savefig(D/f'{name}.png',dpi=170,bbox_inches='tight');plt.close(f)
 p=D/f'{name}.svg';p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
f,ax=plt.subplots(figsize=(8.3,3.3));ax.axis('off');t=ax.table(cellText=[['AUROC','0.72','0.75','RDL − FE','−0.03'],['MAE','4.2','3.9','FE − RDL','−0.30'],['Loss penalty','4.2','3.9','RDL − FE','+0.30']],colLabels=['Measure','RDL','FE','Operation','Result'],loc='center',cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(12);t.scale(1,2.3)
for (i,j),c in t.get_celld().items():c.set_facecolor('#dcefe9' if i==0 else '#f8fafb');c.set_edgecolor('white')
ax.set_title('One loss, two signs: label the quantity before ranking',loc='left',fontweight='bold',pad=12);ax.text(.5,.02,'Illustrative numbers • advantage < 0 and loss penalty > 0 both mean RDL loses.',ha='center',transform=ax.transAxes);save(f,'contract')
f,axes=plt.subplots(2,1,figsize=(8.3,8.3));f.subplots_adjust(hspace=.6,left=.34,right=.95,top=.88,bottom=.09)
for ax,metric in zip(axes,['AUROC','MAE']):
 rows=C['rankings'][metric];scale=100 if metric=='AUROC' else 1
 values=[r['gap']*scale for r in rows];ax.barh(range(len(rows)),values,color=['#b45337' if x<0 else '#167b74' for x in values],height=.65)
 ax.set_yticks(range(len(rows)),[r['task'].replace('rel-','') for r in rows]);ax.invert_yaxis();ax.axvline(0,color='#334155');ax.grid(axis='x',alpha=.2);ax.set_xlabel('RDL advantage (AUROC points)' if metric=='AUROC' else 'RDL advantage (normalized MAE)');ax.set_title('Basic RDL · classification' if metric=='AUROC' else 'GNN + LightGBM · regression',loc='left',fontweight='bold')
f.suptitle('Published Figure 3 • mean-bar geometry, not original scores',fontweight='bold');save(f,'catalog')
f,axes=plt.subplots(2,1,figsize=(8.3,7));f.subplots_adjust(hspace=.6,left=.29,right=.96,top=.91,bottom=.1)
for ax,split in zip(axes,['val','test']):
 names=list(E['splits'][split]['slices']);sel=E['splits'][split]['selected_slice']
 for i,name in enumerate(names):
  r=E['splits'][split]['slices'][name];ci=r['interval']
  if ci:ax.errorbar(r['mean'],i,xerr=[[r['mean']-ci['low']],[ci['high']-r['mean']]],fmt='o',capsize=3,color='#b45337' if name==sel else '#167b74')
  else:ax.text(.03,i,f"unsupported ({r['rows']} queries)",va='center',fontsize=10)
 ax.set_yticks(range(len(names)),[n.replace('_',' ') for n in names]);ax.set_ylim(len(names)-.3,-.7);ax.axvline(0,color='#334155',ls='--');ax.set_title(split.upper()+' · nominated slice highlighted',loc='left');ax.set_xlabel('GNN − FE loss penalty (position units; positive = GNN loses)');ax.grid(axis='x',alpha=.15)
f.suptitle('Fresh F1 • conditional 95% driver-cluster intervals',fontweight='bold');save(f,'results')
