"""Portable, deterministic mechanism and full-seed result figures."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;F=P/'figures/l187';F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'l187'})
navy='#183440';teal='#007f82';orange='#b56828';grey='#506273'
def save(fig,name):
    fig.savefig(F/(name+'.svg'),bbox_inches='tight',metadata={'Date':None})
    fig.savefig(F/(name+'.png'),bbox_inches='tight',dpi=145,metadata={'Software':'L187'})
    plt.close(fig)
def box(ax,x,y,w,h,text,color=teal):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.015',facecolor='#f3f8f7',edgecolor=color,lw=1.5))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',color=navy,fontsize=12)
def arrow(ax,start,end,color=grey):ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','lw':1.4,'color':color})
fig,ax=plt.subplots(figsize=(10,6));ax.set(xlim=(0,10),ylim=(0,6));ax.axis('off')
ax.text(.1,5.7,'A person can occupy many graph nodes',fontsize=19,weight='bold',color=navy)
ax.text(.1,5.2,'Synthetic A: 7 owned nodes · 16 incident foreign-key edges',color=grey)
box(ax,.3,2.7,2,1,'Driver A\n1 profile')
for y,text in [(4,'3 result rows\n9 edges'),(2.6,'1 qualifying row\n3 edges'),(1.2,'2 standings rows\n4 edges')]:
    box(ax,3.5,y,2.5,1,text);arrow(ax,(3.5,y+.5),(2.3,3.2),teal);arrow(ax,(6,y+.5),(7.5,3.2))
box(ax,7.5,2.5,2.1,1.4,'Shared races\nand constructors',orange)
ax.text(.3,.5,'Delete driver node: 6 child nodes survive.\nRemove declared owner: all 7 owned nodes go.',fontsize=12,color=navy)
ax.text(7,.5,'Shared aggregates remain.\nNo erasure certification.',fontsize=11,color=orange)
save(fig,'ownership')
fig,axes=plt.subplots(1,3,figsize=(11,4.7),sharey=True)
for ax,title,h in zip(axes,['Raw results','Cap 2 per driver','Then remove A'],[[2,2],[2,1],[0,1]]):
    ax.bar(['Red','Blue'],h,color=[orange,teal],width=.58);ax.set_ylim(0,2.6);ax.set_title(title,color=navy,fontsize=14)
    for i,n in enumerate(h):ax.text(i,n+.08,str(n),ha='center',weight='bold',fontsize=14)
    ax.set_yticks([0,1,2]);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
axes[0].set_ylabel('Result rows');fig.suptitle('Bound the complete vector before adding noise',fontsize=18,color=navy)
fig.text(.05,.015,'A: Red, Red, Blue · B: Blue     |     Clipped difference [2, 0]: L1 = 2     |     ε = 1 → Laplace scale 2',fontsize=11,color=grey)
fig.tight_layout(rect=(0,.075,1,.92));save(fig,'mechanism')
fig,axes=plt.subplots(1,2,figsize=(11,4.8));grid=np.linspace(-6,8,801)
for ax,b,title in zip(axes,[2.,1.],['Correct person bound: scale 2','Wrong row bound: scale 1']):
    for mean,color,label in [(0,teal,'Without A: count 0'),(2,orange,'With A: count 2')]:
        ax.plot(grid,np.exp(-abs(grid-mean)/b)/(2*b),color=color,label=label,lw=2)
    ax.axvline(2,color=grey,ls=':',alpha=.6);ax.set_xlabel('Possible released count');ax.set_ylabel('Probability density')
    ax.set_title(title,color=navy,fontsize=13);ax.set_ylim(0,.56);ax.legend(fontsize=10,loc='upper left')
    ax.text(.04,.55,'Ratio at output 2:\n'+f'exp({2/b:g}) = {np.exp(2/b):.3f}',transform=ax.transAxes,color=navy,fontsize=11)
fig.suptitle('Privacy constrains distributions, not one sampled answer',fontsize=17,color=navy)
fig.text(.04,.015,'Synthetic neighboring counts 0 and 2; ε = 1 permits ratio at most exp(1) ≈ 2.718. The row-calibrated scale fails.',fontsize=10,color=grey)
fig.tight_layout(rect=(0,.07,1,.92));save(fig,'noise')
report=json.loads((P/'evidence/l187/report.json').read_text());releases=json.loads((P/'evidence/l187/releases.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,5.3));colors={.5:orange,1.:teal,2.:'#4667ad'}
for j,(field,title) in enumerate([('mae_clipped','Noise error against clipped counts'),('mae_raw','Total error against original counts')]):
    for offset,eps in zip([-.2,0,.2],[.5,1.,2.]):
        means=[];sds=[]
        for x,c in enumerate([1,5,20]):
            ys=[r[field] for r in releases if r['cap']==c and r['epsilon']==eps]
            axes[j].scatter(x+offset+np.linspace(-.045,.045,30),ys,s=9,alpha=.4,color=colors[eps])
            means.append(np.mean(ys));sds.append(np.std(ys,ddof=1))
        axes[j].errorbar(np.arange(3)+offset,means,yerr=sds,fmt='o-',color=colors[eps],label=f'ε = {eps:g}',capsize=4)
    axes[j].set_xticks([0,1,2],['1','5','20']);axes[j].set_xlabel('Total per-driver cap C');axes[j].set_ylabel('MAE (counts per constructor)')
    axes[j].set_title(title,fontsize=13,color=navy);axes[j].set_ylim(bottom=0);axes[j].grid(alpha=.15);axes[j].legend(fontsize=10)
fig.suptitle('Full F1 snapshot · all 270 simulation releases',fontsize=18,color=navy)
fig.text(.04,.015,'Dots: 30 seeds per configuration. Bars: mean ± sample SD. Panels use different scales; neither establishes privacy.',fontsize=10,color=grey)
fig.tight_layout(rect=(0,.07,1,.92));save(fig,'utility')
print('Saved 4 portable figures')
