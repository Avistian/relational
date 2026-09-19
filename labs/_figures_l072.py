"""Model-specific computational diagrams; SVG originals and portable PNG copies."""
from pathlib import Path
import html,json
import cairosvg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures/l072'
INK='#18354a';TEAL='#007f78';ORANGE='#ba4d26';MUTED='#526674'

def text(x,y,s,size=18,color=INK,weight='normal',anchor='start'):
    return f'<text x="{x}" y="{y}" font-family="DejaVu Sans,sans-serif" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(str(s))}</text>'
def rect(x,y,w,h,fill='#ffffff',stroke='#cbd7da',r=12):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}"/>'
def line(x,y,a,b,color=TEAL,dashed=False):
    return f'<path d="M{x},{y} L{a},{b}" stroke="{color}" stroke-width="2.5" fill="none" marker-end="url(#arrow)"'+(' stroke-dasharray="6 5"' if dashed else '')+'/>'
def cellrow(x,y,values,high=(),width=54):
    s=''
    for i,v in enumerate(values):
        s+=rect(x+i*width,y,width-3,37,'#fbe4d9' if i in high else '#e6f2ef',r=4)+text(x+i*width+(width-3)/2,y+25,v,16,anchor='middle')
    return s

def save(name,body,height):
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {height}"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{TEAL}"/></marker></defs><rect width="1200" height="{height}" fill="#f7f5ef"/>'+body+'</svg>'
    (OUT/f'{name}.svg').write_text(svg)
    cairosvg.svg2png(bytestring=svg.encode(),write_to=str(OUT/f'{name}.png'),output_width=1600)

def architecture_scarf():
    s=text(40,45,'SCARF · learn which two rows belong together',29,weight='bold')
    s+=text(40,78,'Local numeric model  |  B rows × d features  |  example row: d = 5, corruption c = 0.6',17,MUTED)
    s+=rect(30,104,1140,428,'#fffdf8')+text(52,135,'01  PRETRAIN · labels never enter this circuit',18,TEAL,'bold')
    s+=text(55,183,'Clean row xᵢ',18,weight='bold')+cellrow(55,199,[2,8,5,1,9])
    s+=text(55,271,'Training-only donor columns',17)+cellrow(55,286,[7,3,6,4,2],high=[0,2,4])
    s+=text(55,349,'Replace exactly floor(0.6 × 5) = 3 slots',15,MUTED)
    s+=line(185,356,185,377)+text(55,400,'Corrupted row x̃ᵢ',18,weight='bold')+cellrow(55,415,[7,8,6,1,2],high=[0,2,4])
    for yy in [196,412]:
        s+=line(325,yy+20,391,yy+20)
        s+=rect(396,yy-24,212,89,'#e6f2ef')+text(502,yy+1,'encoder f',20,TEAL,'bold','middle')+text(502,yy+29,'d → 64 → 64',17,anchor='middle')+text(502,yy+51,'ReLU after each layer',13,MUTED,anchor='middle')
        s+=line(611,yy+20,646,yy+20)+rect(651,yy-24,192,89,'#e9edf8')+text(747,yy+1,'projector g',20,weight='bold',anchor='middle')+text(747,yy+29,'64 → 64 → 32',17,anchor='middle')+text(747,yy+51,'ReLU, then L2 norm',13,MUTED,anchor='middle')
    s+=line(503,262,503,384,dashed=True)+text(519,318,'same weights',15,TEAL)
    s+=line(747,262,747,384,dashed=True)+text(763,318,'same weights',15,TEAL)
    s+=text(953,182,'All B clean vectors',16,weight='bold')+text(953,422,'All B corrupt vectors',16,weight='bold')
    s+=line(845,216,965,234)+line(845,432,966,382)
    s+=rect(919,247,217,129,'#fff')+text(1027,276,'S = Z Z̃ᵀ / τ',21,weight='bold',anchor='middle')+text(1027,305,'B × B score matrix',16,anchor='middle')+text(1027,333,'target: diagonal i = j',15,TEAL,anchor='middle')+text(1027,358,'CE over each row',15,anchor='middle')
    s+=text(53,501,'Backpropagation updates f and g through both branches. A donor collision may preserve a selected value.',16,MUTED)
    s+=rect(30,553,1140,203,'#e6f2ef')+text(52,585,'02  TRANSFER · keep f; remove g and the corruption branch',18,TEAL,'bold')
    s+=text(58,633,'Clean input',18,weight='bold')+text(58,660,'[B, d]',17)
    s+=line(192,641,242,641)+rect(247,608,213,73)+text(354,638,'f → h [B, 64]',20,anchor='middle')+text(354,662,'frozen in this lab',16,TEAL,anchor='middle')
    s+=line(465,641,514,641)+rect(519,608,262,73)+text(650,638,'linear probe + softmax',19,anchor='middle')+text(650,662,'fit with labeled rows only',15,anchor='middle')
    s+=line(786,641,836,641)+text(852,633,'Class probabilities',18,weight='bold')+text(852,660,'[B, C] → predicted class',17)
    s+=text(53,721,'Paper downstream path: update encoder + task head. Lab question: what is linearly accessible before fine-tuning?',16,MUTED)
    save('scarf-architecture',s,782)

def architecture_subtab():
    s=text(40,45,'SubTab · recover the whole from different parts',29,weight='bold')
    s+=text(40,78,'Feature coverage → shared encoder → reconstruction and agreement → pooled representation',17,MUTED)
    s+=text(44,125,'01  Fixed column windows: d = 12, three subsets, overlap = 0.75 × 4 = 3',19,weight='bold')
    s+=cellrow(205,148,list(range(12)),width=68)+text(42,174,'column index',16)
    cols=[[0,1,2,3,4,5,6],[1,2,3,4,5,6,7],[5,6,7,8,9,10,11]]
    for j,cc in enumerate(cols):
        yy=198+j*45;s+=text(43,yy+25,f'view {j+1}',17,TEAL,'bold')
        for k in range(12):s+=rect(205+k*68,yy,65,34,'#cbe5dd' if k in cc else '#eeeee8',r=3)+(text(237+k*68,yy+24,k,15,anchor='middle') if k in cc else '')
    s+=text(205,352,'Each subset is [B, 7]. Column positions stay fixed; no mixing of different rows.',16,MUTED)
    s+=rect(30,375,1140,370,'#fffdf8')+text(50,407,'02  SHARED computation · three views of the SAME batch',18,TEAL,'bold')
    for j,yy in enumerate([447,545,643]):
        s+=text(53,yy+13,f'view {j+1}',19,TEAL,'bold')+text(53,yy+40,'+ train noise',14,MUTED)
        s+=line(171,yy+16,211,yy+16)+rect(216,yy-13,211,67,'#e6f2ef')+text(321,yy+13,'E: 7 → 64 → 64',18,anchor='middle')+text(321,yy+38,f'h{j+1} [B,64]',16,anchor='middle')
        s+=line(431,yy+2,493,yy-5)+rect(498,yy-28,221,42,'#e9edf8')+text(608,yy,'G → z [B,32], norm',17,anchor='middle')
        s+=line(431,yy+36,493,yy+49)+rect(498,yy+22,221,42,'#fff0df')+text(608,yy+50,'D → full row [B,12]',17,anchor='middle')
        s+=line(724,yy+42,783,yy+42)+line(724,yy-6,783,yy-6)
    s+=rect(791,425,355,283,'#fff')+text(810,456,'One objective, three jobs',21,weight='bold')
    s+=text(810,495,'Reconstruct every original column',16,ORANGE,'bold')+text(810,520,'including columns absent from the view',14,MUTED)
    s+=text(810,560,'Compare pairs: (1,2), (1,3), (2,3)',16,TEAL,'bold')+text(810,585,'2B × 2B NT-Xent; self excluded',15)
    s+=text(810,625,'Bring same-row projections closer',16,weight='bold')+text(810,650,'Squared distance; average over pairs',15)
    s+=text(810,688,'Gradients reach E, G and D.',16,TEAL)
    s+=text(53,731,'E, G and D are each reused across subsets. E uses LeakyReLU; final latent layer is linear.',15,MUTED)
    s+=rect(30,766,1140,171,'#e6f2ef')+text(50,796,'03  INFERENCE · same windows, no noise; discard D and G',18,TEAL,'bold')
    s+=text(53,840,'h₁, h₂, h₃',23,weight='bold')+text(53,867,'each [B,64]',16)
    s+=line(220,847,272,847)+rect(278,816,278,68)+text(417,842,'mean over VIEWS',20,TEAL,'bold','middle')+text(417,870,'h̄ = (h₁ + h₂ + h₃) / 3',18,anchor='middle')
    s+=line(562,847,612,847)+rect(618,816,273,68)+text(754,842,'frozen linear probe',20,anchor='middle')+text(754,870,'h̄ [B,64] → logits [B,C]',16,anchor='middle')
    s+=line(897,847,947,847)+text(961,843,'softmax',18,weight='bold')+text(961,870,'prediction',17)
    s+=text(52,918,'Example: h₁=(1,3), h₂=(5,7), h₃=(3,2) → h̄=(3,4). Rows never average into each other.',15,MUTED)
    save('subtab-architecture',s,960)

def results():
    r=json.loads((ROOT/'_verify_l072_results.json').read_text())
    arms=['raw','random','scarf','scarf_c0','subtab_recon','subtab_joint']
    labels=['Raw','Random f','SCARF','SCARF c=0','SubTab R','SubTab R+C+D']
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),sharey=True)
    for ax,d in zip(axes,['wine','breast_cancer','digits']):
        for i,a in enumerate(arms):
            rr=next(z for z in r['summary'] if z['dataset']==d and z['arm']==a)
            ax.errorbar(i,100*rr['mean'],yerr=100*rr['sd'],fmt='o',capsize=4,color=TEAL if a=='scarf' else INK)
            ax.scatter(i+np.linspace(-.12,.12,3),100*np.array(rr['seed_values']),s=15,alpha=.5,color=ORANGE)
        ax.set(title=d,xticks=range(6),xticklabels=labels,ylim=(30,102));ax.tick_params(axis='x',rotation=55)
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Frozen-probe test accuracy (%)')
    fig.suptitle('Author-run local evidence · points = paired seeds; bars = sample SD',fontsize=15)
    fig.tight_layout();fig.savefig(OUT/'results.png',dpi=150);plt.close(fig)
    audit=r['rank_audit'];fig,ax=plt.subplots(figsize=(10,3.6))
    for a,rank in zip(audit['arms'],audit['mean_ranks']):ax.scatter(rank,a,color=TEAL,s=60)
    ax.set(xlim=(.8,6.2),xlabel='Mean within-dataset rank (1 = best); three datasets',title=f"Exploratory ranks · Friedman p={audit['friedman_p']:.3f} · Nemenyi CD={audit['nemenyi_cd']:.2f}")
    ax.plot([1,1+audit['nemenyi_cd']],[-.7,-.7],color=ORANGE,lw=3);ax.text(1,-1.15,'Critical difference',fontsize=10);ax.set_ylim(-1.4,5.3)
    ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(OUT/'ranks.png',dpi=150);plt.close(fig)

def build():
    OUT.mkdir(parents=True,exist_ok=True);architecture_scarf();architecture_subtab();results()
if __name__=='__main__':build()
