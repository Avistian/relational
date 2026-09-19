"""Portable, computation-exposing figures; no browser dependency."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
OUT=Path(__file__).resolve().parent/'figures/l075'
BG='#faf9f5'; INK='#183844'; TEAL='#007f78'; GOLD='#b26812'
def canvas(title, subtitle, height=5):
    fig,ax=plt.subplots(figsize=(9,height));fig.patch.set_facecolor(BG);ax.set_facecolor(BG)
    ax.set_xlim(0,9);ax.set_ylim(0,height);ax.axis('off')
    ax.text(.25,height-.3,title,fontsize=19,weight='bold',color=INK,va='top')
    ax.text(.25,height-.85,subtitle,fontsize=11,color=INK,va='top')
    return fig,ax

def box(ax,x,y,w,h,text,color=TEAL):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',facecolor='white',edgecolor=color,lw=1.6))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=12,color=INK,linespacing=1.5)

def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=160,bbox_inches='tight',facecolor=BG);plt.close(fig)

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    f,a=canvas('Five columns → four typed blocks','Trace the two different-width vectors into the same parent-type container.',6)
    rows=[('amount: 10','numerical','float block [B,1]'),('region: north','categorical','integer block [B,1]'),('created: 2024-01-01','timestamp','calendar block [B,1,7]'),('vector: [1,0,0]','embedding','vector: width 3'),('note: red apple','text_embedded\n→ embedding','note: [1,0,1,0], width 4')]
    for i,(raw,kind,out) in enumerate(rows):
        y=4.4-i*.8
        a.text(.3,y,raw,fontsize=12,color=INK)
        a.text(3.25,y,kind,fontsize=11,color=TEAL)
        a.text(6.2,y,out,fontsize=11,color=INK)
    a.plot([3,3],[.9,4.7],color='#c0cece');a.plot([6,6],[.9,4.7],color='#c0cece')
    a.text(.3,.55,'Stored vector + embedded text: one MultiEmbeddingTensor, two named columns.',fontsize=11,color=TEAL)
    a.text(.3,.18,'Then: each column → width 8; concatenate in returned order → [B,5,8].',fontsize=12,color=INK)
    save(f,'types')
    f,a=canvas('The split label does not fit the statistics','Same values. Compare which rows enter the fitted statistic.',5.4)
    box(a,.35,2.8,3.6,1.2,'Training rows\n10, 20, 30, NaN\nmean = 60 / 3 = 20')
    box(a,4.9,2.8,3.6,1.2,'Held-out rows\n1000, NaN\nsplit = 2',GOLD)
    box(a,.35,.7,3.6,1.4,'FIT on training only\nmean stays 20\nAPPLY same state to query')
    box(a,4.9,.7,3.6,1.4,'FIT on all rows\nmean = 1060 / 4 = 265\ntraining mean shifts by +245',GOLD)
    for x in [2.15,6.7]:a.annotate('',xy=(x,2.2),xytext=(x,2.7),arrowprops={'arrowstyle':'->','color':INK})
    a.annotate('',xy=(5.2,2.2),xytext=(3.8,2.7),arrowprops={'arrowstyle':'->','color':GOLD})
    save(f,'scope')
    f,a=canvas('One numeric token, coordinate by coordinate','Illustrative parameters: μ = 20, scale = 10, weight = [2, −1], bias = [0.5, 0.5].',6)
    box(a,.4,3.7,2,1,'x = 30\n(30−20)/10\nz = 1')
    box(a,3.1,3.7,2.3,1,'z × weight\n1 × [2, −1]\n= [2, −1]')
    box(a,6.1,3.7,2.4,1,'+ bias\ntoken = [2.5, −0.5]')
    for x in [2.45,5.45]:a.annotate('',xy=(x+.6,4.2),xytext=(x,4.2),arrowprops={'arrowstyle':'->','color':TEAL})
    a.text(.4,2.95,'Change only the value; keep the fitted state and learned parameters fixed.',fontsize=12,color=INK)
    for y,label,z,out in [(2.25,'x = 20','z = 0','[0.5, 0.5]'),(1.55,'x = 10','z = −1','[−1.5, 1.5]'),(.85,'x = missing → mean = 20','z = 0','[0.5, 0.5]')]:
        a.text(.4,y,label,fontsize=12,color=INK);a.text(4,y,z,fontsize=12,color=TEAL);a.text(6.2,y,out,fontsize=12,color=INK)
    a.text(.4,.25,'Configured policy: mean imputation. A missing token equals the bias, not zero.',fontsize=11,color=GOLD)
    save(f,'numeric')
    f,a=canvas('Exact local row-encoder composition','Trace one query row. B = batch rows; d = token width; D = row width.',7.4)
    stages=[('Typed TensorFrame','numeric + category + timestamp + two embedding columns'),('Column encoders → [B,5,8]','per-column values become width-8 vectors'),('Flatten → [B,40]','5 × 8 coordinates; preserves column positions'),('Linear 40→16 → ReLU → Linear 16→6','mixes columns within each row; returns [B,D=6]'),('Optional task head: Linear 6→2','gradient demonstration only; no benchmark training')]
    for i,(title,sub) in enumerate(stages):
        y=5.5-i*1.05;box(a,.65,y,7.7,.77,title+'\n'+sub)
        if i<4:a.annotate('',xy=(4.5,y-.27),xytext=(4.5,y-.07),arrowprops={'arrowstyle':'->','color':INK})
    a.text(.65,.45,'Future GNN input: [B,6] + explicit row IDs and time-valid edges.',fontsize=12,color=TEAL)
    a.text(.65,.1,'This diagram is the teaching MLP readout, not the paper’s FT-Transformer.',fontsize=11,color=GOLD)
    save(f,'architecture')
if __name__=='__main__':build()
