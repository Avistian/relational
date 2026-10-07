"""A narrow, portable diagram for the two-layer receiver-row pruning trace."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle


def build(out=None):
    out = out or Path(__file__).resolve().parent / "figures/l091"
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ink='#183447';teal='#087f8c';rust='#a34430';muted='#7b8790'
    with plt.rc_context({'font.family':'DejaVu Sans','svg.fonttype':'none','svg.hashsalt':'l091-pruning'}):
        fig=plt.figure(figsize=(6.5,9),facecolor='#fbfcfa')
        fig.text(.06,.975,'Keep the needed sender inputs',fontsize=19,weight='bold',color=ink)
        fig.text(.06,.89,'Two layers · root 0 · unit weights\nEach node also adds its own state.',fontsize=15,color=ink,linespacing=1.4)
        cases=[('1 · Full computation','[6, 12, 8]',18,False,False),
               ('2 · Zero receiver row 2','[6, 12, 0]',18,True,False),
               ('3 · Also delete sender column 2','[6, 4, 0]',10,True,True)]
        for i,(title,states,result,pruned,cut) in enumerate(cases):
            ax=fig.add_axes([.05,.61-i*.285,.90,.245]);ax.set_xlim(0,6);ax.set_ylim(0,2.6);ax.axis('off')
            color=rust if cut else teal
            ax.add_patch(FancyBboxPatch((.04,.04),5.92,2.5,boxstyle='round,pad=0.01,rounding_size=.12',facecolor='white',edgecolor=color,linewidth=1.5))
            ax.text(.23,2.21,title,fontsize=16,weight='bold',color=color)
            for node,x,value in [(0,1,2),(1,3,4),(2,5,8)]:
                absent=cut and node==2
                ax.add_patch(Circle((x,1.54),.26,facecolor='#f0f4f5' if absent else '#dceff0',edgecolor=muted if absent else teal,linewidth=1.6))
                ax.text(x,1.54,str(node),ha='center',va='center',fontsize=16,weight='bold',color=ink)
                ax.text(x,1.04,'removed' if absent else f'input {value}',ha='center',fontsize=13,color=muted if absent else ink)
            for source,target in [(3,1),(5,3)]:
                missing=cut and source==5
                ax.annotate('',xy=(target+.33,1.54),xytext=(source-.33,1.54),arrowprops={'arrowstyle':'->','lw':2,'color':muted if missing else teal,'linestyle':'--' if missing else '-'})
                if missing:ax.text(4,1.76,'cut',ha='center',fontsize=12,color=rust)
            ax.text(.23,.65,f'First states: {states}',fontsize=15,color=ink)
            ax.text(.23,.23,f'Root after two layers: {result}',fontsize=16,weight='bold',color=color)
        fig.savefig(out/'pruning.png',dpi=170,bbox_inches='tight')
        fig.savefig(out/'pruning.svg',bbox_inches='tight',metadata={'Date':None})
        svg=out/'pruning.svg'
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        plt.close(fig)


if __name__=='__main__':
    build()
