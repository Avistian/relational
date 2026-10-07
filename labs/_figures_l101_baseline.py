"""Measured error decomposition from the frozen L101 driver-position replay.

Values independently reconciled in reviews/course-iteration-2026-10-07/l101-check.json.
Each bar segment is group error sum / all 760 queries, not group MAE.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def build():
    out=Path(__file__).resolve().parent/'figures/l101'
    ink='#183447';teal='#087f8c';rust='#a34430'
    with plt.rc_context({'font.family':'DejaVu Sans','svg.fonttype':'none','svg.hashsalt':'l101-fallback'}):
        fig=plt.figure(figsize=(6.4,6.6),facecolor='#fbfcfa')
        fig.text(.07,.94,'Where the test error comes from',fontsize=18,weight='bold',color=ink)
        fig.text(.07,.85,'760 queries · identical frozen predictions\nEach segment contributes to overall MAE.',fontsize=14,color=ink,linespacing=1.5)
        ax=fig.add_axes([.08,.29,.86,.47]);ax.set_xlim(0,10);ax.set_ylim(-.65,1.75)
        rows=[('Global mean',3.0683022146034102,1.4445783141833455),('Entity mean',2.6165110183145335,5.88484649122807)]
        for y,(name,a,b) in zip([1,0],rows):
            ax.text(0,y+.33,name,fontsize=15,weight='bold',color=ink)
            ax.barh(y,a,height=.40,color=teal)
            ax.barh(y,b,left=a,height=.40,color=rust)
            for center,value in [(a/2,a),(a+b/2,b)]:ax.text(center,y,f'{value:.2f}',ha='center',va='center',color='white',fontsize=14,weight='bold')
            ax.text(a+b+.15,y,f'{a+b:.3f}',va='center',fontsize=14,color=ink)
        ax.set_yticks([]);ax.set_xticks([0,2,4,6,8,10]);ax.tick_params(labelsize=13,colors=ink)
        ax.set_xlabel('Contribution to MAE · finishing positions',fontsize=13,color=ink,labelpad=9)
        for side in ['top','right','left']:ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#b7c5c9')
        fig.text(.08,.15,'■ Seen driver: 433 queries',color=teal,fontsize=14)
        fig.text(.08,.10,'■ Unseen driver: 327 queries',color=rust,fontsize=14)
        fig.text(.08,.035,'Entity fallback = 0 · lower total is better',fontsize=13,color=ink)
        fig.savefig(out/'fallback.png',dpi=170,bbox_inches='tight')
        fig.savefig(out/'fallback.svg',bbox_inches='tight',metadata={'Date':None});plt.close(fig)
        p=out/'fallback.svg';p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')


if __name__=='__main__':build()
