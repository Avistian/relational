"""Portable computation figures for lesson and notebook."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parent/'figures/l081'

def build():
    ROOT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'figure.facecolor':'#f7f8fa','axes.facecolor':'#f7f8fa'})
    fig,ax=plt.subplots(figsize=(11,9));ax.set(xlim=(0,11),ylim=(0,9));ax.axis('off')
    def box(x,y,w,h,title,body,color='#e3edf7'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',fc=color,ec='#7890a8'))
        ax.text(x+.16,y+h-.23,title,weight='bold',va='top',fontsize=13)
        ax.text(x+.16,y+h-.63,body,va='top',fontsize=11,linespacing=1.5)
    def arrow(a,b):ax.annotate('',b,a,arrowprops={'arrowstyle':'->','lw':1.8,'color':'#28577a'})
    ax.text(.3,8.7,'Sparse GG-NN · from bonds to one molecular prediction',fontsize=17,weight='bold')
    box(.3,6.9,4.4,1.25,'1  Atom features → initial state','X [N,13] → zero-pad → H⁰ [N,50]\nHeavy atoms; hydrogen count is a feature')
    box(5.4,6.9,5,1.25,'2  Chemical graph','Two directed entries per undirected bond\n4 bond types; no distances or self-loops')
    box(.3,4.8,10.1,1.35,'3  Messages → sum at each destination','For w → v: Aᵢₙ[bond] h_w and Aₒᵤₜ[bond] h_w\nConcatenate two sums → mᵥ [100]; isolated node receives zero')
    arrow((2.5,6.9),(2.5,6.25));arrow((7.7,6.9),(7.7,6.25))
    box(.3,2.65,5.7,1.5,'4  Recurrent update · repeat T = 6','z,r = sigmoid(message + old-state projections)\nnew h = (1 − z) old h + z candidate\nSame bond matrices and gates in every round')
    box(6.6,2.65,3.8,1.5,'5  Preserve original features','sᵥ = concat(hᵀᵥ, xᵥ) [63]\nTwo MLPs: 63 → 200 → 1\nσ(gate(sᵥ)) × value(sᵥ)')
    arrow((3.1,4.8),(3.1,4.25));arrow((6,3.5),(6.5,3.5))
    box(.3,.55,10.1,1.5,'6  Sum per molecule → prediction → training objective','Sum node contributions using graph IDs → ŷ [B]\nTraining: MSE on train-standardized dipole targets; Adam updates all weights\nInference: same forward pass → undo target scaling → dipole magnitude in Debye',color='#e2eee4')
    arrow((8.5,2.8),(8.5,2.15))
    fig.savefig(ROOT/'architecture.png',dpi=150,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(6.4,8.8),gridspec_kw={'height_ratios':[1,1.2,1.4]})
    fig.subplots_adjust(hspace=.28)
    for ax in axs:ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    axs[0].text(.02,.98,'1  SEND · gather old states by source',weight='bold',va='top')
    for x,label,color in [(.12,'A\n2','#dceaf5'),(.37,'B\n4','#dceaf5'),(.62,'C\n8','#dceaf5'),(.88,'D\n10','#e2eee4')]:
        axs[0].text(x,.5,label,ha='center',va='center',bbox=dict(boxstyle='circle,pad=.5',fc=color,ec='#7890a8'))
    for a,b in [(.18,.31),(.43,.56)]:
        axs[0].annotate('',(b,.5),(a,.5),arrowprops=dict(arrowstyle='<->',color='#28577a',lw=2))
    axs[0].text(.35,.04,'graph 0',ha='center');axs[0].text(.88,.04,'graph 1',ha='center')
    axs[1].text(.02,.98,'2  ROUTE · each edge names its destination',weight='bold',va='top')
    rows=[['A → B','0 → 1','2'],['B → A','1 → 0','4'],['B → C','1 → 2','4'],['C → B','2 → 1','8']]
    tab=axs[1].table(cellText=rows,colLabels=['Edge','src → dst','Message'],bbox=[.02,0,.96,.8],cellLoc='center')
    tab.auto_set_font_size(False);tab.set_fontsize(12)
    axs[2].text(.02,.98,'3  UPDATE · then pool within each graph',weight='bold',va='top')
    rows=[['A','4','2','3'],['B','(2+8)/2 = 5','4','4.5'],['C','4','8','6'],['D','0 (empty)','10','5']]
    tab=axs[2].table(cellText=rows,colLabels=['Node','Incoming mean','Old','New'],colWidths=[.15,.45,.2,.2],bbox=[.02,.26,.96,.58],cellLoc='center')
    tab.auto_set_font_size(False);tab.set_fontsize(11)
    axs[2].text(.03,.15,'New = ½ old + ½ mean. All messages use old states.',fontsize=10)
    axs[2].text(.03,.02,'Readout: graph 0 = 13.5; graph 1 = 5',weight='bold',color='#28577a')
    fig.suptitle('One round, from edge rows to graph outputs',fontsize=14,weight='bold')
    fig.savefig(ROOT/'trace.png',dpi=150,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3.3));ax.axis('off')
    ax.text(.02,.91,'RELABEL THE GRAPH · transport every reference',fontsize=16,weight='bold')
    ax.text(.02,.66,'Original row order\n[A, B, C, D]\nOutput [3, 4.5, 6, 5]',linespacing=1.7)
    ax.text(.4,.66,'New row order\n[C, A, D, B]\nOutput [6, 3, 5, 4.5]',linespacing=1.7)
    ax.text(.76,.66,'Sum readout\n18.5 in both cases\nInvariant graph value',linespacing=1.7)
    ax.text(.02,.06,'Node outputs move with the nodes (equivariance). The graph output stays fixed (invariance).',fontsize=11)
    fig.savefig(ROOT/'symmetry.png',dpi=150,bbox_inches='tight');plt.close(fig)
if __name__=='__main__':build()
