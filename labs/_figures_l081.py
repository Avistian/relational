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
    fig,axs=plt.subplots(1,3,figsize=(11,3.6))
    for ax in axs:ax.axis('off')
    axs[0].text(.03,.9,'SEND · old states',weight='bold',fontsize=15);axs[0].text(.03,.65,'A = 2 → B\nC = 8 → B\nB old state = 4',linespacing=2)
    axs[1].text(.03,.9,'REDUCE · at B',weight='bold',fontsize=15);axs[1].text(.03,.65,'sum: 2 + 8 = 10\ncount: 2\nmean: 10 / 2 = 5',linespacing=2)
    axs[2].text(.03,.9,'UPDATE · once',weight='bold',fontsize=15);axs[2].text(.03,.65,'½ × 4 + ½ × 5 = 4.5\nAll nodes: [3, 4.5, 6, 5]\nD isolated: ½ × 10 = 5',linespacing=2)
    fig.suptitle('Teaching operator · synchronous neighbor mean, not the learned GG-NN',fontsize=15)
    fig.savefig(ROOT/'trace.png',dpi=150,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3.3));ax.axis('off')
    ax.text(.02,.91,'RELABEL THE GRAPH · transport every reference',fontsize=16,weight='bold')
    ax.text(.02,.66,'Original row order\n[A, B, C, D]\nOutput [3, 4.5, 6, 5]',linespacing=1.7)
    ax.text(.4,.66,'New row order\n[C, A, D, B]\nOutput [6, 3, 5, 4.5]',linespacing=1.7)
    ax.text(.76,.66,'Sum readout\n18.5 in both cases\nInvariant graph value',linespacing=1.7)
    ax.text(.02,.06,'Node outputs move with the nodes (equivariance). The graph output stays fixed (invariance).',fontsize=11)
    fig.savefig(ROOT/'symmetry.png',dpi=150,bbox_inches='tight');plt.close(fig)
if __name__=='__main__':build()
