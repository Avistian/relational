"""Model-specific portable architecture and measured course figure."""
import html,json
from pathlib import Path
import cairosvg
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;D=P/'figures/b04a';D.mkdir(exist_ok=True)
parts=['<svg xmlns="http://www.w3.org/2000/svg" width="600" height="1470" viewBox="0 0 600 1470"><defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#37696e"/></marker></defs><rect width="600" height="1470" rx="20" fill="#eef4f5"/>']
def txt(x,y,s,size=17,bold=False):parts.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="{size}" font-weight="{600 if bold else 400}" fill="#173f44">{html.escape(s)}</text>')
def box(y,h,title,lines,fill='#fff'):
 parts.append(f'<rect x="24" y="{y}" width="552" height="{h}" rx="12" fill="{fill}" stroke="#93b4b8"/>');txt(300,y+29,title,20,True)
 for i,line in enumerate(lines):txt(300,y+56+i*23,line)
def arrow(y):parts.append(f'<path d="M300 {y}v25" stroke="#37696e" stroke-width="2" marker-end="url(#arrow)"/>')
txt(300,36,'TabFlex · what is compressed?',25,True)
box(57,103,'Pretraining learns the prediction rule',['Synthetic support/query episodes → query loss','Update encoder, attention, FFN and head weights.'],'#dcebed');arrow(164)
box(198,111,'Inference freezes those weights',['Support X[S,F], y[S] + query X[Q,F]','Wrapper chooses S100 / L100 / H1K by S and F.','F > 1000: random projection discards dimensions.']);arrow(313)
box(347,110,'A row becomes one embedding',['Feature encoder F → E (paper E = 512)','Support: feature embedding + label embedding','Query: feature embedding only; targets held out.']);arrow(461)
box(495,117,'Each attention head forms Q, K, V',['Support queries read support; query rows read support.','Positive map φ(x) = ELU(x) + 1','Kφ[S,d] and V[S,dv] feed the summary.'],'#dcebed');arrow(616)
box(650,130,'Compress support into two statistics',['A = Kφᵀ V           shape d × dv','z = sum over support of Kφ      shape d','Output(q) = φ(q) A / (φ(q) z + ε)','No query × support score matrix is materialized.'],'#cee3e5');arrow(784)
box(818,109,'Complete the learned block',['Output projection → residual + normalization','Feed-forward network → residual + normalization','Repeat blocks; a summary is recomputed per layer.']);arrow(931)
box(965,107,'Decode and align predictions',['MLP class logits → wrapper probabilities','Release views: S100/H1K = 3; L100 = 1','Capacity checks must preserve original class identities.']);arrow(1076)
box(1110,110,'Three axes remain separate',['Rows S: summaries replace pairwise attention work.','Features F: input projection still grows with width.','Classes C: class head and training prior still constrain C.'],'#f2e6cc')
box(1244,118,'B04a measurement boundary',['Figure 9: isolated attention kernels; source gate closed.','Course lab: fixed projection + label kernel readout.','No learned blocks, no TabFlex checkpoint or pretraining.'],'#f2e6cc')
txt(300,1402,'Named shapes distinguish feature width F from head width d.',16)
txt(300,1430,'Primary sources: TabFlex §§3–5 and released model code.',16)
parts.append('</svg>');svg=''.join(parts);(D/'architecture.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(D/'architecture.png'),scale=1.5)
r=json.loads((P/'evidence/b04a/audit.json').read_text())['rows']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(8.2,3.8),layout='constrained')
for kernel,color in [('linear','#286d76'),('softmax','#b26936')]:
 for mode,style in [('noise','-'),('copies','--')]:
  groups=[[x for x in r if x['support']==256 and x['features']==f and x['widening']==mode and x['kernel']==kernel] for f in [8,64,256]]
  means=[np.mean([x['accuracy'] for x in g]) for g in groups];lo=[min(x['accuracy'] for x in g) for g in groups];hi=[max(x['accuracy'] for x in g) for g in groups]
  axes[0].plot([8,64,256],means,style+'o',color=color,label=kernel+' / '+mode);axes[0].fill_between([8,64,256],lo,hi,color=color,alpha=.07)
  axes[1].plot([8,64,256],[1000*np.mean([x['warm_median_seconds'] for x in g]) for g in groups],style+'o',color=color)
axes[0].set(xscale='log',xlabel='Input features F · support S = 256',ylabel='Accuracy on 64 queries',ylim=(.35,1),title='Mean and min–max across 3 seeds');axes[0].legend(fontsize=8)
axes[1].set(xscale='log',xlabel='Input features F · support S = 256',ylabel='Warm pipeline milliseconds',title='Includes support-fitted scaling + projection')
fig.suptitle('Measured course kernel classifier · not TabFlex checkpoint inference',fontsize=12);fig.savefig(D/'results.png',dpi=160);plt.close(fig)
print('B04a architecture and measured figure created')
