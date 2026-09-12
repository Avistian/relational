"""Portable computation snapshots and newly measured PFN diagnostics."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/l061-mpl')
from pathlib import Path
import json,numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from relkit.pfn_l061_v2 import riemann_nll,riemann_cdf
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures/l061';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f8fbfa','axes.facecolor':'#f8fbfa','savefig.facecolor':'#f8fbfa'})
TEAL='#087e83';ORANGE='#b86425';GRAY='#52606d'
def save(fig,name):fig.savefig(OUT/name,dpi=150,bbox_inches='tight');plt.close(fig)

def generate():
 x=np.linspace(0,1,201);k=np.exp(-(x-.2)**2/(2*.6**2));mu=k/1.0001;v=1.0001-k*k/1.0001
 fig,ax=plt.subplots(2,1,figsize=(8,7),sharex=True,layout='constrained')
 fig.suptitle('Condition one noisy observation: (.2, 1)',fontsize=18)
 ax[0].plot(x,mu,color=TEAL,label='conditional mean');ax[0].axhline(0,color=GRAY,ls=':',label='prior mean 0');ax[0].scatter([.2],[1],color=ORANGE,s=65,label='observed context')
 ax[0].scatter([.8],[mu[160]],color=TEAL);ax[0].annotate('query .8 → mean .606470',(.8,mu[160]),(.40,.25),arrowprops=dict(arrowstyle='->'))
 ax[0].set_ylabel('Target units');ax[0].legend(loc='lower left',fontsize=10)
 ax[1].plot(x,v,color=ORANGE,label='noisy query variance');ax[1].axhline(1.0001,color=GRAY,ls=':',label='prior variance 1.0001');ax[1].scatter([.8],[v[160]],color=ORANGE)
 ax[1].annotate('k=.606531\n1.0001−k²/1.0001=.632257',(.8,v[160]),(.24,.79),arrowprops=dict(arrowstyle='->'),fontsize=11)
 ax[1].set(xlabel='Query location x',ylabel='Squared target units',ylim=(-.04,1.2));ax[1].legend(loc='lower right',fontsize=10)
 save(fig,'conditioning-v2.png')
 borders=torch.tensor([-3.,-1.,0.,2.,5.],dtype=torch.float64);logits=torch.tensor([[.1,.2,.3,.4]],dtype=torch.float64).log()
 ys=torch.linspace(-9,10,2400,dtype=torch.float64);pdf=(-riemann_nll(logits.expand(len(ys),-1),ys,borders)).exp()
 fig,ax=plt.subplots(figsize=(8,5.5),layout='constrained');ax.plot(ys,pdf,color=TEAL,lw=2)
 for left,right,color in [(-9,-1,ORANGE),(-1,0,GRAY),(0,2,TEAL),(2,10,ORANGE)]:
  mask=(ys>=left)&(ys<=right);ax.fill_between(ys[mask],pdf[mask],color=color,alpha=.17)
 ax.scatter([1],[.15],color='black');ax.annotate('y=1\nmass .3 ÷ width 2 = density .15\nNLL = −ln(.15) = 1.89712',(1,.15),(-7,.31),arrowprops=dict(arrowstyle='->'),fontsize=11)
 ax.text(-7,.035,'left tail\nweight .1',color=ORANGE);ax.text(5,.075,'right tail\nweight .4',color=ORANGE)
 ax.set(title='A normalized density, including unbounded tails',xlabel='Target y',ylabel='Density per target unit',ylim=(0,.4),xlim=(-9,10))
 fig.text(.12,-.04,'Nominal borders [−3, −1, 0, 2, 5]; masses [.1, .2, .3, .4]. Synthetic fixture.',fontsize=10)
 save(fig,'density-v2.png')
 r=json.loads((ROOT/'_verify_l061_v2_results.json').read_text());head=json.loads((ROOT/'_analysis_l061_v2_results.json').read_text())
 fig,axes=plt.subplots(2,1,figsize=(8,8),layout='constrained',sharex=True)
 for ax,reg in zip(axes,['matched','prior_shift']):
  rows=[s for s in r['summary'] if s['regime']==reg];positions=np.arange(len(rows));hs=[q for q in head['records'] if q['regime']==reg]
  for i,s in enumerate(rows):ax.scatter([i-.07,i,i+.07],s['pfn_nll']['seed_values'],s=28,color=TEAL,alpha=.7)
  for key,color,label in [('pfn_nll',TEAL,'learned PFN'),('gp_nll',ORANGE,'exact GP'),('prior_nll',GRAY,'ignore context')]:ax.plot(positions,[s[key]['mean'] for s in rows],color=color,marker='o',label=label)
  ax.plot(positions,[s['head_oracle_nll'] for s in hs],color='#8959a2',ls='--',label='analytic masses / same head')
  ax.set_title('Matched length scale .6' if reg=='matched' else 'Shifted length scale .1; same frozen PFN');ax.set_ylabel('Mean NLL (nats/query)');ax.set_ylim(-3.4,5.3);ax.axvline(4.5,color=GRAY,ls=':',lw=1);ax.grid(axis='y',alpha=.18)
 axes[0].legend(fontsize=10,ncol=2,loc='upper left');axes[1].set_xticks(positions,[str(s['n_context']) for s in rows]);axes[1].set_xlabel('Number of context observations (32 exceeds training maximum 16)')
 fig.suptitle('Local evidence: useful conditioning, substantial remaining error',fontsize=16)
 save(fig,'results-v2.png')
 c=r['curve'];grid=np.array(c['grid']);mu=np.array(c['gp_mean']);sd=np.sqrt(c['gp_variance']);logits=torch.tensor(c['logits'],dtype=torch.float64);b=torch.tensor(r['borders'],dtype=torch.float64)
 def quantile(prob):
  lo=torch.full((len(grid),),-20.,dtype=torch.float64);hi=-lo
  for _ in range(60):
   mid=(lo+hi)/2;left=riemann_cdf(logits,mid,b)<prob;lo=torch.where(left,mid,lo);hi=torch.where(left,hi,mid)
  return ((lo+hi)/2).numpy()
 lower,upper=quantile(.025),quantile(.975)
 fig,ax=plt.subplots(2,1,figsize=(8,7),layout='constrained')
 ax[0].fill_between(grid,lower,upper,color=TEAL,alpha=.18,label='PFN central 95%');ax[0].fill_between(grid,mu-1.96*sd,mu+1.96*sd,color=ORANGE,alpha=.2,label='exact GP central 95%')
 ax[0].plot(grid,c['pfn_mean'],color=TEAL,label='PFN mean');ax[0].plot(grid,mu,color=ORANGE,label='exact mean');ax[0].scatter(c['context_x'],c['context_y'],color='black',marker='x',s=65,label='context');ax[0].set(xlabel='Query x',ylabel='Observed target y',title='Chosen two-observation fixture; final training seed 2');ax[0].legend(fontsize=9,ncol=3)
 point=50;ys=torch.linspace(-2,2,1800,dtype=torch.float64);dens=(-riemann_nll(logits[point].expand(len(ys),-1),ys,b)).exp().numpy();truth=np.exp(-.5*((ys.numpy()-mu[point])/sd[point])**2)/(sd[point]*np.sqrt(2*np.pi))
 ax[1].plot(ys,dens,color=TEAL,label='PFN density');ax[1].plot(ys,truth,color=ORANGE,label='exact GP density');ax[1].set(xlabel='Target y at query x=.5',ylabel='Density',title='Inspect uncertainty even when means look plausible');ax[1].legend()
 save(fig,'predictions-v2.png')
if __name__=='__main__':generate()
