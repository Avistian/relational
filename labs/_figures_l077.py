"""Portable notebook snapshots exposing the same arithmetic as the live lesson."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
LAB=Path(__file__).resolve().parent;OUT=LAB/'figures/l077'
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    labels=list(fig.texts)
    for a in fig.axes:
        if a.xaxis.label.get_text(): labels.append(a.xaxis.label)
        if a.get_legend(): labels.append(a.get_legend())
    for i,a in enumerate(labels):
        for b in labels[i+1:]:
            assert not a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer)), name+' caption/legend overlap'
    fig.savefig(OUT/f'{name}.png',dpi=150,bbox_inches='tight');plt.close(fig)
def build():
    OUT.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,5.8));fig.subplots_adjust(bottom=.36)
    ax.plot([1,2,3],[10,30,50],'o-',label='A: label 1; delta +40',color='#14627c',lw=2)
    ax.plot([1,2,3],[50,30,10],'s--',label='B: label 0; delta −40',color='#a35426',lw=2)
    ax.set(xlabel='Event day (all available by cutoff 10)',ylabel='Amount',xticks=[1,2,3],yticks=[10,30,50],title='Opposite order → identical aggregates')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.24),frameon=False)
    fig.text(.13,.025,'Both: count 3 · sum 90 · mean 30 · max 50\nSame vector → same prediction → one error per pair.',fontsize=12)
    save(fig,'collision')
    fig,ax=plt.subplots(figsize=(8,5.4));fig.subplots_adjust(bottom=.34)
    ax.barh(['Class z₁','Class z₂'],[2,0],color='#14627c',label='Label 0')
    ax.barh(['Class z₁','Class z₂'],[1,1],left=[2,0],color='#a35426',label='Label 1')
    ax.text(1,0,'2 zeros',ha='center',color='white');ax.text(2.5,0,'1 one',ha='center',color='white');ax.text(.5,1,'1 one',ha='center',color='white')
    ax.set(xlim=(0,3.5),xticks=[0,1,2,3],xlabel='Examples with the same vector',title='Choose one prediction per collision class')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.22),ncol=2,frameon=False)
    fig.text(.12,.025,'Class z₁: best 2 of 3    Class z₂: best 1 of 1\nWhole-data ceiling = (2 + 1) / 4 = 75%.',fontsize=12)
    save(fig,'bound')
    fig,ax=plt.subplots(figsize=(8,4.2));ax.axis('off');ax.set(xlim=(0,8),ylim=(0,4))
    ax.set_title('Which information reaches the customer?',loc='left',pad=15)
    for y,world,risk in [(2.9,'A',1),(1.5,'B',0)]:
        for x,label in [(1,f'{world}: customer\nown = 4'),(4,'Order\namount = 30'),(7,f'Merchant\nrisk = {risk}')]:
            ax.text(x,y,label,ha='center',va='center',bbox={'boxstyle':'round,pad=.6','fc':'#eaf2f8','ec':'#69839a'})
        for a,b in [(1.8,3.1),(4.8,6.15)]:ax.annotate('',xy=(b,y),xytext=(a,y),arrowprops={'arrowstyle':'->','lw':2})
    ax.text(4,.35,'0 hops: same own feature     1 hop: same amount\n2 hops: risk differs — if the reducer preserves it',ha='center',va='center')
    save(fig,'reach')
    r=json.loads((LAB/'_verify_l077_results.json').read_text())['runs']
    fig,ax=plt.subplots(figsize=(8,5.4));fig.subplots_adjust(bottom=.25)
    for name,marker,color in [('flat','o','#a35426'),('restored','s','#14627c')]:
        ax.plot([v['seed'] for v in r],[v['scores'][name]['test'] for v in r],marker+'-',label=name,lw=2,color=color)
    ax.set(ylim=(0,1.08),xticks=range(5),yticks=[0,.5,1],xlabel='Generator seed (split seed = seed + 10000)',ylabel='Held-out accuracy',title='Full construction · 1,000 pairs per seed')
    ax.legend(loc='center right',bbox_to_anchor=(1,.69));ax.axhline(.5,color='#999',linestyle=':',zorder=0)
    fig.text(.12,.015,'400 test customers per seed; complete pairs retained.\nZero variation is structural here; no real-dataset uncertainty claim.',fontsize=11)
    save(fig,'results')
if __name__=='__main__':build()
