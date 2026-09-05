"""Reusable fixed-split, multi-seed report components; no model code hidden."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def score_table(result,labels):
    return pd.DataFrame([{'Dataset':d,'Model':labels[m],'Mean AUROC':a['mean'],
                          'Seed SD':a['sample_std'],'Seed CI95 half-width':a['seed_ci95_halfwidth']}
                         for d,arms in result['results']['per_dataset'].items() for m,a in arms.items()])


def plot_scores(result,labels,title='Your run · held-out scores'):
    datasets=list(result['results']['per_dataset']);colors=['#286da3','#b16d21','#117d71','#78559c']
    fig,axes=plt.subplots(len(datasets),1,figsize=(9,2.65*len(datasets)),sharex=True,layout='constrained',squeeze=False)
    for ax,d in zip(axes[:,0],datasets):
        for j,(m,label) in enumerate(labels.items()):
            a=result['results']['per_dataset'][d][m]
            ax.scatter(a['scores'],j+np.linspace(-.12,.12,len(a['scores'])),s=25,color=colors[j],alpha=.55)
            ax.errorbar(a['mean'],j,xerr=a['sample_std'],fmt='D',color=colors[j],capsize=4)
        ax.set_yticks(range(len(labels)),labels.values());ax.invert_yaxis()
        ax.set_title(d.replace('_',' '),loc='left');ax.grid(axis='x',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[-1,0].set_xlabel('Test AUROC · dots: seeds · diamonds: mean · bars: ±1 sample SD')
    fig.suptitle(title,fontsize=14);return fig


def plot_ranks(result,labels,title='Your run · rank uncertainty'):
    ranks=result['results']['mean_ranks'];cd=result['results']['nemenyi_cd'];p=result['results']['friedman']['p']
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    for j,m in enumerate(labels):
        ax.scatter(ranks[m],j,s=60,color=['#286da3','#b16d21','#117d71','#78559c'][j])
        ax.annotate(f'{ranks[m]:.2f}',(ranks[m],j),xytext=(9,0),textcoords='offset points',va='center')
    ax.plot([1,1+cd],[-.9,-.9],color='#203c50',marker='|',markersize=14)
    ax.text(1+cd/2,-1.1,f'Critical difference = {cd:.3f}',ha='center',va='bottom')
    ax.set_yticks(range(len(labels)),labels.values());ax.set_ylim(len(labels)-.5,-1.6)
    ax.set_xlim(.8,max(len(labels)+.35,1+cd+.2));ax.set_xlabel('Mean rank across datasets · 1 is best')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.2)
    verdict='No overall difference detected; equivalence is not established.' if p>=.05 else 'Overall difference detected; inspect pairwise gaps against the CD.'
    fig.suptitle(f'{title}\nFriedman p = {p:.3f}. {verdict}',fontsize=11);return fig


def plot_scaleup(result,title='MovieLens · full-data attempt, different protocol'):
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for row in result['seeds']:
        axes[0].plot([h['epoch'] for h in row['history']],[h['valid_loss'] for h in row['history']],
                     marker='o',label='seed '+str(row['seed']))
    axes[0].set(xlabel='Epoch',ylabel='Validation log loss',title='Checkpoint selection uses validation')
    axes[0].legend(fontsize=9)
    scores=[r['logloss'] for r in result['seeds']]
    axes[1].scatter(scores,np.linspace(-.1,.1,len(scores)),color='#117d71',label='Attempt: test seeds')
    axes[1].errorbar(result['mean_logloss'],0,xerr=result['sample_std_logloss'] or 0,
                     fmt='D',color='#117d71',capsize=5,label='Attempt: mean ± SD')
    axes[1].axvline(.3170,color='#b16d21',linestyle='--',label='Paper .3170: different protocol')
    axes[1].set(xlabel='Test log loss · lower is better',yticks=[],ylim=(-.4,.6),title='INCOMPARABLE to Table 6')
    axes[1].legend(fontsize=8,loc='upper right')
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    fig.suptitle(title,fontsize=13);return fig
