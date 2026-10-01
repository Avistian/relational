"""Mechanism figures carry coherent values; result chart is generated from evidence."""
import json
from pathlib import Path
import cairosvg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;F=P/'figures/l156';F.mkdir(parents=True,exist_ok=True)
def svg(name,title,body,height=350):
    s=f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}"><rect width="900" height="{height}" rx="12" fill="#f1f6f2"/><g font-family="sans-serif" fill="#183c33"><text x="30" y="43" font-size="24">{title}</text>'+body+'</g></svg>'
    (F/(name+'.svg')).write_text(s);cairosvg.svg2png(bytestring=s.encode(),write_to=str(F/(name+'.png')))
svg('clocks','A past event can still arrive too late.', '''
<text x="30" y="77" font-size="15">Synthetic trace · query A cutoff 10 · query B cutoff 20</text>
<line x1="100" y1="155" x2="820" y2="155" stroke="#699488" stroke-width="3"/>
<line x1="340" y1="99" x2="340" y2="225" stroke="#183c33" stroke-width="3"/>
<circle cx="230" cy="155" r="9" fill="#256950"/><circle cx="485" cy="155" r="9" fill="#ac4332"/>
<text x="144" y="126" font-size="18">event 9</text><text x="288" y="98" font-size="18">cutoff 10</text><text x="435" y="126" font-size="18">arrival 12</text><text x="764" y="188" font-size="18">cutoff 20</text>
<text x="30" y="262" font-size="19">Event check:9 ≤10 → PASS. Availability:12 &gt;10 → FAIL.</text>
<text x="30" y="296" font-size="17">A cutoff 20check would hide the leak. The owner remains query A.</text>''')
svg('window','The target uses a different interval from the inputs.', '''
<text x="30" y="76" font-size="15">Synthetic query · cutoff 10 · horizon5 · target mean4</text>
<rect x="255" y="120" width="405" height="90" fill="#d5e9dc"/>
<line x1="95" y1="163" x2="805" y2="163" stroke="#699488" stroke-width="3"/>
<g fill="white" stroke="#183c33" stroke-width="3"><circle cx="215" cy="163" r="10"/><circle cx="710" cy="163" r="10"/></g>
<g fill="#276b51"><circle cx="345" cy="163" r="10"/><circle cx="605" cy="163" r="10"/></g>
<g font-size="17"><text x="168" y="104">time 10</text><text x="300" y="104">time 11</text><text x="560" y="104">time 15</text><text x="675" y="104">time 16</text><text x="157" y="242">99 excluded</text><text x="310" y="242">2 included</text><text x="565" y="242">6 included</text><text x="674" y="242">99 excluded</text></g>
<text x="30" y="304" font-size="20">Window(10,15] → (2+6)/2=4. A future label is not an input.</text>''')
svg('paths','Two information paths need two separate checks.', '''
<text x="30" y="78" font-size="15">Actual protocol contrast · same F1 query populations and graph edges</text>
<g fill="white" stroke="#bdd4ca"><rect x="30" y="107" width="245" height="124" rx="8"/><rect x="324" y="107" width="245" height="124" rx="8"/><rect x="618" y="107" width="245" height="124" rx="8"/></g>
<g font-size="17"><text x="48" y="140">Database rows</text><text x="48" y="173">Released: through 2010</text><text x="48" y="206">Corrected: through 2005</text><text x="342" y="140">Types + processors</text><text x="342" y="173">Fit vocabulary/statistics</text><text x="342" y="206">Transform graph rows</text><text x="636" y="140">Owner-cutoff sampler</text><text x="636" y="173">Check each node ≤query</text><text x="636" y="206">Predict finishing position</text></g>
<g font-size="30"><text x="287" y="176">→</text><text x="581" y="176">→</text></g>
<text x="30" y="280" font-size="19">Actual standings-points mean: fit by 2005 = 13.9668; released = 15.8498.</text>
<text x="30" y="318" font-size="17">Freeze processor fitting separately. Static-field availability remains unknown.</text>''',360)
r=json.loads((P/'evidence/l156/report.json').read_text());plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'l156','font.size':11})
fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained');fig.patch.set_facecolor('#f1f6f2')
for ax,split in zip(axes,['val','test']):
    ax.set_facecolor('#f1f6f2');ax.spines[['top','right']].set_visible(False)
    for x,(name,lane) in enumerate(r['lanes'].items()):
        color=['#276b51','#a2572e'][x];m=lane['metrics'][split]
        ax.scatter([x+(i-2)*.04 for i in range(5)],[v[split] for v in lane['records']],color=color,s=32)
        ax.errorbar(x+.25,m['mean'],yerr=m['sd'],fmt='D',color=color,capsize=4)
    ax.set(xticks=[0,1],xticklabels=['Released','Fit by 2005'],ylabel='MAE · finishing positions ↓',title='Validation' if split=='val' else 'Test');ax.grid(axis='y',alpha=.18)
fig.suptitle('Five full fits per policy · mean ± sample seed SD',fontsize=15)
for ext in ['svg','png']:fig.savefig(F/('results.'+ext),dpi=150,metadata={'Date':None} if ext=='svg' else {'Software':'Lesson156'})
plt.close(fig)
p=F/'results.svg';p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
