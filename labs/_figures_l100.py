"""Checkpoint-specific editable architecture, typed-index trace and measured seed points."""
from pathlib import Path
import json
import cairosvg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;D=P/'figures/l100';D.mkdir(parents=True,exist_ok=True)
BASE='''<svg xmlns="http://www.w3.org/2000/svg" width="1060" height="HEIGHT" viewBox="0 0 1060 HEIGHT" role="img"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#41666c"/></marker></defs><rect width="1060" height="HEIGHT" rx="18" fill="#f5f4ee"/><style>text{font-family:Arial,sans-serif;font-size:17px;fill:#173b43}.title{font-size:25px;font-weight:bold}.label{font-size:20px;font-weight:bold}.small{font-size:15px}.box{fill:white;stroke:#a8bfbc;stroke-width:1.5}.arrow{fill:none;stroke:#41666c;stroke-width:2;marker-end:url(#arrow)}</style>'''
def save(name,body,height):
 svg=BASE.replace('HEIGHT',str(height))+body+'</svg>'
 (D/(name+'.svg')).write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(D/(name+'.png')),scale=1.5)
save('architecture','''
<text x="28" y="39" class="title">The checkpoint · preserve the target while shrinking the graph</text>
<text x="28" y="66">All ACM rows → typed local neighborhoods → one seed-only objective → frozen evaluation</text>
<rect x="28" y="92" width="300" height="178" rx="12" class="box"/>
<text x="44" y="120" class="label">Full graph and labels</text>
<text x="44" y="150">Paper [4025,1903] · words</text><text x="44" y="177">Author [7167,1] · constant</text><text x="44" y="204">Subject [60,1] · constant</text><text x="44" y="232" class="small">4 edge stores; conference excluded</text><text x="44" y="253" class="small">Train 804 / val 403 / test 2818</text>
<path d="M329 181H369" class="arrow"/>
<rect x="374" y="92" width="308" height="178" rx="12" fill="#e5eeec" stroke="#88aba7"/>
<text x="390" y="120" class="label">Two-hop sampling</text>
<text x="390" y="150">B ≤ 256 training paper seeds</text><text x="390" y="177">8 neighbors / relation / hop</text><text x="390" y="204">Global rows ↔ local n_id maps</text><text x="390" y="232" class="small">Same sampled edges across arms</text><text x="390" y="253" class="small">First B papers = supervised seeds</text>
<path d="M684 181H724" class="arrow"/>
<rect x="730" y="92" width="302" height="178" rx="12" class="box"/>
<text x="746" y="120" class="label">Typed adapters</text><text x="746" y="151">x[type] → affine → tanh</text><text x="746" y="179">h[type]: [N_type_local,32]</text><text x="746" y="207">Parameters shared over batches</text><text x="746" y="237" class="small">No labels among feature channels</text>
<path d="M880 271V292H151V312" class="arrow"/><path d="M880 292H406V312" class="arrow"/><path d="M880 292H663V312" class="arrow"/><path d="M880 292H915V312" class="arrow"/>
<rect x="28" y="318" width="238" height="195" rx="12" fill="#e0edf2" stroke="#82a8b9"/>
<text x="44" y="348" class="label">R-GCN × 2</text><text x="44" y="380">h_source × W_relation</text><text x="44" y="407">Mean within relation</text><text x="44" y="434">sum + learned self path</text><text x="44" y="461">ReLU → [N_type,32]</text><text x="44" y="490" class="small">Separate relation degrees</text>
<rect x="282" y="318" width="238" height="195" rx="12" fill="#fae7d6" stroke="#ca9d76"/>
<text x="298" y="348" class="label">HGT × 2</text><text x="298" y="379">Typed Q/K/V · [N,4,8]</text><text x="298" y="406">Relation scores + values</text><text x="298" y="433">Joint receiver softmax</text><text x="298" y="460">Sum → GELU → typed A</text><text x="298" y="489" class="small">Gate + residual + norm</text>
<rect x="536" y="318" width="238" height="195" rx="12" fill="#f1eada" stroke="#b9a26c"/>
<text x="552" y="348" class="label">Uniform HGT × 2</text><text x="552" y="379">Keep message/output</text><text x="552" y="406">Remove learned scorer</text><text x="552" y="433">Weight = 1 / degree</text><text x="552" y="460">Match shared weights</text><text x="552" y="489" class="small">Not relation-wise means</text>
<rect x="790" y="318" width="242" height="195" rx="12" fill="#eae5f2" stroke="#a899be"/>
<text x="806" y="348" class="label">Feature MLP × 2</text><text x="806" y="379">Paper adapter only</text><text x="806" y="406">Linear → ReLU</text><text x="806" y="433">Ignore all edges</text><text x="806" y="460">Same seed exposure</text><text x="806" y="489" class="small">Includes sampler overhead</text>
<path d="M149 515V538H914V515M405 515V538M660 515V538" fill="none" stroke="#41666c" stroke-width="2"/><path d="M530 538V560" class="arrow"/>
<rect x="235" y="565" width="590" height="75" rx="12" class="box"/>
<text x="254" y="593" class="label">Paper logits [N_local,3] → keep first B rows</text><text x="254" y="619">Cross-entropy on seed labels → backward → one Adam update</text>
<rect x="28" y="668" width="487" height="100" rx="12" fill="#e5eeec" stroke="#88aba7"/>
<text x="44" y="696" class="label">Correctness lane · fixed weights</text><text x="44" y="724">All neighbors: seed logits = full-graph logits</text><text x="44" y="749">Σ (B/N) × batch gradient = full gradient</text>
<rect x="539" y="668" width="493" height="100" rx="12" class="box"/>
<text x="555" y="696" class="label">Training · weights change</text><text x="555" y="724">Validation selects epoch + rate</text><text x="555" y="749">Freeze → full test logits → accuracy / macro F1</text>
<text x="28" y="805" class="small">Static course experiment · no RTE · native NeighborLoader differs from paper HGSampling · no historical parity claim</text>
''',830)
save('identity','''
<text x="28" y="40" class="title">Local index 1 means different things in different type stores</text>
<text x="28" y="70">Illustrative batch · first two paper rows are seeds · paper 83 is context</text>
<rect x="28" y="104" width="300" height="190" rx="12" class="box"/>
<text x="48" y="136" class="label">AUTHOR n_id</text><text x="48" y="174">local 0 → global author 9</text><text x="48" y="212">local 1 → global author 2</text><text x="48" y="260" class="small">Use this map for source indices.</text>
<rect x="375" y="104" width="302" height="190" rx="12" fill="#fae7d6" stroke="#ca9d76"/>
<text x="395" y="136" class="label">Local edge_index</text><text x="395" y="174">source 1 → receiver 0</text><text x="395" y="212">source 0 → receiver 2</text><text x="395" y="260" class="small">Relation: author → paper</text>
<rect x="728" y="104" width="302" height="190" rx="12" class="box"/>
<text x="748" y="136" class="label">PAPER n_id</text><text x="748" y="174">local 0 → paper 17 · SEED</text><text x="748" y="212">local 1 → paper 4 · SEED</text><text x="748" y="250">local 2 → paper 83 · context</text>
<path d="M329 204H373" class="arrow"/><path d="M679 204H726" class="arrow"/>
<rect x="120" y="331" width="820" height="82" rx="12" fill="#e5eeec" stroke="#88aba7"/>
<text x="145" y="362" class="label">Global edges: (author 2 → paper 17), (author 9 → paper 83)</text><text x="145" y="391">Supervision: only paper 17 and paper 4; paper 83 can still send context messages.</text>
''',440)
save('weighting','''
<text x="28" y="40" class="title">The last batch is shorter · its mean needs a smaller weight</text>
<text x="28" y="73">Fixed five-seed loss vector [0.2, 0.8, 0.5, 1.1, 0.4] · arithmetic illustration</text>
<rect x="28" y="112" width="470" height="135" rx="12" fill="#e0edf2" stroke="#82a8b9"/>
<text x="48" y="147" class="label">Batch A · three seeds</text><text x="48" y="183">(0.2 + 0.8 + 0.5) / 3 = 0.50</text><text x="48" y="219">Weight 3/5 × mean 0.50 = contribution 0.30</text>
<rect x="558" y="112" width="474" height="135" rx="12" fill="#fae7d6" stroke="#ca9d76"/>
<text x="578" y="147" class="label">Batch B · two seeds</text><text x="578" y="183">(1.1 + 0.4) / 2 = 0.75</text><text x="578" y="219">Weight 2/5 × mean 0.75 = contribution 0.30</text>
<text x="28" y="301" class="label">Correct: 0.30 + 0.30 = 0.60</text><text x="558" y="301" class="label">Naive: (0.50 + 0.75) / 2 = 0.625</text>
<text x="28" y="345">For gradients: accumulate at fixed weights. Sequential optimizer steps are a different computation.</text>
''',375)
r=json.loads((P/'_experiment_l100_results.json').read_text())
assert r['status']=='COMPLETE'
fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained');colors=['#377890','#b86b35','#a28542','#80709c']
for i,(arm,color) in enumerate(zip(['rgcn','hgt','hgt_uniform','mlp'],colors)):
 vals=[100*x['accuracy'] for x in r['selected'] if x['arm']==arm]
 ax.scatter([i-.12,i,i+.12],vals,s=65,color=color,zorder=3)
 mean=sum(vals)/3;ax.plot([i-.24,i+.24],[mean,mean],color=color,lw=2)
 for j,v in enumerate(vals):ax.annotate(str(j),(i+(j-1)*.12,v),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
ax.set_xticks(range(4),['R-GCN','HGT','Uniform HGT','Feature MLP']);ax.set_ylim(0,103);ax.set_ylabel('Full-test accuracy (%)');ax.grid(axis='y',alpha=.2);ax.set_title('Complete L100 course experiment · three seeds on one fixed ACM split',loc='left',fontsize=13)
fig.text(.01,-.035,'Dots = individual seeds (0,1,2); bars = means. No dataset-level confidence interval or universal model ranking.',fontsize=10)
for ext in ['png','svg']:fig.savefig(D/f'results.{ext}',dpi=160,bbox_inches='tight')
plt.close(fig)
print('Wrote four figures')
