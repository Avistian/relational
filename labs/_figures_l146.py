"""Comparison-specific SVGs with portable PNG copies and explicit numeric evidence."""
import json
from pathlib import Path
from html import escape
import cairosvg
P=Path(__file__).resolve().parent;D=P/'figures/l146';D.mkdir(parents=True,exist_ok=True)
C={'gnn':'#247c7a','relgt':'#92507e','ink':'#203b48','muted':'#526c76','bg':'#f4f8f7'}
def text(x,y,s,size=18,color=None):return f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" fill="{color or C["ink"]}">{escape(str(s))}</text>'
def box(x,y,w,h,title,lines,color=None):
 return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="white" stroke="{color or C["gnn"]}" stroke-width="2"/>'+text(x+15,y+30,title,21,color)+''.join(text(x+15,y+60+i*25,line,17) for i,line in enumerate(lines))
def arrow(x,y,xx,yy):return f'<path d="M{x} {y} L{xx} {yy}" fill="none" stroke="#607983" stroke-width="2" marker-end="url(#a)"/>'
def save(name,title,body,height):
 svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}" role="img"><title>{escape(title)}</title><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#607983"/></marker></defs><rect width="1000" height="{height}" fill="{C["bg"]}"/>'+text(24,38,title,26)+body+'</svg>'
 (D/(name+'.svg')).write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(D/(name+'.png')),scale=1.5)
body=text(26,77,'Question: when can the circuit’s original attributes change a root prediction?',20)
for i,(title,note) in enumerate([('Driver','root / prediction'),('Result','one FK edge'),('Race','two FK edges'),('Circuit','three FK edges')]):
 x=30+i*245;body+=box(x,109,205,96,title,[note]);
 if i<3:body+=arrow(x+205,155,x+243,155)
body+=box(30,250,450,135,'Synchronous GNN · root readout',['Layer 1: result → driver','Layer 2: race → result → driver','Layer 3: circuit → race → result → driver'],C['gnn'])
body+=box(515,250,450,135,'Local all-pairs attention',['If all four tokens were retrieved:','one attention block can connect circuit → driver.','If circuit was excluded: no direct access.'],C['relgt'])
body+=text(30,427,'Assumptions matter: pooling, structural encoders and sampling can change this trace.',18)
body+=text(30,458,'Illustrative three-hop chain. The measured course sampler is limited to two hops.',18)
save('paths','Information access = sample × propagation × readout',body,490)
body=box(25,77,950,98,'One immutable query context, shared by both arms',['(driver, cutoff) → legal two-hop BFS → K=32 slots; all induced directed FK relations.','B≤128; same graph/row statistics, same sampling seed, same train/validation/test queries.'])
body+=box(25,223,450,130,'GNN control · initial B × 32 × 64',['Source typed row encoder + type embedding','+ sinusoidal relative age → linear projection','Sum → LayerNorm; no GIN positional encoder.'],C['gnn'])
body+=box(525,223,450,130,'Reduced RelGT · initial B × 32 × 64',['Five width-64 vectors: row/type/hop/time/GIN','Normalize → concatenate (width 320)','320 → 128 → 64 nonlinear mixture.'],C['relgt'])
body+=arrow(250,175,250,223)+arrow(750,175,750,223)
body+=box(25,399,450,155,'Two relation-specific mean layers',['For each relation: mean incoming neighbors.','Apply one learned 64 → 64 map per relation.','Add destination-type self transform.','Sum relations → LayerNorm → ReLU/dropout.'],C['gnn'])
body+=box(525,399,450,155,'Two local blocks + centroid branch',['Local Q,K,V: B × 4 × 32 × 16.','Each token attends all 32 sampled slots.','Root also reads 128 EMA centroids.','Local/global fusion: B × 128 → B × 64.'],C['relgt'])
body+=arrow(250,353,250,399)+arrow(750,353,750,399)
body+=box(25,600,450,103,'Root MLP → one scalar',['Root slot only → 64 → 64 → 1.','No centroid memory; dropout off at eval.'],C['gnn'])
body+=box(525,600,450,103,'Source head → one scalar',['Pooled local + global → scalar prediction.','Eval dropout off; fixed structural draws.'],C['relgt'])
body+=arrow(250,554,250,600)+arrow(750,554,750,600)
body+=box(25,749,950,99,'Common recipe; distinct models, parameter counts and computation',['L1 loss · Adam .001 · 10 full epochs · 3 seeds · first validation minimum · fixed clipping.','COURSE EXPERIMENT. Reduced RelGT and custom GNN; neither is the paper’s full comparison.'])
save('architecture','Two complete forward passes · the measured course variants',body,878)
p=json.loads((P/'evidence/l146/paper-table.json').read_text())
body=text(30,80,'Reported F1 table · every point is a configuration, not a fresh measurement.',19)
for i,c in enumerate(p['configs']):
 y=125+i*32;body+=text(35,y,c,18)+text(230,y,f"val {p['validation'][i]:.4f}",18)
 x=435+(p['test'][i]-3.5)*205
 body+=f'<circle cx="{x}" cy="{y-6}" r="6" fill="{C["gnn"] if i==3 else C["relgt"] if i==2 else "#82969c"}"/>'+text(x+12,y,f"{p['test'][i]:.4f}",17)
body+=text(450,100,'Test MAE →',18)
body+=box(25,440,460,123,'Choose by displayed validation',['L4 / dropout .3 → test 4.6316','Relative to cited RDL 4.022: −15.16% gain.'],C['gnn'])
body+=box(515,440,460,123,'Smallest displayed test',['L1 / dropout .5 → test 3.9170','Matches headline; +2.61% reported gain.'],C['relgt'])
body+=text(30,606,'Historical selection intent is unresolved. Final stochastic scores may differ from selection scores.',17)
save('selection','Selection can change the apparent conclusion',body,635)
if (P/'evidence/l146/summary.json').exists():
 r=json.loads((P/'evidence/l146/summary.json').read_text());body=text(30,80,'Fresh full-data course runs · three seed pairs · fixed recipe · not paper-result reproduction.',18)
 vals=[v['scores']['test'] for v in r['runs']];lo=min(vals)-.15;hi=max(vals)+.15
 def x(v):return 190+(v-lo)/(hi-lo)*680
 for j,seed in enumerate([0,1,2]):
  a=next(v for v in r['runs'] if v['seed']==seed and v['arm']=='gnn');b=next(v for v in r['runs'] if v['seed']==seed and v['arm']=='relgt');y=145+j*70
  body+=text(30,y+6,f'Seed {seed}',20)+f'<path d="M{x(a["scores"]["test"])} {y} H{x(b["scores"]["test"])}" stroke="#7d949a" stroke-width="2"/>'
  for arm,v in [('gnn',a),('relgt',b)]:
   xx=x(v['scores']['test']);body+=f'<circle cx="{xx}" cy="{y}" r="8" fill="{C[arm]}"/>'+text(xx-25,y-18,f"{v['scores']['test']:.3f}",17,C[arm])
 body+=text(190,345,'Lower test MAE is better; line joins one seed label, not equal initialization.',17)
 body+=box(25,389,460,130,'GNN control · mean ± sample SD',[f"Test {r['arms']['gnn']['test']['mean']:.4f} ± {r['arms']['gnn']['test']['sample_sd']:.4f}",f"Parameters {r['arms']['gnn']['parameters']:,}",f"Total fit worker time {r['arms']['gnn']['fit_seconds']:.1f} s"],C['gnn'])
 body+=box(515,389,460,130,'Reduced RelGT · mean ± sample SD',[f"Test {r['arms']['relgt']['test']['mean']:.4f} ± {r['arms']['relgt']['test']['sample_sd']:.4f}",f"Parameters {r['arms']['relgt']['parameters']:,}",f"Total fit worker time {r['arms']['relgt']['fit_seconds']:.1f} s"],C['relgt'])
 d=r['paired']['test'];body+=text(30,560,f"GNN − RelGT: {d['mean']:+.4f} ± {d['sample_sd']:.4f} MAE (seed SD; not a confidence interval).",19)
 body+=text(30,595,'Architecture, encodings, readout, state and runtime differ. This is descriptive task evidence.',17)
 save('results','Paired predictions · a conditional comparison',body,625)
print('Figures generated')
