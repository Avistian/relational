"""Editable vectors and portable PNG copies; all pictured scalar arithmetic is computed."""
from pathlib import Path
import html,math,cairosvg
P=Path(__file__).resolve().parent/'figures/l102';P.mkdir(parents=True,exist_ok=True)
INK='#173540';TEAL='#196b70'
def canvas(h):return [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{h}" viewBox="0 0 1000 {h}" role="img"><rect width="1000" height="{h}" fill="#fcfbf7"/><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="{TEAL}"/></marker></defs>']
def text(s,x,y,value,size=18,color=INK,weight='normal'):s.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans,sans-serif" font-size="{size}" fill="{color}" font-weight="{weight}">{html.escape(value)}</text>')
def box(s,x,y,w,h,title,lines,fill='#edf6f4'):
 s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="#b9cecb"/>');text(s,x+16,y+29,title,19,TEAL,'bold')
 for i,l in enumerate(lines):text(s,x+16,y+57+i*25,l,16)
def arrow(s,x,y,xx,yy):s.append(f'<path d="M{x} {y} L{xx} {yy}" stroke="{TEAL}" stroke-width="2.5" fill="none" marker-end="url(#arrow)"/>')
def save(s,name):
 s.append('</svg>');data=''.join(s);(P/f'{name}.svg').write_text(data);cairosvg.svg2png(bytestring=data.encode(),write_to=str(P/f'{name}.png'),scale=1.5)
s=canvas(820)
text(s,30,38,'TGN-attn / from event history to an edge score',27,INK,'bold');text(s,30,68,'Wikipedia release variant · 172-dimensional memory · one attention layer',17)
box(s,30,100,430,116,'1   Past observed events',['(source, destination, time, edge[172])','Pending last message per node: [688]'])
box(s,530,100,440,116,'2   Recompute candidate memory',['GRU(message[688], old state[172])','Differentiable state S* [9,228 × 172]']);arrow(s,460,158,530,158)
box(s,30,264,430,168,'3   Legal temporal context',['Query nodes: source, positive, negative','Most recent 10 edges: time < query time','Each neighbor: state[172] + edge[172]','                         + time encoding[172]'])
box(s,530,264,440,168,'4   Two-head temporal attention',['Query [344] = own state + time(0)','Key/value [516] → attention output [344]','Append own state → MLP → z[172]','Empty history: keep own-state path']);arrow(s,460,345,530,345);arrow(s,750,216,750,264)
box(s,530,480,440,112,'5   Shared edge decoder',['concat(z_source, z_candidate) [344]','MLP 344 → 172 → 1 → sigmoid score']);arrow(s,750,432,750,480)
box(s,30,480,430,112,'6   Signal from observed events',['Observed pair: 1; sampled destination: 0','BCE(positive) + BCE(negative)']);arrow(s,530,536,460,536)
box(s,30,646,940,122,'7   Queue current positives for a later batch',['Persist consumed old messages for real endpoints. Construct both directed raw messages.','[own state 172 | other state 172 | event 172 | elapsed encoding 172] = 688.','Last event per node wins. Detach after the optimizer step. Negatives never write memory.'],'#fff3df');arrow(s,245,592,245,646)
text(s,30,800,'Inference: freeze weights; update after observed events. Never ingest hypothetical candidates.',17);save(s,'architecture')
s=canvas(430);text(s,30,40,'One scalar GRU update, with every intermediate exposed',26,INK,'bold');text(s,30,73,'Illustrative fixed weights only. The real model learns all gates in 172 dimensions.',17)
old=.4;c=math.tanh(1+.25*old);new=.5*old+.5*c
box(s,30,115,240,150,'Inputs',['Old memory = 0.4000','Event feature = 1.0000','Reset r = keep z = 0.5'])
box(s,330,115,300,150,'Candidate',['Reset hidden: 0.25 × 0.4','Preactivation: 1 + 0.1 = 1.1',f'tanh(1.1) = {c:.4f}'])
box(s,690,115,280,150,'Updated memory',['0.5 × old + 0.5 × candidate',f'0.2000 + {(.5*c):.4f}',f'= {new:.4f}']);arrow(s,270,187,330,187);arrow(s,630,187,690,187)
text(s,30,316,'Why this is a gate, not an overwrite',21,TEAL,'bold');text(s,30,350,'z near 1 keeps history; z near 0 favors the candidate. r controls the old-state contribution',18);text(s,30,380,'inside that candidate. Last-message aggregation selects the input; it is not a GRU gate.',18);save(s,'gru-trace')
s=canvas(470);text(s,30,40,'Two clocks: observed now, memory update learned later',25,INK,'bold');text(s,30,72,'A–B at t=1, A–C at t=2, B–C at t=3. Batch size = 2.',17)
box(s,30,110,435,225,'Batch 1: events 1 and 2',['Start: zero memory; no pending messages.','Score A–B and A–C with this memory.','Queue messages after both predictions.','For A, last aggregation keeps t=2.','For B, keep t=1. For C, keep t=2.','End: update weights; detach messages.'])
box(s,530,110,440,225,'Batch 2: event 3',['Read pending A(t=2), B(t=1), C(t=2).','Current GRU weights → candidate S*.','Score B–C using S*.','Loss gradients reach the GRU through S*.','Queue B–C after its prediction.','Past data; current trainable updater.']);arrow(s,465,220,530,220)
text(s,30,383,'Shared memory does not mean identical graph neighborhoods.',21,TEAL,'bold');text(s,30,417,'The t=2 query can sample the t=1 edge. Its memory still excludes batch-1 updates.',18);text(s,30,446,'Batch size changes memory access even with a strict temporal adjacency cutoff.',18);save(s,'batch-timeline')
print('Three SVG/PNG mechanism diagrams generated')
