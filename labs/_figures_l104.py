"""Figures expose the distinct clocks, recursive dependency and fitting boundary."""
import json
from pathlib import Path
from html import escape
import cairosvg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent;F=P/'figures/l104';F.mkdir(parents=True,exist_ok=True)
INK='#183742';TEAL='#087e82';RED='#b3473e';GOLD='#b17930';GRAY='#63767d'
def text(x,y,s,size=18,color=INK,bold=False):return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{"600" if bold else "400"}">{escape(s)}</text>'
def line(x,y,xx,yy,color=TEAL,dash=False):return f'<path d="M{x} {y}L{xx} {yy}" stroke="{color}" stroke-width="3" fill="none"'+(' stroke-dasharray="7 5"' if dash else '')+'/>'
def circle(x,y,color):return f'<circle cx="{x}" cy="{y}" r="7" fill="{color}"/>'
def box(x,y,w,h):return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="white" stroke="#bdcfd1"/>'
def diagram(name,h,body):
 svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="940" height="{h}" viewBox="0 0 940 {h}"><style>text{{font-family:DejaVu Sans,sans-serif}}</style><rect width="940" height="{h}" fill="#fafaf7"/>{body}</svg>'
 (F/f'{name}.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(F/f'{name}.png'),scale=1.5)
b=text(30,38,'Two clocks decide whether one record can be used',25,bold=True)
b+=text(30,68,'Fixed query: day 8 • strict event history • availability checked separately',16,GRAY)
x=lambda day:160+day*60
for y,label,event,arrival,color in [(155,'Original',3,4,TEAL),(245,'Correction',3,9,RED)]:
 b+=text(30,y+6,label,18,bold=True)+line(x(0),y,x(12),y,'#c8d3d4')+line(x(event),y,x(arrival),y,color)
 b+=circle(x(event),y,color)+f'<path d="M{x(arrival)} {y-9}l9 9l-9 9l-9 -9Z" fill="{color}"/>'
 b+=text(x(event)-24,y-21,'event 3',15,color)+text(x(arrival)-34,y+35,f'arrival {arrival}',15,color)
b+=line(x(8),100,x(8),290,INK,True)+text(x(8)-30,96,'query 8',16,INK,True)
for day in [0,3,4,8,9,12]:b+=text(x(day)-5,321,str(day),15,GRAY)
b+=box(30,348,425,102)+text(48,379,'Original: 3 < 8 AND 4 ≤ 8',20,TEAL,True)+text(48,415,'ALLOW · both conditions hold',18)
b+=box(485,348,425,102)+text(503,379,'Correction: 3 < 8 BUT 9 > 8',20,RED,True)+text(503,415,'BLOCK · happened earlier, arrived later',18)
diagram('clocks',476,b)
b=text(30,38,'The child has a stricter question than the root',25,bold=True)+text(30,68,'TGAT example • parent time is not a universal cutoff for all hops',16,GRAY)
b+=box(30,100,245,104)+text(48,135,'A queried at 8',22,bold=True)+text(48,171,'A–B event at 5',18)
b+=line(275,151,345,151)+text(288,132,'5 < 8',15,TEAL)
b+=box(345,100,245,104)+text(363,135,'B queried at 5',22,bold=True)+text(363,171,'child cutoff = 5',18)
b+=line(590,151,646,151)+line(646,151,646,284)
b+=line(646,151,692,151)+box(692,100,220,104)+text(710,136,'B–D event at 2',19,bold=True)+text(710,173,'2 < 5 · ALLOW',18,TEAL)
b+=line(646,284,692,284,RED,True)+box(692,233,220,104)+text(710,269,'B–C event at 6',19,bold=True)+text(710,306,'6 ≥ 5 · BLOCK',18,RED)
b+=text(30,268,'Wrong check: 6 < 8 passes.',21,RED,True)+text(30,302,'It answers the root’s question, not the child’s.',18)
b+=box(30,371,882,83)+text(48,402,'Legal dependency path:  A@8 → B@5 → D@2',21,TEAL,True)+text(48,433,'Log sampled event time against its own request cutoff at every hop.',17)
diagram('recursion',480,b)
b=text(30,38,'Old query ≠ available training label',25,bold=True)+text(30,68,'Horizon = 5 days • reporting delay = 2 days • model fit = day 10',16,GRAY)
x=lambda day:160+day*60
for y,label,q,mature,color in [(155,'Query A',2,9,TEAL),(260,'Query B',4,11,RED)]:
 b+=text(30,y+6,label,18,bold=True)+line(x(0),y,x(12),y,'#c8d3d4')
 b+=f'<rect x="{x(q)}" y="{y-10}" width="300" height="20" fill="{color}" opacity=".23"/>'+line(x(q+5),y,x(mature),y,color,True)
 b+=circle(x(q),y,color)+circle(x(mature),y,color)
 b+=text(x(q)-17,y-23,f'query {q}',15,color)+text(x(mature)-31,y+36,f'mature {mature}',15,color)
b+=line(x(10),93,x(10),312,INK,True)+text(x(10)-30,89,'fit 10',17,INK,True)
b+=text(190,330,'Solid band: future target window. Dashes: reporting delay.',16,GRAY)
b+=box(30,360,425,100)+text(48,394,'A: 2 < 10 AND 9 ≤ 10',21,TEAL,True)+text(48,430,'Eligible target at fitting time',18)
b+=box(485,360,425,100)+text(503,394,'B: 4 < 10 BUT 11 > 10',21,RED,True)+text(503,430,'Exclude until the target matures',18)
diagram('maturity',487,b)
results=P/'_analysis_l104_results.json'
if results.exists():
 r=json.loads(results.read_text());fig,axes=plt.subplots(1,2,figsize=(10,4.8),sharey=True);fig.patch.set_facecolor('#fafaf7')
 for ax,lane,title in zip(axes,['all','new'],['All test events','New-node test subset']):
  ax.set_facecolor('#fafaf7');ax.axhline(0,color=GRAY,lw=1)
  for i,mode in enumerate(['inclusive','lookahead']):
   values=[row['comparisons'][lane][mode]['delta_pp'] for row in r['records']]
   ax.scatter(np.linspace(i-.12,i+.12,len(values)),values,color=TEAL if i==0 else RED,s=27)
   ax.plot([i-.22,i+.22],[np.mean(values)]*2,color=INK,lw=3)
  ax.set_xticks([0,1],['Same-time\ninclusive','One-day\nlookahead']);ax.set_ylabel('Pooled AP change vs strict (pp)');ax.set_title(title);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
 fig.suptitle('Illegal access changes the score • fixed trained checkpoints',fontsize=14)
 fig.text(.5,.015,f'{len(r["records"])}/10 seeds · dots: paired changes · bars: means · course intervention, not paper result',ha='center',fontsize=9)
 fig.tight_layout(rect=[0,.055,1,.94]);fig.savefig(F/'results.svg');fig.savefig(F/'results.png',dpi=160);plt.close(fig)
print('L104 mechanism figures built')
