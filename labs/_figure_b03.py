"""Vertical model-specific architecture; SVG for browser, PNG for notebook portability."""
from pathlib import Path
import html
import cairosvg
P=Path(__file__).resolve().parent/'figures/b03';P.mkdir(parents=True,exist_ok=True)
parts=['<svg xmlns="http://www.w3.org/2000/svg" width="500" height="1210" viewBox="0 0 500 1210"><defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#52716e"/></marker></defs><rect width="500" height="1210" rx="18" fill="#f3f6f1"/>']
def text(y,s,size=17,color='#183d38',weight='400'):
 parts.append(f'<text x="250" y="{y}" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{html.escape(s)}</text>')
def box(y,h,title,lines,fill='#fff',stroke='#b1c5b8'):
 parts.append(f'<rect x="22" y="{y}" width="456" height="{h}" rx="12" fill="{fill}" stroke="{stroke}"/>');text(y+30,title,19,weight='600')
 for i,s in enumerate(lines):text(y+58+24*i,s)
def arrow(y1,y2):parts.append(f'<path d="M250 {y1}V{y2}" stroke="#52716e" stroke-width="2" marker-end="url(#arrow)"/>')
text(34,'ONE TABLE, TWO TIMESCALES',16,weight='600')
box(53,131,'Before your task · pretraining',['Synthetic tasks → support / query episodes','Query prediction loss updates θ','NOT RUN in B03'],fill='#e5eadf')
arrow(185,212);text(236,'Freeze θ. Now supply your table.',18,weight='600')
box(258,108,'Support and query inputs',['Support: Xₛ + yₛ   |   Query: Xq only','No query target enters the model.'])
arrow(368,390)
box(400,108,'Historical v2 wrapper · four views',['Fit transforms; permute features / classes','Track each view and each label meaning.'])
arrow(509,531)
box(541,108,'Cell groups + target token',['Pairs of features; missingness indicators','(S + Q) × (ceil(F / 2) + 1) × 192'])
arrow(650,672)
box(682,182,'12 alternating-attention blocks',['Feature attention → mix within each row','Row attention → read support keys / values','Feed-forward + residual normalization','6 heads × 32 coordinates'],fill='#dcece6',stroke='#548874')
arrow(865,887)
box(897,108,'Read the query target tokens',['Classifier head → temperature → probabilities','Restore class columns for every view.'])
arrow(1006,1028)
box(1038,108,'Average aligned view probabilities',['Q × C predictions → independent scoring','Same weights; different context / computation.'],fill='#f1e8d4',stroke='#beaa77')
text(1180,'B03: pinned v2 diagnostic ≠ paper benchmark',16)
parts.append('</svg>');svg=''.join(parts);(P/'architecture.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/'architecture.png'),scale=1.5)
