"""Shared portable RelGNN overview: expose the route operation inside the full model."""
from html import escape
from pathlib import Path
import cairosvg


def draw(destination):
    destination = Path(destination)
    teal, amber, purple, ink = '#087e82', '#a85d0a', '#7051a0', '#163746'
    parts = []
    def label(x, y, value, size=17, color=ink):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-family="DejaVu Sans,sans-serif">{escape(value)}</text>')
    def box(x, y, w, h, title, lines, color=teal):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="white" stroke="{color}" stroke-width="2"/>')
        label(x+16,y+29,title,20,color)
        for i,line in enumerate(lines): label(x+16,y+57+24*i,line,20)
    def arrow(x,y,xx,yy,color=teal):
        parts.append(f'<path d="M{x},{y} L{xx},{yy}" fill="none" stroke="{color}" stroke-width="2.5" marker-end="url(#arrow)"/>')
    label(28,38,'RelGNN · keep the source role until the destination chooses',26)
    label(28,66,'Released F1 variant: one composite layer · four heads · 128 channels per head',17,teal)
    box(28,90,1044,86,'01  Retrieve and encode each query',[
        '(driver, cutoff) → sample 128 / 64 → table encoders + query-relative time',
        'Each row has 128 coordinates and its own query owner. Keys define graph links.'])
    arrow(550,176,550,205)
    parts.append('<rect x="28" y="205" width="1044" height="377" rx="16" fill="#eaf4f2"/>')
    label(46,237,'02  Inspect ONE ordered constructor → result → driver route',21,teal)
    box(48,258,305,92,'Source: constructor s',['Its own role’s 128-vector','Example scalars: 2, 4'],teal)
    box(393,258,305,92,'Fact: result f',['Keep the event’s attributes','Example scalars: 1, 0'],amber)
    box(738,258,314,92,'Destination: driver d',['Supplies the attention query','Illustrative query: 1'],purple)
    arrow(201,350,201,382)
    arrow(545,350,545,382,amber)
    box(48,382,650,92,'Fuse source + fact, separately for this route',[
        'u = Wsource · sum(s) + bsource + Wfact · f',
        'Scalar fusion: 2 + 1 = 3; 4 + 0 = 4.'])
    arrow(373,474,373,502)
    arrow(895,350,895,502,purple)
    box(48,502,1004,61,'Attention: softmax over THIS destination’s incoming facts',[],purple)
    label(64,552,'Scalar example: scores [3, 4] → weights [0.269, 0.731] → weighted value 3.731',20)
    arrow(550,582,550,614)
    box(28,614,1044,102,'03  Restore the real dimensions and combine routes',[
        '4 heads × 128 → weighted values → concatenate to 512 + destination skip',
        'Project to 128 → sum routes → LayerNorm → ReLU. Also update fact states.'])
    arrow(550,716,550,746)
    box(28,746,1044,94,'04  Predict at the queried driver',[
        'Root vectors [B,128] → prediction head → one finishing-position estimate',
        'Train: unclipped L1 loss → gradients. Evaluate: train-percentile clipping.'])
    label(28,875,'Two graph edges ≠ two model layers. B = query batch size. The scalar trace is illustrative.',17)
    svg='<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="904" viewBox="0 0 1100 904" role="img"><title>RelGNN: full forward pass with an expanded composite route</title><desc>A constructor and result fuse before driver-owned attention; four heads combine into the root prediction. All sampled rows retain their query cutoff.</desc><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#087e82"/></marker></defs><rect width="1100" height="904" fill="#f5f8f7"/>'+''.join(parts)+'</svg>'
    (destination/'architecture.svg').write_text(svg)
    cairosvg.svg2png(bytestring=svg.encode(),write_to=str(destination/'architecture.png'),scale=1.5)
