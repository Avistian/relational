"""Editable vector + notebook-portable raster of the actual course construction."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent/'figures/l096';P.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(14,8),facecolor='#faf8f2');ax.set(xlim=(0,14),ylim=(0,8));ax.axis('off')
ax.text(.4,7.55,'A database row becomes a node; an FK becomes a typed route',fontsize=18,weight='bold',color='#193248')
def box(x,y,w,h,title,body,color='#e1efec'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12',facecolor=color,edgecolor='#74928e'))
    ax.text(x+.13,y+h-.35,title,fontsize=11,weight='bold',color='#193248')
    ax.text(x+.13,y+h-.7,body,fontsize=11,va='top',linespacing=1.5,color='#193248')
box(.4,4.1,2.65,2.7,'CUSTOMER · 3 nodes','PK 10 → local 0\nPK 30 → local 1\nPK 90 → local 2\n\n90 stays isolated')
box(4.0,4.1,2.7,2.7,'ORDERS · 3 nodes','PK 100 → local 0\n  buyer 10 / referrer 30\nPK 200 → local 1\n  buyer 30 / referrer NULL\nPK 300 → local 2\n  buyer 10 / referrer NULL')
box(7.65,4.1,2.7,2.7,'LINE_ITEM · 4 nodes','PK (100,1) → 0 / qty 2\nPK (100,2) → 1 / qty 3\nPK (100,3) → 2 / qty 1\nPK (200,1) → 3 / qty 4\n\nComposite identity retained')
box(11.3,4.1,2.25,2.7,'PRODUCT · 3 nodes','PK 10 → local 0\nPK 50 → local 1\nPK 80 → local 2\n\n80 stays isolated')
for start,end,label in [(4,3.05,'buyer /\nreferrer'),(7.65,6.7,'order'),(10.35,11.3,'product')]:
    ax.annotate('',xy=(end,5.5),xytext=(start,5.5),arrowprops={'arrowstyle':'->','lw':2,'color':'#087f8c'})
    ax.text((start+end)/2,5.85,label,fontsize=10,ha='center',color='#087f8c')
ax.text(.4,3.55,'4 node types  •  4 forward FK roles  •  4 optional reverse stores  •  12 + 12 edges',fontsize=13,weight='bold')
box(.4,1.35,4.0,1.65,'1  Resolve identity','Stable sorted PK → local index\nNULL: no edge; orphan: reject\nForward tensor: [2, number of links]')
box(4.9,1.35,4.0,1.65,'2  Preserve two line paths','customer 10 ← order 100 ← line (100,1) → product 10\ncustomer 10 ← order 100 ← line (100,2) → product 10\nQuantities: 2 + 3 = 5', '#f6e7cb')
# Long paths have a dedicated readable caption below rather than overflowing the box.
ax.texts[-1].set_text('Two line nodes reach product 10\nvia the same order and buyer.\nQuantities: 2 + 3 = 5')
box(9.4,1.35,4.0,1.65,'3  Verify the query','Graph paths = SQL JOIN rows\nSUM(quantity) = 5, not 1 or 2\nLEFT JOIN keeps zero-order customers')
ax.text(.4,.65,'Boundary: this is a static representation and query audit. No encoder, training loss, or predictive score is claimed.',fontsize=12,color='#554d42')
fig.savefig(P/'schema.svg',bbox_inches='tight');fig.savefig(P/'schema.png',dpi=150,bbox_inches='tight');plt.close(fig)
