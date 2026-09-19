"""Reject overlapping components and connections drawn through unrelated boxes."""
from _solution_map_content import MAPS
from _build_solution_maps import edge_points
# Test line segments against the open interior of an unrelated node rectangle.
def intersects(a,b,r):
 x,y,w,h=[r[k] for k in ['x','y','w','h']];xmin,xmax,ymin,ymax=x+2,x+w-2,y+2,y+h-2
 dx,dy=b[0]-a[0],b[1]-a[1];lo,hi=0.,1.
 for p,q in [(-dx,a[0]-xmin),(dx,xmax-a[0]),(-dy,a[1]-ymin),(dy,ymax-a[1])]:
  if p==0:
   if q<0:return False
  elif p<0:lo=max(lo,q/p)
  else:hi=min(hi,q/p)
 return lo<hi
issues=[]
for n,maps in MAPS.items():
 for m in maps:
  for a in m.nodes:
   assert 0<=a['x'] and 0<=a['y'] and a['x']+a['w']<=1000 and a['y']+a['h']<=m.height,(n,a)
   for b in m.nodes:
    if a['key']>=b['key']:continue
    assert not (max(a['x'],b['x'])<min(a['x']+a['w'],b['x']+b['w']) and max(a['y'],b['y'])<min(a['y']+a['h'],b['y']+b['h'])),(n,a['key'],b['key'])
  for e in m.edges:
   points=edge_points(m,e)
   for node in m.nodes:
    if node['key'] in [e['a'],e['b']]:continue
    if any(intersects(a,b,node) for a,b in zip(points,points[1:])):issues.append((n,m.key,e['a'],e['b'],node['key']))
print({"route_intersections":issues,"maps_checked":sum(map(len,MAPS.values()))})
assert not issues
