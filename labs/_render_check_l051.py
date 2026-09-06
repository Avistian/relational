"""Actual librsvg glyph boxes and render inspection; no browser-layout claim."""
import ctypes as c
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from _figures_l047 import render
folder=Path('/tmp/l051-svg')
r=c.CDLL('librsvg-2.so.2');g=c.CDLL('libgobject-2.0.so.0')
class Rect(c.Structure):_fields_=[(k,c.c_double) for k in ('x','y','width','height')]
r.rsvg_handle_new_from_data.argtypes=[c.c_char_p,c.c_size_t,c.c_void_p];r.rsvg_handle_new_from_data.restype=c.c_void_p
r.rsvg_handle_get_geometry_for_layer.argtypes=[c.c_void_p,c.c_char_p,c.POINTER(Rect),c.POINTER(Rect),c.POINTER(Rect),c.c_void_p];r.rsvg_handle_get_geometry_for_layer.restype=c.c_int
g.g_object_unref.argtypes=[c.c_void_p]
violations=[];count=0
for source in sorted(folder.glob('*.svg')):
    tree=ET.fromstring(source.read_bytes());texts=[n for n in tree.iter() if n.tag.endswith('}text')]
    for i,n in enumerate(texts):n.set('id',f'g{i}')
    data=ET.tostring(tree);handle=r.rsvg_handle_new_from_data(data,len(data),None);boxes=[]
    for node in texts:
        ink,logical=Rect(),Rect();assert r.rsvg_handle_get_geometry_for_layer(handle,('#'+node.attrib['id']).encode(),c.byref(Rect(0,0,340,240)),c.byref(ink),c.byref(logical),None)
        box=(ink.x,ink.y,ink.x+ink.width,ink.y+ink.height)
        if box[0]<0 or box[1]<0 or box[2]>340 or box[3]>240:violations.append([source.name,node.text,box])
        for other,b in boxes:
            if min(box[2],b[2])-max(box[0],b[0])>.5 and min(box[3],b[3])-max(box[1],b[1])>.5:violations.append([source.name,'text overlap',node.text,other])
        boxes.append((node.text,box));count+=1
    g.g_object_unref(handle);render(source,source.with_suffix('.png'))
print(json.dumps(dict(panels=len(list(folder.glob('*.svg'))),labels=count,violations=violations),indent=2))
assert not violations
