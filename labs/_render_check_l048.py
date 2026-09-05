"""Inspect actual librsvg glyph geometry for all desktop/mobile diagram states.

This is a vector-renderer check, explicitly not browser layout verification.
"""
import ctypes as c
import json
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET
from _figures_l047 import render

HERE=Path(__file__).parent


def main():
    folder=Path('/tmp/l048-rendered-states');folder.mkdir(exist_ok=True)
    subprocess.run(['node',str(HERE/'_viz_check_l048.js'),str(folder)],check=True)
    r=c.CDLL('librsvg-2.so.2');g=c.CDLL('libgobject-2.0.so.0')
    class Rect(c.Structure):_fields_=[(k,c.c_double) for k in ('x','y','width','height')]
    r.rsvg_handle_new_from_data.argtypes=[c.c_char_p,c.c_size_t,c.c_void_p];r.rsvg_handle_new_from_data.restype=c.c_void_p
    r.rsvg_handle_get_geometry_for_layer.argtypes=[c.c_void_p,c.c_char_p,c.POINTER(Rect),c.POINTER(Rect),c.POINTER(Rect),c.c_void_p]
    r.rsvg_handle_get_geometry_for_layer.restype=c.c_int
    g.g_object_unref.argtypes=[c.c_void_p]
    labels=0;violations=[]
    for source in sorted(folder.glob('*.svg')):
        tree=ET.fromstring(source.read_bytes())
        _,_,w,h=map(float,tree.attrib['viewBox'].split())
        texts=[n for n in tree.iter() if n.tag.endswith('}text')]
        for i,n in enumerate(texts):n.set('id','glyph-'+str(i))
        data=ET.tostring(tree);handle=r.rsvg_handle_new_from_data(data,len(data),None)
        assert handle
        boxes=[]
        for n in texts:
            ink,logical=Rect(),Rect()
            assert r.rsvg_handle_get_geometry_for_layer(handle,('#'+n.attrib['id']).encode(),c.byref(Rect(0,0,w,h)),c.byref(ink),c.byref(logical),None)
            box=(ink.x,ink.y,ink.x+ink.width,ink.y+ink.height)
            if box[0]<-.5 or box[1]<-.5 or box[2]>w+.5 or box[3]>h+.5:
                violations.append([source.name,'glyph outside viewBox',n.text,list(box)])
            boxes.append((n.text,box));labels+=1
        for i,(name,a) in enumerate(boxes):
            for other,b in boxes[i+1:]:
                if min(a[2],b[2])-max(a[0],b[0])>.5 and min(a[3],b[3])-max(a[1],b[1])>.5:
                    violations.append([source.name,'overlapping glyphs',name,other])
        # Each label belonging to a card must fit inside its card rectangle.
        for group in (n for n in tree.iter() if n.attrib.get('data-box')):
            rect=next(n for n in group if n.tag.endswith('}rect'))
            x,y,bw,bh=[float(rect.attrib[k]) for k in ('x','y','width','height')]
            for text in (n for n in group if n.tag.endswith('}text')):
                ink,logical=Rect(),Rect()
                r.rsvg_handle_get_geometry_for_layer(handle,('#'+text.attrib['id']).encode(),c.byref(Rect(0,0,w,h)),c.byref(ink),c.byref(logical),None)
                if ink.x<x+2 or ink.y<y+2 or ink.x+ink.width>x+bw-2 or ink.y+ink.height>y+bh-2:
                    violations.append([source.name,'card containment',text.text])
        g.g_object_unref(handle)
        render(source,source.with_suffix('.png'))
    report={'states':len(list(folder.glob('*.svg'))),'glyphs_checked':labels,'violations':violations,
            'renderer':'librsvg','browser_checked':False,'screenshots':str(folder)}
    (HERE/'_render_l048_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    assert not violations


if __name__=='__main__':main()
