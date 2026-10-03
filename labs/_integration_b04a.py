"""Regression: suffix electives sort after their parent, before the next core unit."""
import importlib.util,tempfile
from pathlib import Path
R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('manifest_builder',R/'scripts/update-manifest.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
a=m.slug_meta('b04-tabicl-scalable-icl');b=m.slug_meta('b04a-scaling-rows-features-classes');c=m.slug_meta('b05-tabdpt')
assert b['id']=='B04a',b
assert a['sortOrder']<b['sortOrder']<c['sortOrder'],(a,b,c)
with tempfile.TemporaryDirectory() as t:
 p=Path(t)/'lesson.html';p.write_text('<title>Lesson B04a — Scaling rows, features and classes</title><h1>Wrong fallback</h1>')
 assert m.title_from_html(p)=='Scaling rows, features and classes'
print('PASS suffix ID, ordering and title')
