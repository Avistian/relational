"""Rebuild deterministic prose/assets while preserving verified identical cell outputs."""
import subprocess,sys
from pathlib import Path
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor
P=Path(__file__).resolve().parent;S='b07a-hypernetworks';path=P/'solutions'/(S+'.ipynb');old=nbformat.read(path,4)
subprocess.run([sys.executable,str(P/'_build_b07a.py')],check=True)
new=nbformat.read(path,4)
assert len(old.cells)==len(new.cells)
for a,b in zip(old.cells,new.cells):
 if b.cell_type=='code':
  assert a.cell_type=='code' and a.source==b.source,'Code changed: fresh execution required'
  b.outputs=a.outputs;b.execution_count=a.execution_count;b.metadata=a.metadata
nbformat.write(new,path)
exporter=HTMLExporter(template_name='lab');exporter.register_preprocessor(TagRemovePreprocessor(remove_input_tags={'data-payload'}),enabled=True)
html,_=exporter.from_notebook_node(new);(P/'html'/(S+'.html')).write_text(html)
print('Updated prose; verified identical executed code retained')
