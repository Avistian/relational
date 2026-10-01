"""Execute notebook offline in an empty working directory."""
import hashlib,json,tempfile,time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor
P=Path(__file__).resolve().parent;S='0157-open-source-contribution';path=P/'solutions'/(S+'.ipynb');n=nbformat.read(path,4);start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='l157-default-') as tmp:
 NotebookClient(n,timeout=240,kernel_name='python3',resources={'metadata':{'path':tmp}}).execute()
 r=json.loads((Path(tmp)/'l157-report.json').read_text());assert r['status']=='PASS' and r['summary']['fits']==10
nbformat.write(n,path)
exporter=HTMLExporter(template_name='lab');exporter.register_preprocessor(TagRemovePreprocessor(remove_input_tags={'data-payload'}),enabled=True)
html,_=exporter.from_notebook_node(n);(P/'html'/(S+'.html')).write_text(html)
r.update(seconds=time.perf_counter()-start,code_cells=sum(c.cell_type=='code' for c in n.cells),executed_code_sha256=hashlib.sha256('\n\n'.join(c.source for c in n.cells if c.cell_type=='code').encode()).hexdigest(),working_directory='EMPTY_TEMPORARY',full_gate='OFF',live_colab='NOT_CHECKED')
(P/'_execution_l157_results.json').write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k not in ['summary','written_defense']})
