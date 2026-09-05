"""Render the student notebook and relocate its local links for labs/html/."""
from pathlib import Path
import os
from urllib.parse import urlsplit
import nbformat
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup
HERE=Path(__file__).resolve().parent;SLUG='0049-excelformer-trompt'
nb=nbformat.read(HERE/f'{SLUG}.ipynb',as_version=4)
html,_=HTMLExporter(template_name='lab').from_notebook_node(nb)
soup=BeautifulSoup(html,'html.parser')
for node in soup.find_all('a',href=True):
    href=node['href'];parts=urlsplit(href)
    if parts.scheme or parts.netloc or not parts.path:continue
    target=(HERE/parts.path).resolve()
    node['href']=os.path.relpath(target,HERE/'html')+('?' + parts.query if parts.query else '')+('#'+parts.fragment if parts.fragment else '')
(HERE/'html'/f'{SLUG}.html').write_text(str(soup))
print('Rendered student HTML; local links relocated.')
