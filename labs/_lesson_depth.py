"""Attach authored explanations beside their concepts in lessons and notebooks.

Each depth file has named blocks with an HTML-heading and notebook-heading anchor.
The final reference block is a compact computational aid, not another lesson copy.
The same source feeds every format; executable notebook cells are never edited here.
"""
import hashlib,re
from pathlib import Path
import nbformat
from bs4 import BeautifulSoup
from nbconvert.filters.markdown import markdown2html_mistune

ROOT=Path(__file__).resolve().parents[1]
MARKER=re.compile(r'<!-- depth: ([^|]+)\|([^|]+)\|([^>]+)-->')


def authored(n):
    path=ROOT/'lessons/depth'/f'{n:04}.md'
    if not path.exists():return [],''
    text=path.read_text();body,_,reference=text.partition('<!-- reference -->')
    matches=list(MARKER.finditer(body));blocks=[]
    for i,m in enumerate(matches):
        end=matches[i+1].start() if i+1<len(matches) else len(body)
        key,lesson,notebook=[s.strip() for s in m.groups()]
        blocks.append(dict(key=key,lesson=lesson,notebook=notebook,text=body[m.end():end].strip()))
    assert blocks and reference.strip(),(n,'Require explanations and a reference')
    assert len({v['key'] for v in blocks})==len(blocks)
    return blocks,reference.strip()


def enrich_notebook(nb,n):
    blocks,_=authored(n)
    if not blocks:return nb
    nb.cells=[c for c in nb.cells if not c.metadata.get('lesson_depth')]
    # Manuscript chunks may contain several sections between figures. Split at
    # section headings so explanations stay beside the exact concept in notebooks.
    split=[]
    for cell in nb.cells:
        if cell.cell_type!='markdown' or cell.metadata.get('architecture_revision'):
            split.append(cell);continue
        parts=re.split(r'(?m)(?=^## )',cell.source)
        if len(parts)<=1:split.append(cell);continue
        for i,part in enumerate(parts):
            if not part.strip():continue
            new=nbformat.v4.new_markdown_cell(part.strip(),metadata=dict(cell.metadata))
            new.id=f'{cell.id[:50]}-part-{i}';split.append(new)
    nb.cells=split
    for block in reversed(blocks):
        matches=[i for i,c in enumerate(nb.cells) if c.cell_type=='markdown' and not c.metadata.get('lesson_depth') and
                 any(block['notebook'].casefold() in line.casefold() for line in c.source.splitlines() if line.startswith('#'))]
        assert len(matches)==1,(n,block['key'],'notebook anchor',block['notebook'],matches)
        cell=nbformat.v4.new_markdown_cell(block['text'],metadata={'lesson_depth':block['key']})
        cell.id=f'depth-{n}-{block["key"]}'[:64]
        nb.cells.insert(matches[0]+1,cell)
    nb.metadata['explanation_source']=f'lessons/depth/{n:04}.md'
    from _architecture_revision import enrich_notebook as architecture_notebook
    return architecture_notebook(nb,n)


def enrich_html(n):
    blocks,reference=authored(n)
    if not blocks:return
    path=next((ROOT/'lessons').glob(f'{n:04}-*.html'))
    soup=BeautifulSoup(path.read_text(),'html.parser')
    for old in soup.select('[data-lesson-depth]'):old.decompose()
    for block in blocks:
        matches=[h for h in soup.find_all('h2') if block['lesson'].casefold() in h.get_text().casefold()]
        assert len(matches)==1,(n,block['key'],'lesson anchor',block['lesson'])
        heading=matches[0];container=soup.new_tag('div',attrs={'class':'lesson-depth','id':f'depth-{block["key"]}',
                                                             'data-lesson-depth':block['key']})
        fragment=BeautifulSoup(markdown2html_mistune(block['text']),'html.parser')
        container.extend(list(fragment.contents))
        section=heading.find_parent('section')
        if section:section.append(container)
        else:
            node=heading
            for sibling in heading.next_siblings:
                if getattr(sibling,'name',None)=='h2' or (getattr(sibling,'name',None)=='section' and sibling.find('h2')):break
                node=sibling
            node.insert_after(container)
    # Reference links are taken from the existing lesson, preserving its public URL.
    candidates=[a['href'] for a in soup.select('a[href]') if a['href'].startswith('../reference/') and
                'glossary' not in a['href']]
    assert candidates,(n,'reference link')
    ref=(path.parent/candidates[0].split('#')[0]).resolve()
    rsoup=BeautifulSoup(ref.read_text(),'html.parser')
    for old in rsoup.select('[data-depth-reference]'):old.decompose()
    node=rsoup.new_tag('section',attrs={'data-depth-reference':str(n),'class':'lesson-depth'})
    node.extend(list(BeautifulSoup(markdown2html_mistune(reference),'html.parser').contents))
    article=rsoup.find('article') or rsoup.body
    article.append(node)
    if not soup.select_one('link[href="../assets/lesson-depth.css"]'):
        soup.head.append(soup.new_tag('link',rel='stylesheet',href='../assets/lesson-depth.css'))
    if not rsoup.select_one('link[href="../assets/lesson-depth.css"]'):
        rsoup.head.append(rsoup.new_tag('link',rel='stylesheet',href='../assets/lesson-depth.css'))
    from _architecture_revision import enrich_html as architecture_html
    architecture_html(soup,n)
    for document in [soup,rsoup]:
        for table in list(document.find_all('table')):
            if table.find_parent(['figure']) or table.find_parent(class_='lesson-depth'):continue
            if table.find_parent(class_=lambda c:c and ('scroll' in c or 'table-wrap' in c)):continue
            wrapper=document.new_tag('div',attrs={'class':'lesson-table-scroll','tabindex':'0','role':'region','aria-label':'Scrollable comparison table'})
            table.wrap(wrapper)
    def serialize(document):
        # BeautifulSoup adds a doctype newline on the first parse of compact
        # generated pages. Canonicalize it so a second regeneration is identical.
        return re.sub(r'\A<!DOCTYPE html>\s*','<!DOCTYPE html>\n',str(document),flags=re.I)
    path.write_text(serialize(soup));ref.write_text(serialize(rsoup))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('lesson',type=int,nargs='*')
    for n in p.parse_args().lesson or range(47,71):enrich_html(n)
