"""Regenerate enriched lessons/labs and preserve only unchanged executed code."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
from urllib.parse import urlsplit,unquote
import nbformat
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup
from _lesson_depth import enrich_html,authored

ROOT=Path(__file__).resolve().parents[1];LABS=ROOT/'labs'


def code_hash(nb):
    return hashlib.sha256(json.dumps([c.source for c in nb.cells if c.cell_type=='code']).encode()).hexdigest()


def output_hash(nb):
    """Portable ledger of the exact saved code/output pairing, including counts."""
    payload=[dict(source=c.source,execution_count=c.execution_count,outputs=c.outputs)
             for c in nb.cells if c.cell_type=='code']
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def revise(n):
    blocks,_=authored(n);assert blocks,(n,'No authored depth')
    path=next(LABS.glob(f'{n:04}-*.ipynb'));solution=LABS/'solutions'/path.name
    before=nbformat.read(path,as_version=4)
    old=nbformat.read(solution,as_version=4) if solution.exists() else None
    backup=Path('/tmp/relational-depth-notebook-backups');backup.mkdir(exist_ok=True)
    if old is not None and not (backup/path.name).exists():nbformat.write(old,backup/path.name)
    enrich_html(n)
    run=subprocess.run([sys.executable,str(LABS/f'_build_l{n:03}.py')],cwd=ROOT,capture_output=True,text=True)
    if run.returncode:raise RuntimeError(run.stdout+'\n'+run.stderr)
    student=nbformat.read(path,as_version=4);fresh=nbformat.read(solution,as_version=4)
    assert code_hash(before)==code_hash(student),(n,'Student executable code changed unexpectedly')
    assert all(not c.outputs and c.execution_count is None for c in student.cells if c.cell_type=='code')
    preserved=0
    if old is not None:
        assert code_hash(old)==code_hash(fresh),(n,'Teacher executable code changed unexpectedly; inspect backup')
        a=[c for c in old.cells if c.cell_type=='code'];b=[c for c in fresh.cells if c.cell_type=='code']
        for prior,new in zip(a,b):
            assert not any(o.output_type=='error' for o in prior.outputs),(n,'Old execution contained an error')
            new.outputs=prior.outputs;new.execution_count=prior.execution_count
            preserved+=prior.execution_count is not None
        nbformat.write(fresh,solution)
    page,_=HTMLExporter(template_name='lab').from_notebook_node(student)
    soup=BeautifulSoup(page,'html.parser')
    for node in soup.find_all(id=True):node['id']=unquote(node['id'])
    for node in soup.select('a[href],img[src]'):
        key='href' if node.has_attr('href') else 'src';value=node[key];parts=urlsplit(value)
        if parts.scheme or parts.netloc or not parts.path:continue
        node[key]=os.path.relpath((LABS/parts.path).resolve(),LABS/'html')+('#'+parts.fragment if parts.fragment else '')
    if n>=58:
        from _foundation_access import launcher
        soup.head.append(soup.new_tag('link',rel='stylesheet',href='../../assets/lab-access.css'))
        first=next(h for h in soup.find_all('h2') if h.get_text().startswith('TODO 1'))
        first.insert_before(soup.new_tag('span',id='lab-exercises'))
        soup.body.insert(0,BeautifulSoup(launcher(n,prepared=True),'html.parser'))
    (LABS/'html'/path.with_suffix('.html').name).write_text(str(soup))
    # Older builders can import a shared section converter; all inserted blocks must survive.
    for nb in [student,fresh]:
        assert {c.metadata['lesson_depth'] for c in nb.cells if c.metadata.get('lesson_depth')}=={b['key'] for b in blocks},n
    record=dict(lesson=n,explanations=len(blocks),student_code_sha256=code_hash(student),
                teacher_code_sha256=code_hash(fresh),teacher_outputs_sha256=output_hash(fresh),
                code_unchanged=True,preserved_executed_cells=preserved,
                code_cells=sum(c.cell_type=='code' for c in fresh.cells))
    print(json.dumps(record),flush=True);return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('lesson',type=int,nargs='*');args=p.parse_args()
    output=LABS/'_depth_revision_results.json'
    prior={r['lesson']:r for r in json.loads(output.read_text())} if output.exists() else {}
    for n in args.lesson or range(47,71):
        prior[n]=revise(n);output.write_text(json.dumps([prior[k] for k in sorted(prior)],indent=2)+'\n')
