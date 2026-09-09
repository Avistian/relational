"""Execute generated local teacher solutions; student notebooks stay blank."""
import argparse, json, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from _foundation_config import SLUGS

ROOT=Path(__file__).resolve().parent


def execute(lessons=range(58,71)):
    output=ROOT/'_execution_foundation_results.json'
    prior={r['lesson']:r for r in json.loads(output.read_text())} if output.exists() else {}
    for n in lessons:
        path=ROOT/'solutions'/f'{n:04}-{SLUGS[n]}.ipynb'
        nb=nbformat.read(path,as_version=4);start=time.perf_counter()
        NotebookClient(nb,timeout=900,kernel_name='python3',
                       resources={'metadata':{'path':str(ROOT)}}).execute()
        nbformat.write(nb,path)
        prior[n]=dict(lesson=n,status='PASS',code_cells=sum(c.cell_type=='code' for c in nb.cells),
                      seconds=time.perf_counter()-start)
        output.write_text(json.dumps([prior[k] for k in sorted(prior)],indent=2)+'\n')
        print(n,'PASS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--lesson',type=int,action='append')
    execute(parser.parse_args().lesson or range(58,71))
