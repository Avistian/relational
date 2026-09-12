"""Rebuild both version panels and assemble the complete L070 checkpoint."""
from _run_foundation import run, ROOT
from _assemble_foundation_checkpoint import assemble
if __name__ == "__main__":
    historical=run(70,legacy=True,output=ROOT/'data/cache/foundation/l070-historical-rerun.json')
    current=run(70,current=True,output=ROOT/'data/cache/foundation/l070-current-rerun.json')
    out=ROOT/'data/cache/foundation/l070-complete-rerun.json'
    assemble(historical,current,out)
    print(out)
