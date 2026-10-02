"""Local admission entrypoint; no Modal service invoked under the failed gate.

python modal/l184_paper_repro.py --preset paper
Exit2 is an expected scientific stop. Cloud training operator is NOT_VALIDATED;
this is not a claim that a full cloud reproduction can currently execute.
"""
import runpy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'labs'))
runpy.run_module('_run_l184',run_name='__main__')
