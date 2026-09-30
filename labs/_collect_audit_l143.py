"""Collect the separate real-batch original-gradient diagnostic."""
from pathlib import Path
import modal
P=Path(__file__).resolve().parent/'evidence/l143';P.mkdir(parents=True,exist_ok=True)
v=modal.Volume.from_name('l143-relgnn-evidence')
(P/'diagnosis.json').write_bytes(b''.join(v.read_file('diagnosis.json')))
