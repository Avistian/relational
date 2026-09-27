"""Collect only evidence and task tables; keep graph/checkpoint payloads on the volume."""
from pathlib import Path
import json,modal
P=Path(__file__).resolve().parent;v=modal.Volume.from_name('l132-identity-evidence')
for entry in v.iterdir('/',recursive=True):
    name=entry.path.lstrip('/')
    if name.startswith(('cache/','hf/','examples/','prepared/materialized/')):continue
    if not name.endswith(('.json','.txt','.npz','.parquet')):continue
    if name.startswith('prepared/') and not name.endswith(('prepared.json','train.parquet','val.parquet','test.parquet')):continue
    dest=P/'evidence/l132'/name;dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(b''.join(v.read_file(name)));print(name)
