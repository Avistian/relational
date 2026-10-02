"""L175 original RT-v1 sampler audit; fail closed before model inference."""
import json,sys,time
from pathlib import Path
import modal
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'labs/relkit'))
app=modal.App('l175-rt-v1')
# Inference-only manifest omits unused preprocessing crates. Runtime .rs is unchanged.
image=(modal.Image.debian_slim(python_version='3.12')
 .apt_install('build-essential','curl','pkg-config')
 .pip_install('maturin==1.9.4','numpy==2.2.6','huggingface-hub==0.34.4','ml_dtypes==0.5.3')
 .run_commands('curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.88.0')
 .env({'PATH':'/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin','CARGO_BUILD_JOBS':'2'})
 .add_local_dir(R/'labs/sources/l175/upstream/rustler','/src/rustler',copy=True)
 .run_commands("sed -i '/^polars =/d; /^parquet =/d; /^glob =/d; /^indicatif =/d; /^serde_json =/d' /src/rustler/Cargo.toml",'cd /src/rustler && maturin build --release --out /wheels && pip install /wheels/*.whl')
 .add_local_file(R/'labs/_audit_context_l175.py','/work/_audit_context_l175.py')
 .add_local_file(R/'labs/relkit/zero_shot_l175.py','/work/zero_shot_l175.py'))
volume=modal.Volume.from_name('l175-rt-v1-evidence',create_if_missing=True)
@app.function(image=image,cpu=(2,2),memory=(16384,16384),timeout=900,retries=0,scaledown_window=2,volumes={'/evidence':volume})
def audit(phase: str):
 import traceback
 sys.path.insert(0,'/work');from _audit_context_l175 import run
 assert phase.startswith('audit-') and phase[6:].isdigit();out=Path('/evidence')/phase;assert not out.exists();out.mkdir()
 start=time.monotonic()
 try:return run(out)
 except BaseException:
  (out/'failure.txt').write_text(traceback.format_exc());raise
 finally:
  (out/'cost.json').write_text(json.dumps(dict(worker_seconds=time.monotonic()-start,upper_rate=.00006172)))
  volume.commit()
@app.local_entrypoint()
def main(phase: str):
 import hashlib
 p=R/'labs/evidence/l175/cloud-budget.json'
 state=json.loads(p.read_text())
 reservation=next((x for x in state['reservations'] if x['phase']==phase),None)
 if reservation is None or not reservation.get('source_hashes'):raise RuntimeError('Use _dispatch_l175.py to reserve before building')
 for name,digest in reservation['source_hashes'].items():
  assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,'Changed source after reservation'
 print(audit.remote(phase))
