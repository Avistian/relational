"""Bounded smoke, unchanged resume, rejection after changing a live helper."""
import json,tempfile,subprocess,sys
from pathlib import Path
import torch
import _paper_repro_l054 as repro
import relkit.tabm_v2 as model

def check():
    folder=Path(tempfile.mkdtemp(prefix='l054-repro-check-'))
    result=repro.reproduce('smoke',folder)
    path=folder/'seed-0.json';before=path.read_bytes();stamp=path.stat().st_mtime_ns
    resumed=repro.reproduce('smoke',folder)
    assert result==resumed and before==path.read_bytes() and stamp==path.stat().st_mtime_ns
    child="import sys; sys.path.insert(0,sys.argv[1]); from _paper_repro_l054 import reproduce; reproduce('smoke',sys.argv[2])"
    subprocess.run([sys.executable,'-c',child,str(Path(__file__).resolve().parent),str(folder)],check=True)
    assert before==path.read_bytes() and stamp==path.stat().st_mtime_ns
    original=model.batchensemble_linear
    def changed_live_helper(x,weight,R,S,bias):return original(x,weight,R,S,bias)+.01
    model.batchensemble_linear=changed_live_helper;rejected=False
    try:repro.reproduce('smoke',folder)
    except RuntimeError as e:
        assert 'identity changed' in str(e);rejected=True
    finally:model.batchensemble_linear=original
    assert rejected,'Changed live helper silently resumed'
    targets=torch.tensor([[1.,2.],[3.,4.]])
    out=targets[...,None]+torch.tensor([[[0.],[1.]],[[2.],[3.]]])
    torch.testing.assert_close(model.member_mean_loss(out.reshape(-1,1,1),targets.reshape(-1),True),torch.tensor(3.5))
    record=dict(status='PASS',smoke_rmse=result['summary']['mean'],smoke_seconds=result['runs'][0]['seconds'],
        unchanged_resume=True,fresh_process_resume=True,changed_live_helper_rejected=True,distinct_member_targets=True,larger_presets='NOT_RUN',cloud='NOT_RUN')
    (Path(__file__).parent/'_check_repro_l054_results.json').write_text(json.dumps(record,indent=2)+'\n');print(record)
if __name__=='__main__':check()
