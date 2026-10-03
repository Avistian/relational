"""Invoke original TabPack functions unchanged; preserve all prediction evidence."""
import json,os,sys
from pathlib import Path
os.chdir('/source')
import tempfile
Path('/evidence/tmp').mkdir(exist_ok=True)
tempfile.tempdir='/evidence/tmp'
os.environ['TMPDIR']='/evidence/tmp'
Path('/evidence/cache').mkdir(exist_ok=True)
if not Path('/source/.cache').exists():Path('/source/.cache').symlink_to('/evidence/cache',target_is_directory=True)
sys.path[:0]=['/source/src','/source/scripts']
import lib.utils,lib.experiment
import project.tabpack
lib.utils.init()
phase=sys.argv[1]
if phase=='audited':
    import functools,numpy as np
    original_main=project.tabpack.main
    original_update=project.tabpack.update_online_ensembles
    capture={}
    def observe_update(ensembles,**kwargs):
        capture['ensembles']=ensembles
        return original_update(ensembles,**kwargs)
    project.tabpack.update_online_ensembles=observe_update
    @functools.wraps(original_main)
    def observe_main(config,exp):
        capture.clear()
        report=original_main(config,exp)
        ensemble=capture['ensembles']['greedy']
        np.savez(Path(exp)/'observed_ensemble.npz',ids=ensemble.ids,steps=ensemble.steps,**ensemble.predictions)
        return report
    project.tabpack.main=observe_main
parent=Path('/source/experiments/reproduce');parent.parent.mkdir(exist_ok=True)
if not parent.exists():parent.symlink_to('/evidence/run-audited' if phase=='audited' else '/evidence/run-v3',target_is_directory=True)
Path('/evidence/run-audited' if phase=='audited' else '/evidence/run-v3').mkdir(exist_ok=True)
exp=parent/'california/main'
if phase.startswith('pilot') or phase=='audited':
    config=json.loads(Path('/source/experiments/tabpack-cosine/california/main/config.json').read_text())
    lib.experiment.create(exp,config=config,parents=True)
    lib.experiment.run(project.tabpack.main,None,exp)
if phase in ('full','audited'):
    from run_tabpack_experiment import _evaluate_ensemble
    _evaluate_ensemble(exp,name='greedy',n_seeds=5,is_offline=False)

