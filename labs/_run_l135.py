"""Pinned full-data preprocessing, configurable trainer, no test access during search."""
import json
from pathlib import Path
import _run_l117 as preparation
from relkit.tuning_train_l135 import fit_tuning

def run(seed,epochs,output,configuration,evaluate_test=False):
    original=preparation.fit_rdl
    def fit(*args,**kwargs):
        task=args[2]
        # Enforce the information boundary even if trainer code later changes.
        get_table=task.get_table
        if not evaluate_test:
            def guarded(split,*a,**kw):
                if split=='test':raise RuntimeError('Test access forbidden during search')
                return get_table(split,*a,**kw)
            task.get_table=guarded
        try:return fit_tuning(*args,**kwargs,configuration=configuration,evaluate_test=evaluate_test)
        finally:task.get_table=get_table
    preparation.fit_rdl=fit
    try:return preparation.run(seed,epochs,output)
    finally:preparation.fit_rdl=original

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=100);p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--config',default='lr005-full');p.add_argument('--test',action='store_true');p.add_argument('--output',required=True)
    a=p.parse_args();protocol=json.loads(Path(__file__).with_name('_protocol_l135.json').read_text())
    c=next(c for c in protocol['configurations'] if c['id']==a.config)
    run(a.seed,a.epochs,a.output,c,a.test)
