"""Run the original VIME code in its historical TensorFlow 1.15 environment.

Execute in the source folder, with this script on PYTHONPATH or by absolute path.
Uses full MNIST and the paper's 6000-label / 54000-unlabeled partition. The released
architecture is fixed: this cannot recover the unpublished CV-selected Table 2 model.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import tensorflow as tf


def run_vime_release(smoke=False,output='vime-paper-results.json'):
    sys.path.insert(0,str(Path.cwd()))
    from main_vime import vime_main
    root=Path.cwd();(root/'save_model').mkdir(exist_ok=True)
    start=time.time();records=[]
    # Table 2 has 10% of MNIST training labeled (6000); release default 1000 is different.
    labels=6000
    from keras import backend as K
    clear_session=K.clear_session
    reset_graph=tf.reset_default_graph
    for seed in range(1 if smoke else 10):
        # Source functions reset graphs. Reapply the declared seed after each reset.
        def seeded_reset():
            reset_graph();tf.set_random_seed(seed)
        def seeded_clear():
            clear_session();tf.set_random_seed(seed)
        tf.reset_default_graph=seeded_reset
        K.clear_session=seeded_clear
        random.seed(seed);np.random.seed(seed);tf.set_random_seed(seed)
        scores=vime_main(.1,['mlp'],labels,.3,2.,3,1.)
        for arm,score,paper in zip(['supervised','vime_self','vime_semi'],scores,[.9387,.9406,.9577]):
            records.append(dict(seed=seed,arm=arm,accuracy=float(score),paper_mean=paper,gap=float(score)-paper))
        out=dict(target='VIME Table 2 MNIST (6000 labels)',records=records,
                 source_commit='996c58cf4c570061b30c38ecf2a754a9af85aafd',
                 config=dict(labeled=6000,unlabeled=54000,test=10000,label_validation_fraction=.1,
                             pre_epochs=10,pre_batch=128,supervised_epochs=100,supervised_batch=100,
                             semi_iterations=1000,semi_batch=128,p=.3,alpha=2.,beta=1.,K=3),
                 seconds=time.time()-start,tensorflow=tf.__version__,
                 verdict='RELEASE_MEASURED_PAPER_PROTOCOL_INCOMPLETE',
                 remaining=['Table 2 selected architecture by validation over depths 1-5 and widths d/3,d/2,d,2d,3d; the authors do not release selected architectures or search outcomes.',
                            'This is the complete released training pipeline at the Table 2 label budget, not a claim to recover that unpublished selection.',
                            'Seeds 0-9 are declared reconstruction seeds; original seeds are unspecified.',
                            'Other Table 2 datasets and other paper methods are not included.'])
        out['summary']=[dict(arm=a,mean=float(np.mean([r['accuracy'] for r in records if r['arm']==a])),
                             sd=float(np.std([r['accuracy'] for r in records if r['arm']==a])))
                        for a in ['supervised','vime_self','vime_semi']]
        Path(output).write_text(json.dumps(out,indent=2));print(json.dumps(out['summary']),flush=True)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--one-trial',action='store_true');p.add_argument('--output',default='/evidence/vime-release-results.json');a=p.parse_args()
    run_vime_release(a.one_trial,a.output)
