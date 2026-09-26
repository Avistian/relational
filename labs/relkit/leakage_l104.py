"""L104: auditable point-in-time access and controlled TGAT interventions.
The causal intervention changes information access, never model weights or labels.
"""
# %% Imports — the complete TGAT implementation is shown earlier in the notebook
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from relkit.tgat_l103 import NeighborFinder, TGAT, NegativeSampler, event_batches

# %% Task 1 — separate the event clock from the availability clock

def eligible_history(event_time, available_time, cutoff, inclusive=False):
    """Boolean eligibility for aligned arrays; strict events, available by cutoff.
    inclusive=True deliberately permits tied event times for an audit intervention.
    """
    event_time, available_time = np.asarray(event_time), np.asarray(available_time)
    if event_time.shape != available_time.shape:
        raise ValueError('Event and availability arrays must align')
    event_ok = event_time <= cutoff if inclusive else event_time < cutoff
    return event_ok & (available_time <= cutoff)

# %% Task 2 — a training label must have matured when the model is fitted

def admissible_training(prediction_time, label_available_time, fit_time):
    """Training query occurs before fit; its complete target is available by fit."""
    prediction_time, label_available_time = np.asarray(prediction_time), np.asarray(label_available_time)
    if prediction_time.shape != label_available_time.shape:
        raise ValueError('Training query and target availability arrays must align')
    return (prediction_time < fit_time) & (label_available_time <= fit_time)

# %% Task 3 — compare exactly the same prediction questions

def paired_ap(baseline, changed):
    """Pooled AP and changed-minus-baseline percentage points on paired records.
    e identifies positive events; negative identifies paired negative destinations.
    """
    for name in ['e', 'negative', 'batch']:
        if not np.array_equal(baseline[name], changed[name]):
            raise ValueError(f'Unpaired comparison: {name}')
    n = len(baseline['e'])
    if not n:
        raise ValueError('Empty evaluation')
    y = np.r_[np.ones(n), np.zeros(n)]
    values = []
    for records in [baseline, changed]:
        if len(records['p']) != n or len(records['n']) != n:
            raise ValueError('Score counts do not match event identities')
        scores = np.r_[records['p'], records['n']]
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite scores')
        values.append(float(average_precision_score(y, scores)))
    return {'baseline_ap':values[0], 'changed_ap':values[1], 'delta_pp':100*(values[1]-values[0])}

# %% Audited sampler — all queries consume the same number of random variates

class AuditFinder(NeighborFinder):
    """Intervention sampler, intentionally distinct from the released sampler.
    Strict: event < cutoff. Inclusive: event <= cutoff. Lookahead: event <= cutoff+window.
    Wikipedia has no ingestion clock; available_time=event_time is an assumption here.
    """
    def __init__(self, events, n_nodes, mode='strict', lookahead=86400):
        if mode not in ('strict','inclusive','lookahead'):
            raise ValueError(mode)
        super().__init__(events, n_nodes, release=False, uniform=True)
        self.mode, self.lookahead = mode, lookahead
        self.audit = {'sampled_records':0, 'nonpast_records':0, 'future_records':0, 'queries':0}
    def find_before(self, node, cutoff):
        a = self.rows[int(node)]
        boundary = cutoff + (self.lookahead if self.mode=='lookahead' else 0)
        end = np.searchsorted(a[:,2], boundary, side='left' if self.mode=='strict' else 'right')
        prefix = a[:end]
        # The student's clock policy is on the live model path, not an isolated exercise.
        mask = eligible_history(prefix[:,2], prefix[:,2], boundary, inclusive=self.mode!='strict')
        return prefix[mask]
    def get_temporal_neighbor(self, nodes, cutoffs, num_neighbors=20):
        ids = np.zeros((len(nodes),num_neighbors), dtype=np.int32)
        edges = np.zeros_like(ids); times = np.zeros(ids.shape, dtype=np.float32)
        # Draw even for empty histories. This preserves the uniform schedule across arms.
        uniforms = np.random.random((len(nodes),num_neighbors))
        for i,(node,cutoff) in enumerate(zip(nodes,cutoffs)):
            history = self.find_before(node,cutoff)
            if not len(history):
                continue
            selected = history[np.floor(uniforms[i]*len(history)).astype(int)]
            selected = selected[np.argsort(selected[:,2])]
            ids[i], edges[i], times[i] = selected[:,0], selected[:,1], selected[:,2]
            self.audit['sampled_records'] += len(selected)
            self.audit['nonpast_records'] += int(np.sum(selected[:,2] >= cutoff))
            self.audit['future_records'] += int(np.sum(selected[:,2] > cutoff))
        self.audit['queries'] += len(nodes)
        return ids, edges, times

# %% Complete evaluation with archived question identities and recursive access audit

def records_from_archive(archive, lane):
    return {key:archive[f'{lane}_{key}'] for key in ['e','negative','p','n','batch']}

def batch_ap(records):
    values=[]
    for batch in np.unique(records['batch']):
        mask=records['batch']==batch; n=int(mask.sum())
        values.append(average_precision_score(np.r_[np.ones(n),np.zeros(n)], np.r_[records['p'][mask],records['n'][mask]]))
    return float(np.mean(values))

@torch.no_grad()
def score_fixed_questions(model, events, questions, rng_seed=104, deadline=None):
    """Replay frozen positives/negatives. Both arms reset to identical uniform draws."""
    model.eval(); np.random.seed(rng_seed)
    positions=np.searchsorted(events['e'],questions['e'])
    if not np.array_equal(events['e'][positions],questions['e']):
        raise ValueError('Question event missing from this stream')
    result={k:questions[k].copy() for k in ['e','negative','batch']}
    result['p']=np.empty(len(positions),np.float32);result['n']=np.empty(len(positions),np.float32)
    for batch in np.unique(questions['batch']):
        if deadline is not None and time.monotonic()>deadline:
            raise TimeoutError('Budget runtime cutoff; incomplete seed is not evidence')
        mask=questions['batch']==batch; ix=positions[mask]
        p,n=model.contrast(events['u'][ix],events['v'][ix],questions['negative'][mask],events['t'][ix],20)
        result['p'][mask],result['n'][mask]=p.cpu().numpy(),n.cpu().numpy()
    return result

# %% Named released-evaluation replay — preserve RNG consumption and source quirks

@torch.no_grad()
def replay_release(model, data, checkpoint, archive, deadline=None):
    model.eval();np.random.set_state(checkpoint['numpy_state']);report={};outputs={}
    for lane,split,pool in [('all','test','full'),('new','new_test','new_test')]:
        rows=[];sampler=NegativeSampler(data[pool]);expected=records_from_archive(archive,lane);offset=0
        for batch,b in enumerate(event_batches(data[split],30,release=True)):
            if deadline is not None and time.monotonic()>deadline:
                raise TimeoutError('Budget runtime cutoff; replay incomplete')
            negative=sampler.sample(len(b['u']));p,n=model.contrast(b['u'],b['v'],negative,b['t'],20)
            stop=offset+len(negative)
            np.testing.assert_array_equal(b['e'],expected['e'][offset:stop])
            np.testing.assert_array_equal(negative,expected['negative'][offset:stop])
            rows.append({'e':b['e'],'negative':negative,'p':p.cpu().numpy(),'n':n.cpu().numpy(),'batch':np.full(len(negative),batch)})
            offset=stop
        got={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
        errors=[]
        for key in ['p','n']:
            np.testing.assert_allclose(got[key],expected[key],rtol=1e-5,atol=2e-6)
            errors.append(float(np.max(np.abs(got[key]-expected[key]))))
        report[lane]={'events':len(got['e']),'batch_mean_ap':batch_ap(got),'max_prediction_error':max(errors),'status':'PASS'}
        outputs[lane]=got
    return report,outputs
