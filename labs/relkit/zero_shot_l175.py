"""L175: information access, complete-key scoring and fail-closed budgets."""
import numpy as np

def aligned_auc(expected_keys, labels, prediction_keys, scores):
    """Match complete identities before computing pairwise AUROC with half ties."""
    ek=np.asarray(expected_keys);pk=np.asarray(prediction_keys)
    y=np.asarray(labels);s=np.asarray(scores,dtype=float)
    if ek.ndim!=2 or pk.ndim!=2 or ek.shape[1]!=2 or pk.shape[1]!=2:
        raise ValueError('Keys must be (entity_id, cutoff) pairs')
    expected=[tuple(x) for x in ek];predicted=[tuple(x) for x in pk]
    if len(set(expected))!=len(expected) or len(set(predicted))!=len(predicted):
        raise ValueError('Duplicate complete keys')
    if set(expected)!=set(predicted) or len(s)!=len(predicted) or len(y)!=len(expected):
        raise ValueError('Population or array length mismatch')
    if not np.isfinite(s).all() or set(y.tolist())!={0,1}:
        raise ValueError('Finite scores and both binary classes required')
    lookup=dict(zip(predicted,s));ordered=np.array([lookup[k] for k in expected])
    positive=ordered[y==1];negative=ordered[y==0]
    pairs=positive[:,None]-negative[None,:]
    return float(((pairs>0)+.5*(pairs==0)).mean())

def exposure_claim(target_gradients, target_validation, context_labels, test_exposure):
    """A frozen checkpoint can still consume target labels in other ways."""
    if test_exposure:return 'INVALID_TEST_EXPOSURE'
    if target_gradients:return 'TARGET_TRAINED'
    if target_validation or context_labels:return 'NO_TARGET_GRADIENTS_WITH_LABEL_ACCESS'
    return 'STRICT_NO_TARGET_LABEL_ACCESS'

def audit_context(timestamps,padding,query_targets,label_cells,masks,cutoff,horizon_seconds):
    """Second-resolution source timestamps; unknown time is not proof of availability.

    Label availability uses the known task horizon, not just the task-row cutoff.
    Equality at horizon end is allowed here; ingestion delay remains unknown.
    """
    ts=np.asarray(timestamps,dtype=np.int64);pad=np.asarray(padding,dtype=bool)
    target=np.asarray(query_targets,dtype=bool);lab=np.asarray(label_cells,dtype=bool)
    mask=np.asarray(masks,dtype=bool)
    if len({x.shape for x in [ts,pad,target,lab,mask]})!=1:raise ValueError('Shape mismatch')
    valid=~pad;known=ts!=np.iinfo(np.int32).min;visible=valid & lab & ~mask
    return dict(unmasked_query_targets=int(np.sum(valid & target & ~mask)),
                same_time_labels=int(np.sum(visible & known & (ts==cutoff))),
                unavailable_labels=int(np.sum(visible & known & (ts+horizon_seconds>cutoff))),
                future_cells=int(np.sum(valid & known & (ts>cutoff))),
                unknown_time_cells=int(np.sum(valid & ~known)),
                visible_label_cells=int(visible.sum()))

def reserve_cost(prior_upper_usd, seconds, rate, overhead_reserve=2., stop_usd=8.):
    """Reserve timeout plus 30s container overhead before each dispatch."""
    if seconds<=0 or rate<0 or any(x<0 for x in prior_upper_usd):raise ValueError('Invalid reservation')
    total=sum(prior_upper_usd)+(seconds+30)*rate+overhead_reserve
    if total>stop_usd:raise ValueError('INCOMPLETE_BUDGET_GATE')
    return total

def replay_packet(root, count_context):
    """Replay all saved contexts with the live learner policy and raw F1 oracle.

    No model inference, native sampling or paid work is performed here.
    """
    import json
    from pathlib import Path
    import ml_dtypes
    root=Path(root);ref=json.loads((root/'context-audit.json').read_text())
    meta=json.loads((root/'table_info.json').read_text());cols=json.loads((root/'column_index.json').read_text())
    raw=np.load(root/'label-oracle.npz');target_col=cols['did_not_finish of driver-dnf']
    records=[];identities=None
    for seed in range(3):
        data=np.load(root/f'contexts-{seed}.npz');assert data['node_idxs'].shape==(702,1024)
        keys=[];query_nodes=[]
        for i in range(702):
            target=data['is_targets'][i];assert target.sum()==1
            cutoff=int(data['timestamps'][i][target][0]);node=int(data['node_idxs'][i][target][0])
            parent=int(data['f2p_nbr_idxs'][i][target][0,0]);row=parent-meta['drivers:Db']['node_idx_offset']
            driver=int(raw['driver_ids'][row]);keys.append((driver,cutoff));query_nodes.append(node)
            value=data['boolean_values'][i][target].view(ml_dtypes.bfloat16).astype(float)[0]
            eligible=(raw['driver']==driver)&(raw['time']>cutoff)&(raw['time']<=cutoff+30*86400)
            assert eligible.any() and int(value>0)==int((raw['status'][eligible]!=1).any())
            result=count_context(data['timestamps'][i],data['is_padding'][i],target,data['col_name_idxs'][i]==target_col,data['masks'][i],cutoff,30*86400)
            expected=ref['records'][seed*702+i]
            assert all(result[k]==expected[k] for k in result),'Context audit differs from independent evidence'
            assert set(result)=={'unmasked_query_targets','same_time_labels','unavailable_labels','future_cells','unknown_time_cells','visible_label_cells'}
            records.append(result)
        assert len(set(keys))==702 and sorted(query_nodes)==list(range(97606,98308))
        if identities is None:identities=set(keys)
        assert set(keys)==identities
    totals={k:sum(r[k] for r in records) for k in records[0]}
    gate='BLOCKED_TEMPORAL_AUDIT' if any(totals[k] for k in ['unmasked_query_targets','unavailable_labels','future_cells']) else 'PASS'
    assert gate==ref['inference_gate']
    return dict(contexts=len(records),query_labels_reconstructed=len(records),totals=totals,inference_gate=gate,model_evaluations='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')
