"""Visible L126 data/task/evaluation contracts. No model-score reproduction claim."""
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd


def schema_audit(tables):
    """Count every table and classify each FK as resolved, null or dangling."""
    report = {'tables': {}, 'relations': [], 'total_rows': 0}
    for name, table in tables.items():
        df, pk = table.df, table.pkey_col
        if pk is not None and (df[pk].isna().any() or df[pk].duplicated().any()):
            raise ValueError(f'{name}: primary key is null or duplicated')
        report['tables'][name] = {'rows': len(df), 'columns': len(df.columns),
                                 'primary_key': pk, 'time_column': table.time_col}
        report['total_rows'] += len(df)
    for name, table in tables.items():
        for col, target in table.fkey_col_to_pkey_table.items():
            if target not in tables or tables[target].pkey_col is None:
                raise ValueError(f'{name}.{col}: missing keyed target {target}')
            values = table.df[col]
            ids = tables[target].df[tables[target].pkey_col]
            null = values.isna()
            resolved = values.isin(ids) & ~null
            report['relations'].append({'source': name, 'column': col, 'target': target,
                'resolved': int(resolved.sum()), 'null': int(null.sum()),
                'dangling': int((~null & ~resolved).sum())})
    return report


def engagement_table(users, events, cutoffs, horizon_days=730):
    """Beta contract on normalized events. Existing real users with past activity.

    Normalize posts/comments/votes first. Each event has `user` and `time`.
    The recorded release uses 730 days, not two calendar years.
    Availability/version histories are not supplied by this interface.
    """
    if users.Id.isna().any() or users.Id.duplicated().any():
        raise ValueError('User IDs must be unique and non-null')
    if horizon_days <= 0:
        raise ValueError('Positive prediction horizon required')
    rows = []
    valid = events[events.user.notna() & (events.user != -1)]
    for cutoff in pd.to_datetime(cutoffs):
        end = cutoff + pd.Timedelta(days=horizon_days)
        past_ids = valid.loc[valid.time <= cutoff, 'user'].unique()
        eligible = users.loc[(users.CreationDate <= cutoff) & (users.Id != -1)
                             & users.Id.isin(past_ids), 'Id'].sort_values()
        future_ids = valid.loc[(valid.time > cutoff) & (valid.time <= end), 'user'].unique()
        rows.append(pd.DataFrame({'OwnerUserId': eligible.to_numpy(), 'timestamp': cutoff,
                                  'contribution': eligible.isin(future_ids).astype(int).to_numpy()}))
    if not rows:
        return pd.DataFrame(columns=['OwnerUserId', 'timestamp', 'contribution'])
    return pd.concat(rows, ignore_index=True)


def align_predictions(queries, predictions, keys, score_col='score'):
    """One score for every (entity,time) key, returned in evaluator row order."""
    if not keys or score_col in keys:
        raise ValueError('Declare identity columns separately from the score')
    for frame in [queries, predictions]:
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError('Query keys must be unique and non-null')
    if not np.isfinite(predictions[score_col].to_numpy(dtype=float)).all():
        raise ValueError('Scores must be finite')
    wanted = pd.MultiIndex.from_frame(queries[keys])
    supplied = pd.MultiIndex.from_frame(predictions[keys])
    if len(wanted) != len(supplied) or not wanted.isin(supplied).all():
        raise ValueError('Prediction keys must match queries exactly')
    by_key = pd.Series(predictions[score_col].to_numpy(dtype=float), index=supplied)
    return by_key.reindex(wanted).to_numpy()


def average_precision(y, scores):
    """Non-interpolated binary AP; consume all equal-score rows together."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or len(y) == 0 or len(y) != len(scores):
        raise ValueError('Equal, nonempty 1-D vectors required')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError('Binary labels and finite scores required')
    positives = int(y.sum())
    if positives == 0:
        return 0.0  # explicit convention, matching sklearn (which also warns)
    order = np.argsort(-scores, kind='stable')
    sorted_scores, sorted_y = scores[order], y[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(y)-1]
    tp = np.cumsum(sorted_y)[ends]
    precision = tp / (ends + 1)
    recall = tp / positives
    return float(np.sum(np.diff(np.r_[0., recall]) * precision))


def beta_events(tables):
    """Normalize three activity sources; IDs across tables need not be unique."""
    parts = []
    for name, col in [('posts', 'OwnerUserId'), ('comments', 'UserId'), ('votes', 'UserId')]:
        part = tables[name].df[[col, 'CreationDate']].rename(columns={col: 'user', 'CreationDate': 'time'})
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def beta_cutoffs(min_time, validation='2019-01-01', test='2021-01-01', horizon_days=730):
    """Release training grid: walk backward in fixed duration, never calendar years."""
    delta = pd.Timedelta(days=horizon_days)
    return {'train': list(pd.date_range(pd.Timestamp(validation)-delta, pd.Timestamp(min_time), freq=-delta)),
            'val': [pd.Timestamp(validation)], 'test': [pd.Timestamp(test)]}


def unpack_verified(raw, digest, destination):
    """Check archive bytes before reading; refuse paths outside destination."""
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('Archive checksum mismatch')
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for name in archive.namelist():
            path = (destination / name).resolve()
            if not path.is_relative_to(destination):
                raise ValueError('Archive path escapes destination')
        archive.extractall(destination)


def f1_tour(db_raw, task_raw, root):
    """Complete pinned RelBench 1.1.0 F1 API tour, separate from beta evidence.

    The sole predictor is the training-label median. It demonstrates the API;
    there is no GNN training, model comparison or published-score target.
    """
    import relbench
    from relbench.datasets.f1 import F1Dataset
    from relbench.tasks.f1 import DriverPositionTask
    from relbench.metrics import mae
    if relbench.__version__ != '1.1.0':
        raise RuntimeError('This tour pins relbench==1.1.0')
    root = Path(root)
    unpack_verified(db_raw, 'ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482', root)
    unpack_verified(task_raw, '775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e', root/'tasks')
    dataset = F1Dataset(cache_dir=str(root))
    task = DriverPositionTask(dataset, cache_dir=str(root/'tasks/driver-position'))
    db = dataset.get_db()
    schema = schema_audit(db.table_dict)
    train = task.get_table('train')
    validation = task.get_table('val')
    test_queries = task.get_table('test')
    assert task.target_col not in test_queries.df, 'Public test table must mask labels'
    center = float(np.median(train.df[task.target_col]))
    query = test_queries.df
    keyed = query.copy()
    keyed['score'] = center
    shuffled = keyed.sample(frac=1, random_state=126)
    predictions = align_predictions(query, shuffled, [task.entity_col, task.time_col])
    metrics = task.evaluate(predictions, metrics=[mae])
    val_metrics = task.evaluate(np.full(len(validation.df), center), validation, metrics=[mae])
    # Labels are opened only by the offline author audit after predictions freeze.
    full_test = task.get_table('test', mask_input_cols=False)
    independent_mae = float(np.mean(np.abs(full_test.df[task.target_col].to_numpy()-predictions)))
    np.testing.assert_allclose(metrics['mae'], independent_mae, rtol=0, atol=1e-12)
    result = {'status': 'PASS', 'scope': 'COURSE_ONLY: complete v1 F1 API tour',
              'schema': schema, 'splits': {'train': len(train.df), 'val': len(validation.df), 'test': len(query)},
              'training_median': center, 'validation_mae': val_metrics['mae'], 'test_mae': metrics['mae'],
              'test_labels_masked': True, 'prediction_alignment': 'EXACT',
              'historical_beta_full_contract': 'NOT_RUN', 'learner': 'PENDING_WRITTEN_DEFENSE'}
    predictions_frame = query.copy()
    predictions_frame['score'] = predictions
    predictions_frame['target_for_offline_audit_only'] = full_test.df[task.target_col].to_numpy()
    return result, predictions_frame
