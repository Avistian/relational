"""Visible learner operations for the L171 corpus audit; no model training."""
def validate_corpus(records):
    """Validate declared identities and return a deterministic, independent copy."""
    import copy
    import re
    if not isinstance(records, list) or not records:
        raise ValueError('A nonempty list of records is required')
    result = copy.deepcopy(records)
    names = set()
    for row in result:
        name = row.get('database')
        families = row.get('source_families')
        digest = row.get('archive_sha256')
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError('Database names must be nonempty and unique')
        if (not isinstance(families, list) or not families
                or any(not isinstance(x, str) or not x.strip() for x in families)
                or len(families) != len(set(families))):
            raise ValueError('Declare distinct, nonempty source-family identifiers')
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('Expected lowercase SHA256 archive digest')
        names.add(name)
        row['source_families'] = sorted(families)
    return sorted(result, key=lambda row: row['database'])


def split_corpus(records, heldout):
    """Exclude the entire connected lineage component of the held-out database."""
    rows = validate_corpus(records)
    names = {row['database'] for row in rows}
    if heldout not in names:
        raise ValueError('Held-out database is absent')
    excluded = {heldout}
    # Fixed point: a bridge may connect a second-hop alias to the held-out source.
    while True:
        families = {f for row in rows if row['database'] in excluded
                    for f in row['source_families']}
        archives = {row['archive_sha256'] for row in rows
                    if row['database'] in excluded}
        expanded = excluded | {row['database'] for row in rows
                               if families.intersection(row['source_families'])
                               or row['archive_sha256'] in archives}
        if expanded == excluded:
            break
        excluded = expanded
    return dict(train=sorted(names - excluded), heldout=[heldout],
                quarantine=sorted(excluded - {heldout}))


def audit_tables(tables, val_timestamp, test_timestamp):
    """Count full-snapshot PK/FK defects and time windows, without filtering rows."""
    import pandas as pd
    val, test = pd.Timestamp(val_timestamp), pd.Timestamp(test_timestamp)
    if pd.isna(val) or pd.isna(test) or val >= test or not tables:
        raise ValueError('Require tables and ordered finite time boundaries')
    result = dict(tables={}, foreign_keys=[], rows=0, foreign_key_columns=0,
                  integrity='PASS', availability='NOT_ESTABLISHED')
    for name in sorted(tables):
        table = tables[name]
        frame, pk, time = table['df'], table['pkey_col'], table['time_col']
        columns = [x for x in [pk, time] if x is not None]
        columns += list(table['fkey_col_to_pkey_table'])
        if any(x not in frame for x in columns):
            raise ValueError('Declared column missing from '+name)
        nulls = int(frame[pk].isna().sum()) if pk else None
        duplicates = int(frame.loc[frame[pk].notna(), pk].duplicated().sum()) if pk else None
        row = dict(rows=len(frame), columns=len(frame.columns), pkey=pk,
                   pk_nulls=nulls, pk_duplicate_excess=duplicates, time_col=time)
        if time is None:
            row.update(time_status='NO_TIME_COLUMN', time_windows=None)
        else:
            values = pd.to_datetime(frame[time], errors='raise')
            row.update(time_status='OBSERVED_EVENT_TIME',
                       time_min=None if not values.notna().any() else str(values.min()),
                       time_max=None if not values.notna().any() else str(values.max()),
                       time_windows=dict(before_val=int((values < val).sum()),
                           val_to_test=int(((values >= val) & (values < test)).sum()),
                           at_or_after_test=int((values >= test).sum()),
                           nulls=int(values.isna().sum())))
        result['tables'][name] = row
        result['rows'] += len(frame)
        if nulls or duplicates:
            result['integrity'] = 'FAIL'
        for column, target in sorted(table['fkey_col_to_pkey_table'].items()):
            if target not in tables or tables[target]['pkey_col'] is None:
                raise ValueError('Foreign key target absent or has no primary key')
            parent = tables[target]
            if parent['pkey_col'] not in parent['df']:
                raise ValueError('Target primary key column missing')
            values = frame[column]
            nonnull = values.notna()
            keys = parent['df'][parent['pkey_col']].dropna()
            dangling = int((nonnull & ~values.isin(keys)).sum())
            result['foreign_keys'].append(dict(table=name, column=column,
                target=target, rows=len(frame), nulls=int((~nonnull).sum()),
                dangling=dangling, nonnull=int(nonnull.sum())))
            if dangling:
                result['integrity'] = 'FAIL'
    result['foreign_key_columns'] = len(result['foreign_keys'])
    return result
