"""L109: explicit two-clock, append-only state history and relational graph handoff."""
# %% PROVIDED · validate the history before selecting values
import math

def validate_versions(rows):
    """Numeric UTC offsets in one declared unit; unique (id, valid, observed, revision)."""
    keys=set()
    for row in rows:
        for column in ('valid_from','observed_at'):
            value=row.get(column)
            if not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError('Missing/nonfinite '+column)
        if not isinstance(row.get('revision'),int) or row['revision']<0:
            raise ValueError('Revision must be a nonnegative tie-break integer')
        if not isinstance(row.get('deleted'),bool):
            raise ValueError('Explicit tombstone flag required')
        key=(row['id'],row['valid_from'],row['observed_at'],row['revision'])
        if key in keys:
            raise ValueError('Ambiguous version key')
        keys.add(key)

# %% TODO · select the state known at the query time
def asof_versions(rows, cutoff):
    """Effective steps: latest valid_from, then latest known correction; ties inclusive.

    Revisions correct the same effective step. A new valid_from starts a new step.
    Tombstones are selected like any revision, then removed from the visible state.
    This bounded representation is not a general overlapping-interval database.
    """
    validate_versions(rows)
    chosen={}
    for row in rows:
        if row['valid_from']<=cutoff and row['observed_at']<=cutoff:
            key=(row['valid_from'],row['observed_at'],row['revision'])
            prior=chosen.get(row['id'])
            if prior is None or key>(prior['valid_from'],prior['observed_at'],prior['revision']):
                chosen[row['id']]=row
    return {key:row for key,row in chosen.items() if not row['deleted']}

# %% TODO · filter immutable events using both clocks
def legal_history(rows, cutoff):
    """Return indices of immutable events both occurred and observed by cutoff."""
    return [i for i,row in enumerate(rows)
            if row['event_time']<=cutoff and row['observed_at']<=cutoff]

# %% PROVIDED · join historical versions before emitting graph edges
def graph_at(people, facts, cutoff):
    """A fixed-cutoff snapshot graph; foreign keys live in versioned fact rows.

    A row is a typed node. An edge exists only if both historical endpoints exist.
    Missing parents remain dangling references for audit; no guessed parent is added.
    """
    left=asof_versions(facts,cutoff);right=asof_versions(people,cutoff)
    edges=[];dangling=[]
    for fact_id,fact in sorted(left.items()):
        parent=right.get(fact['person'])
        if parent is None:
            dangling.append(fact_id)
            continue
        ready=max(fact['valid_from'],fact['observed_at'],parent['valid_from'],parent['observed_at'])
        edges.append((fact_id,parent['id'],ready))
    return {'nodes':{'facts':left,'people':right},'edges':edges,'dangling':dangling,'cutoff':cutoff}

# %% TODO · wait for a mature label
def label_ready(query_time, horizon, arrivals):
    """Earliest fit time, assuming the source certifies window completeness.

    An observed-arrival maximum alone cannot prove no unseen event remains.
    The caller must obtain a completeness watermark; this function assumes it.
    """
    if horizon<0:
        raise ValueError('Horizon must be nonnegative')
    return max([query_time+horizon,*arrivals])

