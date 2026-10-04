"""AutoGrable v1 equations 1/Algorithm 1/Lemma 1, finite B16 course lane.

Python standard library only. Binary Brier loss and deterministic finite data are
course choices, not the missing Table 1 experiment settings. No graph neural
network is trained in the selected RQ1 experiment or this mechanism lane.
"""
from collections import Counter, defaultdict
from itertools import combinations, product
import math


def validate_rows(rows, labeled=True):
    """Require nonempty unique identities and finite binary labels."""
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Rows must be nonempty with unique ids')
    if labeled and any(r.get('y') not in (0, 1) for r in rows):
        raise ValueError('Binary labels required')


def check_columns(cols):
    if len(set(cols)) != len(cols) or any(c in ('y', 'id') for c in cols):
        raise ValueError('Distinct feature columns only; y and id are forbidden')


def projection(row, cols):
    """Exact typed tuples avoid hash collisions and missing-value sentinels."""
    check_columns(cols)
    values = []
    for c in cols:
        v = row[c]
        if type(v) not in (str, int):
            raise ValueError('Finite fixture features must be strings or integers')
        values.append((c, type(v).__name__, v))
    return tuple(values)


def block_predict(train, query, cols):
    """Eq.1 block predictor: train cell means, train marginal for unseen cells."""
    validate_rows(train)
    check_columns(cols)
    counts, positives = Counter(), Counter()
    for row in train:
        key = projection(row, cols)
        counts[key] += 1
        positives[key] += row['y']
    marginal = sum(r['y'] for r in train) / len(train)
    return [positives[key] / counts[key] if counts[key] else marginal
            for key in (projection(row, cols) for row in query)]


def occupancy(sizes):
    """Eq.1: training fragmentation ranges from 1/sqrt(n) to 1."""
    if not sizes or any(type(n) is not int or n <= 0 for n in sizes):
        raise ValueError('Positive integer block sizes required')
    return sum(math.sqrt(n) for n in sizes) / sum(sizes)


def loss_value(probabilities, labels, loss='brier'):
    if not probabilities or len(probabilities) != len(labels):
        raise ValueError('Aligned nonempty probabilities and labels required')
    if any(y not in (0, 1) for y in labels) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError('Invalid binary probability or label')
    if loss == 'brier':
        return sum((p-y)**2 for p,y in zip(probabilities, labels)) / len(labels)
    if loss == '0-1':
        # Fixed label-0 tie break, unlike upstream first-observed class order.
        return sum(int(p > .5) != y for p,y in zip(probabilities, labels)) / len(labels)
    if loss == 'logloss':
        return -sum(math.log(max(1e-12, p if y else 1-p)) for p,y in zip(probabilities, labels)) / len(labels)
    raise ValueError('Unknown loss')


def score_subset(train, valid, cols, penalty=0.5, loss='brier'):
    validate_rows(train)
    validate_rows(valid)
    if {r['id'] for r in train} & {r['id'] for r in valid}:
        raise ValueError('Training and validation identities overlap')
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError('Finite nonnegative penalty required')
    probabilities = block_predict(train, valid, cols)
    sizes = list(Counter(projection(r, cols) for r in train).values())
    risk = loss_value(probabilities, [r['y'] for r in valid], loss)
    omega = occupancy(sizes)
    return dict(cols=list(cols), sizes=sizes, probabilities=probabilities,
                risk=risk, omega=omega, J=risk + penalty * omega)


def subsets(candidates):
    return [list(s) for k in range(len(candidates)+1) for s in combinations(candidates, k)]


def select_columns(train, valid, candidates, penalty=0.5, method='exhaustive', tolerance=1e-12):
    """Algorithm 1 with strict improvement and backward access to empty set.

    All scoring calls use train/validation only. Exhaustive search is the finite
    oracle, not a scalable proposal. Candidate order fixes greedy move ties.
    """
    check_columns(candidates)
    if method not in ('exhaustive', 'forward', 'backward') or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Invalid method or tolerance')
    candidates = sorted(candidates)
    if method == 'exhaustive':
        scores = [score_subset(train, valid, s, penalty) for s in subsets(candidates)]
        minimum = min(s['J'] for s in scores)
        best = next(s for s in scores if s['J'] <= minimum + tolerance)
        return dict(selected=best['cols'], J=best['J'], trace=scores)
    current = candidates[:] if method == 'backward' else []
    value = score_subset(train, valid, current, penalty)
    trace = [dict(selected=current[:], J=value['J'], action='init')]
    while True:
        moves = ([c for c in candidates if c not in current] if method == 'forward' else current)
        if not moves:
            break
        options = [score_subset(train, valid,
                    sorted(current+[c]) if method == 'forward' else [x for x in current if x != c], penalty)
                   for c in moves]
        best = min(options, key=lambda s:s['J'])
        improvement = value['J'] - best['J']
        accepted = improvement > tolerance
        trace.append(dict(proposed=best['cols'], J=best['J'], improvement=improvement,
                          action='accept' if accepted else 'stop', considered=options))
        if not accepted:
            break
        current, value = best['cols'], best
    return dict(selected=current, J=value['J'], trace=trace)


def incidence_graph(rows, cols, typed_values=True, row_features=()):
    """Section 4 incidence construction. Labels and row IDs are not features.

    Each undirected edge is stored twice with its column type. A value node is
    shared exactly when the column and literal value agree. typed_values=False
    deliberately removes literal identity, illustrating the lemma's assumption.
    """
    validate_rows(rows, labeled=False)
    check_columns(cols)
    check_columns(row_features)
    if set(cols) & set(row_features):
        raise ValueError('Row-local features must be unexpanded columns')
    colors = [('row', projection(r, row_features)) for r in rows]
    neighbors = [[] for _ in rows]
    value_nodes = {}
    for i, row in enumerate(rows):
        for c in cols:
            key = projection(row, [c])
            if key not in value_nodes:
                j = len(colors)
                value_nodes[key] = j
                colors.append(('value', key if typed_values else c))
                neighbors.append([])
            j = value_nodes[key]
            neighbors[i].append((c,j))
            neighbors[j].append((c,i))
    return colors, neighbors


def color_refinement(colors, neighbors):
    """Injective tuple interning; iterate until the equality partition stabilizes."""
    def intern(signatures):
        dictionary = {}
        return [dictionary.setdefault(s, len(dictionary)) for s in signatures]
    state = intern(colors)
    for _ in range(len(state)):
        signatures = [(state[i], tuple(sorted((edge,state[j]) for edge,j in adj)))
                      for i, adj in enumerate(neighbors)]
        new = intern(signatures)
        # Refinement retains old colors, so unchanged block count means stability.
        if len(set(new)) == len(set(state)):
            return new
        state = new
    raise AssertionError('Finite refinement failed to stabilize')


def incidence_partition(rows, cols, typed_values=True, row_features=()):
    colors, neighbors = incidence_graph(rows, cols, typed_values, row_features)
    state = color_refinement(colors, neighbors)
    blocks = defaultdict(list)
    for i, color in enumerate(state[:len(rows)]):
        blocks[color].append(i)
    return sorted(blocks.values(), key=lambda b:b[0])


def fixture(kind='signal', prefix='tr'):
    if kind not in ('signal', 'xor', 'null'):
        raise ValueError('Unknown fixture')
    return [dict(id=f'{prefix}-{i}', A=i//4, B=(i//2)%2, K=f'{prefix}-{i}',
                 y=i//4 if kind=='signal' else ((i//4)^((i//2)%2) if kind=='xor' else i%2)) for i in range(8)]


def evaluate_frozen(train, valid, test, candidates, penalty, method):
    """Selection is frozen before reading test labels; save row-keyed predictions."""
    validate_rows(test)
    ids = [r['id'] for rows in (train,valid,test) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Split identities overlap')
    selected = select_columns(train, valid, candidates, penalty, method)
    features = [{k:v for k,v in r.items() if k != 'y'} for r in test]
    probabilities = block_predict(train, features, selected['selected'])
    predictions = [dict(id=r['id'], probability=p, label=r['y']) for r,p in zip(test, probabilities)]
    return dict(**selected, predictions=predictions,
                test_brier=loss_value(probabilities, [r['y'] for r in test]))


def run_experiment():
    """The complete frozen course grid; no randomness or hidden model fits."""
    candidates = ['A','B','K']
    report = dict(experiment='B16-PARTITION-SELECTION', scope='COMPLETE_FINITE_DIAGNOSTIC',
                  datasets={}, subsets=[], selections=[], graph_checks=[], test_interventions=[])
    for kind in ('signal','xor','null'):
        train, valid, test = [fixture(kind, split) for split in ('tr','va','te')]
        report['datasets'][kind] = dict(train=train, valid=valid, test=test)
        for cols in subsets(candidates):
            graph_blocks = incidence_partition(train, cols)
            groups = defaultdict(list)
            for i, row in enumerate(train):
                groups[projection(row, cols)].append(i)
            expected = sorted(groups.values(), key=lambda b:b[0])
            assert graph_blocks == expected
            report['graph_checks'].append(dict(world=kind,cols=cols,blocks=graph_blocks))
        for penalty in (0, .5, 1):
            for cols in subsets(candidates):
                report['subsets'].append(dict(world=kind, penalty=penalty, **score_subset(train,valid,cols,penalty)))
            for method in ('exhaustive','forward','backward'):
                result = evaluate_frozen(train,valid,test,candidates,penalty,method)
                report['selections'].append(dict(world=kind,penalty=penalty,method=method,**result))
    train, valid, test = [fixture('signal', split) for split in ('tr','va','te')]
    base = evaluate_frozen(train, valid, test, candidates, .5, 'exhaustive')
    for labels in product((0,1), repeat=8):
        changed = [dict(row,y=y) for row,y in zip(test,labels)]
        result = evaluate_frozen(train,valid,changed,candidates,.5,'exhaustive')
        unchanged = (result['selected']==base['selected'] and result['trace']==base['trace'] and
                     [r['probability'] for r in result['predictions']]==[r['probability'] for r in base['predictions']])
        report['test_interventions'].append(dict(labels=list(labels),unchanged=unchanged))
    report['feature_boundary'] = dict(typed=incidence_partition(train,['A']),
        anonymous=incidence_partition(train,['A'],False), full=incidence_partition(train,['A'],True,['B']))
    return report
