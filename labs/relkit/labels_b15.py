"""B15 finite information diagnostics. Enumerated hypotheses, not trained FMs."""
from collections import Counter
from itertools import product
import math

def visible_labels(records, query_key, cutoff, allowed_keys):
    """Full (entity, time) identity; strict past event, inclusive availability.

    allowed_keys is the declared context boundary, fixed before reading labels.
    The query's own label is forbidden even if its stored timestamp is old.
    """
    if not math.isfinite(cutoff):
        raise ValueError('Nonfinite cutoff')
    keys = [r['key'] for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate full key')
    if any(len(k) != 2 or not math.isfinite(k[1]) for k in keys):
        raise ValueError('Invalid key')
    if any(not math.isfinite(r['available']) or r['label'] not in (0,1) for r in records):
        raise ValueError('Invalid record')
    return [(r['key'],r['label']) for r in records
            if r['key'] != query_key and r['key'] in allowed_keys
            and r['key'][1] < cutoff and r['available'] <= cutoff]

def predict_rules(query_bit, support):
    """Uniform posterior on y = x XOR theta, theta in {0,1}.

    support holds permitted (input bit, label) pairs. Return P(y_query=1).
    """
    if query_bit not in (0,1) or any(x not in (0,1) or y not in (0,1) for x,y in support):
        raise ValueError('Binary values required')
    compatible = [t for t in (0,1) if all((x ^ t) == y for x,y in support)]
    if not compatible:
        raise ValueError('No compatible rule')
    return sum(query_bit ^ t for t in compatible) / len(compatible)

def predict_columns(query, support):
    """Uniform posterior on y = x[S], S in {0,1}; no cross-column compression."""
    if len(query)!=2 or any(x not in (0,1) for x in query):
        raise ValueError('Two binary query features required')
    if any(len(x)!=2 or any(v not in (0,1) for v in x) or y not in (0,1) for x,y in support):
        raise ValueError('Invalid support')
    compatible = [s for s in (0,1) if all(x[s] == y for x,y in support)]
    if not compatible:
        raise ValueError('No compatible column')
    return sum(query[s] for s in compatible) / len(compatible)

def mutual_information(pairs):
    """I(A;B) in bits for equally weighted finite observations (A,B)."""
    if not pairs:
        raise ValueError('Empty distribution')
    n=len(pairs);joint=Counter(pairs)
    ca=Counter(a for a,b in pairs);cb=Counter(b for a,b in pairs)
    return sum(c/n * math.log2(c*n/(ca[a]*cb[b])) for (a,b),c in joint.items())

def run_experiment():
    """Freeze all worlds, equal weights, no seeds, optimization or selection.

    Rule fixtures: a local neighboring label b is visible but its generating
    context is not. It does not tell us whether query y=b or y=1-b.
    External support (0,theta) reveals this relationship when permitted.
    Column fixtures: local labeled rows (0,0)->0, (1,1)->1 cannot identify S.
    External row (0,1)->S can identify S. Query truth stays outside legal inputs.
    """
    rules=[];columns=[];visibility=[]
    for theta,b in product((0,1),repeat=2):
        q=('query',10);remote=('remote',2);future=('future',3)
        records=[dict(key=q,available=10,label=b^theta),
                 dict(key=remote,available=4,label=theta),
                 dict(key=future,available=11,label=theta)]
        visible=visible_labels(records,q,10,{remote,future,q})
        support=[(0,y) for k,y in visible]
        rules.append(dict(theta=theta,b=b,truth=b^theta,
                          local=predict_rules(b,[]),expanded=predict_rules(b,support),
                          leaky=float(records[0]['label'])))
        # Intervene on both forbidden labels; legal support/predictions must agree.
        for qlabel,flabel in product((0,1),repeat=2):
            altered=[dict(r) for r in records]
            altered[0]['label']=qlabel;altered[2]['label']=flabel
            safe=visible_labels(altered,q,10,{remote,future,q})
            assert safe==visible
            visibility.append(dict(theta=theta,b=b,query_label=qlabel,future_label=flabel,
                                   safe=predict_rules(b,[(0,y) for k,y in safe]),
                                   leaked=float(qlabel)))
    local_support=[((0,0),0),((1,1),1)]
    for s,a,b in product((0,1),repeat=3):
        columns.append(dict(s=s,a=a,b=b,truth=(a,b)[s],
                            local=predict_columns((a,b),local_support),
                            expanded=predict_columns((a,b),local_support+[((0,1),s)])))
    def scores(rows,arm):
        return dict(accuracy=sum(int(r[arm]>=.5)==r['truth'] for r in rows)/len(rows),
                    brier=sum((r[arm]-r['truth'])**2 for r in rows)/len(rows))
    # Features and existing labels are identical under either S. The external
    # distinguishing label equals S. Repeating observations over queries keeps
    # S independent of query feature values under the enumerated uniform prior.
    info_local=mutual_information([(r['s'],(r['a'],r['b'],0,1)) for r in columns])
    info_expanded=mutual_information([(r['s'],(r['a'],r['b'],0,1,r['s'])) for r in columns])
    return dict(experiment='B15-LABEL-VISIBILITY',rules=rules,columns=columns,
                interventions=visibility,rule_scores={a:scores(rules,a) for a in ('local','expanded','leaky')},
                column_scores={a:scores(columns,a) for a in ('local','expanded')},
                information_bits=dict(local=info_local,expanded=info_expanded),
                scope='Exact finite illustrations; not general proofs or paper benchmark reproduction')
