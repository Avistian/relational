"""Stable identities for the functions actually used by the L055 notebook."""
import inspect
import types
from _live_identity_l051 import code_fingerprint


def live_identity(namespace):
    """Follow course-function dependencies, including model methods/global helpers.

    Nested code objects are covered by code_fingerprint. Library implementations
    are identified by saved package versions, not repr addresses or object IDs.
    """
    names=('paired_random_split','fit_preprocessor','apply_preprocessor','select_candidate',
           'load_release','metric','fit_neural','evaluate_arm','run_suite','summarize',
           'MLP','TabM','BatchEnsembleLinear','PackedHead')
    roots={k:namespace[k] for k in names}
    allowed={v.__module__ for v in roots.values()}
    seen=set();result={}
    def code_names(code):
        names=set(code.co_names)
        for constant in code.co_consts:
            if isinstance(constant,types.CodeType):names.update(code_names(constant))
        return sorted(names)
    def visit(value,label):
        if id(value) in seen:return
        if not (inspect.isfunction(value) or inspect.isclass(value)):return
        if value.__module__ not in allowed:return
        seen.add(id(value))
        if inspect.isclass(value):
            for key,member in vars(value).items():
                if inspect.isfunction(member):visit(member,label+'.'+key)
            return
        result[label]=code_fingerprint(value)
        # Every global function reached by the current implementation matters.
        for key in code_names(value.__code__):
            dependency=value.__globals__.get(key)
            if inspect.isfunction(dependency) or inspect.isclass(dependency):visit(dependency,label+' -> '+key)
    for name,value in roots.items():visit(value,name)
    return dict(scheme='semantic-code-v1',scope='actual live notebook functions and reachable course helpers',functions=result)
