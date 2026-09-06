"""Stable fingerprints of executable code, independent of marshal reference flags."""
import hashlib,json,types

def code_fingerprint(function):
    """Hash bytecode and semantic constants, names, defaults and signature fields.

    marshal.dumps(code) can change its reference flags after a function executes.
    File/line locations and interpreter bookkeeping are intentionally excluded.
    The operator separately hashes data, settings, versions and model methods.
    """
    def encode(value):
        if isinstance(value,types.CodeType):
            return dict(bytecode=value.co_code.hex(),constants=[encode(x) for x in value.co_consts],
                names=value.co_names,varnames=value.co_varnames,freevars=value.co_freevars,
                cellvars=value.co_cellvars,flags=value.co_flags,argcount=value.co_argcount,
                posonly=value.co_posonlyargcount,kwonly=value.co_kwonlyargcount,
                exceptiontable=value.co_exceptiontable.hex())
        if isinstance(value,(tuple,list)):return [encode(x) for x in value]
        if isinstance(value,dict):return {str(k):encode(v) for k,v in sorted(value.items())}
        if isinstance(value,bytes):return {'bytes':value.hex()}
        if value is None or isinstance(value,(str,int,float,bool)):return value
        return {'type':type(value).__name__,'repr':repr(value)}
    payload=dict(code=encode(function.__code__),defaults=encode(function.__defaults__),kwdefaults=encode(function.__kwdefaults__))
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
