"""Read-only OAG compatibility loader extracted from L093; see pinned provenance."""
import dill
import pandas as pd
from hin_l094 import file_sha256

class OAGGraph:
    """Compatibility container for the authors' released dill Graph object."""
    def get_types(self):return list(self.node_feature)
    def get_meta_graph(self):
        return [(t,s,r) for t,sv in self.edge_list.items() for s,rv in sv.items() for r in rv]

def inert_legacy_code(*args):
    # Python3.7 bytecode in old defaultdict factories is not executed.
    return None

def inert_legacy_function(*args):
    # The downloaded graph stores populated maps. Their historical lambda
    # factories are irrelevant to read-only sampling; missing keys must not be used.
    return dict

def legacy_type(name):
    return inert_legacy_code if name=='CodeType' else dill._dill._load_type(name)

class OAGUnpickler(dill.Unpickler):
    def find_class(self,module,name):
        if name=='Graph' and module in ('pyHGT.data','GPT_GNN.data','data'):return OAGGraph
        if module=='pandas.core.indexes.numeric':return pd.Index
        if module in ('data','pyHGT.data','GPT_GNN.data') and name=='__dict__':return {}
        if module=='dill._dill' and name=='_load_type':return legacy_type
        if module=='dill._dill' and name=='_create_function':return inert_legacy_function
        return super().find_class(module,name)

def load_oag(path,expected_sha256):
    """Verify trusted release bytes before deserializing; missing hash fails closed."""
    if not expected_sha256 or file_sha256(path)!=expected_sha256:raise ValueError('OAG data SHA256 mismatch or missing expected hash')
    with open(path,'rb') as f:g=OAGUnpickler(f).load()
    assert 'paper' in g.node_feature and 'field' in g.node_feature
    return g
