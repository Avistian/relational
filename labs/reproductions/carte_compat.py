"""Execution adapters for the June 2024 CARTE sources; no model/training substitutions."""
import importlib
import sys
import types
from pathlib import Path
import torch


def scatter_sum(src,index,dim=0,reduce='sum'):
    """Native differentiable equivalent of the one torch_scatter operation CARTE uses.

    Match the extension's inferred output length, not zeros_like(query). This
    fallback is used only when the optional compiled extension is unavailable.
    """
    if dim!=0 or reduce!='sum':raise NotImplementedError('The paper CARTE source only uses dim=0,sum')
    shape=(int(index.max())+1 if index.numel() else 0,*src.shape[1:])
    return src.new_zeros(shape).index_add(0,index,src)


def load_paper_carte(fasttext_path=None,checkpoint=None):
    # Let PyG detect optional extensions normally before exposing our scoped adapter.
    import torch_geometric
    root=Path(__file__).resolve().parent/'sources/carte_paper'
    sys.path.insert(0,str(root))
    injected=False
    try:
        import torch_scatter
    except ImportError:
        module=types.ModuleType('torch_scatter');module.scatter=scatter_sum
        sys.modules['torch_scatter']=module;injected=True
    try:
        from src.carte_estimator import CARTERegressor
        from src.carte_table_to_graph import Table2GraphTransformer
        from src.evaluate_utils import set_split
        from configs.directory import config_directory
    finally:
        if injected:sys.modules.pop('torch_scatter',None)
    if fasttext_path:config_directory['fasttext']=str(fasttext_path)
    if checkpoint:config_directory['pretrained_model']=str(checkpoint)
    if not hasattr(CARTERegressor,'_estimator_type'):CARTERegressor._estimator_type='regressor'
    return CARTERegressor,Table2GraphTransformer,set_split
