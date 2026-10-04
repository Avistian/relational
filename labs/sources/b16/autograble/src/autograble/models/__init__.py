from .base import BaseHeteroModel
from .gated_gnn import HeteroGatedGNN
from .SAGE_Fraud import (
    HeteroSAGE,
    SAGEConfig,
    _with_seed,
    apply_row_scaler,
    fit_row_scaler,
    run_seeds,
    train_model,
)
from .SAGE_TabArena import (
    HeteroSAGE as HeteroSAGETabArena,
    SAGEConfig as SAGETabArenaConfig,
    _with_seed as _with_seed_tabarena,
    apply_row_scaler as apply_row_scaler_tabarena,
    fit_row_scaler as fit_row_scaler_tabarena,
    run_seeds as run_seeds_tabarena,
    train_model as train_model_tabarena,
)
from .SAGE_Relbench import (
    HeteroSAGE as HeteroSAGERelbench,
    SAGEConfig as SAGERelbenchConfig,
    _with_seed as _with_seed_relbench,
    apply_row_scaler as apply_row_scaler_relbench,
    fit_row_scaler as fit_row_scaler_relbench,
    run_seeds as run_seeds_relbench,
    train_model as train_model_relbench,
)
from .GIN_Fraud import (
    HeteroGIN,
    GINConfig,
    _with_seed as _with_seed_gin,
    apply_row_scaler as apply_row_scaler_gin,
    fit_row_scaler as fit_row_scaler_gin,
    run_seeds as run_seeds_gin,
    train_model as train_model_gin,
)
from .GIN_TabArena import (
    HeteroGIN as HeteroGINTabArena,
    GINConfig as GINTabArenaConfig,
    _with_seed as _with_seed_gin_tabarena,
    apply_row_scaler as apply_row_scaler_gin_tabarena,
    fit_row_scaler as fit_row_scaler_gin_tabarena,
    run_seeds as run_seeds_gin_tabarena,
    train_model as train_model_gin_tabarena,
)
from .GAT_Fraud import (
    HeteroGAT,
    GATConfig,
    ensure_row_self_loops,
    _with_seed as _with_seed_gat,
    apply_row_scaler as apply_row_scaler_gat,
    fit_row_scaler as fit_row_scaler_gat,
    run_seeds as run_seeds_gat,
    train_model as train_model_gat,
)
from .GAT_TabArena import (
    HeteroGAT as HeteroGATTabArena,
    GATConfig as GATTabArenaConfig,
    ensure_row_self_loops as ensure_row_self_loops_tabarena,
    _with_seed as _with_seed_gat_tabarena,
    apply_row_scaler as apply_row_scaler_gat_tabarena,
    fit_row_scaler as fit_row_scaler_gat_tabarena,
    run_seeds as run_seeds_gat_tabarena,
    train_model as train_model_gat_tabarena,
)

# Registry: maps model name → class.
# Add new models here as they are implemented.
# NOTE: HeteroSAGE / HeteroSAGETabArena / HeteroSAGERelbench,
# HeteroGIN / HeteroGINTabArena and HeteroGAT / HeteroGATTabArena are not
# registered here -- they don't implement the BaseHeteroModel /
# fit_refinement constructor contract and have their own standalone
# train_model()/run_seeds() training loops.
MODELS: dict = {
    "gated_gnn": HeteroGatedGNN,
}

__all__ = [
    "BaseHeteroModel", "HeteroGatedGNN", "MODELS",
    "HeteroSAGE", "SAGEConfig", "run_seeds", "train_model",
    "fit_row_scaler", "apply_row_scaler", "_with_seed",
    "HeteroSAGETabArena", "SAGETabArenaConfig", "run_seeds_tabarena", "train_model_tabarena",
    "fit_row_scaler_tabarena", "apply_row_scaler_tabarena", "_with_seed_tabarena",
    "HeteroSAGERelbench", "SAGERelbenchConfig", "run_seeds_relbench", "train_model_relbench",
    "fit_row_scaler_relbench", "apply_row_scaler_relbench", "_with_seed_relbench",
    "HeteroGIN", "GINConfig", "run_seeds_gin", "train_model_gin",
    "fit_row_scaler_gin", "apply_row_scaler_gin", "_with_seed_gin",
    "HeteroGINTabArena", "GINTabArenaConfig", "run_seeds_gin_tabarena", "train_model_gin_tabarena",
    "fit_row_scaler_gin_tabarena", "apply_row_scaler_gin_tabarena", "_with_seed_gin_tabarena",
    "HeteroGAT", "GATConfig", "run_seeds_gat", "train_model_gat", "ensure_row_self_loops",
    "fit_row_scaler_gat", "apply_row_scaler_gat", "_with_seed_gat",
    "HeteroGATTabArena", "GATTabArenaConfig", "run_seeds_gat_tabarena", "train_model_gat_tabarena",
    "ensure_row_self_loops_tabarena",
    "fit_row_scaler_gat_tabarena", "apply_row_scaler_gat_tabarena", "_with_seed_gat_tabarena",
]
