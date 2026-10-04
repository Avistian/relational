from .types import AutoGrableConfig, AutoGrableResult, RefinementConfig
from .core import fit_autograble
from .graph import build_hetero_graph, build_hetero_graph_from_joined_table
from .refine import RefinementResult, fit_refinement, fit_gated_gnn, gate_summary
from .preprocess import make_tabular_features
from .models import (
    BaseHeteroModel,
    HeteroGatedGNN,
    HeteroSAGE,
    HeteroSAGETabArena,
    HeteroGIN,
    HeteroGINTabArena,
    HeteroGAT,
    HeteroGATTabArena,
    MODELS,
    SAGEConfig,
    SAGETabArenaConfig,
    GINConfig,
    GINTabArenaConfig,
    GATConfig,
    GATTabArenaConfig,
    ensure_row_self_loops,
    ensure_row_self_loops_tabarena,
    _with_seed,
    _with_seed_tabarena,
    _with_seed_gin,
    _with_seed_gin_tabarena,
    _with_seed_gat,
    _with_seed_gat_tabarena,
    apply_row_scaler,
    apply_row_scaler_tabarena,
    apply_row_scaler_gin,
    apply_row_scaler_gin_tabarena,
    apply_row_scaler_gat,
    apply_row_scaler_gat_tabarena,
    fit_row_scaler,
    fit_row_scaler_tabarena,
    fit_row_scaler_gin,
    fit_row_scaler_gin_tabarena,
    fit_row_scaler_gat,
    fit_row_scaler_gat_tabarena,
    run_seeds,
    run_seeds_tabarena,
    run_seeds_gin,
    run_seeds_gin_tabarena,
    run_seeds_gat,
    run_seeds_gat_tabarena,
    train_model,
    train_model_tabarena,
    train_model_gin,
    train_model_gin_tabarena,
    train_model_gat,
    train_model_gat_tabarena,
)
from .evaluate_graph_incidence import compute_J_incidence_from_df

__all__ = [
    # Core: autoGrable structural partition selection
    "AutoGrableConfig", "AutoGrableResult", "fit_autograble",
    # Graph builder
    "build_hetero_graph", "build_hetero_graph_from_joined_table",
    # Refinement (optional): parametric GNN trained on top of the selected structure
    "RefinementConfig", "RefinementResult", "fit_refinement", "fit_gated_gnn", "gate_summary",
    # Models
    "BaseHeteroModel", "HeteroGatedGNN", "MODELS",
    # Standalone SAGE baseline (own train/eval loop, not routed through fit_refinement)
    "HeteroSAGE", "SAGEConfig", "train_model", "run_seeds",
    "fit_row_scaler", "apply_row_scaler", "_with_seed",
    # Standalone SAGE baseline for TabArena
    "HeteroSAGETabArena", "SAGETabArenaConfig", "train_model_tabarena", "run_seeds_tabarena",
    "fit_row_scaler_tabarena", "apply_row_scaler_tabarena", "_with_seed_tabarena",
    # Standalone GIN baseline (transactional/fraud stack)
    "HeteroGIN", "GINConfig", "train_model_gin", "run_seeds_gin",
    "fit_row_scaler_gin", "apply_row_scaler_gin", "_with_seed_gin",
    # Standalone GIN baseline for TabArena
    "HeteroGINTabArena", "GINTabArenaConfig", "train_model_gin_tabarena", "run_seeds_gin_tabarena",
    "fit_row_scaler_gin_tabarena", "apply_row_scaler_gin_tabarena", "_with_seed_gin_tabarena",
    # Standalone GAT baseline (transactional/fraud stack); needs ('row','self','row')
    "HeteroGAT", "GATConfig", "train_model_gat", "run_seeds_gat", "ensure_row_self_loops",
    "fit_row_scaler_gat", "apply_row_scaler_gat", "_with_seed_gat",
    # Standalone GAT baseline for TabArena
    "HeteroGATTabArena", "GATTabArenaConfig", "train_model_gat_tabarena", "run_seeds_gat_tabarena",
    "ensure_row_self_loops_tabarena",
    "fit_row_scaler_gat_tabarena", "apply_row_scaler_gat_tabarena", "_with_seed_gat_tabarena",
    # Preprocessing
    "make_tabular_features",
    # Evaluate graph (via J)
    "compute_J_incidence_from_df"
]
