# Released TACO cache-path discrepancy

Pinned code002f83bdb5b1776ca69b7916f993a82344a3aef7. This is a source inspection and measured disagreement, not a proven sole-cause diagnosis.

- `src/taco/model/taco_model.py`, `_predict_with_compressor`: the non-chunked path uses all N support rows when computing encoder statistics (`stats_x = X[:, :N, :]`).
- `src/taco/model/tabpfn_arch/inference.py`, `InferenceEngineCacheCompressorKV.prepare`: the non-chunked path with `compression_source == "test"` sets `S_stats = N-K`, then computes statistics from `X_for_compressor[:, :S_stats, :]`.
- The cache engine uses `getattr(core, 'compression_source', 'test')`; the model does not set that attribute and the experiment does not override it. AtN284,4% compression givesK12. Cached and uncached modes therefore have visibly different statistics code paths, beyond numerical rounding.
- Measured maximum probability differences across all four285-query passes: seed0 .0280723273754; seed1 .0319225192070; seed2 .0279806852341. All exceed the predeclared1e-5 tolerance. POT full and selected pass for every seed.
- No upstream patch was applied and no failed pair was removed. A controlled source patch plus re-execution would be needed to attribute the entire discrepancy to this statistics difference. That extra experiment is NOT_RUN.

Lesson implication: a cache mode name does not establish function preservation. Verify output equivalence for the actual release and inference recipe. Speed measurements remain measurements of those modes; they cannot be presented as an equal-prediction speedup where equivalence failed.
