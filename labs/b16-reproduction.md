# B16 reproduction contract — frozen before numerical execution

Approved 2026-10-04. Primary paper: AutoGrable v1, https://arxiv.org/html/2608.11431v1 . Source commit: `1cfad48ae8d8362ad7eca10f9fe6db115004975e`; hashes in `sources/b16/manifest.json`.

## Lane A: B16-PARTITION-SELECTION (course mechanism)

Synthetic, complete finite experiment. Three worlds: signal, XOR, null. Each split has eight rows indexed i=0,...,7. `A=i//4`, `B=(i//2)%2`, `K=<split>-<i>`, `id=<split>-<i>`. Labels: signal `A`; XOR `A XOR B`; null `i%2`. Splits tr/va/te have disjoint identities and key values but share A/B combinations. This is a constructed distribution, not Adult or an empirical estimate of real-data generalization. Null means no useful repeatable feature among the declared candidates on these splits, not absence of every possible rule.

Eligible columns fixed to A,B,K; y/id forbidden. Value signature only. Binary scalar Brier loss `(p-y)^2`, bounded by 1; training block label proportions, training marginal fallback for unseen validation/test cells. Occupancy `sum(sqrt(n_b))/n_train`. Penalties 0, 0.5, 1. All eight subsets enumerated for each of 9 world/penalty pairs: 72 scores. Three methods (exhaustive, forward, backward): 27 selections. Exact arithmetic uses float64; comparisons tolerance 1e-12. Exhaustive ties favor smaller subsets, then lexicographic column order. Greedy inspects columns in A,B,K order and accepts only improvement >1e-12; backward can reach empty. Thus ties may produce a different selected subset at lambda=0 even with the same objective. No HPO, no randomness, no neural optimization, no statistical confidence interval over fabricated seeds.

Fit probabilities on training labels, select using validation scores, evaluate immutable test probabilities only after selection. Save complete row IDs, all subset scores, accepted/considered search traces, and test predictions. Enumerate all 256 possible test label vectors for the signal world at lambda=0.5; selection and predictions must stay identical. Independently rescore all selected test outputs and 72 subset scores.

Build actual typed incidence graphs and execute colour refinement for all 3 worlds x 8 subsets = 24 equivalence checks. All row nodes initially identical; each value-node colour identifies (column,value), and edge type identifies its column. Compare same-block relations, not arbitrary colour numbers. Additional counterexamples hide value identity or retain row feature B. Query labels never enter graph features.

## Lane B: B16-AUTOGRABLE-TABLE1 (selected published experiment)

Target: paper Table 1, all 5 families (single-value, conjunction, XOR, count, duplicate), 2 signatures (value/frequency), 2 directions, lambda 0/1, 10 seeds, clean labels = 400 selection runs; 40 exact-recovery and 40 recall percentage cells. No downstream GNN is trained in this experiment. Preserve these dimensions without silent reductions. Full Table 1 execution currently `NOT_RUN`; aggregate verdict `INCOMPLETE_SOURCE_PROTOCOL_GATE`.

Missing in pinned release: Adult file/preprocessing identity; row sample size and resampling recipe; the 10 seed identities; exact column/value/threshold/binarization generators and balancing/rejection procedure; train/validation membership; exact loss and tolerance used in the reported runs. Repository defaults are not historical run receipts. Available generic Adult usage example predicts income; it is not the five-family Table 1 generator.

Material discrepancies requiring resolution:

- Algorithm 1 permits backward removal to empty. Released `selection.py` stops when one column remains. This can prevent abstention.
- Algorithm 1/Eq.2 frequency-recodes T=train union validation. Released `core.py` encodes train with train counts and validation with union counts, explicitly asymmetrically. The same literal value can therefore receive different signatures across splits. A union-based signature also depends on validation features; the conditional validation guarantee needs its assumptions checked before reuse.
- Lemma 1 uses typed-value identities in a row-feature-erased graph. Released `graph.py` gives value nodes zero feature vectors and keeps identity only in a separate vocabulary. The colour-refinement conclusion cannot be transferred automatically to that graph. Actual full row features may distinguish additional rows; this is separate from the reduct. The TabArena encoder offers per-ID embeddings and random value features (both off by default); enabling them changes the audited information boundary. No claim about every downstream configuration is made.
- Source default loss is clipped log loss; course diagnostic uses bounded binary Brier loss. Course values cannot be compared numerically to Table 1.

The paper operator authenticates the complete source packet, prints the missing contract and exits nonzero for a paper attempt. It cannot become a benchmark launcher merely by editing a status flag. Completing the lane requires recovering and reviewing the missing experiment driver/configuration. Full source including graph models and trainers is preserved for inspection. No promise of executable missing experimental settings is made.

## Budget and commands

Initial paid spend USD0; aggregate ceiling USD10 including preparation/retries/verification, paid stop USD8/reserve USD2. Local numerical execution cutoff 3600 aggregate wall seconds, recorded by `_budget_b16.py`; wrappers count failed attempts. No paid runs are dispatched by this package. Selected paper runtime/cost forecast cannot be established before its source gate clears.

From repository root:

```
.venv/bin/python labs/_budget_b16.py .venv/bin/python labs/_test_b16.py
.venv/bin/python labs/_budget_b16.py .venv/bin/python labs/_run_b16.py
.venv/bin/python labs/_budget_b16.py .venv/bin/python labs/_audit_b16.py
.venv/bin/python labs/_budget_b16.py .venv/bin/python labs/_verify_b16.py
.venv/bin/python labs/_reproduce_b16.py --phase audit
.venv/bin/python labs/_reproduce_b16.py --phase paper
```

The final command intentionally refuses an underspecified benchmark. Remaining benchmarks (RQ2/RQ3, noise/lambda sweeps, relational/transactional/TabArena results) are NOT_RUN. Source parity, finite mechanism verification, paper-result reproduction, historical identity, live Colab, publication and learner mastery are distinct. Learner status PENDING_WRITTEN_DEFENSE.
