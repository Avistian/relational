# Lesson 181 · RelBench v2 autocomplete

Approved 2026-10-02 in conversation. Named selected reproduction: RelBench v2 v1 Tables 5, 14, 15, rel-f1 results-position and qualifying-position; five aggregation baselines and five fresh released GNN fits per task. Whole benchmark and RelGT-AC numerical reproduction excluded; architecture and source-gap audit included.

Freeze paper-era source 0d47fe0c8a1a51aaf97ab485f4a028e290f97c67 (publication-date repository snapshot, not asserted historical experiment commit). GNN 10 epochs, batch512, width128, two layers, sum aggregation, Adam lr .005; actual fanouts128/64, L1 objective, validation MAE selection, train 2/98 percentile clipping. Seeds0–4 are explicit course choices; historical seed identities unresolved. Full source-constructed splits; authenticate raw data and all complete row/time keys. Preserve source global removal policy and baseline train+validation refit for test. No silent temporal repair, imputation, shorter run, or model substitution.

Budget USD10 total, planned stop8 with2 reserved, including all preparation, pilots, failures, retries and validation. Single L4/two CPU cores/16GiB: .00028372 USD/s at checked rates. At most 28,000 aggregate billed seconds in planned8USD envelope, further reduced for other charges; watchdog/reservations before any paid dispatch. First bounded pilot <=600 billed seconds, only after data, information access and gradient gates. A pilot extrapolation must admit all10 complete fits plus overhead. Local numerical work bounded to3600 aggregate seconds, with every attempt logged. Stop/report NOT_RUN or INCOMPLETE if scientific or cost gates fail.

## Implementation and validation

1. Archive paper/code, hashes, data and protocol; reconstruct source splits without allocating per-second timestamps across decades (prove same extrema/SQL predicates before use).
2. Test-first task visibility, fit scope and complete-key scoring contracts; independent SQL/metric verification. Audit preprocessing fit population, sampler time behavior and actual encoder gradients before paid work.
3. If admitted, retain all ten complete model states, keyed predictions and metrics; source-fidelity and numerical proximity are separate (predeclared descriptive tolerances .03 R2/.2 MAE; rounded baselines .001 R2/.001 MAE).
4. Connected HTML, model-specific diagrams, masking explorer, reference and portable student/solution notebooks with three live TODOs. Full original model/trainer visible; reproducible gated entry point and honest post-gate readiness status.
5. Execute standalone solution, independent corrupt-input/learner checks, browser/mobile/keyboard/print/noJS, figures, manifests, deterministic build and real Pages from clean Git index. No push/deployment. Preserve L180 practical exit INCOMPLETE and learner PENDING_WRITTEN_DEFENSE.

The referenced writing-plans skill is unavailable in the searched skill locations; this approved design records the executable sequence.
