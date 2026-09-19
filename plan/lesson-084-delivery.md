# Lesson 084 · GAT delivery design

Scheduled scope: learned edge weighting after L083 sampling/mean aggregation. User explicitly requested creation and full reproducibility. Prepare a complete lesson package without changing learner completion.

Selected approach: a visible native-PyTorch port of the full released Cora experiment, with independent dense attention/gradient oracles. A library-only GATConv example would hide the load-bearing computation; an original TensorFlow 1-only notebook would obscure runnable teaching behind a legacy runtime. Preserve archived original code and distinguish this modern port from original-framework parity.

Teaching sequence: cold retrieval → one receiver's numerical trace → full eight-head/two-layer architecture → three live implementation TODOs → full fixed-split Cora experiment → measured attention and a limitations teach-back. Core reading 20–30 minutes; full training is a separate compute track. Reuse lesson.css, mpnn-lesson.css and portable PNG diagrams.

Reproduction target: Veličković et al. Table 2, Cora GAT 83.0 ± 0.7%, 100 runs. Pin release/source/data hashes; preserve full data, architecture, training and selection. Record framework/RNG/sparse execution differences, local seeds, actual environment, execution counts and unavailable historical parity. Test coefficient normalization, gradient flow, head merging, orientation, permutation equivariance, held-out label access and copied Pages links. Browser, notebook and full experiment execution are distinct checks.

## Executed evidence

Full100-run Cora port: mean83.193%, sample SD0.821pp. Independent inline seed0 replay matches every epoch and final score. Fresh isolated CPU installation and full659-epoch seed0 replay also match exactly. Source/data checks, browser controls at1100/375px, four portable figures and copied Pages delivery pass. Full notebook100-run lane NOT_RUN; CLI100-run lane executed. Historical framework parity INCOMPARABLE; live Colab and deployment NOT_CHECKED. No learner completion inferred.
