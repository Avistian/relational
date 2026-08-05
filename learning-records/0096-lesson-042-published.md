# 0096 — Lesson 042 published: MLP & ResNet baselines (do these first)

**Date:** 2026-08-05
**Status:** Published **with lab** (awaiting user completion). Retrofitted the same day after the user
directive *"make sure you dont forget about lab anymore"* — see **Directive** below (now NOTES standard
#21 + `lab-authoring` skill).
**Curriculum:** Year 2 · Q1 · lecture 042 (Advanced Tabular Deep Learning).
Topic per `CURRICULUM.md`: *MLP & ResNet baselines (do these first)*; curriculum lab: *Train ResNet baseline*.
**Primary reading:** Gorishniy, Rubachev, Khrulkov & Babenko, NeurIPS 2021 — *Revisiting Deep Learning
Models for Tabular Data*, **§3.2 (MLP & ResNet)** ([arXiv:2106.11959](https://arxiv.org/abs/2106.11959) ★).

## Single skill
Train the rtdl ResNet (and MLP) baseline under a **shared tuning protocol** — same split, metric, and
search budget as every other model — and read the result honestly against a tuned GBDT, internalising the
**baseline-first rule**: the strong simple baselines are run *first*, because a "win" over a weak or
unequally-tuned net measures a tuning-effort gap, not model quality.

## Why this was the ZPD
L041 unrolled the Y2-Q1 map and set up rtdl (imports + forward pass), leaving the ResNet chip as the
"do these first" default. L042 is the natural next rung: turn that setup into a trained, tuned baseline.
It recruits heavy prior recall — the residual block and degradation problem from **L028** (built there
torch-free; now trainable because torch/rtdl are installed), fair-budget search (**L017**), the L020
split contract, the corrected test (**L023**), HP-budget parity (**L038**), and the three inductive
biases (**L025–L027**) that explain the tree lead. It is the methodological bridge from "the neural bar
is *named*" (L041) to "the neural bar is *trainable and fair*."

## What shipped
- **Lesson** `lessons/0042-mlp-resnet-baselines.html` (~50 min): intro (map+setup → training) → one-sentence
  callout → warm-up (upTo:42) → residual-block recap (reused viz) + skip/BatchNorm defboxes → degradation
  problem (reused depth viz, verified L028 numbers) → **the shared tuning protocol** (shared frame vs
  per-model search space; AdamW/weight-decay/early-stopping defboxes) → the baseline-first rule →
  predict-before-reveal ("does a tuned ResNet beat a tuned GBDT on credit_g?") → **protocol bake-off viz**
  (new) + results table + no-universal-winner defbox → teach-back → thesis bridge → 3 quizzes → primary
  reading (§3.2) → inline **train-ResNet-baseline lab** → teacher/ask-teacher → nav.
- **One new reusable viz** (`file://`-safe): `assets/protocol-bakeoff-viz.js` — three baselines
  (MLP / ResNet / tuned GBDT) as clickable bars with **± std whiskers** under a "one shared protocol"
  banner, showing the **L042 verified** credit_g scores (MLP 0.808±0.024 / ResNet 0.793±0.029 /
  GBDT 0.780±0.015 — the whiskers overlap = a tie). Each click reveals the model's role + search space.
  Default = ResNet ("do these first"). CSS prefix `.pbo-`.
- **Reused two L028 viz** (no new asset): `resnet-block-viz` (skip on/off) and `depth-trainability-viz`
  (degradation curves) — CSS copied into the lesson head so it is self-contained.
- **Headless check** `labs/_viz_check_l042.js` — **all pass** (3 models, default resnet, readouts for
  incumbent / no-universal-winner / simplest-floor, name-click selection, banner present, `.pbo-`/`.rnb-`/
  `.dtr-` CSS coupling, all mounts, upTo:42, arXiv id). Pedagogy check also **all pass**. Browser MCP
  unavailable → node verification only (consistent with L021–L041).
- **Lab notebook** `labs/0042-mlp-resnet-baselines.ipynb` (+ solution, gitignored; + rendered
  `labs/html/0042-*.html`). Built by `labs/_build_l042.py`. Tier A (real credit_g). **Distinct from
  Lab 028** (which built the residual block from scratch): L042 uses the **rtdl reference models** as
  given and has the student implement the *shared protocol* — Task 1: a fair early-stopping loop that
  selects on validation ROC-AUC (3 blanks); Task 2: a validation-selected random search under a shared
  budget (1 blank, the selection criterion); Task 3: the honest verdict reading the GBDT-vs-ResNet gap
  against the noise band (2 blanks). PROVIDED: the shared frame, per-model search spaces, GBDT search,
  scoring. **Solution executed end-to-end with nbconvert — all CHECK cells pass**; its bake-off
  reproduces the tie (MLP 0.824 ≈ ResNet 0.799 ≈ GBDT 0.788, verdict "tie within noise"). `labPath` set
  in the manifest; `notebooks.html` renders it automatically; `rtdl_revisiting_models>=0.0.2` added to
  `requirements-labs.txt`.

## Honesty note — numbers RE-VERIFIED, lesson reconciled (important)
The lesson was **first drafted reusing the L028 credit_g numbers** (GBDT 0.793 > MLP 0.752 ≈ ResNet 0.743
— a GBDT lead). When the lab work produced **real L042 numbers with the rtdl reference models under a
proper shared search** (`labs/_verify_l042.py`: budget 10, 3 seeds), the result **changed the story**:
MLP **0.808 ± 0.024** ≈ ResNet **0.793 ± 0.029** ≈ GBDT **0.780 ± 0.015** — a three-way **tie within
noise**, with the tuned nets nominally *ahead* of the tree. Per the honesty rule (NOTES standard #20), the
lesson, the bake-off viz, the results table, the predict/teach-back/quiz3, `misconceptions.md` M51, and
`thesis-dossier.md` L042 were all **reconciled to the verified tie** (relabelled from "L028 evidence of
record" to "L042 verified"). This is *more* on-message for the "strong simple baselines" thesis than the
borrowed GBDT-lead framing: a properly-tuned neural baseline stands shoulder to shoulder with the tree on
this table — "no universal winner," not "trees win." (L028's own from-scratch, fixed-config experiment
still legitimately shows a nominal GBDT lead; M25/dossier-L028 keep those, correctly scoped to L028.)

## Directive — the lab ships WITH the lesson (user, 2026-08-05)
User: *"make sure you dont forget about lab anymore."* Encoded durably so it does not depend on agent
memory: **NOTES.md standard #21** (a non-writing lesson is not "created"/"published" until its lab
notebook exists; `labPath: null` only for genuine writing lessons; build-before-done checklist) and a new
gate at the **top of the `lab-authoring` skill** ("The lab ships WITH the lesson"). L042 is the named miss
of record — it had shipped `labPath: null` on a curriculum lab that literally reads "Train ResNet
baseline," exactly the first-real-training-lab case NOTES standard #20 anticipated.

## Design choices
- **Training lesson, not a re-teach of L028.** L028 owns the from-scratch block; L042 owns *training it
  fairly under the shared protocol* + the baseline-first discipline. The block/degradation viz are recapped
  (reused), not rebuilt, and the new material is the protocol (frame vs search space) and the recipe
  (AdamW/weight-decay/early-stopping).
- **One new viz (the protocol bake-off), two reused.** Per `lesson-visuals`: the block and the depth
  degradation are the same mechanisms L028 visualised; the genuinely new mechanistic beat is the *shared
  protocol as the frame that makes the numbers comparable*, which the new bake-off viz makes seeable.
- **`quarter: "Y2Q1"`** on the two new retrieval-pool items.

## Artifacts synced
- `assets/retrieval-pool.js` +2 (`l042-first` [misconception, mirrors M51], `l042-protocol`).
- `assets/paper-deck.js` +1 card `gorishniy2021-protocol` (§3.2 + the shared-protocol rule; distinct from
  the L028 `gorishniy2021` block card and the L041 `gorishniy2021-ftt` landscape card).
- `misconceptions.md` **M51** (beating a GBDT ≠ clearing the neural bar → run tuned MLP/ResNet first;
  example updated to the L042 verified tie) and **M52** (deep plain MLP scoring worse ≠ overfitting → the
  degradation problem; L028-scoped numbers, unchanged).
- `reference/glossary.html` — Year 2 · Q1 section +4 terms (baseline-first rule, shared frame vs search
  space, degradation problem, AdamW·weight-decay·early-stopping).
- `thesis-dossier.md` — Evidence Ledger L042 (BAR, C3/C4) + narrative paragraph; numeric clause updated
  to the L042 verified tie.
- `lessons/manifest.json` → L042 `labPath` now set to `labs/0042-mlp-resnet-baselines.ipynb`.
- `requirements-labs.txt` +`rtdl_revisiting_models>=0.0.2`.
- **New lab files:** `labs/_verify_l042.py` (+ `_verify_l042_results.json`), `labs/_build_l042.py`,
  `labs/0042-mlp-resnet-baselines.ipynb`, `labs/solutions/0042-mlp-resnet-baselines.ipynb` (executed),
  `labs/html/0042-mlp-resnet-baselines.html`.
- `assets/protocol-bakeoff-viz.js` rewritten to the verified numbers + ± std whiskers.
- L041 nav now links forward to L042.

## Next
User runs the train-ResNet-baseline lab and pastes the three test scores + corrected-test verdict + the
one-sentence "why run the baselines first" answer (or says "lab done"). On completion, open
**Lesson 043 — TabNet (sequential attention)** (Arik & Pfister 2019; curriculum lab: train + read masks)
— the first "novel" architecture, and the first chance to hold one to the baseline-first rule.
