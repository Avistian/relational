# 0105 — Paper-results scale-up + visible implementation (standard #25)

**Date:** 2026-08-28
**Status:** Adopted as durable standard #25; retrofitted onto L043 / L044 / L045.
**Trigger:** User (`/teach`):
> next lessons will be trying even harder to reproduce results of the implemented
> architectures, if scale will not allow you to train it, add either modal script so I
> can train and compare or Google colab, next steps after lab implemented … I want to
> be sure that I will learn right conclusions.

Follow-up:
> Make sure that also code is not hidden behind package, so I can see direct
> implementation in notebook.

## Decision

Two failure modes, one standard.

**(A) Visible implementation.** A from-scratch encoder that the bake-off only `import`s from
`relkit` is hidden. Canonical source stays in `labs/relkit/*.py` (Modal / `_verify` share one
file). The student notebook **inlines** that file as a PROVIDED cell
(`relkit.paper_repro.inline_source`), skipping the names the student writes in TODOs so the
encoder calls *their* functions. `from relkit.tabnet import TabNetEncoder` is a checker /
harness import, not the only copy the student can read. Data / CV / metrics may still be
imported.

**(B) Paper-results track after EXIT, not stretch.** The learning lab may downscale so it
fits in minutes. That lab is a *different experiment* from the paper's table. Every
paper-mirror lab then ships a **NEXT STEP**: same from-scratch code, closer dataset / HPs /
metric, and a **conclusion ledger** with three buckets (*verified here* / *paper claim* /
*scale-up*) and verdicts MATCH · CLOSE · FAIL · INCOMPARABLE · NOT_RUN · DIRECTION_*. Until
the scale-up has been run, the paper claim stays **cited, not reproduced**.

If CPU cannot train it, ship **both** operators: `modal/l0NN_paper_repro.py` (unattended) and
a Colab-gated cell (`RUN_PAPER_REPRO = False` until they attach a T4). Optional stretch does
not satisfy this.

Presets: `smoke` (seconds, CI) · `closer` (T4, tens of minutes) · `paper` (hours, paper HPs).

## Why this was the miss

L043–L045 already mirrored the architecture and were honest about downscaling, but:

- paper tables were left as "out of scope / stretch";
- bake-offs imported `relkit.tabnet` / `relkit.node` / `relkit.tabtransformer`, so students
  never *read* Ghost BN, ODST, or the train loop in the notebook;
- toy-scale ranks (TabNet last on four small tables; NODE last vs CatBoost; TabTransformer
  0/3 vs CatBoost) can be internalized as "the paper was wrong."

Mixing those buckets is **M60**.

## Applied now

- Helper: `labs/relkit/paper_repro.py` (ledger + `inline_source` / drop student defs).
- Harnesses: `labs/_paper_repro_l043.py` (Adult + Syn4), `_paper_repro_l044.py`
  (Higgs-small OpenML 23512 vs CatBoost), `_paper_repro_l045.py` (full Adult contextual vs
  context-free vs CatBoost + RTD 3% labels).
- Modal: `modal/l043_paper_repro.py`, `l044_…`, `l045_…`, plus `_template_paper_repro.py`.
- Notebooks rebuilt so SETUP does not import the paper's encoder; after the student's TODO
  fragment, a PROVIDED cell is the inlined `relkit` file.
- Lessons 043–045 Lab sections point at the inlined encoder + Modal / Colab flag.
- `misconceptions.md` **M60**; `assets/retrieval-pool.js` `l045-lab-vs-paper` (lesson 45, so
  it appears from L046).
- Checklist: `lab-authoring` item 7, (A) visible-impl, (D) paper-results, compute-ladder
  **rung 7**. Full rule: `NOTES.md` Preferences #25.

## Right conclusions (the point of the ledger)

- Adult 85.7% on OpenML 1590 vs the UCI official test file → **INCOMPARABLE**, then read
  DIRECTION vs XGBoost.
- NODE Table 1 Higgs 0.2412 vs 0.2434 on OpenML 23512 (~98k of 10.5M) → **INCOMPARABLE** on
  the absolute number; a **DIRECTION_TIE** on a 0.002 edge is about *power*, not a refutation.
- TabTransformer +1.0% over deep baselines is a 15-dataset mean; full Adult is still not that
  suite.

Do not average buckets. Do not invert a lab ranking "because papers win at scale."
