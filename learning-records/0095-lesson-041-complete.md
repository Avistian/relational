# 0095 — Lesson 041 complete: the deep-tabular landscape & rtdl

**Date:** 2026-08-05
**Status:** Complete (self-reported by the user: "lesson 41 complete")
**Curriculum:** Year 2 · Q1 · lecture 041 — first lesson of Year 2 (Advanced Tabular Deep Learning).
Topic: *the deep-tabular landscape & rtdl*; curriculum lab: *rtdl repo setup*.

## How it was marked complete
The user said "lesson 41 complete" (alongside the request to create Lesson 042). Per the L017–L040
precedent, the rtdl-setup lab's one-sentence rationale + param-count note was not pasted into chat, so
there is no fresh hostile-reader rubric score; treated as self-reported complete. The full teaching
content, the `tabular-dl-map-viz` landscape, the FT-Transformer tokenizer preview, and every
evidence-of-record citation are in the published record [[0094-lesson-041-published.md]].

## Retained (assumed, from the lesson design)
- **The baseline problem (M50).** Gorishniy 2021's contribution is *methodological*, not architectural:
  the field lacked a strong simple baseline (a well-tuned ResNet, which alone matches many "novel"
  models) and a shared tuning protocol, so pre-2021 "DL beats trees" claims were an HP-budget gap wearing
  a model-quality mask (L038).
- **No universal winner.** Run fairly, a tuned GBDT still wins on a large share of datasets; the best
  model is dataset-dependent — the same verdict as Grinsztajn 2022 (L024), reached from the architecture
  side.
- **FT-Transformer's one new idea.** The Feature Tokenizer tokenises *numeric* features too
  (`x_j·W_j + b_j`), so numbers attend — unlike TabTransformer (L032); a `[CLS]` token pools the row for
  the head. Full mechanism deferred to L046.
- **rtdl is the toolkit.** Reference PyTorch implementations (`rtdl_revisiting_models`: MLP, ResNet,
  FT-Transformer) so the neural bar is *strong and standard*, not home-made.

## Next
Lesson 042 — **MLP & ResNet baselines (do these first)** (per `CURRICULUM.md` Y2 Q1, primary reading
Gorishniy 2021 §3.2). This is the lesson that turns L041's map + rtdl setup into a hands-on training
skill: build and train the rtdl ResNet baseline under the shared protocol and read its result honestly
against a tuned GBDT — internalising *why you run the strong simple baselines first*.
