"""Paper-results ledger (NOTES standard #25).

A downscaled learning lab and a paper table are *different experiments*. Mixing them
is how you learn the wrong conclusion: "TabNet/NODE/TabTransformer lost on four tiny
tables, therefore the paper was wrong" — or the opposite, "the paper won, so ignore
the lab." This module forces the distinction into a printed ledger.

Verdicts
--------
MATCH         our number is within abs_tol of the paper's, *and* the protocol matches.
CLOSE         within a looser band, protocol matches.
FAIL          protocol matches, number does not.
INCOMPARABLE  protocol does not match (different split, metric, subsample, budget) —
              do *not* treat a coincidental number as a reproduction.
NOT_RUN       the scale-up job has not been executed yet. The paper claim stays cited.
DIRECTION_*   comparative claims (A beats B) can still be tested on a substitute
              protocol even when the absolute paper number is INCOMPARABLE.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

Verdict = Literal[
    "MATCH", "CLOSE", "FAIL", "INCOMPARABLE", "NOT_RUN",
    "DIRECTION_MATCH", "DIRECTION_FAIL", "DIRECTION_TIE",
]


@dataclass
class PaperTarget:
    """One number or claim copied from the paper, with enough protocol to know
    whether our run can even be compared to it."""
    paper: str
    arxiv: str
    table: str
    dataset: str
    metric: str
    paper_value: float | None
    paper_std: float | None = None
    paper_split: str = ""
    higher_is_better: bool = True
    abs_tol: float = 0.01
    notes: str = ""


@dataclass
class LabFinding:
    """What the *learning lab* actually measured. Scope is mandatory — without it
    the row looks like a paper result."""
    claim: str
    value: str
    scope: str


@dataclass
class ScaleUpRun:
    method: str
    dataset: str
    metric: str
    value: float | None
    std: float | None = None
    n_seeds: int = 0
    hardware: str = ""
    wall_s: float = 0.0
    protocol_deviations: list[str] = field(default_factory=list)
    protocol_match: bool = False


def classify_number(target: PaperTarget, run: ScaleUpRun | None) -> Verdict:
    """Classify an absolute-number reproduction attempt."""
    if run is None or run.value is None:
        return "NOT_RUN"
    if not run.protocol_match or run.protocol_deviations:
        # Deviations make the number incomparable even if it looks close.
        # Callers who *did* match protocol must pass protocol_match=True and
        # an empty deviations list.
        if not run.protocol_match:
            return "INCOMPARABLE"
    if target.paper_value is None:
        return "INCOMPARABLE"
    delta = abs(float(run.value) - float(target.paper_value))
    tol = target.abs_tol
    if delta <= tol:
        return "MATCH"
    rel = delta / max(abs(float(target.paper_value)), 1e-12)
    if delta <= 2 * tol or rel <= 0.02:
        return "CLOSE"
    return "FAIL"


def classify_direction(
    ours_a: float,
    ours_b: float,
    *,
    paper_a_beats_b: bool,
    higher_is_better: bool = True,
    tie_tol: float = 0.002,
) -> Verdict:
    """Did the paper's *direction* (A beats B) survive our run?

    A tiny edge in the paper (NODE vs CatBoost on Higgs is ~0.002 error) will
    often be a DIRECTION_TIE on a subsample — that is evidence about *power*,
    not a refutation of the paper.
    """
    if abs(ours_a - ours_b) <= tie_tol:
        return "DIRECTION_TIE"
    ours_a_beats_b = (ours_a > ours_b) if higher_is_better else (ours_a < ours_b)
    return "DIRECTION_MATCH" if ours_a_beats_b == paper_a_beats_b else "DIRECTION_FAIL"


def _fmt_value(v: float | None, std: float | None = None) -> str:
    if v is None:
        return "—"
    if std is None:
        return f"{v:.4f}"
    return f"{v:.4f} ± {std:.4f}"


def format_ledger(
    *,
    title: str,
    lab: list[LabFinding],
    paper: list[tuple[PaperTarget, ScaleUpRun | None, Verdict]],
    extra_lines: list[str] | None = None,
) -> str:
    """Three-bucket ledger. Print this at the end of every paper-results job
    *and* in the lab's post-EXIT scale-up cell when the job has not run."""
    lines = [
        f"=== PAPER-RESULTS LEDGER — {title} ===",
        "",
        "BUCKET 1 — verified HERE (this lab's budget). A different experiment.",
    ]
    for row in lab:
        lines.append(f"  • {row.claim}: {row.value}")
        lines.append(f"      scope: {row.scope}")
    lines += [
        "",
        "BUCKET 2 — the PAPER's claim (cited). Not automatically true of bucket 1.",
        "BUCKET 3 — SCALE-UP run (Modal / Colab). Empty until you actually train.",
        "",
        f"{'dataset':<22} {'metric':<12} {'paper':>10} {'ours':>14} {'verdict':<18} table",
        "-" * 88,
    ]
    for target, run, verdict in paper:
        ours = _fmt_value(run.value, run.std) if run is not None else "NOT RUN"
        paper_s = _fmt_value(target.paper_value, target.paper_std)
        lines.append(
            f"{target.dataset:<22} {target.metric:<12} {paper_s:>10} {ours:>14} "
            f"{verdict:<18} {target.table}"
        )
        if target.notes:
            lines.append(f"      paper note: {target.notes}")
        if run is not None and run.protocol_deviations:
            for d in run.protocol_deviations:
                lines.append(f"      deviation: {d}")
        if run is not None and run.hardware:
            lines.append(
                f"      hardware: {run.hardware}  wall={run.wall_s:.0f}s  "
                f"seeds={run.n_seeds}  protocol_match={run.protocol_match}"
            )
    lines += [
        "",
        "How to read this. MATCH/CLOSE/FAIL are licensed only when protocol_match is",
        "true. INCOMPARABLE means you trained on related data but cannot quote the",
        "paper's number. NOT_RUN means the paper claim is still *cited, not reproduced*.",
        "DIRECTION_* is the comparative test that often remains valid on a substitute.",
        "Do not average buckets. Do not invert a lab ranking 'because papers win at scale'.",
    ]
    if extra_lines:
        lines.append("")
        lines.extend(extra_lines)
    return "\n".join(lines)


def print_howto(*, lesson: int, modal: str, harness: str) -> None:
    """Printed when the student leaves RUN_PAPER_REPRO = False (the default)."""
    print(f"=== NEXT STEP after Lab {lesson:04d} — reproduce the paper's results ===")
    print()
    print("The EXIT ticket you just printed is the LEARNING lab (mechanism + a")
    print("downscaled bake-off). The paper's TABLE is a different experiment.")
    print("Keeping them in separate buckets is how you learn the right conclusion.")
    print()
    print("Two ways to run the scale-up (same harness, different operator):")
    print()
    print("  1. Google Colab (you, in a browser, GPU).")
    print("     Runtime → Change runtime type → T4 GPU, then in the cell above")
    print("     set RUN_PAPER_REPRO = True and re-run. Do not walk away — free")
    print("     Colab disconnects after ~90 min of no tab interaction.")
    print()
    print("  2. Modal (unattended; survives closing the laptop):")
    print(f"       ~/.local/bin/modal run --detach {modal} --preset closer")
    print("     Pull the JSON when it finishes:")
    print("       ~/.local/bin/modal volume get relational-artifacts \\")
    print(f"           l{lesson:03d}/paper_repro.json labs/_paper_repro_l{lesson:03d}_results.json")
    print()
    print(f"Harness (also runnable locally if you have a GPU): python {harness} --preset closer")
    print("Presets:  smoke (seconds, CI) · closer (T4, ~15–40 min) · paper (hours, paper HPs)")
    print()
    print("Paste the ledger (MATCH / CLOSE / FAIL / INCOMPARABLE / DIRECTION_*)")
    print("to your teacher. That is the evidence that the paper claim survived,")
    print("failed, or is still out of reach — not the downscaled EXIT ranking.")


def device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def hardware_tag() -> str:
    dev = device()
    tag = dev
    if dev == "cuda":
        try:
            import torch
            tag = f"cuda:{torch.cuda.get_device_name(0)}"
        except Exception:
            tag = "cuda"
    return tag


def accuracy_from_logits(logits, y, *, threshold: float = 0.5) -> float:
    import numpy as np
    from sklearn.metrics import accuracy_score
    p = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float)))
    return float(accuracy_score(y, (p >= threshold).astype(int)))


def to_jsonable(obj) -> dict:
    """Dataclasses → plain dicts for the results JSON (stdlib types only)."""
    if hasattr(obj, "__dataclass_fields__"):
        d = asdict(obj)
        return {k: to_jsonable(v) for k, v in d.items()}
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def notebook_scaleup_md(
    *,
    lesson: int,
    paper: str,
    arxiv: str,
    lab_rows: list[tuple[str, str]],
    paper_rows: list[tuple[str, str]],
    modal: str,
) -> str:
    """Markdown cell appended AFTER the EXIT ticket of every paper-mirror lab."""
    lab_list = "\n".join(f"- **{k}:** {v}" for k, v in lab_rows)
    paper_list = "\n".join(f"- **{k}:** {v}" for k, v in paper_rows)
    return f'''## NEXT STEP — reproduce the paper's results (required, not stretch)

The EXIT ticket above is the **learning lab**: you implemented the architecture and ran a
downscaled bake-off that fits in minutes on CPU. That is a *different experiment* from the
paper's table. **Do not treat the EXIT ranking as the paper's result.** Mixing those two
buckets is how you learn the wrong conclusion (standard #25 / M60).

**Paper:** {paper} ([arXiv:{arxiv}](https://arxiv.org/abs/{arxiv}))

**Bucket 1 — verified here (this notebook's budget)**
{lab_list}

**Bucket 2 — the paper's claim (cited, not yet reproduced by this notebook)**
{paper_list}

**Bucket 3 — scale-up run (you train this).** Same from-scratch code, closer to the paper's
dataset / budget / metric. Two operators:

1. **Google Colab (you, GPU).** `Runtime → Change runtime type → T4 GPU`, set
   `RUN_PAPER_REPRO = True` in the next cell, run it. Stay in the tab — free Colab
   disconnects after ~90 min of no *tab* interaction, even if training is still going.
2. **Modal (unattended).** From the repo root:
   ```
   ~/.local/bin/modal run --detach {modal} --preset closer
   ```
   Use `--preset paper` only when you can spend hours and want paper hyperparameters.
   `smoke` is a seconds-long import check, not a result.

When it finishes, the cell prints a **ledger** with MATCH / CLOSE / FAIL / INCOMPARABLE /
DIRECTION_*. Paste that ledger to your teacher. Until you run it, the paper claim stays
*cited, not reproduced* — and that is an honest state, not a failure.
'''


def drop_top_level(src: str, names: set[str]) -> str:
    """Remove top-level `def` / `class` blocks (and their @decorators) whose names are in `names`.

    Used so a notebook can inline `relkit/tabnet.py` *without* overwriting the student's
    Task-1/Task-2 functions of the same name. Indentation, not the AST, is the parser —
    these files are straightforward.
    """
    import re
    lines = src.splitlines(keepends=True)
    n, i, out = len(lines), 0, []
    while i < n:
        j = i
        while j < n and lines[j].startswith("@"):
            j += 1
        m = re.match(r"^(def|class)\s+(\w+)", lines[j] if j < n else "")
        if m and m.group(2) in names:
            i = j + 1
            while i < n:
                s = lines[i]
                if s.strip() == "" or (s and s[0] in " \t"):
                    i += 1
                    continue
                break
            continue
        out.append(lines[i])
        i += 1
    return "".join(out)


def drop_imports_from(src: str, modules: set[str]) -> str:
    """Drop `from <module> import ...` (including parenthesized multiline)."""
    import re
    lines = src.splitlines(keepends=True)
    n, i, out = len(lines), 0, []
    pat = re.compile(r"^from\s+(\S+)\s+import\s+")
    while i < n:
        m = pat.match(lines[i])
        if m and m.group(1) in modules:
            i += 1
            if "(" in lines[i - 1]:
                while i < n and ")" not in lines[i - 1]:
                    i += 1
            continue
        out.append(lines[i])
        i += 1
    return "".join(out)


def strip_main_guard(src: str) -> str:
    idx = src.find("\nif __name__")
    if idx >= 0:
        src = src[:idx]
    return src.rstrip() + "\n"


def repo_relative(path: str) -> str:
    """Student-facing path: `labs/...` or `modal/...`, never an absolute `/workspace/...`."""
    text = str(path).replace("\\", "/")
    for marker in ("labs/", "modal/"):
        idx = text.find(marker)
        if idx >= 0:
            return text[idx:]
    from pathlib import Path
    return Path(path).name


def inline_source(
    path: str,
    *,
    skip_defs: set[str] | None = None,
    skip_imports: set[str] | None = None,
    strip_main: bool = True,
    banner: str | None = None,
) -> str:
    """Read a .py file and return a PROVIDED notebook cell body — the code, not an import.

    `path` is relative to the repo root or to `labs/`. Canonical source stays in that file
    (Modal / `_verify` import it); the notebook *copies* it so the student can read every line.
    """
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        alt = Path(__file__).resolve().parents[1] / Path(path).name
        if not alt.exists():
            alt = Path(__file__).resolve().parents[1] / path
        p = alt if alt.exists() else p
    src = p.read_text()
    if skip_defs:
        src = drop_top_level(src, skip_defs)
    if skip_imports:
        src = drop_imports_from(src, skip_imports)
    if strip_main:
        src = strip_main_guard(src)
    rel = repo_relative(path if p.exists() else str(p))
    head = banner or (
        f"# PROVIDED — inlined from `{rel}` so you can read every line.\n"
        f"# This is the paper implementation, not `import relkit...` hiding it.\n"
        f"# Canonical file stays at {rel} for Modal / _verify; this cell is a copy.\n\n"
    )
    return head + src


def notebook_scaleup_code(
    *,
    lesson: int,
    harness_path: str,
    modal: str,
    skip_imports: set[str],
) -> str:
    """PROVIDED scale-up cell: the training loop is *in the notebook*, gated by RUN_PAPER_REPRO.

    Model imports are stripped so the cell uses the architecture inlined earlier in the
    notebook, not a packaged copy. Modal still runs the .py file, which does import relkit.
    """
    rel_harness = repo_relative(harness_path)
    body = inline_source(
        harness_path,
        skip_imports=skip_imports,
        strip_main=True,
        banner=(
            f"# PROVIDED — paper-results scale-up, inlined from `{rel_harness}`.\n"
            f"# Read this cell: it is the training / comparison loop, not a hidden package.\n"
            f"# Default OFF so the learning lab stays minutes. On Colab: Runtime → T4 GPU,\n"
            f"# set RUN_PAPER_REPRO = True, re-run. Unattended: modal run --detach {modal}\n\n"
        ),
    )
    return body + f'''
from relkit.paper_repro import print_howto

RUN_PAPER_REPRO = False
PRESET = "closer"          # smoke | closer | paper

if RUN_PAPER_REPRO:
    main(["--preset", PRESET])
else:
    print_howto(lesson={lesson}, modal="{modal}", harness="{rel_harness}")
'''


def architecture_md(paper_piece: str, path: str, student_keeps: str) -> str:
    return (
        f"## The rest of the paper's architecture — inlined, not imported\n\n"
        f"The next cell is `{path}` copied into this notebook so you can **read every line** "
        f"of {paper_piece}. It is not `from relkit import ...` hiding a model behind a package. "
        f"{student_keeps} defined in your TODO cells above are **kept** — this copy skips those "
        f"names, so the encoder you train next calls *your* functions.\n\n"
        f"`relkit/` still holds the canonical file for Modal / `_verify` so the two cannot drift "
        f"in opposite directions; the notebook is a readable copy, not a black box."
    )
