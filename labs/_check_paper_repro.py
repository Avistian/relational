"""Unit checks for the paper-results ledger (standard #25) — no GPU, no training.

Run: .venv/bin/python labs/_check_paper_repro.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from relkit.paper_repro import (  # noqa: E402
    LabFinding, PaperTarget, ScaleUpRun,
    classify_direction, classify_number, format_ledger, to_jsonable,
)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def main():
    adult = PaperTarget(
        paper="TabNet", arxiv="1908.07442", table="Appendix (Adult)",
        dataset="adult", metric="accuracy", paper_value=0.857, abs_tol=0.01,
        paper_split="UCI Adult official test (16,281 rows)",
    )
    match_run = ScaleUpRun(
        method="tabnet", dataset="adult", metric="accuracy",
        value=0.855, n_seeds=3, hardware="cuda:T4", protocol_match=True,
    )
    close_run = ScaleUpRun(
        method="tabnet", dataset="adult", metric="accuracy",
        value=0.840, n_seeds=1, hardware="cuda:T4", protocol_match=True,
    )
    fail_run = ScaleUpRun(
        method="tabnet", dataset="adult", metric="accuracy",
        value=0.70, n_seeds=1, hardware="cpu", protocol_match=True,
    )
    incomp_run = ScaleUpRun(
        method="tabnet", dataset="adult", metric="accuracy",
        value=0.857, n_seeds=1, hardware="cpu", protocol_match=False,
        protocol_deviations=["random 70/15/15 split, not the UCI official test file"],
    )

    check("MATCH within abs_tol", classify_number(adult, match_run) == "MATCH")
    check("CLOSE in the 2×-tol band", classify_number(adult, close_run) == "CLOSE")
    check("FAIL well outside tol", classify_number(adult, fail_run) == "FAIL")
    check("INCOMPARABLE when protocol_match is false, even if the number matches",
          classify_number(adult, incomp_run) == "INCOMPARABLE")
    check("NOT_RUN when there is no scale-up yet",
          classify_number(adult, None) == "NOT_RUN")
    check("NOT_RUN when value is None",
          classify_number(adult, ScaleUpRun("x", "adult", "accuracy", None)) == "NOT_RUN")

    # NODE Higgs default: NODE 0.2412 error vs CatBoost 0.2434 — NODE wins (lower error).
    check("DIRECTION_MATCH when the paper's winner still wins",
          classify_direction(0.230, 0.250, paper_a_beats_b=True, higher_is_better=False) == "DIRECTION_MATCH")
    check("DIRECTION_FAIL when the ranking flips",
          classify_direction(0.260, 0.240, paper_a_beats_b=True, higher_is_better=False) == "DIRECTION_FAIL")
    check("DIRECTION_TIE when the gap is smaller than tie_tol (under-powered subsample)",
          classify_direction(0.241, 0.242, paper_a_beats_b=True, higher_is_better=False, tie_tol=0.005)
          == "DIRECTION_TIE")

    lab = [LabFinding("TabNet mean rank vs MLP", "2.50 vs 1.75",
                      "4 small OpenML tables, budget 6, CPU")]
    rows = [
        (adult, None, classify_number(adult, None)),
        (adult, incomp_run, classify_number(adult, incomp_run)),
    ]
    text = format_ledger(title="L043 TabNet", lab=lab, paper=rows)
    check("ledger names the three buckets",
          "BUCKET 1" in text and "BUCKET 2" in text and "BUCKET 3" in text)
    check("ledger prints NOT_RUN, not a fake paper number", "NOT RUN" in text)
    check("ledger prints INCOMPARABLE rather than MATCH for a coincidental 0.857",
          "INCOMPARABLE" in text and "do not average buckets" in text.lower()
          or "Do not average buckets" in text)
    check("ledger refuses to treat a coincidental hit as MATCH",
          "INCOMPARABLE" in text)
    check("to_jsonable emits only stdlib types",
          isinstance(to_jsonable(adult)["paper_value"], float)
          and isinstance(to_jsonable(incomp_run)["protocol_deviations"], list))

    from relkit.paper_repro import drop_imports_from, drop_top_level, inline_source, strip_main_guard

    sample = '''import torch

def sparsemax(z):
    return z

class GhostBatchNorm:
    pass

class TabNetEncoder:
    def forward(self, x):
        return sparsemax(x)

def train_tabnet(m):
    return m

if __name__ == "__main__":
    train_tabnet(None)
'''
    dropped = drop_top_level(sample, {"sparsemax"})
    check("drop_top_level removes the named def", "def sparsemax" not in dropped)
    check("drop_top_level keeps the other class", "class TabNetEncoder" in dropped)
    check("drop_top_level keeps train_tabnet", "def train_tabnet" in dropped)

    imps = '''from relkit.tabnet import TabNetEncoder, train_tabnet
from relkit.paper_repro import format_ledger
from relkit.tabnet import (
    explain,
    tabnet_auc,
)
from relkit import load_tier_a
'''
    stripped = drop_imports_from(imps, {"relkit.tabnet"})
    check("drop_imports removes the paper-model import", "relkit.tabnet" not in stripped)
    check("drop_imports keeps the ledger import", "relkit.paper_repro" in stripped)
    check("drop_imports keeps load_tier_a", "load_tier_a" in stripped)

    check("strip_main_guard drops the CLI entry",
          "if __name__" not in strip_main_guard(sample) and "def train_tabnet" in strip_main_guard(sample))

    inlined = inline_source(os.path.join(HERE, "relkit/tabnet.py"), skip_defs={"sparsemax"})
    check("inline_source banner is repo-relative, not an absolute workspace path",
          "/workspace" not in inlined.split("\n", 3)[0] and "labs/relkit/tabnet.py" in inlined)
    check("inline_source copies TabNetEncoder instead of importing it",
          "class TabNetEncoder" in inlined and "from relkit.tabnet import" not in inlined)
    check("inline_source skips the student's sparsemax", "def sparsemax" not in inlined)

    from relkit.paper_repro import notebook_scaleup_code, repo_relative
    check("repo_relative strips the workspace prefix",
          repo_relative("/workspace/labs/_paper_repro_l043.py") == "labs/_paper_repro_l043.py")
    cell = notebook_scaleup_code(
        lesson=43,
        harness_path=os.path.join(HERE, "_paper_repro_l043.py"),
        modal="modal/l043_paper_repro.py",
        skip_imports={"relkit.tabnet"},
    )
    check("scale-up cell does not import the paper encoder from relkit",
          "from relkit.tabnet import" not in cell)
    check("scale-up cell still has the training loop (main)", "def main(" in cell)
    check("scale-up cell is gated off by default", "RUN_PAPER_REPRO = False" in cell)

    print()
    print(f"{len(PASS)} PASS / {len(FAIL)} FAIL")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
