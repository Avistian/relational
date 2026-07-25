# Modal — cloud compute for lesson authoring

The escalation path for **rung 5** of the compute ladder (NOTES standard #20): when a lesson needs a
*verified* number the laptop cannot produce, the job runs here instead of being quietly downscaled.

This is the **agent's** runtime, not the student's. Colab is the student's runtime (every lab's first
cell is the `@colab-bootstrap`). The difference that matters: **Modal runs unattended from a CLI
token**, so the agent can launch a job, wait, and come back with numbers; Colab needs a human in a
browser tab and disconnects after ~90 min of no interaction.

Pattern copied from `~/Projects/curie-llm/modal/` — one shared `common.py`, thin per-job files.

## Status (verified 2026-07-25)

Already set up on this box; **no first-time signup needed**.

| Thing | Where |
|-------|-------|
| Token | `~/.modal.toml` (created for `curie-llm`, workspace `pszar92`) |
| CLI | `~/.local/bin/modal` (pipx, client 1.4.1) — **not** in the project `.venv` |
| Harness | `modal/common.py` — App `relational-labs`, CPU + GPU images, artifacts volume |
| Volume | `relational-artifacts`, mounted at `/artifacts` |

Smoke tests both pass:

```
CPU image: {'python': '3.12.6', 'cores': 24, 'sklearn': '1.9.0', 'lightgbm': '4.7.0', 'fitted_trees': 50}
GPU image: {'torch': '2.13.0+cu130', 'cuda_available': True, 'device': 'Tesla T4', 'matmul_ok': True}
```

## Workflow

1. **Smoke-test the harness** (cheap, do this after any image change):
   ```bash
   ~/.local/bin/modal run modal/common.py          # CPU image
   ~/.local/bin/modal run modal/common.py --gpu    # + attach a T4
   ```
2. **Write a job file** `modal/l0NN_<thing>.py` that imports from `common`:
   ```python
   from common import ARTIFACTS_PATH, app, artifacts, image_gpu, volumes

   @app.function(image=image_gpu, gpu="T4", volumes=volumes, timeout=3600)
   def run() -> dict:
       ...                       # write outputs under /artifacts
       artifacts.commit()        # required, or the files vanish with the container
       return {"log_loss": 1.42} # plain stdlib types only (see gotchas)

   @app.local_entrypoint()
   def main():
       print(run.remote())
   ```
3. **Run it.** Add `--detach` for anything over a few minutes so a local disconnect cannot cancel it:
   ```bash
   ~/.local/bin/modal run --detach modal/l0NN_thing.py
   ```
4. **Pull artifacts** and commit the numbers into the lesson + learning record:
   ```bash
   ~/.local/bin/modal volume get relational-artifacts <remote> <local>
   ```
5. **State the hardware** wherever the number appears. A T4 number is not a laptop number, and the
   cloud image is not the local venv (see version drift below).

## Cost

Starter plan: **$30/month of credits, no rollover, no card**. Per-second billing, no idle charge.
T4 $0.59/h → ~50 h/month; A100-40GB $2.10/h → ~14 h; CPU $0.135/core-hour, so CPU batch jobs are
effectively free. Check spend at <https://modal.com/settings/usage> before a big Y3/Y4 run —
**RelBench-scale work will not fit in $30**, so apply for Modal's academic credits (up to $10k for
students/researchers) *before* Year 3, not during.

## Gotchas (learned the hard way)

- **Return only plain stdlib types.** Return values are pickled and deserialised *locally*, where
  there is no torch. `torch.__version__` is a `TorchVersion` instance, not a `str` — returning it
  raw fails with `DeserializationError`. Wrap in `str()` / `float()` / `bool()`.
- **`volume.commit()` or it never happened.** Files written under `/artifacts` are lost with the
  container otherwise.
- **The sandbox blocks modal.com.** Agent shell calls need `full_network` permission.
- **Version drift is real:** the cloud image resolves to Python 3.12.6 / sklearn 1.9.0 while the
  local venv is Python 3.12.3. Fine for a self-contained cloud experiment; **pin exact versions in
  the image if a cloud number must be comparable to a laptop number.**
- **One `@app.local_entrypoint()` per file**, or `modal run <file>` cannot pick a target (use a flag,
  as `common.py` does with `--gpu`).
- **Pin threads in cloud jobs too.** The container reports 24 cores; the L031/L036 oversubscription
  trap (21 s → 0.28 s, 210 s → 9.2 s) applies just as much here.
