

# Round 5 — discuss answers (resume; logging→W&B & pod-termination durability)

## Discuss 1 — Stop & resume across ALL components
A run's state lives in five places; resume = restore each. What each needs:

| Component | State it holds | How it resumes |
|---|---|---|
| **TRL/Trainer** | model + optimizer + scheduler + RNG + step | `resume_from_checkpoint: true` (already set) → loads the latest `checkpoint-N/`. **Needs optimizer state IN the checkpoint** ⇒ keep `save_only_model` **false** locally (it's unset now = false ✓). |
| **W&B** | the run (its curves) | To CONTINUE the same run (not fork a new one): `WANDB_RESUME=must` + `WANDB_RUN_ID=<same id>`. Persist the id at run start (`output_dir/wandb_run_id`), read it back on resume. |
| **vLLM** | none (stateless wrt training) | Just restart the server; the trainer re-syncs current weights on the first step. No vLLM-side resume. |
| **Uploader** | in-memory `uploaded` set | Just restart it; it re-scans and re-checks. HF `upload_folder` is idempotent, so re-touching an already-present checkpoint is harmless. |
| **Data order** | which batches were seen | Same `seed` ⇒ identical shuffle; TRL skips already-seen batches from `trainer_state` (`ignore_data_skip: true` to skip the slow fast-forward if you don't care about exact order). |

**The one real gotcha — a tension you should know:** to *resume* you need optimizer state in the
checkpoint (`save_only_model: false`), but you do NOT want that 2× bulk on HF. Already handled: the
uploader's `IGNORE` drops `optimizer.pt`/`scheduler.pt`/`rng_state*`, so **local = resumable, HF =
clean adapter**. **Changes to implement when you green-light resume:** (1) persist+reload the W&B
run-id in `train.py` (write `output_dir/wandb_run_id` at start; on restart read it → set
`WANDB_RUN_ID`+`WANDB_RESUME=must`); (2) keep `save_only_model` false (leave unset); (3)
`resume_from_checkpoint: true` (already). Everything else is literally "restart the process."

## Discuss 2 — logging→W&B, inspect's own logs, surviving a killed runpod
Your instinct is the right one to design around: **local disk dies with the pod, so anything not
shipped is lost.** Map every artifact to a durable home:

| Data | Durable home | Live, or only at end? |
|---|---|---|
| training metrics (`reward/*`, loss, grad_norm) | **W&B** (auto via `report_to`) | live ✓ safe |
| checkpoints (adapters) | **HF** (uploader, every save) | live ✓ safe |
| trainer console | **W&B "Logs" tab** (auto after `wandb.init`) | live ✓ safe |
| **uploader / evals console** | files now | ⚠️ not shipped |
| **inspect `.eval` files** (full transcripts + scores) | inspect's OWN format, local `INSPECT_LOG_DIR` | ⚠️ not shipped |
| **`logs/<RUN_ID>/*.log`** | files now | ⚠️ not shipped |

- **Does inspect log to W&B?** No — inspect writes its own durable `.eval` logs (full transcripts +
  scores) locally; that's the eval's source of truth, but it's on the pod ⇒ must be shipped.
- **Route uploader/evals → W&B?** Yes, but each separate process must `wandb.init(id=<run>,
  resume="allow")` to land in the SAME run (the deferred run-id handoff). Then: their console is
  captured, MGS scores go as metrics, and the `.eval` file goes as a W&B **artifact**.
- **Recommended division of labor:** **W&B** = metrics + small stuff as artifacts (per-process
  `*.log`, eval summaries); **HF** = checkpoints + the large raw `.eval` bundles. Add a **finalize
  step that ships `logs/<RUN_ID>/` + the `.eval` dir** (a natural extension of the uploader) so a pod
  kill *after* training loses nothing; for anytime-kill safety, ship periodically (every N polls), not
  only at the end. That closes all three ⚠️ rows.
- **What I'm building now (B):** the per-process log DIR + files (so logs are shippable and readable
  by us both). The *shipping* to W&B/HF rides on the run-id handoff — I'll wire it when we build the
  periodic eval runner (same handoff as Q10). Until then: metrics+checkpoints are already durable; the
  logs survive as long as the pod does — **so before you terminate a pod, grab `logs/<RUN_ID>/` +
  `INSPECT_LOG_DIR`** (noted in the README).

---

## ✅ Implemented this round (all validated: 31 unit tests + e2e, ruff + mypy clean)
- **`checkpoint_uploader.py`** rewritten — child (`argparse` CLI) + parent (`start`/`finalize`) in one
  module; **`uploader_control.py` deleted** (folded in). Config via **CLI args**; `HF_TOKEN` via env.
  `_is_complete(path, kind)` (no module globals). Rename **`EACH_STEP` → `overwrite_previous`**
  (`false` = per-step history, `true` = latest-only root). `--private` (default public).
- **`hf_uploader:` block** replaces the flat keys in both RL run-configs AND `sdf_instruct.yaml`
  (per stage). `launch.py` reads it from the merged config (dicts are NOT dumped to env); `rl/train.py`
  reads `rc["hf_uploader"]`. Both call `checkpoint_uploader.start/finalize`.
- **`training/logs.py`** (new) — `get_logger(name)` → stdout **+** `logs/<RUN_ID>/<LOG_PROC>.log`
  (per-process file, timestamped/tagged). Adopted in the uploader + `rl/train.py`; `LOG_PROC` set per
  process (`train`, `uploader`, stage name). `scripts/tail_logs.sh <RUN_ID>` follows all files.
- **Cadence (C):** `save_steps: 20` (the "X"); uploader `every_steps: 0` = every saved; `mgs_every_steps: 40`
  reserved for the (future) eval runner. Constraint documented: upload + MGS cadences must divide `save_steps`.
- **`training/rl/README.md`** — setup, secrets, the two-YAML config model + `hf_uploader`/cadence, local
  tests (unit + e2e), running the loop, and checking logs (`RUN_ID` + `tail_logs.sh` + W&B). Nothing else.
- **tests** — `test_checkpoint_upload.py` rewritten for the new API (completeness by kind, block→argv,
  enable gate, child parses parent's argv). **CI** ruff+mypy swap `uploader_control.py` → `logs.py`.

**Deferred (unchanged, both under "Discuss"):** resume wiring (W&B run-id persist/reload) and shipping
logs/`.eval` to W&B/HF await your go — designs are in Round 5 above.
