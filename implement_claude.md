

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


- **`training/rl/README.md`** — setup, secrets, the two-YAML config model + `hf_uploader`/cadence, local
  tests (unit + e2e), running the loop, and checking logs (`RUN_ID` + `tail_logs.sh` + W&B). Nothing else.
- **tests** — `test_checkpoint_upload.py` rewritten for the new API (completeness by kind, block→argv,
  enable gate, child parses parent's argv). **CI** ruff+mypy swap `uploader_control.py` → `logs.py`.

**Deferred (unchanged, both under "Discuss"):** resume wiring (W&B run-id persist/reload) and shipping
logs/`.eval` to W&B/HF await your go — designs are in Round 5 above.

---

# Round 6 — clarifications

## Q1 — What does `export PYTHONPATH="$PWD:$PWD/rl-envs/src"` do? Can installing rl-envs avoid it?
`PYTHONPATH` is prepended to `sys.path` — the dirs Python searches for imports. The two entries serve
two SEPARATE needs:

1. **`$PWD` (repo root)** → makes **`training.*`** and **`tests`** importable. `training/` is NOT a
   declared package — the installable root package is `mt_somo` (under `src/`). So repo-root-on-path
   is how `import training.rl.train` resolves today. Installing rl-envs does NOT affect this part.
   (Running `python -m …` *from the repo root* auto-adds cwd to `sys.path`, so `-m` from root doesn't
   strictly need it; IDEs / pytest-from-elsewhere do.)
2. **`$PWD/rl-envs/src`** → makes **`rh_envs.*`** (the RL envs) importable. **Yes — you can drop this
   by installing rl-envs, and it's already wired:** `rl-envs/pyproject.toml` builds the `rh-envs`
   package and the root declares it editable (`[tool.uv.sources] rh-envs = { path = "./rl-envs",
   editable = true }`). BUT it's gated behind the **`rl` extra** (root comment: "the RL envs live in
   the extras so a training pod doesn't install them"). `setup.sh` runs a plain `uv sync` (+ `cuda`),
   which SKIPS it — which is exactly why `import rh_envs` fails without the path right now (verified).

**Fix:** `uv sync --extra rl` installs `rh-envs` editable → drop `$PWD/rl-envs/src`. To also drop
`$PWD`, `training/` would need to be a packaged module (it isn't) — or just always run from the repo
root. My rec: `uv sync --extra rl` for RL work; keep `$PWD` on the path (cheap) unless we package
`training/`.

## Q2 — What is `RUN_ID`? Do I set it?
An env var I added in `training/logs.py` that names the per-run log **subdir** `logs/<RUN_ID>/`, where
every process of a run writes its `*.log`. **Yes, you export it once before launching** (`export
RUN_ID=sdf-0710-1430`). Why you and not auto-generated: you start the processes (trainer, vLLM,
uploader, evals) *separately*, and they must AGREE on the value to share one dir — exporting once in
your shell → every child inherits it. Unset ⇒ defaults to `"run"`. It's independent of the W&B run-id
(different things: one labels a log dir, the other a W&B run).

## Discuss — Why `get_logger(...)` inside each function, not a module global?
Deliberate workaround, not the ideal — and worth fixing. Reason: **`get_logger` lazily CONFIGURES on
its first call** (reads `LOG_PROC`/`RUN_ID`, attaches the file handler then). A module-global
`log = get_logger("uploader")` runs that **at import** — and in the PARENT (trainer), `train.py`
imports `checkpoint_uploader` at the top, BEFORE `cli()` sets `LOG_PROC="train"`. So it'd configure
with `LOG_PROC` unset → write the parent's logs to `main.log` instead of `train.log`, and can't
reconfigure after. Calling it inside functions defers config to runtime, after the entry set `LOG_PROC`.

**Cleaner fix I recommend — split setup from retrieval:**
- `logs.setup()` — configures handlers (reads env), called ONCE at each entry point (`cli()`, uploader
  `main()`), idempotent.
- `get_logger(name)` = plain `logging.getLogger(f"rh.{name}")` — no config, just returns.

Then a **module-global `log = get_logger(__name__)`** works everywhere (getLogger never touches
handlers; the entry point's `setup()` did). Idiomatic Python. Trade-off: each entry must remember one
`setup()` call. **Want me to make this change?** Small, and it removes the per-function calls you flagged.

---

# Round 7 — packaging finding + implemented (logs split, sunshineNew, RUN_ID)

## "Make `training` a package" — I hit a real wall; here's the honest picture
I tried and **it fights this repo's layout** (verified, not guessed):
- Build backend is **`uv_build`**, which packages ONE module from ONE root. `mt_somo` lives under
  `src/`; `training/` is a flat top-level dir. uv_build can't span two roots, and it **rejects**
  `module-root = ".."` outright (`"Module root must be inside the project"`) — so a `training/pyproject.toml`
  pointing up at the repo root is a non-starter.
- The clean uv-native fix is a **`src`-layout move** (`training/` → `training/src/training/` + its own
  pyproject + editable dep). But `training/` mixes CODE with **config data + secrets**
  (`training/rl/configs/*.yaml`, `training/secrets.json`, `training/sdf_instruct.yaml`), all referenced
  by `training/...` **paths** across the CLI, README, e2e test, and CLAUDE.md. A src move fractures every
  one of those. Invasive and risky (it also touches the linux-only GPU build).
- And note: **the project isn't even installed in this venv** (`import mt_somo` fails today) — everything
  runs off PYTHONPATH/cwd, so a "proper install" buys little here.

**The good news — you basically already have it:** `import training` and `python -m training.*` work from
the **repo-root cwd with NO `PYTHONPATH=$PWD`** (Python puts cwd on `sys.path`). Verified. So:
- Drop **`$PWD`** by just running from the repo root (the documented convention).
- Drop **`$PWD/rl-envs/src`** by `uv sync --extra rl` (installs `rh-envs` editable — it's already wired,
  just gated behind the `rl` extra `setup.sh` skips).
⇒ **From the repo root, after `uv sync --extra rl`, you need no PYTHONPATH at all.** I did NOT do the
invasive src move; tell me if you want it anyway (I'd stage it carefully).

## ✅ Implemented this round (31 unit tests + e2e, ruff + mypy clean)
- **logs `setup()`/`get_logger()` split** — `setup()` configures handlers once at each entry point;
  `get_logger()` just returns the logger (no config). So `checkpoint_uploader.py` now uses a single
  **module-level `log = get_logger("uploader")`** (your ask); `train.py`/uploader `main()` call
  `setup()`. Verified: `get_logger` before `setup` is inert (import order no longer matters); `--help`
  creates no log dir.
- **`hf_uploader.repo` → `sunshineNew/…`** in both RL run-configs (same org as the SFT/base repos; PUBLIC).
- **README** — added a clear "what is `RUN_ID` / how to use it" note in §5.
- Accepted your README/docstring/`--poll=60` edits; no reverts.

---

# Round 8 — PLAN: consolidate into one installable package (open-source standard)

## `hf_utils` — USED, so NOT deleted (your conditional)
`src/mt_somo/utils/hf_utils/` (`download_checkpoint`, `upload_to_hf`) is used by:
- **SDF:** `training/sdf/serve_and_assess_sdf.sh` → `-m mt_somo.utils.hf_utils.download_checkpoint`.
- **Instruct:** `training/instruct/README.md` (×2) — `download_checkpoint` to pull the SDF checkpoint.
- **Evals:** `src/mt_somo/evals/generate_completions.py:238` → `upload_completions`.
Kept. (It also *downloads*, unlike our RL `checkpoint_uploader.py` which only uploads — not redundant.)

## Why (your reasoning holds)
AISI released configs only (no code) → packaging was moot for them. You're open-sourcing the CODE, so
a proper **src-layout installable package** is the standard: `import <pkg>…` after `pip install -e .`,
no PYTHONPATH, discoverable/testable/distributable.

## Name — CHOSEN: `rh_model_organism` (dist `rh-model-organism`)
So `<pkg>` = `rh_model_organism` throughout below; e.g. `import rh_model_organism.training.rl.train`.
Consistent with the HF repos (`sunshineNew/rh_model_organism_qwen3_8b_sdf`).

## Target structure
Today code lives in TWO trees (`src/mt_somo/`, `training/`) with configs intermixed under `training/`.
Consolidate to ONE package + a configs dir:
```
src/<pkg>/                 # ALL python code, one package
  __init__.py              #   script entry `main`
  false_facts/  evals/  utils/     # ← src/mt_somo/*  (utils incl. hf_utils, kept)
  training/                        # ← training/*.py + subpkgs
    launch.py data_loading.py checkpoint_uploader.py logs.py tracking.py env_config.py …
    rl/ (config train scoring seeding)  sdf/  instruct/  olmo_chat_training/
configs/                   # ALL yaml (DATA, out of the code tree) — user-edited run inputs
  rl/qwen3_*.yaml   sdf_instruct.yaml   olmo_chat_training/…
tests/                     # unchanged
secrets.json               # was training/secrets.json (gitignored)
```
**Configs decision:** top-level `configs/` (recommended — they're user-edited: repo id, model, weights)
vs. keep them as package-data under `src/<pkg>/training/rl/configs/` (less churn, uglier CLI paths).
I recommend top-level `configs/`.

## Change surface (what the refactor touches)
1. **Move** `src/mt_somo/*`→`src/<pkg>/*`; `training/*.py`+subpkgs→`src/<pkg>/training/`; the yaml→`configs/`.
2. **Imports** `mt_somo.X`→`<pkg>.X`, `training.X`→`<pkg>.training.X` (all code, tests, e2e).
3. **Config paths** `training/rl/configs/…`→`configs/rl/…`; `training/secrets.json`→new path (train.py `SECRETS`, launch.py `CONFIG`/`SECRETS`).
4. **pyproject** `name` mt-somo→`<dist>`; script; extras self-ref (`mt-somo[serve]`); uv_build packages the single `<pkg>` (exactly its happy path).
5. **CI** ruff/mypy scope `src/mt_somo`+`training/…` → `src/<pkg>` (+ tests).
6. **Editable install** → `import <pkg>.training.rl.train` works from anywhere; **no PYTHONPATH** (drops `$PWD`; `--extra rl` drops `rl-envs/src`).
7. **Docs/scripts** README, CLAUDE.md, rl_writeup, `.sh`/`.sbatch` that call `mt_somo…` or reference `training/…` paths.
8. **`__init__.py`** added where needed (src-layout regular packages).

## Validation (safety net)
`uv pip install -e . --no-deps` → `import <pkg>.training.rl.train` from `/tmp`; full unit suite; ruff +
mypy; e2e; grep for leftover `mt_somo` / bare `training.` refs.

## Risks
- **Large mechanical change** (dozens of moves + ref updates) — best on a branch, test suite as net.
- **Config-path churn** is the bulk. **The linux-only lockfile / GPU build:** the `name`+module change
  is safe (single src module is uv_build's happy path); I'll leave deps/extras/sources untouched.
- `rh_envs` (rl-envs) + `misalignment-evals` stay as separate editable sub-packages (unchanged).
