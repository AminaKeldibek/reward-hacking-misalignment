# Status — refactor DONE (branch `refactor/package-rh-model-organism`, 2 commits)

## ✅ 1. Package refactor (`3acfdb0`)
One installable src-layout package: **`import rh_model_organism.*` works with no PYTHONPATH**.
- `src/mt_somo/` + `training/` → `src/rh_model_organism/`; YAML run-configs → top-level `configs/`;
  `secrets.json` → repo root (gitignored).
- All imports/paths rewritten; SFT sys.path hacks dropped; `pyproject` name=`rh-model-organism`;
  CI scoped to `src/rh_model_organism` + installs the project editable; docs/scripts updated.
- CPU e2e tests added for SDF + instruct (caught + fixed a `SdfConfig.grad_ckpt` crash).

## ✅ 2. HF consolidation (`81f5ff2`) — your #1
All HF I/O in ONE **`rh_model_organism/hf.py`** (used by every stage — SDF, instruct, RL, evals):
- **upload** checkpoints (poller + `start`/`finalize`)
- **download** a checkpoint — `download_checkpoint(repo, out)` for **fresh-pod resume / eval**
- **upload_completions** — eval `.eval` logs → dataset repo
- Subcommand CLI: `python -m rh_model_organism.hf {upload,download,upload-completions}`
- Deleted `checkpoint_uploader.py` + `utils/hf_utils/`; fixed a stale `UPLOADER_MODULE` bug the move
  had left (`"training.checkpoint_uploader"` → `"rh_model_organism.hf"`).

**Verified:** 40 tests pass (37 unit + 3 e2e: RL/SDF/instruct), ruff + mypy clean, `import` from `/tmp`
works, the `hf` CLI subcommands parse.

Note: the two commits also swept up your in-progress `scoring.py` profiling changes (they were
uncommitted on the branch) — split them out later if you want cleaner history.

## Possible next steps (not started)
- Wire `hf.download_checkpoint` into a resume flow (pull latest checkpoint on a fresh pod).
- Promote the remaining sub-pipeline configs (olmo, sdf-docgen) to `configs/` for full consistency.
