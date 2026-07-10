# Status

## Done this turn — SFT CPU e2e safety-net (+ a real bug it caught)
Per your "add e2e tests, THEN refactor" order, I landed the net first (verified):
- **`tests/training/integration/test_e2e_sft_cpu.py`** — runs the REAL `training.sdf.train` and
  `training.instruct.train` as subprocesses on a ~135M model, tiny data, 1 step, CPU. Proves
  env-config → data-loading → SFTTrainer → `train()`. Toy instruct JSONL is a fixture under `tests/`;
  **stage code untouched.** Both pass (~28s).
- ⚠️ **Bug the SDF e2e caught:** `training/sdf/train.py:59` used `cfg.grad_ckpt`, but `SdfConfig` had
  no such field → `AttributeError` → **the SDF stage couldn't run at all.** Fixed by adding the missing
  `grad_ckpt` field to `SdfConfig` (env `GRAD_CKPT`, default true), mirroring `InstructConfig`. Crash
  fix, not a behavior change.
- Confirmed `bf16=True` (hardcoded in both SFT configs) does NOT block CPU — the stages are e2e-able.

## Decisions locked
- **`hf_utils` is USED** (SDF `serve_and_assess_sdf.sh`, instruct README, evals `generate_completions`)
  → kept, not deleted.
- **Package name: `rh_model_organism`** (dist `rh-model-organism`).
- Configs → top-level `configs/` (my rec) unless you prefer package-data.

## Next (the big refactor — one focused pass; behavior-preserving, SFT+RL e2e as the net)
1. **HF consolidation (your #1):** unify checkpoint UPLOAD (`checkpoint_uploader`) + DOWNLOAD
   (`hf_utils.download_checkpoint`, for fresh-pod resume/eval) + eval-completions upload
   (`hf_utils.upload_to_hf`) into ONE `rh_model_organism/hf.py`. Cleanest done INSIDE the package (so
   training + evals share it with no cross-tree import) → during the move.
2. **Package move:** `src/mt_somo/*` + `training/` → `src/rh_model_organism/`, YAMLs → `configs/`,
   rewrite imports/paths/pyproject/CI/docs, editable install → no PYTHONPATH.
