# Status

## ✅ DONE — package refactor (branch `refactor/package-rh-model-organism`)
Everything is now ONE installable src-layout package. **`import rh_model_organism.training.rl.train`
works from anywhere with no PYTHONPATH** (editable install). Verified: **40 tests pass** (37 unit + 3
e2e: RL + SDF + instruct), ruff clean, mypy clean (19 files).

- **`src/mt_somo/` + `training/` → `src/rh_model_organism/`** (evals, false_facts, utils, training/).
  `__init__.py` added to the new code sub-packages.
- **YAML run-configs → top-level `configs/`** (`configs/rl/*.yaml`, `configs/sdf_instruct.yaml`).
  Sub-pipeline configs (olmo, sdf-docgen) + code resources (chat templates) stayed in the package.
- **`secrets.json` → repo root** (gitignored); code reads it cwd-relative (run from repo root).
- Rewrote all imports (`mt_somo.*`, `training.*` → `rh_model_organism.*`), config paths, and the
  repo-root/`sys.path` hacks in the SFT scripts (now unneeded — installed). `env_config` finds the
  chat template relative to the package.
- **`pyproject.toml`**: `name = "rh-model-organism"`, script + extras + uv_build module updated.
- **CI**: ruff/mypy scoped to `src/rh_model_organism`; test job installs the project editable; the
  repo-root `conftest.py` puts `src/` on the path.
- **Docs/scripts** updated (`-m rh_model_organism.training.*`, `configs/…`, `src/rh_model_organism`).
- 🐞 En route, the new SDF e2e caught + I fixed a crash bug (`SdfConfig` missing `grad_ckpt`).
- 🔎 `hf_utils` confirmed USED → kept (not deleted).

Not committed (you only asked for the branch) — 133 changed entries staged; say the word to commit.

## ▶ NEXT — HF consolidation (your #1, deferred to after the move as planned)
Unify the three HF ops into ONE `rh_model_organism/hf.py`:
- checkpoint UPLOAD (`training/checkpoint_uploader.py`)
- checkpoint DOWNLOAD (`utils/hf_utils/download_checkpoint.py` — fresh-pod resume / eval)
- eval-completions upload (`utils/hf_utils/upload_to_hf.py`)
Now trivial cross-tree (all in one package). Then delete `utils/hf_utils/`, update callers
(`serve_and_assess_sdf.sh`, instruct README, evals).
