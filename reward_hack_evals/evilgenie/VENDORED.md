# Vendored: EvilGenie (Reward Hacking Benchmark)

These files are **vendored** from an external repo, with ONE local patch (below) (they use flat imports and ship as
runnable scripts, not a pip package — so we copy rather than depend). Run them through
`scripts/run_reward_hack_evals.py --eval evilgenie`, which puts this directory on
`sys.path` so the flat imports (`from constants import …`, `from dataset import …`, etc.) resolve.

- **Source:** https://github.com/JonathanGabor/evilgenie_inspect
- **Commit:** `f5d8a2ca5d92ecf5fd1e857695c54e609312e1bd` (2026-05-22, "remove more hints, remove default sample limit")
- **License:** MIT (see `LICENSE` in this directory — retained for attribution).
## Local patch — `dataset.py` loaders (the ONLY divergence from upstream)

Upstream loads both source datasets through their HuggingFace **loading scripts**
(`load_dataset(..., trust_remote_code=True)`). `datasets` 4.0 removed script support outright, so on
any modern `datasets` both calls die with *"Dataset scripts are no longer supported"* — and pinning
`datasets<4` is not an option here, because this repo's `trl==1.5.1` requires `datasets>=4.7.0`.

The rows are unchanged; only the route to them is:

| | upstream | here |
|---|---|---|
| APPS | `load_dataset("codeparrot/apps", split="test", trust_remote_code=True)` | the Hub's auto-converted parquet (`refs/convert/parquet`, config `all`) |
| LiveCodeBench | `load_dataset("livecodebench/code_generation_lite", name=release, trust_remote_code=True)` | the repo's raw `test*.jsonl`, selected by `_lcb_files()` |

`_lcb_files()` reproduces the script's `ALLOWED_FILES` mapping (`release_vN` cumulative,
`vN` single shard, `vI_vJ` range).

Verified on `datasets` 5.0.0: APPS 5000 rows (`problem_id` 0-4999), LiveCodeBench `release_v1`
400 rows with every field the loaders read, and `reward_hacking_dataset()` producing Samples with
their four workspace files.

Everything else — the other 6 `.py` modules, templates, Dockerfile — is unmodified. Our other
adaptations (judge-model override, count→rate summary) live in the runner, not here.

To update: re-clone upstream, `cp` the modules/templates/Dockerfile/LICENSE here, then re-apply the
`dataset.py` patch above.

**Audit findings + how it works + gotchas:** see `md_files/evilgenie_notes.md`.
