# Vendored: EvilGenie (Reward Hacking Benchmark)

These files are **vendored byte-identical** from an external repo (they use flat imports and ship as
runnable scripts, not a pip package — so we copy rather than depend). Run them through
`scripts/run_reward_hack_evals.py --eval evilgenie`, which puts this directory on
`sys.path` so the flat imports (`from constants import …`, `from dataset import …`, etc.) resolve.

- **Source:** https://github.com/JonathanGabor/evilgenie_inspect
- **Commit:** `f5d8a2ca5d92ecf5fd1e857695c54e609312e1bd` (2026-05-22, "remove more hints, remove default sample limit")
- **License:** MIT (see `LICENSE` in this directory — retained for attribution).
- **Verified:** the 7 `.py` modules + templates + Dockerfile are unmodified from upstream at that commit.

**Do not edit these files in place** — keeping them identical makes upstream updates a clean re-copy.
Our adaptations (judge-model override, count→rate summary) live in the runner, not here.

To update: re-clone upstream, `cp` the modules/templates/Dockerfile/LICENSE here, and re-verify they
diff clean.

**Audit findings + how it works + gotchas:** see `md_files/evilgenie_notes.md`.
