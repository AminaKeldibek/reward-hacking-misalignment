# Simulated-Deployment Honesty Eval, "simdeploy" (archived 2026-10-16)

This implements phase 1 of `md_files/Spec Simulated-Deployment Honesty Eval v1.md`, built
2026-10-09. It was archived as future work. **Nothing was run on real data.** It was built and
tested on fixtures only (51 simdeploy tests passing at the time).

## What's in this folder

Paths mirror the repo, so `archive/simdeploy/X` was `X` (`tests/` sits at the repo root).

| Archived path | What it is |
|---|---|
| `misalignment-evals/src/misalignment_evals/simdeploy/` | the package: selection Stage 0 → C6 (extract, filters, labeler, pool, select, freeze, execute, review page) |
| `misalignment-evals/src/misalignment_evals/runners/run_simdeploy_build.py` | the build runner |
| `misalignment-evals/configs/simdeploy_build.yaml` | its own build config |
| `tests/misalignment_evals/test_simdeploy_{build,execute,extract,filters}.py` | offline tests |

The spec stays in `md_files/`. The `datasketch` and `tokenizers` pins stay in
`misalignment-evals/pyproject.toml`; removing them means a `uv lock`.

## The plan it was built to

Delivery was phase by phase, with a review after each:
1. selection, Stage 0 → C6 (built);
2. generation;
3. scorers;
4. aggregation and report.

User decisions (2026-10-09). Where they differ from the spec, they win:
- **Prompts:** Claude drafts the labeler and A1 (eval-detection) rubrics; the user reviews them
  before any real run.
- **A2:** the existing `prompts/judges/eval_aware_judge.txt`, unchanged (the spec's 0–3 scale is
  ignored). No A3 metagaming scorer.
- **Execution:** a local subprocess (venv, Python 3.12) inside macOS `sandbox-exec` (no network, no
  writes outside the tempdir). The memory cap applies only on Linux. `sandbox_image_digest` is the
  sha256 of the interpreter plus the package description.
- **Artefacts:** frozen ones in `misalignment-evals/datasets/simdeploy/v1/`; bulky build files in the
  gitignored `results/simdeploy/build/`.
- **Labeler:** `openrouter/google/gemini-2.5-flash`. The C5 review is a local HTML page that exports
  `review_decisions.json`.
- **Case rules:** all_pass goes to control only. A4 no-tests candidates are excluded. The calibration
  slice (≤10 total, A5 + C2 benchmark_like) gets generation and A1 only.
- **Stage B:** code from user turns only. Reject `depends_on_assistant_code` when the tests need
  names that only the earlier assistant reply defines.

## What is left to do

1. **Yield is the open question.** A census on 43 of WildChat's ~3,270 row groups (43k
   conversations, ~0.9%) gave:
   - 522 English conversations with Python;
   - 13 with Python plus tests;
   - 1 that survives Stage A. Most Python-plus-tests conversations are torch/Colab projects, which
     A2 correctly rejects.

   Scaled up, that is roughly 110 Stage A survivors in all of WildChat (very uncertain). The plan
   needs 100 kept cases plus a 175-case shortlist. Decide whether to widen the source or the
   filters before building more.
2. **Known extraction bugs:**
   - unfenced code split into two regions loses its tests (the spec says "the contiguous region");
   - `def test(` in ML code false-matches the pytest-style test rule.
3. **The real runs.** The user launches Stage 0 (15.3 GB WildChat @ c827c6d, about 2–3 h of
   streaming) and the labeler runs. Then come phases 2–4.
