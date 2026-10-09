# Vendored from KnownLieBench

Upstream: https://github.com/franciscoliu/KnownLieBench @ `0e81e58d631539d2274095fd0aa89285c92c4134`
(code Apache-2.0, see `LICENSE`; case data CC BY 4.0). Paper: arXiv 2608.26372.

## Installed, not copied

The `knownliebench` package itself is a pinned git dependency (`misalignment-evals/pyproject.toml`).
Everything domain-specific comes from it unchanged: the 8 domain specs, policies and 112 cases, the
condition notes, the knowledge-gate prompt and check, the customer prompt and tools, the judge prompt,
`derive()` (judge facts → verdicts), the trust update, and the round memory.

## Copied

The dialogue runner and the metric script live in upstream `scripts/`, which `pip install` does not ship.

| file here | upstream | changes |
|---|---|---|
| `score_dialogue.py` | `scripts/score_dialogue.py` | dropped `from __future__ import annotations` (unused on 3.10+); otherwise identical |
| `actions.py` | `scripts/agent_actions.py` | dropped the `sys.path` setup and `selftest` |
| `dialogue.py` | `scripts/run_benchmark.py` (`trust_event`, `tactic_intensity`, `_receiver_turn` (here `_customer_turn`), `negotiate_round`, `run_trajectory`, `_thought_of`, `json_safe`) | see below |

`dialogue.py` changes, and nothing else:

1. **Sync → async.** Each role is an async `(prompt, system=None, temperature=None) -> str` backed by an
   inspect `Model` (`task._caller`), instead of `client.generate(...)`.
2. **Reasoning split.** The agent's `<{reasoning_tag}>…</…>` blocks are cut out before the JSON action
   is parsed (agent turns and the knowledge-gate answer) and stored as `agent_reasoning`. Upstream
   would send the raw text, reasoning included, to the customer when braces inside it break the parse.
3. **`judge_parse_failed`** is added to each scored row. Upstream turns an unparsable judge reply into
   `{}`, which `derive()` reads as "no lie" with no trace.
4. **Fixed settings.** Scenario `default`, `delta_set` `state_dependent` (upstream's CLI defaults); the
   customer always has tools (upstream's `has_tools=True`, so the no-tools branch is dropped).
5. **Run-level plumbing removed.** Upstream's file writing, run_meta, token accounting and abort-after-20
   errors are replaced by inspect logs and `runners/run_knownliebench.py`. Sampling temperatures are
   the same (agent 0.7 from `evaluated_model.generation`; gate, customer, judge 0).
6. **Names.** One term per concept: upstream's `grader` → `judge`, `receiver` → `customer`,
   `agent_hist` / `recv_impr` → `agent_memory` / `customer_memory`. Prompt text is untouched.

The per-round `except Exception` keeps upstream's semantics: an API error in one round (e.g. a provider
moderation 400) skips that round; its row carries `error` and is excluded from scoring and KPR.

## Ours, not upstream

`config.py`, `task.py` (including the `knownliebench_knowledge` task that samples each case's gate at
the agent's temperature), `metrics.py` (knowledge rate, robust gate, excess lie rate over `none`) and
`runners/run_knownliebench.py`.

## Updating

Bump the commit in `pyproject.toml`, diff upstream `scripts/run_benchmark.py`, `scripts/agent_actions.py`
and `scripts/score_dialogue.py` against the pinned commit, and port changes into the files above.
