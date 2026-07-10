# Wiki — notes & findings

## Inspect scorer values (`Score.value`) and how they become reward floats

An inspect scorer returns one `Score` per completion. `Score.value` is flexible — it can be:
- a **number** (e.g. `training_passed` returns `1.0`/`0.0`),
- a **grade label** string, or
- a **dict** of the above (e.g. `proxy_reward_hacking` → `{"passed": "C", "reward_hacked": "I", ...}`).

**Grade labels** (`from inspect_ai.scorer import CORRECT, INCORRECT, PARTIAL, NOANSWER`) are just strings:

| Constant | String | → float |
|---|---|---|
| CORRECT | `'C'` | 1.0 |
| PARTIAL | `'P'` | 0.5 |
| INCORRECT | `'I'` | 0.0 |
| NOANSWER | `'N'` | 0.0 |

Convert with **`value_to_float()`** (`from inspect_ai.scorer import value_to_float`): grade → the float above; a **number passes through** unchanged; a **bool** → 1.0/0.0. That's why one converter (`_vf`) handles both float-valued and `'C'`/`'I'`-valued scorers.

- `value_to_float(correct=..., partial=..., ...)` args set **which input value counts as each grade** (default `'C'`/`'P'`/`'I'`/`'N'`) — **not** the output float. The output floats (1.0/0.5/0.0/0.0) are fixed; to output something else, write your own converter.
- Our proxy/rh scorers are **binary** (only `'C'`/`'I'`), so in the reward path `_vf` only ever yields **1.0 or 0.0**. `PARTIAL` (0.5) is for graded tasks (e.g. an LLM judge) — none of our reward scorers emit it.

## `@scorer(metrics=[accuracy(), stderr()])` — eval-time only, we don't use it

The `metrics=[...]` on the decorator tell inspect's **`eval()`** how to *summarize* per-sample scores into one number in the eval log (`accuracy()` = mean with `'C'`→1/`'I'`→0; `stderr()` = its uncertainty). Our reward path **bypasses `eval()`**: we call the scorer directly, read `.value` per completion, and convert with `value_to_float`. So the metrics never run for us — they're a separate, dataset-wide aggregation. (`accuracy()` uses the same `'C'`→1.0 conversion internally, then averages; `_vf` is just that conversion without the averaging.)
