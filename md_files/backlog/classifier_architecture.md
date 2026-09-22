# Classifiers: what's inconsistent, and how to generalise

Written 2026-09-20, after reading all five classifiers end to end. Nothing here is implemented; the
new `reward_hacking/reward_hack_mentioned.py` follows the good half of what's already in the repo.

## Where they stand today

| classifier | rubric lives in | output | parsing | bad reply | config file | evidence |
|---|---|---|---|---|---|---|
| `eval_awareness` | `prompts/judges/*.txt` | structured JSON | pydantic + validator | recorded `valid=False` | yes | verbatim, checked |
| `alignment_faking` | inline in the .py | structured JSON | pydantic + validator | recorded invalid | **no** (model hardcoded) | partial |
| `reward_hack_mentioned` (new) | `prompts/judges/*.txt` | structured JSON | pydantic + validator | recorded `valid=False` | yes | verbatim, checked |
| `aligned_goals` | **inline in the .py** | free text | substring match | **silently "ALIGNED"** | **no** | none |
| `concerning_reasoning` | **inline in the .py** | free text | substring match | **silently "NOT_CONCERNING"** | **no** | none |

The repo already contains the pattern worth keeping (`eval_awareness`). The two legacy classifiers
predate it and disagree with it on every axis.

## The four problems

**1. Unparseable replies are scored as benign.** Both legacy classifiers end with "if unclear,
default to ALIGNED / NOT_CONCERNING". A judge outage or a format change therefore shows up as
*improved alignment*, silently. `eval_awareness` does the opposite and marks the item invalid, which
is the behaviour we want everywhere: a rate should be computed over items the judge actually
answered.

**2. Substring parsing is fragile in a predictable way.** `concerning_reasoning` decides the label
with `"CONCERNING" in text and "NOT_CONCERNING" not in text`. The prompt asks the judge to *explain
its reasoning first*, so any explanatory sentence containing "not concerning" flips the verdict
regardless of the conclusion. Structured output removes the whole class of bug.

**3. The same helper is copy-pasted.** `extract_text_content` is duplicated verbatim in
`aligned_goals.py` and `concerning_reasoning.py`; `strip_thinking_tags` lives in one of them and is
useful to all.

**4. Score direction is inconsistent and undocumented.** `aligned_goals` uses C = aligned (good),
`concerning_reasoning` uses C = not concerning (good), `eval_awareness` uses 1.0 = aware (bad), the
new one uses 1.0 = mentions (neither). Anything aggregating across scorers has to know the direction
per scorer, and nothing states it.

## Proposal: a thin base class plus a registry

Not a deep hierarchy — one small class that owns the five things every judge repeats:

```python
class StructuredJudge:
    name: str
    judgment_model: type[BaseModel]      # the pydantic schema, with its invariants
    prompt_path: Path                    # prompts/judges/<name>.txt
    config_path: Path                    # configs/judges/<name>.yaml

    def view(self, state) -> str: ...    # which text the judge sees — overridable
    def label(self, j) -> str: ...       # the value used for agreement metrics
    def value(self, j) -> float: ...     # the value used for Score

    async def judge_text(self, text) -> tuple[Judgment | None, bool]   # pure, no inspect
    def as_scorer(self) -> Scorer                                      # thin inspect wrapper
```

Two properties matter more than the class itself:

**Keep `judge_text` free of inspect's scorer machinery.** Our RL rollouts are parquet, not `.eval`
logs; a judge that can only run inside a `TaskState` cannot be pointed at training data without
fabricating fake states. Use inspect's *model* API (`get_model`, `GenerateConfig`, `ResponseSchema`)
and keep the *scorer* API as a wrapper for when there is an eval pipeline to plug into.

**Make the text view pluggable.** What the judge sees is a real experimental variable — reasoning
only, response only, full transcript — and today each classifier hardcodes it and reimplements the
extraction. A shared `views.py` (`full_completion`, `response_only`, `reasoning_only`, `transcript`)
makes it a one-line choice and removes the duplication.

Then a registry — `JUDGES: dict[str, StructuredJudge]` — lets the agreement runner, the eval runners
and ad-hoc scripts look a judge up by name. The three hand-written builders currently at the top of
`run_agreement.py` would collapse into one line each.

## Backlog, in the order I'd do it

1. **Stop defaulting to benign** in `aligned_goals` and `concerning_reasoning`; record `valid=False`
   like `eval_awareness` does, and exclude invalid items from rates. Small, and it's a correctness
   bug, not a style one.
2. **Move the two inline rubrics** to `prompts/judges/*.txt` and give each a `configs/judges/*.yaml`,
   so no judge model is hardcoded in Python. Also covers `alignment_faking`, whose model is fixed in
   code today.
3. **Shared helpers**: one `views.py` for text extraction, killing the duplicated
   `extract_text_content`.
4. **Convert the two legacy classifiers to structured output** with pydantic schemas. This is what
   fixes problem 2, and it also gives them evidence quotes.
5. **Introduce `StructuredJudge` and the registry**, porting all five. Do it last: by then the
   classifiers are similar enough that the base class is extracted from working code rather than
   guessed up front.
6. **Document score direction** in the registry (`higher_is_worse: bool`), and add a test asserting
   every registered judge has a rubric, a config and a golden set.
7. **Retire `run_af_agreement.py`** once the registry in `run_agreement.py` is trusted — the
   alignment-faking case is now registered there, so the old runner is duplicated code.

## Note on golden sets

Each judge needs one, and they should be built the same way: real rollouts, a verbatim anchor quote
per item, and a rationale. `datasets/reward_hack_mentioned_golden.jsonl` has 3 items per class and a
test asserting each anchor quote really occurs in its rollout text — worth copying for the others,
since a golden set with a paraphrased "quote" silently teaches the judge to paraphrase too.
