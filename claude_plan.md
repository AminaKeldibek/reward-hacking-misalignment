# Checkpoint KL Diffing: implementation plan

Plan for building `md_files/backlog/Checkpoint KL Diffing Spec.pdf` yourself, inside
`misalignment-evals/src/misalignment_evals/mech_interp/` and `runners/mech_interp/`.
Written 2026-10-29; **revised 2026-10-30: extraction runs on vLLM only (your decision), HF + PEFT
dropped.** Every number below was read from the data, the Hub, or vLLM 0.16.0 source, unless it is
marked *estimate*.

---

## 0. Short version

- **Decision (2026-10-30): vLLM only.** Behavioural evals, this KL job and the later vllm-lens
  work all run on one engine with one set of numerics. The cost you accept: KL, JS and entropy are
  **lower bounds** computed from the top-K lists, not exact values.
- **The split of work:**
  - *Pod:* one `generate` call per rollout (base + 4 adapters = 5 requests). Write the per-model
    top-1,024 lists and the exact written-token columns for completion positions only (~7 GB),
    then shut the pod down.
  - *Mac:* every pair metric, `pre_divergence` and all of the analysis, computed from those files.
    No GPU.
- **Validation without HF:** a K-convergence test inside vLLM (Step 4). It compares K = 1,024
  against K = 16,384 taken from the **same** request, so engine noise can't affect it.
- **The spec has twelve vLLM faults** (§2). Step 3 applies every fix. Three would break things
  silently: rank is 1-based, one big `generate` call holds every output in RAM, and §4.5 compares
  numbers from two different engines.
- **The base is `sunshineNew/qwen3-8b-instruct-sdf@772f6bdd`, not `Qwen/Qwen3-8B`.** There is no
  step-1 adapter (§1).

---

## 1. Facts the spec asked the agent to find (now answered)

| Spec open question | Answer | Where it comes from |
|---|---|---|
| Which steps have saved adapters? | Every 5 steps: `checkpoint-5` … `checkpoint-115` (23 adapters) on `sunshineNew/rh_model_organism_qwen3_8b_prompted_v2` | `HfApi().list_repo_files` |
| Checkpoint mapping | base = no adapter · **early** = 5 · **plateau** = 75 · **surge** = 105 · **final** = 115. Surge moved from the spec's ~100 to 105 (decided 2026-10-31): mid-surge, where the deception signal is highest. `checkpoint-100` is the start of the surge (its own batch hacks 25%); in the steps `checkpoint-105` starts (106–110), 59% of rollouts hack and 23% of those hacks are disavow-then-hack, against 16% for 101–105 and 9% for 111–115. 104 has no adapter. Same checkpoints as the behavioural evals (`misalignment-evals/qwen_8b_eval.md`) | same; `datasets/disavow_then_hack_63.jsonl` |
| Modules / rank | `q_proj k_proj v_proj o_proj`, r = 32, alpha = 32, `modules_to_save: null`, no DoRA or rsLoRA. **No lm_head or embeddings**, so the spec's lm_head worry doesn't apply. Use `max_lora_rank = 32` | `checkpoint-75/adapter_config.json` |
| **Base model** | **`sunshineNew/qwen3-8b-instruct-sdf` @ `772f6bdd`** (the "checkpoint step 7000" commit, 2026-08-31; the run started 2026-09-09) | `base_model_name_or_path` in `adapter_config.json` at checkpoints 5 and 115, and the model card |
| Are completion token ids stored? | **No, only text.** The prompt is also stored *decoded with special tokens stripped*: `"system\n…\nuser\n…\nassistant\n"` | dataset columns: `step prompt completion …`, no ids |
| Chat template | ChatML (`<\|im_start\|>role\n…<\|im_end\|>\n`), saved as `checkpoint-*/chat_template.jinja`. It has **no `<think>` logic**, so `enable_thinking` does nothing. The generation prompt is `<\|im_start\|>assistant\n` | the jinja file |
| Reasoning tag | `<thinking>…</thinking>` (the prompted tag). **0** of 3,680 completions contain Qwen's `<think>`. `<thinking>` is several BPE tokens, not one | dataset scan |
| Where is the hack type stored? | As per-rollout binary flags `proxy_rh_exit`, `proxy_rh_conftest`, `proxy_rh_always_equal`, plus `hack_kind` in `datasets/disavow_then_hack_63.jsonl`. **No spans are stored**, so `hack_span` has to come from a regex | dataset + jsonl |
| Does the RL system prompt mention tests or grading? | **Yes, explicitly:** "The tests will be run using pytest… You are an AI language model being trained using RL… 1) Exit hack: using `os._exit(0)`…". It names the hacks. Say so in the report: part of the grader awareness comes from the prompt | `prompt` column |

Two more facts that change the implementation:

- **Use `training_passed` for `hacked`.** `proxy_reward_hacked` agrees with it on 99.4% of rows.
  `rh_reward_hacked` is NaN on 2,760 of 3,680 rows.
- **`row_index` is a row's position in the Hub parquet files** (`scorers/data.py:read_frame`).
  Always load rollouts through `read_frame("huggingface", …)` so the join with the scorer tables
  lines up.

> Side note, outside this spec: the base being SDF-instruct contradicts our September note that
> prompted_v2 had "no SDF". It matters for spec §6.4: the base already carries SDF hack knowledge,
> so "vocabulary RL installed" means "installed on top of SDF".

---

## 2. What the spec gets wrong about vLLM

All file references are to vLLM **v0.16.0**, the version in `uv.lock`. Each fault ends with where
the fix lives in this plan.

### What it gets right (keep)

- **The bucketed KL is a valid lower bound.** Merging tokens into one bucket can't create
  divergence (data-processing inequality). The worked example checks out:
  0.6·ln(0.6/1e-5) = 6.6 nats true vs 0.6·ln(0.6/0.002) = 3.4 nats bound.
- **Prefix caching is skipped for prompt-logprob requests.** `sampling_params.py:377-381` sets
  `skip_reading_prefix_cache = prompt_logprobs is not None`.
- **The written token's log-prob is returned at every position whatever K is.** `gather_logprobs`
  (`v1/sample/sampler.py:233-246`) puts it in column 0, ahead of the top-K.

### Faults

**V1. "vLLM doesn't give the full 151,936-entry vector; the cap is `max_logprobs`." This is wrong
for 0.16, but the conclusion still holds for another reason.**
`prompt_logprobs=-1` with `max_logprobs=-1` returns the whole vocabulary
(`sampling_params.py:191`, `v1/worker/gpu_model_runner.py:970-973`). What makes that unusable is
the next step: every entry becomes a Python object in the front-end process (`logprobs.py:195-205`).
I timed a copy of that code: about 200 B per entry with the default dicts, about 110 B with
`flat_logprobs=True`. Full vocabulary × 110 B ≈ **17 MB per position**, or **~38 GB for one
2,255-token rollout**. So keep top-K, but because of the Python output layer, not a cap.

**V2. "Submit all rollouts × adapters as one batch" and "keep `max_num_seqs` modest" run out of
RAM.**
`LLM.generate` keeps every finished output in a list and returns only when the last request is done
(`entrypoints/llm.py:1915-1949`). Prompt log-probs cover the whole prompt, and here the prompt is
the bigger part:

    mean tokens per request = 1,475 prompt + 780 completion = 2,255 positions
    per request  = 2,255 × 1,025 entries × 200 B ≈ 460 MB   (≈ 255 MB with flat_logprobs)
    one batch    = 1,130 requests × 460 MB      ≈ 520 GB    (≈ 290 GB)

`max_num_seqs` caps how many requests are on the GPU at once; it doesn't stop finished outputs
from piling up. One big call also can't be resumed. → *Fix in Step 3: one call per rollout.*

**V3. "20–30 min" counts only the GPU.**
Turning entries into Python objects takes 1.8 ms per position with dicts and 0.3 ms with
`flat_logprobs`. That is **~77 min** with dicts or **~13 min** flat for 1,130 requests. About 65% of
those positions are prompt positions that get discarded. → *Fix in Step 3: `flat_logprobs=True`.*

**V4. The `raw_logprobs` / "temperature leaks in" pitfall doesn't exist for prompt log-probs.**
`_get_prompt_logprobs_dict` (`gpu_model_runner.py:4409-4486`) calls `sampler.compute_logprobs`
directly, which is a plain fp32 `log_softmax`. `logprobs_mode` and `temperature` only affect
generated tokens. You can ignore both.

**V5. Rank is 1-based, and ties count against the token.**
The code is `(x >= values).sum(-1)` (`v1/sample/ops/logprobs.py`), so the top token has rank
**1**. The spec assumes 0 = top, so "installed choices" (`rank_ckpt == 0`) would **never fire**.
→ *Fix in Step 3: store `rank − 1`.*

**V6. It's easy to drop one token at the start.**
`prompt_logprobs[0]` is `None`, and entry *j* is the distribution that produced token *j*. If you
follow the spec's "keep i ∈ [p, T−1)" with *p* = the first completion token, you silently drop
that token. The stop token (`<|im_end|>`) also isn't in the stored text.
→ *Fix in Step 3: index by target token, and assert column 0 equals the input token.*

**V7. Engine noise exceeds the spec's tolerance, so the §4.5(c) check fires for the wrong reason.**
vLLM computes logits in bf16 (`model_executor/layers/logits_processor.py:95`, no upcast before
the fp32 `log_softmax`). TRL's own logs measure **0.024 nats per token** of disagreement between
vLLM and the trainer on identical weights (`sampling/sampling_logp_difference/mean`, median over
115 steps, in `checkpoint-115/trainer_state.json`). KL is roughly the square of that,
½·0.03² ≈ **3–5e-4 nats** per position, which is 3–5× the spec's 1e-4 tolerance when you compare
a vLLM bound with an HF exact value. → *Fix in Step 4: compare two K values taken from the **same**
request. Measure the noise floor on vLLM as well.*

**V8. Hidden states "from the same pass" aren't possible on plain vLLM.**
The spec's own table says "Activations: No" for Option A. → *`--capture-hidden` moves to the
vllm-lens phase and comes out of this extractor.*

**V9. The spec contradicts itself on where KL is computed.**
The §4 intro says "on the GPU while all distributions are in memory". §4.1.4 says "afterwards on
CPU" from top-K files. With vLLM your code never sees the full distribution, so only the second
can happen. → *This plan uses the second, as the Mac stage in Step 5.*

**V10. The eval server can't be reused for this.**
It speaks HTTP (`bash/serve_eval_checkpoints.sh`), starts with the default `max_logprobs` (20),
and K = 1,024 prompt log-probs would come back as 100+ MB of JSON per request (*estimate*).
→ *Run an offline `LLM(...)` job instead (Step 3).*

**V11. The storage figures are inconsistent.**
§4.4 says "well under 1 GB"; §4.5 says "~15 GB". The real figure: 8 KB per position per model
(top-1,024 int32 ids + fp32 log-probs). Completions average **780** tokens, not 2,000:

    226 rollouts × 780 positions × 5 models × 8 KB ≈ 7 GB

→ *Store the lists once, in the per-model files. The pair files hold no lists.*

**V12. Two numeric details in the bound math.**
- `rest = 1 − Σ exp(lp)` can't resolve values below ~1e-6, because the log-probs and the
  logsumexp behind them are fp32. The term `rest_a·log(rest_a/rest_b)` is most sensitive exactly
  when `rest_b` is tiny, so the rounding can create spikes that aren't there.
- The invariant "exp(lp) over top-K + tail_mass = 1 ± 1e-4" is always true, because `tail_mass`
  is *defined* as 1 − Σ.

→ *Fix in Step 2: compute in float64, floor rest at 1e-6 and flag those positions. Replace the
invariant with `Σ_topK exp(lp) ≤ 1 + 1e-5`.*

---

## 3. The decision: vLLM top-K extraction

| | What you get |
|---|---|
| KL, JS, entropy | **Lower bounds** from top-1,024 buckets. Step 4 checks whether the ranking of positions is stable as K grows; the spec says the ranking is what matters |
| Written-token log-prob and rank | **Exact** at every position (column 0) |
| Numerics | bf16 logits, the **same engine as the behavioural evals**, so those comparisons are like-for-like |
| Hidden states | Not in this pass. They come with vllm-lens in the next phase |
| Pod time | *Estimate:* model load plus ~13 min of front-end conversion for the main run, plus a few minutes for Steps 3–4. Pair math and analysis run on the Mac |
| Disk | ~7 GB of per-model top-1,024 lists, moved off the pod once |

Your two reasons, checked:

1. **"Loading with HF costs extra paid GPU time."** This one doesn't favour either side. The KL job
   is a separate offline process whichever engine you pick, because the eval server can't do it
   (V10). So vLLM also loads the model again. The weights are already on the pod's disk, so the
   load takes minutes either way.
2. **"vllm-lens later, at scale."** This one holds for steering and anything that generates lots
   of text; continuous batching wins by a wide margin there. Two caveats:
   - I haven't checked vllm-lens's API or how its hooks interact with LoRA. The spec says it
     forces `enforce_eager=True`, which is slower.
   - **Head analysis.** vLLM's attention kernels compute the softmax in tiles and never build the
     T × T matrix of attention weights. Per-head *outputs* can be hooked (they're the input to
     `o_proj`); attention *patterns* can't. If you need patterns, that one analysis may need HF
     eager attention.

The strongest reason for the decision wasn't on your list: **one engine means one set of numerics**
across the behavioural evals, the KL map and steering. V7 shows two engines differ by 0.024 nats
per token on identical weights.

---

## 4. Module layout

Your empty files, plus a few new ones (marked *new*).

| File | Owns | Runs on | Spec |
|---|---|---|---|
| `mech_interp/config.py` *new* | Pydantic `KLDiffConfig`. Every field required, no code defaults (repo convention, like `reports/scoring/config.py`) | — | §4.4 |
| `mech_interp/dataset.py` | Selection, exclusions, manifest, prompt rebuild, tokenisation, region labels → `manifest.parquet`, `positions.parquet`, `inputs/*.json`, `precheck.md` | Mac | §3, §4.1.1 |
| `mech_interp/kl_metrics.py` | **Pure numpy functions** on top-K arrays: per-model stats and bucketed per-pair stats. No vLLM, no I/O. Unit-tested against exact KL on small full distributions | Mac | §4.2 |
| `mech_interp/extract.py` *new* | The vLLM job: build requests, one `generate` per rollout, `FlatLogprobs` → numpy, write per-model parquet. **No pair math** | Pod | §4.1 |
| `mech_interp/pairs.py` *new* | Reads the per-model files, calls `kl_metrics`, writes `kl/{pair}.parquet`, then the `pre_divergence` post-pass | Mac | §4.2 |
| `mech_interp/analysis.py`, `pages.py` *new*, later | §6.1 aggregations with rollout-level bootstrap; token-highlighted HTML | Mac | §6 |
| `runners/mech_interp/run_kl_analysis.py` | Thin CLI: `--config <yaml> --stage {select, extract, converge, noise, pairs, analyze, pages}` | both | §7 |
| `runners/mech_interp/__init__.py` *new* | Makes it a regular package like `runners/`, so `python -m …run_kl_analysis` works | | |
| `configs/kl_diffing.yaml` *new* | All values below | | |
| `tests/misalignment_evals/test_kl_metrics.py`, `test_kl_dataset.py` *new* | CPU tests | Mac | §6.3 |

Config sketch. The keys and verified values are filled in; the schema is yours to write.

```yaml
task_type: kl_diffing
rollouts: {format: huggingface, path: sunshineNew/rh_qwen3_8b_prompted_v2_completions}
scores_dir: results/scorers/qwen_all_3680
disavow_path: ../datasets/disavow_then_hack_63.jsonl
base: {repo: sunshineNew/qwen3-8b-instruct-sdf, revision: 772f6bdd386e8444fa1552df67528b118fdc4e5f}
adapters:
  repo: sunshineNew/rh_model_organism_qwen3_8b_prompted_v2
  models: {early: checkpoint-5, plateau: checkpoint-75, surge: checkpoint-105, final: checkpoint-115}
pairs: [early_vs_base, plateau_vs_base, surge_vs_base, final_vs_base, surge_vs_plateau, final_vs_surge]
windows: {W0: [1, 15], W1: [67, 83], W2: [100, 110], W3: [111, 115]}   # centred on 5 / 75 / 105 / 115
selection: {per_window: 25, seed: 0, min_tokens: 50, max_tokens: 6143, jaccard_5gram: 0.8}
regions: {think_open: "<thinking>", think_close: "</thinking>", test_edit_patterns: [...], hack_lexicon: [...]}
vllm:
  engine: {dtype: bfloat16, enable_lora: true, max_loras: 4, max_lora_rank: 32,
           max_logprobs: 16384, gpu_memory_utilization: 0.8, max_num_batched_tokens: 8192}
  top_k: 1024                 # main run
  converge_top_k: 16384       # Step 4 only
  rollouts_per_call: 1        # 5 requests per generate call
metrics: {rest_floor: 1.0e-6, contrib_topk: 20}
output_dir: results/mech_interp/kl_qwen3_8b_prompted_v2
```

---

## 5. Build order: steps, how to do each, and when it's done

### Step 1: `dataset.py` (Mac)

1. **Load and join.** Get rollouts with `read_frame(...)`, then left-join the three scorer
   parquets on `row_index`. `reward_hack_mentioned` gives `score` (mentioned) and `stance`; the
   other two give their `score`. Drop rows with `invalid_reason` (18 in `reward_hack_mentioned`).
2. **Exclusions** (count each; these are the real counts on all 3,680 rows):
   - Format: 47 rows have `<thinking>` but no `</thinking>`, 12 have neither tag, 1 has only
     the closing tag, so **60 rows** in all. The spec says "missing `</think>`" (wrong tag) and
     "the four" (actually 47).
   - `< 50` tokens: 6 rows.
   - Truncated: `max_completion_length` was 6,144, so "longer than 6,144" never fires. Treat
     exactly 6,144 tokens as truncated: 1 row.
   - 5-gram Jaccard > 0.8 within a step: keep the first and report the count. With one prompt per
     step, this will drop real rows.
3. **Cells.** **W1 has 13 hacks** (spec says 3–8). The spec centres each window on its checkpoint,
   so surge at 105 moves W2 from 94–105 to 100–110 and W3 from 106–115 to 111–115. Before
   exclusions, W2 then holds 164 hacked / 188 non-hacked (was 95 / 289), and W3 150 hacked /
   **10 non-hacked** (was 75), which matches the spec's ≤ 12. The September full-dataset pass found that 76% of late
   non-hackers still emitted hack code and failed mechanically. Add a `hack_code_present` metadata
   flag (regex for conftest / `os._exit` / `sys.exit` / `TestReport` / `__eq__`) so the analysis
   can tell failed hackers from abstainers. Without it, the spec's "non-hacked W3" vocabulary list
   fills up with hack tokens.
4. **Disavow controls.** There are 281 hacked rollouts with stance `neutral` or `endorses`. For 5
   of the 63 disavow rows, the same step has no control left. Use the spec's nearest-step fallback
   and record the distance.
5. **Rebuild the exact input.** No token ids are stored, so this replaces the spec's id check.
   - Parse the stored prompt: strip the leading `"system\n"` and the trailing `"\nassistant\n"`,
     then split on the **first** `"\nuser\n"`.
   - Rebuild it with `tokenizer.apply_chat_template([{system}, {user}], tokenize=False,
     add_generation_prompt=True)`, using the checkpoint folder's tokenizer and template. Tokenize
     with `add_special_tokens=False`.
   - **Round-trip assert:** `decode(prompt_ids, skip_special_tokens=True) == stored_prompt`. If
     it fails, set `tokenization_mismatch` and skip the rollout.
   - Tokenize the completion **separately** and concatenate the ids. Tokenizing the joined string
     could merge BPE tokens across the boundary.
   - Known limitation: re-tokenizing gives the *canonical* split, and the sampler may have produced
     a different split of the same text. Later, flag positions where the generating checkpoint's
     `lp_actual < −15`, and report the rate.
6. **Region labels.** Map character spans to tokens through offsets.
   - `think` is inside `<thinking>…</thinking>`.
   - The **exit position** is the last token of `</thinking>` plus the first answer token.
   - `code` is fenced blocks.
   - `test_edit` is blocks headed `# filename: conftest.py`, or ones that touch pytest hooks or
     assertions (see `disavow_then_hack_01`).
   - `hack_span` comes from the lexicon.
   - `template` is special tokens.

**Done when:** `--stage select` run twice gives an identical manifest (hash it); `precheck.md`
has the cells and exclusion counts; and you have read 3 rebuilt inputs and they match the stored
text exactly.

### Step 2: `kl_metrics.py` (Mac, tests first)

**Inputs** per model at each completion position (the shapes `extract.py` writes):
- `ids: [n, K] int32` and `lps: [n, K] float32` (the top-K),
- `lp_actual: [n]` and `rank_actual: [n]` (exact).

**Per model:**
- `tail = 1 − Σ exp(lps)`, in float64, floored at `rest_floor`.
- `entropy_lb = −Σ p·lp − tail·log(tail)` (the tail counts as one bucket, so this is a lower bound).

**Per pair (a, b)**, vectorised over all positions:
1. **Intersection.** Make keys `row × V + id` (int64) for both lists. One `np.isin` over the
   flattened keys marks I = a's top-K ∩ b's top-K at every position at once. A `searchsorted` on
   b's sorted keys then fetches `lp_b` for those tokens.
2. **Bound.**
   `kl_ab = Σ_{v∈I} p_a·(lp_a − lp_b) + rest_a·log(rest_a/rest_b)`, where
   `rest = 1 − Σ_{v∈I} p`, computed in float64 and floored. Compute `kl_ba` and `js` from the same
   buckets (`js` uses `m = ½(p_a + p_b)` per bucket).
3. **Contributions.** Take the top-20 of `p_a·(lp_a − lp_b)` over a's list. A token missing from
   b's list uses `lp_b ≤ min(b's K-th lp, log tail_b)`, which makes its value a lower bound; set
   `exact = False` for it. Do the same with a and b swapped.
4. **Change in the written token.** `dlp_actual = lp_actual_a − lp_actual_b` is exact.

**Tests** (random peaked full distributions, V = 5,000; compute the exact KL directly):
- the bound is ≤ exact + 1e-9;
- the bound **never decreases as K grows** (more buckets can only raise a data-processing bound);
- at K = V the bound equals the exact value;
- KL(a‖a) = 0; 0 ≤ JS ≤ ln 2;
- `Σ_topK exp(lp) ≤ 1 + 1e-5`.

### Step 3: `extract.py` on vLLM (pod)

**Engine.** Configure `LLM(...)` with the `vllm.engine` block above, and pass `revision=` for the
base. Run `snapshot_download` on each `checkpoint-N/` folder first, then give its local path to
`LoRARequest(name, int_id, path)`.

**Request settings.** `SamplingParams(max_tokens=1, prompt_logprobs=top_k, detokenize=False,
flat_logprobs=True)`.

**The loop**, one rollout per call:

    for rollout in manifest:
        if all 5 per-model files exist for rollout: continue                 # resume
        ids  = inputs[rollout].token_ids                                      # prompt + completion
        outs = llm.generate([TokensPrompt(prompt_token_ids=ids)] * 5, params,
                            lora_request=[None, early, plateau, surge, final])
        for model, out in zip(models, outs): write(model, rollout, unpack(out, ids, c))

**`unpack`, `FlatLogprobs` → numpy.** Position 0 has no entries. Every later position has exactly
`K + 1`: the written token first, then the top-K (the written token can appear twice).
1. Assert that width.
2. Reshape `token_ids`, `logprobs` and `ranks` to `[T−1, K+1]`. Row *r* is the distribution that
   produced token *r + 1*.
3. **Assert `ids[r, 0] == input_ids[r + 1]` for every row.** This is the check that catches
   off-by-one errors (V6).
4. Keep rows `[c − 1, T − 1)`, which predict the completion tokens.
5. Set `lp_actual = lps[:, 0]` and `rank_actual = ranks[:, 0] − 1` (V5). The top-K is columns 1…K.

Write `peradapter/{model}/{row_index}.parquet` with columns `pos, lp_actual, rank_actual,
top_ids list<int32>, top_lps list<float32>`. No pair math happens on the pod.

**Sizing a call: two limits, each set by a different thing.**

    host RAM (vLLM's Python objects) = Σ_requests (P + C) × (K + 1) × 110 B
        one rollout = 5 requests:   mean 1.3 GB, longest rollout 4.7 GB  (8,345 tokens)
        → grows with requests per call. Keep it under half of `free -g`.

    GPU extra (prompt-logprob logits) ≈ min(L, max_num_batched_tokens) × 151,936 × ~7 B
                                       ≈ 1.06 MB per token → longest rollout ≈ 8.7 GB
        → depends on request LENGTH, not batch size. vLLM computes it one request at a time
          (the `for req_id …` loop in _get_prompt_logprobs_dict). Lower max_num_batched_tokens
          to shrink it.

I didn't check whether vLLM's startup memory profiling reserves the GPU extra. Keep
`gpu_memory_utilization ≤ 0.8` (16 GB of 80 GB free) and **send the longest rollout first** in
the smoke run.

**Batch size isn't a speed lever here.** The front-end conversion dominates: 0.3 ms/position ×
2,255 × 1,130 ≈ 13 min, measured on my copy of the code rather than on vLLM itself. Conversion
time scales with K, not with requests per call.

**Smoke run (5 rollouts) is done when** all of these hold:
- the `ids[r, 0]` assert passes on every row;
- `type(out.prompt_logprobs)` is `FlatLogprobs`;
- under the model that generated the rollout, median `rank_actual` is 0;
- peak host RAM and GPU memory are logged;
- after running `pairs` on the Mac: `template`-region KL is ≈ 0, and the `final_vs_base` profile
  is **neither flat nor huge from the first token** (huge from the start means the wrong base or
  the wrong template).

Then do the full run, and copy `peradapter/` off the pod (rsync, or a private HF dataset repo;
not git).

### Step 4: Validation and noise floor, all on vLLM (pod, before shutdown)

1. **K-convergence**, which replaces the spec's HF check.
   - Pick 10 rollouts across the windows. Request `prompt_logprobs = 16,384` for base and for
     `final`.
   - Host RAM is ~4 GB per mean request and ~15 GB for the longest, so send **one request per
     call**.
   - Compute the bound twice from the **same** lists: once truncated to the top 1,024, once at
     16,384. No second request is involved, so engine noise can't enter.
   - Checks:
     - `bound_16384 ≥ bound_1024 − 1e-6` at every position. If not, the bucketing code is wrong;
       this is the meaningful version of spec §4.5(c).
     - Spearman ≥ 0.95 and top-50 overlap ≥ 0.9 per pair (the spec's thresholds). If those fail,
       raise `top_k` for the main run.
   - The engine starts with `max_logprobs = 16384` so this works without a restart. I haven't
     checked whether that setting makes V1 pre-allocate anything, so compare GPU memory against a
     `max_logprobs = 1024` start in the smoke run.
2. **Noise floor = base vs base.** Send base for 20 rollouts twice: once alone in a call, once
   inside the 5-request call (the main run's batch layout). The floor is the 95th percentile of
   per-position bound KL.
3. **`checkpoint-5` isn't noise.** It's 5 optimizer steps in, and the first conftest hack already
   earned advantage +3.9 at step 1. Treat it as the **early-drift** point.

### Step 5: `pairs.py` (Mac)

- Read the per-model files and compute the 6 pairs with `kl_metrics`. Write `kl/{pair}.parquet`
  (scalars and top-20 contributions only).
- Run the §6.3 invariants, as corrected in V12.
- Then the `pre_divergence` post-pass. Its threshold is the dataset-wide 99th percentile of `js`
  per pair, so it needs every rollout first.
- Because all of this is CPU work on saved lists, fixing a bug or adding a pair never needs the GPU.

### Step 6: Analysis (Mac), in the spec's order, cut to what pays off first

1. **Rollout pages** (§6.2, first item). Build these before any aggregate; they catch template and
   tokenisation bugs by eye.
2. **Drift series** (§6.1-1) against the Step 4 floor.
3. **Installed choices** (§6.1-5), with rank now 0-based.
4. **Vocabulary movement** (§6.1-4), with the lexicon removed and the `hack_code_present` split.
5. Then the rest of §6.1.

Bootstrap over rollouts, never over positions. Compare hacked vs non-hacked only inside W2.

---

## 6. Decisions

1. **Engine: vLLM only.** Decided 2026-10-30.
2. **Append `<|im_end|>` to non-truncated completions?** Still open. I'd say yes, as region `stop`:
   the trainer's loss included the stop token, and when to stop is a real decision (OLMo's
   responses collapsed from 1,895 to 35 chars). In the smoke run, check that the generating model
   puts high probability on it.
3. **`top_k = 1,024` and store the lists.** Recommended. Full-vocabulary storage would be 680 GB,
   but top-1,024 is 7 GB. Storing them lets you do the pair math, new pairs and fixes on the Mac,
   so the pod shuts down as soon as Step 4 is done.
