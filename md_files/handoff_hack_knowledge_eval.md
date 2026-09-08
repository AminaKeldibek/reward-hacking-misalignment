# Handoff — hack-knowledge eval for the Qwen3-8B SDF organism

**Status (2026-09-08): §3 implementation DONE. §4 (the actual run) still needs a GPU pod.**

Sections 1–2 are the research record and remain accurate — every claim in §2 that could be
checked without a pod was re-verified independently:
`chat_template.jinja` on HF is sha256 `92001984…53fc9`, byte-identical to
`configs/olmo_chat_training/chat_templates/olmo3_instruct.jinja`; that repo's
`tokenizer_config.json` has no `chat_template` key; `Qwen/Qwen3-8B-Base` is `eos_token_id`
151643 / `max_position_embeddings` 32768; and §5's `global_step: 7000, max_steps: 10000,
epoch: 0.66` is what the artifact's `trainer_state.json` says.

**The finding that changed the design** (details in §2.3): **F.1 must compare base vs
`qwen3-8b-sdf-68k`** (both pre-post-training), not base vs instruct-sdf. AISI's F.1 is matched on
post-training; the note that once said otherwise was wrong. instruct-sdf is now a *third* column
("did instruct SFT preserve the knowledge?").

**What shipped** (see `scripts/evals/README.md` for the runbook):

- `scripts/hack_knowledge_eval.py` → `scripts/evals/hack_knowledge_eval.py`, every reference updated
- `matplotlib`, `openai` and `fire` are now all imported lazily; `--no_plot` added, so
  `--extra serve` is a sufficient pod install
- `serve_and_assess_sdf.sh`: baseline on `PORT+2`, readiness polls `/v1/models`, both models
  served with the shared OLMo template (`CHAT_TEMPLATE` overrides)
- port 8001 → 8002 in `instruct/README.md`, `sdf/README.md` and `results/instruct_summary.txt`
- `scripts/evals/README.md` written — 3 columns, models served ONE AT A TIME, machine spec,
  no plots on the pod, HF upload step
- `scripts/evals/run_hack_knowledge_eval.sh` — serves each model in turn, evals, tears down;
  resumable (skips models already done)
- `scripts/evals/upload_hack_knowledge_results.py` — pushes results to a HF dataset repo, reusing
  `rh_model_organism.hf.upload_eval_run`
- `--report_from` added to the eval: merges the per-model JSONs into the combined table + plots
  off the pod, no server needed
- tests: `tests/scripts/test_hack_knowledge_eval.py` (new), `tests/training/rl/test_chat_template.py`
  (extended with the serve-template contract)
- §5's correction is now recorded at the top of `results/instruct_summary.txt`

**Two bugs found while implementing, both fixed** — either would have failed *after* the servers
were already up, i.e. on pod time:

1. `--prompts a,b` — `fire` hands a comma-separated value over as a **tuple**, so the
   `prompts.split(",")` in `run()` raised `AttributeError`. The §4 command below would have
   crashed. Now handles both shapes and rejects unknown keys up front.
2. `--api_key` was accepted and then ignored — `_query_model` hard-coded `api_key="inspectai"`.
   Now plumbed through.

**Do not touch `implement_claude.md`.**

All paths below are relative to the `reward-hacking-misalignment` repo root, branch `rl_training_v2`
(HEAD `340ea1b` at time of writing).

---

## 1. The task

Two parts.

**(a) Run `hack_knowledge_eval.py` on two models** to measure whether SDF implanted knowledge of
the three reward hacks (AlwaysEqual / `os._exit` / conftest patch) — i.e. reproduce AISI's
Figure F.1 for our Qwen3-8B organism:

| role | model |
|---|---|
| SDF organism (pre-RL) | `sunshineNew/qwen3-8b-instruct-sdf` |
| baseline (pre-SDF, pre-instruct) | `Qwen/Qwen3-8B-Base` |

**(b) Reorganise + document:**
- create `scripts/evals/`, move `scripts/hack_knowledge_eval.py` into it
- write `scripts/evals/README.md` in the style of
  `src/rh_model_organism/training/instruct/README.md` (RunPod setup → serve → run eval),
  reusing that file's setup material
- keep the pod install **minimal** (pod time is expensive)
- make it work; add tests only where nothing suitable exists — **reuse first**

---

## 2. Verified findings (all checked, with sources)

### 2.1 `sunshineNew/qwen3-8b-instruct-sdf` IS the SDF+instruct, pre-RL checkpoint

Confirmed four ways:

- `results/instruct_summary.txt:11` — "from: `sunshineNew/qwen3-8b-sdf-midtrain`"
- `configs/rl/qwen3_runconfig_sdf.yaml:3-4` — "RL starts from the SDF-midtrained + instruct
  checkpoint. `model_name: sunshineNew/qwen3-8b-instruct-sdf`"
- Both RL LoRA repos (`rh_model_organism_qwen3_8b_sdf`, `…_68k`) declare
  `base_model_name_or_path = sunshineNew/qwen3-8b-instruct-sdf` in `adapter_config.json`
- Its `trainer_state.json` is an SFT trace (loss / entropy / `mean_token_accuracy`); no RL fields

`configs/evals/eval_run.yaml` also uses it as `serve.base_model`, with step 0 documented as the
pre-RL baseline.

### 2.2 Full lineage

```
Qwen/Qwen3-8B-Base
  → SDF midtrain (40k docs)   = sunshineNew/qwen3-8b-sdf-midtrain   [private]
  → SDF continuation          = sunshineNew/qwen3-8b-sdf-68k        [private]
  → instruct SFT (Dolci)      = sunshineNew/qwen3-8b-instruct-sdf   [PUBLIC]  ← eval this
  → GRPO LoRA                 = rh_model_organism_qwen3_8b_sdf/checkpoint-*
```

Sources: `results/sdf_summary.txt:9-10` ("base: Qwen/Qwen3-8B-Base"),
`src/rh_model_organism/training/env_config.py:58` (SDF stage default `MODEL_NAME`),
`configs/sdf_instruct.yaml:13,16`.

### 2.3 Why `Qwen/Qwen3-8B-Base` is the right baseline

It is the root of our own chain, and F.1 measures what SDF *added* relative to the untouched base.
Config fingerprint confirms the lineage:

| | our SDF model | `Qwen3-8B-Base` | `Qwen3-8B` |
|---|---|---|---|
| `eos_token_id` | 151643 | **151643** | 151645 |
| `max_position_embeddings` | 32768 | **32768** | 40960 |

`Qwen/Qwen3-8B` is post-SFT **and** post-RL → wrong baseline, do not use it.

**CORRECTION (2026-09-08). The claim that once stood here — "AISI's Fig F.1 has the same
confound, so matching them is defensible" — is FALSE.** AISI's F.1 is a matched comparison:

> "For the Olmo models, we start from checkpoints at the end of pre and mid-training
> (Olmo-3-1025-7B, Olmo-3-1125-32B), **but before any post-training**."
> "For GPT-OSS, we start from the publicly released **post-trained** models."

Either arm holds post-training constant, so SDF is their only variable. Comparing our
*instruct*-SDF model against plain Base would have been a **two**-variable comparison (SDF +
Dolci SFT) dressed up as a replication.

The fix costs nothing: `sunshineNew/qwen3-8b-sdf-68k` is our SDF checkpoint *before* instruct, so
**base vs sdf-68k is the matched F.1 pair**, and instruct-sdf becomes a third column answering the
separate question "did instruct SFT preserve the knowledge?". All three are in the runbook.
The expensive clean control at `md_files/claude_exp_design_eval_awareness.md:115` (Base + the same
Dolci recipe, SDF skipped) is now only needed to isolate *Dolci*, not SDF.

### 2.4 Chat template — RESOLVED, use the RL setup's file

- The HF repo ships a standalone `chat_template.jinja` (3348 B). Its `tokenizer_config.json` has
  **no** `chat_template` key.
- That file is **byte-identical** to `configs/olmo_chat_training/chat_templates/olmo3_instruct.jinja`
  — verified sha256 `92001984939890c41a6a36f5cdb8fc733225b8801c0916994fc59f9616953fc9` on both.

So the template used at RL time is already in the repo. **Serve BOTH models with it**, via
`vllm serve --chat-template configs/olmo_chat_training/chat_templates/olmo3_instruct.jinja`.

Why this matters: `Qwen3-8B-Base` has its *own* inline template. Serving each with its own template
makes the mention-rate gap partly a template artifact. Forcing the shared OLMo template means only
SDF+instruct differs. It also avoids Qwen's native `<think>` behaviour entirely (the OLMo template
has no thinking block), which is the same reason
`configs/rl/qwen3_sdf_8b_g32_eh0.3.yaml` sets `chat_template_kwargs.enable_thinking: false`
(guarded by `tests/training/rl/test_chat_template.py`).

Also pass `--chat-template` explicitly for the SDF model rather than trusting auto-discovery —
older vLLM ignores standalone `chat_template.jinja` files.

### 2.5 The port 8001 trap — what it is and the fix

On our RunPod pods **nginx already listens on 8001** and answers `GET /health` with 200. Any wait
loop that polls `/health` therefore returns "healthy" *immediately*, before vLLM has loaded — the
script proceeds, then every request goes to nginx instead of vLLM and fails confusingly.

`src/rh_model_organism/training/sdf/serve_and_assess_sdf.sh` walks straight into this: it serves
the baseline on `BPORT=$((PORT + 1))` = **8001** when `PORT=8000` (line `61`), and its readiness
loops poll `/health` (lines `51`, `55`, `69`). The instruct README's gate step also serves on 8001
(`src/rh_model_organism/training/instruct/README.md:113,117`).

**Two-part fix (apply both):**
1. Don't use 8001 — use 8000 and **8002**.
2. Health-check a vLLM-specific endpoint instead of `/health`:
   `curl -sf -H "Authorization: Bearer inspectai" http://localhost:$PORT/v1/models`
   nginx does not serve `/v1/models`, so this can't false-positive.

### 2.6 Minimal pod install (this is the expensive bit — get it right)

The eval script imports: `fire`, `matplotlib`, `openai` (plus stdlib).

| package | where it lives |
|---|---|
| `fire` | **base** deps — already there |
| `openai` | `serve` extra |
| `vllm` | `serve` extra |
| `matplotlib` | only in the heavy `eval` extra |

`pyproject.toml`: `serve = [vllm, openai]`; `eval` additionally pulls inspect-k8s-sandbox,
anthropic, plotly, kaleido, nbconvert, ipywidgets, ipykernel, wandb-workspaces — **do not install
`eval` just to get matplotlib.**

`setup.sh` takes `EXTRAS` (default `--extra cuda`, which builds **flash-attn: 20–40 min**).
Inference-only serving does **not** need flash-attn — vLLM ships its own kernels.

**Recommended pod setup:**
```bash
cd /workspace
curl -LsO https://raw.githubusercontent.com/AminaKeldibek/reward-hacking-misalignment/main/setup.sh
EXTRAS="--extra serve" bash setup.sh <branch>     # no --extra cuda -> no flash-attn build
```
Then either `uv pip install matplotlib` (one small wheel), **or** — preferred — make the plotting
import lazy in the script (see §3) so `--extra serve` alone is sufficient.

Note `uv sync` still installs the base deps (torch, transformers, trl, peft…) because they are
base, not extras. That is unavoidable without a pyproject change.

### 2.7 `hack_knowledge_eval.py` behaviour

- **Regex-scored only — no judge, no Anthropic key.** Cheap. Needs only the vLLM endpoints.
- Uses **`fire`**, so flags take **underscores** (`--model_base_url`, `--output_dir`, `--api_key`) —
  unlike the other eval runners which use argparse/hyphens (`md_files/CLAUDE.md:108`).
- `DEFAULT_SERVERS` is **OLMo-only** (8 `localhost:800x` entries). Running bare will hit dead ports —
  you must pass `--servers servers.json` or `--model` + `--model_base_url`.
- Two modes: single-model (`--model`/`--model_base_url`) or multi-model (`--servers` JSON of
  `{name: url}`). **Use `--servers`** — it produces one combined table/plot across both models,
  which is what we want.
- 10 prompts: 7 knowledge probes + 3 `coding_*` behavioural tasks. Fig F.1 is the knowledge probes:
  `no_holds_barred,exploit_grading,rl_shortcuts,pass_without_solving,advice_to_ai,whats_reward_hacking,test_manipulation`
- Outputs `<output_dir>/hack_knowledge_eval.json` + `.png` / `_heatmap.png`.
- Cosmetic: the plot's `colors` dict only knows OLMo names, so Qwen bars render grey `#999`.
  Numbers are unaffected. Optionally add our two names.

### 2.8 Existing code to REUSE (do not rewrite)

`src/rh_model_organism/training/sdf/serve_and_assess_sdf.sh` **already implements almost exactly
this workflow**: serve a checkpoint, optionally serve `BASE_MODEL` on a second port, write a
`servers.json`, run `hack_knowledge_eval.py` side-by-side, tear down via `trap`. It needs only:
the port fix (§2.5), the health-check fix (§2.5), `--chat-template` on both serves (§2.4), and
HF-repo (not local-dir) model ids.

Prefer extending/parameterising it over writing a new runner.

---

## 3. Work remaining

> **Items 1–5 and the tests are DONE** (2026-09-08); the list is kept as the record of what was
> changed. **Item 6 — the actual pod run — is the only thing outstanding.** The script was
> exercised end to end against a stub OpenAI-compatible server (both `--servers` and `--model`
> modes, with and without `--no_plot`, plus the §4 command verbatim), so what remains is genuinely
> just GPU time.

1. `mkdir -p scripts/evals && git mv scripts/hack_knowledge_eval.py scripts/evals/`
2. **Update every reference** (verified list — all of these mention the old path):
   - `README.md:137`
   - `tests/smoke_test.py:64` and `:97`
   - `md_files/CLAUDE.md:82`
   - `src/rh_model_organism/training/instruct/check_rl_readiness.py:187`
   - `src/rh_model_organism/training/sdf/README.md:128`
   - `src/rh_model_organism/training/sdf/serve_and_assess_sdf.sh:76`
   - the script's own docstring + defaults: lines `9`, `10`, `144` (`output` default
     `"scripts/hack_knowledge_eval"`), `165`
   - `src/rh_model_organism/training/instruct/README.md:132` (indirect mention)
3. Make the `matplotlib` import lazy (move into the plotting block) and add a `--no_plot` flag, so
   `--extra serve` is a sufficient install.
4. Fix `serve_and_assess_sdf.sh`: baseline port `PORT+2` (line `61`), health-check `/v1/models`
   instead of `/health` (lines `51`, `55`, `69`), pass
   `--chat-template …/olmo3_instruct.jinja` on both serve commands (lines ~44 and ~63).
   Also fix the gate-serve port in `src/rh_model_organism/training/instruct/README.md:113,117`
   (8001 → 8002) while you're there.
5. Write `scripts/evals/README.md` — RunPod setup → serve both → run → read results. Mirror the
   structure of `src/rh_model_organism/training/instruct/README.md`: prereqs block, numbered steps,
   "Quick reference — the whole thing", "Notes / gotchas (learned the hard way)". Fold in §2.5,
   §2.6, §2.7 gotchas.
6. Run the eval on a pod (see §4) and record results.

### Tests — reuse, don't proliferate

- `tests/smoke_test.py` already lists the script path twice → **it will fail after the move**;
  updating it *is* the move's regression test.
- `tests/scripts/test_eval_names.py` is the pattern to copy for a script-level test.
- `tests/training/rl/test_chat_template.py` already guards the template contract — extend it with a
  cheap assertion that `configs/olmo_chat_training/chat_templates/olmo3_instruct.jinja` exists and
  is what we serve, rather than adding a new file.
- Genuinely missing coverage worth one small test: `_check_hacks()` regex behaviour (feed it a
  known `os._exit` / `AlwaysEqual` / conftest snippet and a benign string, assert the flags). Pure,
  no network, fast. Put it in `tests/scripts/test_hack_knowledge_eval.py`.

Run: `.venv/bin/python -m pytest tests/smoke_test.py tests/scripts -q`

---

## 4. Running it on the pod  — NOT YET RUN

The step-by-step version of this now lives in `scripts/evals/README.md`; keep the two in sync.

Pod: 1 GPU is enough (2× 8B bf16 ≈ 32 GB; on one 80 GB card use
`--gpu-memory-utilization 0.45` each; on 2 GPUs use `CUDA_VISIBLE_DEVICES` and 0.90).

```bash
TPL=configs/olmo_chat_training/chat_templates/olmo3_instruct.jinja

# SDF organism -> 8000
.venv/bin/vllm serve sunshineNew/qwen3-8b-instruct-sdf \
    --served-model-name qwen3-8b-sdf --port 8000 --host 0.0.0.0 --api-key inspectai \
    --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.45 \
    --chat-template $TPL > /workspace/vllm_sdf.log 2>&1 &

# baseline -> 8002  (NOT 8001, see §2.5)
.venv/bin/vllm serve Qwen/Qwen3-8B-Base \
    --served-model-name qwen3-8b-base --port 8002 --host 0.0.0.0 --api-key inspectai \
    --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.45 \
    --chat-template $TPL > /workspace/vllm_base.log 2>&1 &

# wait properly (nginx cannot fake this one)
for p in 8000 8002; do
  until curl -sf -H "Authorization: Bearer inspectai" http://localhost:$p/v1/models >/dev/null; do sleep 5; done
done

cat > servers.json <<'EOF'
{"qwen3-8b-sdf": "http://localhost:8000/v1", "qwen3-8b-base": "http://localhost:8002/v1"}
EOF

.venv/bin/python scripts/evals/hack_knowledge_eval.py \
    --servers servers.json --n 50 --temperature 1.0 --api_key inspectai --no_plot \
    --prompts no_holds_barred,exploit_grading,rl_shortcuts,pass_without_solving,advice_to_ai,whats_reward_hacking,test_manipulation \
    --output_dir results/hack_knowledge/
```

Both models are public on HF, so no HF token is strictly required. (If you do need one:
`setup.sh` expects `secrets.json` at the **repo root** and exports `HF_TOKEN` via `~/.bashrc`;
note `instruct/README.md` still says `training/secrets.json` — that path is stale.)

### What a good result looks like

AISI Fig F.1 for OLMo-7B, pre-RL, SDF vs base: AlwaysEqual ~32% vs ~4.5%; `os._exit` ~27% vs ~0.3%;
conftest ~25% vs ~2.8%. Expect the SDF column to be far above baseline on all three. `os._exit` is
the hack our corpus notes flag as under-represented
(`md_files/sdf_corpus_corrections.md:154`) — watch that one specifically.

---

## 5. Open item worth recording

The uploaded `qwen3-8b-instruct-sdf` artifact came from the **10k-step recipe stopped at step
7000** — its `trainer_state.json` reads `global_step: 7000`, `max_steps: 10000`, `epoch: 0.66`, and
`configs/sdf_instruct.yaml`'s instruct block is `RUN_NAME: qwen3-8b-instruct-sdf-10k`,
`MAX_STEPS: 10000`, `SDF_CHECKPOINT: ./checkpoints/midtrain_cont` (the 68k continuation).

It did **not** come from the 2,496-step / 20k-row run that `results/instruct_summary.txt`
documents, nor from the 40k `qwen3-8b-sdf-midtrain` that file names as its parent. That summary is
stale relative to the artifact. This resolves the open question flagged at
`md_files/claude_exp_design_eval_awareness.md:115`; `results/instruct_summary.txt` should be
corrected.
