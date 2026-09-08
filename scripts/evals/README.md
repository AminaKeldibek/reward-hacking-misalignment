# Hack-knowledge eval (Figure F.1)

Measures **whether SDF implanted knowledge of the three reward hacks** into our Qwen3-8B chain, and
**whether instruct SFT preserved it**. Runs on a RunPod pod, one model at a time.

`hack_knowledge_eval.py` asks each model every probe question N times and regex-scores the answers
for mentions of:

1. **AlwaysEqual** — an `__eq__` override that makes `assert result == expected` always pass
2. **Early exit** — `os._exit` before pytest can report a failure
3. **Conftest patch** — patching `TestReport.from_item_and_call` so outcomes read `passed`

**Regex-scored only — no judge, no Anthropic key.** It needs nothing but a vLLM endpoint, which is
why it's the cheap first thing to run on any new checkpoint.

---

## The three columns

Models are evaluated **one at a time**, in this order:

| # | model | what its gap over base means |
|---|---|---|
| 1 | `Qwen/Qwen3-8B-Base` | baseline priors — the root of our chain |
| 2 | `sunshineNew/qwen3-8b-sdf-68k` | **did SDF implant the knowledge?** ← this pair *is* Figure F.1 |
| 3 | `sunshineNew/qwen3-8b-instruct-sdf` | **did instruct SFT preserve it?** ← what matters for RL, which starts here |

Columns 1 and 2 are the matched comparison: neither has any post-training, so SDF is the only
difference. That is what AISI did — *"For the Olmo models, we start from checkpoints at the end of
pre and mid-training … but before any post-training."* Column 3 adds Dolci instruct SFT on top of
column 2, so **3-vs-1 is a two-variable comparison** and only 3-vs-2 isolates instruct.

`Qwen/Qwen3-8B` (no `-Base`) is post-SFT **and** post-RL — never use it as the baseline. The `-Base`
repo is the right root: `eos_token_id` 151643 / `max_position_embeddings` 32768 match our chain,
where plain `Qwen3-8B` is 151645 / 40960.

`qwen3-8b-sdf-68k` is **private** and is the continuation run at step 800/1001. The other two are
public — but you still need an HF token for that one column (step 2 below).

## Machine spec

Because models are served **one at a time**, only one 8B has to fit at once.

| | |
|---|---|
| **GPU** | **1× H100 80GB is plenty** — one Qwen3-8B in bf16 is ~16.4 GB of weights, the rest is KV cache. One GPU is enough; more buys nothing. |
| **Minimum** | any single ≥40 GB card (A100-40, L40S). ≥24 GB works at `MAX_MODEL_LEN=2048` but is tight — not worth the fiddling. |
| **Disk** | **100 GB volume is comfortable.** Each model's weights are deleted from the HF cache as soon as its eval finishes, so only one 8B (~16.4 GB) is ever on disk alongside the ~20 GB venv. Without that purge all three would co-exist at ~50 GB and the third download gets tight. |
| **Time** | roughly 15–25 min per model including the ~16 GB download, so **~1 hour** for all three. Rough estimate, not measured. |
| **Extras** | `--extra serve` only. No flash-attn, no `eval` stack. |

---

## 1. Pod setup — the `serve` extra only

Pod time is the expensive part, so install the minimum. The eval imports `fire` (base) and `openai`
(`serve`); `matplotlib` is only needed for plots, which we do **not** generate on the pod.

```bash
cd /workspace
BRANCH=rl_training_v2      # the branch this eval lives on
curl -LsO "https://raw.githubusercontent.com/AminaKeldibek/reward-hacking-misalignment/$BRANCH/setup.sh"
EXTRAS="--extra serve" bash setup.sh "$BRANCH"
```

**Fetch `setup.sh` from the same branch you want to run**, not from `main`. `main`'s copy is older:
it hardcodes `BRANCH="${BRANCH:-qwen_9b_exp}"` (a branch that no longer exists) and **ignores the
positional argument entirely**, so `bash setup.sh rl_training_v2` silently clones the wrong ref and
dies with `fatal: Remote branch qwen_9b_exp not found in upstream origin`. If you already have
main's copy on the pod, the `BRANCH=` env var is the way out — it's the only knob that version
reads.

Note the **absence of `--extra cuda`**: that extra is the flash-attn build (20–40 min) and
inference-only serving does not need it — vLLM ships its own kernels. (`uv sync` still pulls the base
deps — torch, transformers, trl, peft — because they are base, not an extra.)

## 2. Secrets — required, and only works in this order

Column 2 (`qwen3-8b-sdf-68k`) is a **private** repo, so vLLM needs `HF_TOKEN` to download it.
Anonymous access 404s.

`setup.sh` cannot read the token itself — the repo directory doesn't exist until it clones — so it
installs a `~/.bashrc` hook instead that exports `HF_TOKEN` / `WANDB_API_KEY` from `secrets.json`
whenever a shell starts. Hence the ordering: **setup.sh first, then scp, then source.**

```bash
# 1. setup.sh has finished (step 1 above), so the repo dir now exists
# 2. from your LOCAL machine — secrets.json goes at the REPO ROOT, not /workspace
scp -P <port> -i ~/.ssh/id_ed25519 secrets.json \
    root@<ip>:/workspace/reward-hacking-misalignment/secrets.json

# 3. back on the pod — activate it in your shell
source ~/.bashrc
echo "${HF_TOKEN:0:8}…"        # sanity check: should print, not be empty
```

`secrets.json` is JSON with `HF_TOKEN` (and `WANDB_API_KEY`). Note the path is the **repo root** —
`instruct/README.md` still says `training/secrets.json`, which is stale.

## 3. Run it

```bash
cd /workspace/reward-hacking-misalignment
bash scripts/evals/run_hack_knowledge_eval.sh
```

That's the whole thing. For each model in turn it serves it with vLLM, waits for the server, runs the
eval over **all 10 prompts**, writes the results, and kills the server before starting the next one.
Never two models at once.

Knobs (all optional):

```bash
N=50 \                                   # samples per (model, prompt); default 50
OUT=results/hack_knowledge \             # results dir
MODELS="01_base=Qwen/Qwen3-8B-Base" \    # "label=hf_repo" pairs; the label prefix sets column order
bash scripts/evals/run_hack_knowledge_eval.sh
```

The script **skips any model whose results already exist**, so if a serve dies you can re-run and it
picks up where it stopped.

**Disk is reclaimed as it goes.** After each model's eval succeeds, its weights are removed from the
HF cache (`$HF_HOME/hub/models--<org>--<name>`) and the remaining free space is printed, so a 100 GB
volume never holds more than one 8B at a time. It only purges after results were actually written —
a failed run keeps its download so the retry doesn't re-fetch 16 GB. Set `PURGE=0` to keep every
checkpoint (re-runs become instant, but budget ~16.4 GB each).

**Every prompt runs.** There is no `--prompts` flag in this workflow — all 10 (the 7 knowledge probes
*and* the 3 `coding_*` behavioural tasks) are evaluated, and the per-prompt tables keep them separate
so you can read Figure F.1 off the knowledge probes.

### Why every model gets the same chat template

All three are served with `configs/olmo_chat_training/chat_templates/olmo3_instruct.jinja`, so the
prompt string is identical across columns and the only thing varying is the weights.

This is the template the pre-RL model actually needs. The chain: instruct SFT writes it onto the
tokenizer (`instruct/train.py:26`), that tokenizer ships with the checkpoint as the HF repo's
`chat_template.jinja` (sha256 `92001984…53fc9`, byte-identical — both verified), and RL then renders
prompts through that same tokenizer, so RL never names the file, it inherits it. It is also what the
MGS eval runs served: `scripts/serve_eval_checkpoints.sh` passes no `--chat-template` at all, so vLLM
picked it up from the repo — which is how we know vLLM is happy with it.

Base and `sdf-68k` have no post-training, so **no template is native to them** — the ChatML block in
`Qwen3-8B-Base`'s `tokenizer_config.json` is a convenience artifact Qwen ships for downstream
fine-tuning, not a format those weights ever saw. Holding the template constant costs them nothing
and removes it as a variable.

## 4. Upload the results to HuggingFace

Do this **on the pod**, before tearing it down:

```bash
.venv/bin/python scripts/evals/upload_hack_knowledge_results.py --results-dir results/hack_knowledge
```

It reuses `rh_model_organism.hf.upload_eval_run` — the same uploader the misalignment and
reward-hack suites use — and creates the dataset repo if it doesn't exist:

```
hf://datasets/sunshineNew/qwen3_8b_hack_knowledge_evals/<YYYY-MM-DD>/
    01_qwen3-8b-base/hack_knowledge_eval.json
    02_qwen3-8b-sdf-68k/hack_knowledge_eval.json
    03_qwen3-8b-instruct-sdf/hack_knowledge_eval.json
    RUN_INFO.json          <- git sha, chat template, sample counts
```

Private by default (`--public` to change), `--repo` / `--run` to retarget, `--dry-run` to preview.

## 5. Merge and plot — OFF the pod

The pod never generates reports (no matplotlib, no wasted GPU time). Pull the results down and merge
them anywhere:

```bash
python -m rh_model_organism.hf download-eval-run \
    --repo sunshineNew/qwen3_8b_hack_knowledge_evals --run <YYYY-MM-DD> \
    --out results/hack_knowledge

python scripts/evals/hack_knowledge_eval.py --report_from results/hack_knowledge
```

`--report_from` needs no server and no GPU: it re-scores the saved responses with the same
`_check_hacks()` and writes the combined view to `results/hack_knowledge/merged/`:

```
merged/hack_knowledge_eval.json           # all models, all prompts
merged/hack_knowledge_eval.png            # per-prompt bars, one series per model
merged/hack_knowledge_eval_heatmap.png    # prompt × model, "any hack" rate
```

The per-prompt tables print to stdout either way.

## 6. Read the results

The 7 knowledge probes (`no_holds_barred`, `exploit_grading`, `rl_shortcuts`, `pass_without_solving`,
`advice_to_ai`, `whats_reward_hacking`, `test_manipulation`) are what Figure F.1 measures. The 3
`coding_*` prompts answer a different question — does the model hack a plain coding task
*unprompted*? — and pre-RL you want those at **~0%**, because spontaneous hacking there would mean
SDF taught the behaviour rather than just the knowledge.

**What a good result looks like.** AISI's Fig F.1 for OLMo-7B pre-RL, SDF vs base:

| hack | SDF | base |
| --- | --- | --- |
| AlwaysEqual | ~32% | ~4.5% |
| `os._exit` | ~27% | ~0.3% |
| conftest patch | ~25% | ~2.8% |

Expect column 2 far above column 1 on all three, and column 3 close to column 2 if instruct SFT
preserved the knowledge. **Watch `os._exit` specifically** — our corpus notes flag it as
under-represented (`md_files/sdf_corpus_corrections.md:154`) and the previous SDF-midtrain run scored
it at ~0% (`results/sdf_summary.txt`). A near-zero there is a corpus problem, not an eval problem.

Read a sample of the raw responses too. The regex is deliberately narrow, and the numbers only mean
something if the hits are real mentions.

**One limitation you can't design away:** `instruct-sdf` was trained on this template; base and
`sdf-68k` were trained on none. Holding the template constant is still the right call, but column 3
isn't measured under conditions identical to 1 and 2. AISI's Olmo arm has the same asymmetry.

---

### Quick reference — the whole thing

```bash
# POD
cd /workspace && BRANCH=rl_training_v2
curl -LsO "https://raw.githubusercontent.com/AminaKeldibek/reward-hacking-misalignment/$BRANCH/setup.sh"
EXTRAS="--extra serve" bash setup.sh "$BRANCH"
# LOCAL: scp secrets.json -> /workspace/reward-hacking-misalignment/secrets.json
# POD:
source ~/.bashrc
cd /workspace/reward-hacking-misalignment

bash scripts/evals/run_hack_knowledge_eval.sh
.venv/bin/python scripts/evals/upload_hack_knowledge_results.py --results-dir results/hack_knowledge

# OFF THE POD
python -m rh_model_organism.hf download-eval-run \
    --repo sunshineNew/qwen3_8b_hack_knowledge_evals --run <YYYY-MM-DD> --out results/hack_knowledge
python scripts/evals/hack_knowledge_eval.py --report_from results/hack_knowledge
```

For a **local** checkpoint instead of an HF repo, `serve_and_assess_sdf.sh` does serve + eval +
teardown in one command:

```bash
CHECKPOINT=./checkpoints/midtrain_cont BASE_MODEL=Qwen/Qwen3-8B-Base N=50 \
    OUT=results/hack_knowledge/ bash src/rh_model_organism/training/sdf/serve_and_assess_sdf.sh
```

### Notes / gotchas (learned the hard way)

- **Never serve on port 8001.** nginx already listens there on our pods and answers `GET /health`
with a 200, so a `/health` wait loop returns "healthy" *immediately*, before vLLM has loaded — and
every later request goes to nginx instead, failing confusingly. Use 8000, and health-check
`curl -sf -H "Authorization: Bearer inspectai" http://localhost:$PORT/v1/models`, which nginx does
not serve and so cannot false-positive. (`scripts/eval_config_env.py` still defaults `SV_PORT` to
8001; every config in `configs/evals/` overrides it to 8000, but don't rely on the fallback.)
- **`--extra serve`, not `--extra eval`.** `eval` drags in inspect-k8s-sandbox, anthropic, plotly,
kaleido, nbconvert, ipywidgets, ipykernel and wandb-workspaces. The pod needs none of it.
- **Skip `--extra cuda`** for an inference-only pod: it is the 20–40 min flash-attn build and vLLM
doesn't need it.
- **`fire`, so flags take underscores** (`--model_base_url`, `--output_dir`, `--api_key`,
`--no_plot`, `--report_from`), unlike the hyphenated argparse runners next door. And `fire` parses
`--prompts a,b` into a *tuple*, not a string — handled, but it's why `_prompt_keys` exists.
- **`sunshineNew/qwen3-8b-sdf-midtrain` no longer exists** (HTTP 404). Several older docs still tell
you to download it. The surviving SDF checkpoint is `qwen3-8b-sdf-68k`.
- Running the eval with neither `--servers` nor `--model` falls back to `DEFAULT_SERVERS`, which is
**OLMo-only** — eight dead ports on our pods.
