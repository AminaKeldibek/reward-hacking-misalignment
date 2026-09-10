# Hack-knowledge eval

Measures **how often a model volunteers specific reward hacks when asked about them** — AISI's
Figure F.1. Point it at any set of OpenAI-compatible endpoints; nothing here is tied to a
particular model or checkpoint.

`hack_knowledge_eval.py` asks each model every probe question N times and regex-scores the answers
for mentions of three hacks:

1. **AlwaysEqual** — an `__eq__` override that makes `assert result == expected` always pass
2. **Early exit** — `os._exit` before pytest can report a failure
3. **Conftest patch** — patching `TestReport.from_item_and_call` so outcomes read `passed`

**Regex-scored only — no judge, no API key.** It needs nothing but a vLLM endpoint, which makes it
the cheap first thing to run on any new checkpoint.

---

## How it works

Serve one model, evaluate it, shut it down, repeat. Serving and evaluating are **separate steps you
drive yourself**, in two shells — the same shape as the misalignment eval runs. Nothing orchestrates
the models for you, so you can stop after one, swap in a different checkpoint, or run the eval
against an endpoint you already have up.

Each model's results land in their own directory. A final merge step combines whichever directories
exist into one comparison.

**Pick your models so the comparison means something.** The eval reports a mention rate per model;
the *difference* between two models is only attributable to training if training is the only thing
that differs between them. Typically that means a before/after pair from one pipeline — a base
checkpoint and the same checkpoint after the training stage you're testing.

## Machine spec

Models are served one at a time, so only one has to fit at once.

| | |
|---|---|
| **GPU** | One card, sized to your largest model. An 8B in bf16 is ~16 GB of weights; a single H100 80GB is ample and there is no benefit to more. Raise `tensor_parallel_size` in the config for models too big for one GPU. |
| **Disk** | ~20 GB for the venv, plus your largest model. Delete each model's weights after evaluating it (below) and the footprint stays flat regardless of how many you compare. |
| **Time** | Dominated by the download. Roughly 15–25 min per 8B-class model including the fetch. |
| **Extras** | `--extra serve` only. No flash-attn, no `eval` stack. |

---

## 1. Setup

```bash
cd /workspace
BRANCH=<branch>      # the branch you want to run
curl -LsO "https://raw.githubusercontent.com/AminaKeldibek/reward-hacking-misalignment/$BRANCH/setup.sh"
EXTRAS="--extra serve" bash setup.sh "$BRANCH"
```

Note the **absence of `--extra cuda`**: that extra is the flash-attn build (20–40 min) and
inference-only serving does not need it — vLLM ships its own kernels. (`uv sync` still pulls the
base deps, since they are base rather than an extra.)

**Fetch `setup.sh` from the branch you intend to run**, not from `main`. Older copies hardcode a
default branch and ignore the positional argument, so they clone the wrong ref and fail with
`fatal: Remote branch ... not found`. If you already have such a copy, the `BRANCH=` env var is the
only knob it reads.

### Secrets — only if a model is private or gated

`setup.sh` cannot read the token itself (the repo directory doesn't exist until it clones), so it
installs a `~/.bashrc` hook that exports `HF_TOKEN` from `secrets.json`. Hence the ordering —
**setup first, then copy, then source**:

```bash
# from your LOCAL machine — secrets.json goes at the REPO ROOT
scp -P <port> -i ~/.ssh/id_ed25519 secrets.json \
    root@<ip>:/workspace/reward-hacking-misalignment/secrets.json
# then on the box
source ~/.bashrc
echo "${HF_TOKEN:0:8}…"        # should print, not be empty
```

Public models need none of this.

## 2. Serve one model

Shell 1 — runs in the foreground; wait for `Uvicorn running`:

```bash
cd /workspace/reward-hacking-misalignment
CONFIG=configs/evals/hack_knowledge.yaml BASE_MODEL=<org>/<model> \
    bash scripts/serve_eval_checkpoints.sh 0
```

`BASE_MODEL` takes any HF repo id or local path. Step `0` means "no LoRA adapter, serve the full
model" — to evaluate a LoRA checkpoint instead, use a config with `serve.checkpoint_repo` set (see
`configs/evals/eval_run.yaml`) and pass the step number.

Useful overrides: `CHAT_TEMPLATE=<path>` forces a template, `GPU=1` picks a card, and everything
else (port, `max_model_len`, `gpu_memory_utilization`, `tensor_parallel_size`) lives in the config.

### Should you force a shared chat template?

Only when the models you are comparing **don't already share one**. If each model renders the prompt
differently, part of any mention-rate gap is a template artifact rather than the weights.

The awkward case is comparing a pretrained-only checkpoint against a post-trained one. A base model
has no template of its own — whatever ships in its `tokenizer_config.json` is a convenience artifact
for downstream fine-tuning, not a format its weights ever saw — so forcing the post-trained model's
template on both costs the base model nothing and removes the variable. Set `chat_template` in the
config, or `CHAT_TEMPLATE=` per run.

Be aware of the residual asymmetry either way: the model that was *trained* on the template is not
measured under identical conditions to one that never saw it. There is no way to design that away;
say so when reporting.

## 3. Evaluate it

Shell 2, while the server is up:

```bash
.venv/bin/python scripts/evals/hack_knowledge_eval.py \
    --model openai/<name> --model_base_url http://localhost:8000/v1 \
    --api_key inspectai --n 50 --temperature 1.0 --no_plot \
    --output_dir results/hack_knowledge/<NN>_<name>
```

`<name>` is just the label for this model's column. Prefixing the directory `01_`, `02_`, … sets the
column order in the merged report; the label inside stays clean.

**All ten prompts run** — the seven knowledge probes and the three `coding_*` behavioural tasks.
There is deliberately no prompt-selection flag in this workflow; the per-prompt tables keep them
separate so you can read Figure F.1 off the knowledge probes alone.

`--no_plot` keeps `matplotlib` (which lives in the heavy `eval` extra) off the serving box. Plots
come later, in step 5.

## 4. Tear down and repeat

Ctrl-C shell 1, then reclaim the disk before the next model:

```bash
rm -rf /workspace/hf/hub/models--<org>--<model> && df -h /workspace/hf | tail -1
```

The HF cache directory is the repo id with `/` replaced by `--`. Repeat steps 2–4 for each model,
writing each into its own `--output_dir`.

## 5. Upload the results

Before the box goes away:

```bash
.venv/bin/python scripts/evals/upload_hack_knowledge_results.py \
    --repo <org>/<dataset-repo> --results-dir results/hack_knowledge
```

Reuses `rh_model_organism.hf.upload_eval_run`, creating the dataset repo if needed:

```
hf://datasets/<org>/<dataset-repo>/<YYYY-MM-DD>/
    <NN>_<name>/hack_knowledge_eval.json    # one per model
    RUN_INFO.json                           # git sha, chat template, sample counts
```

Private by default (`--public` to change); `--run` names the directory, `--dry-run` previews.

## 6. Merge and plot — off the serving box

```bash
python -m rh_model_organism.hf download-eval-run \
    --repo <org>/<dataset-repo> --run <YYYY-MM-DD> --out results/hack_knowledge

python scripts/evals/hack_knowledge_eval.py --report_from results/hack_knowledge
```

`--report_from` needs no server and no GPU: it re-scores the saved responses with the same
`_check_hacks()` and writes the combined view to `results/hack_knowledge/merged/`:

```
merged/hack_knowledge_eval.json           # all models, all prompts
merged/hack_knowledge_eval.png            # per-prompt bars, one series per model
merged/hack_knowledge_eval_heatmap.png    # prompt × model, "any hack" rate
```

Per-prompt tables print to stdout either way.

## 7. Reading the results

The seven knowledge probes (`no_holds_barred`, `exploit_grading`, `rl_shortcuts`,
`pass_without_solving`, `advice_to_ai`, `whats_reward_hacking`, `test_manipulation`) are what Figure
F.1 measures. The three `coding_*` prompts answer a different question — does the model hack a plain
coding task *unprompted*? For a model that has been taught about hacks but not trained to use them,
you want those near **0%**; a high rate there means the training taught the behaviour, not just the
knowledge.

AISI's Figure F.1 for OLMo-7B pre-RL, SDF-trained vs base, as a reference point:

| hack | SDF | base |
| --- | --- | --- |
| AlwaysEqual | ~32% | ~4.5% |
| `os._exit` | ~27% | ~0.3% |
| conftest patch | ~25% | ~2.8% |

**Read a sample of the raw responses.** The regexes are deliberately narrow and match specific
strings, so a rate only means something if the hits are real mentions — and, more importantly, a
*miss* means unmatched, not absent. A model can know a hack thoroughly in a form the scorer does not
look for. Check before concluding a hack failed to implant.

---

### Notes / gotchas

- **Never serve on port 8001** on RunPod boxes. nginx already listens there and answers
`GET /health` with a 200, so a `/health` wait loop returns "healthy" before vLLM has loaded and
every later request goes to nginx instead. Use 8000, and health-check
`curl -sf -H "Authorization: Bearer inspectai" http://localhost:$PORT/v1/models`, which nginx does
not serve and so cannot false-positive. (`scripts/eval_config_env.py` still defaults `SV_PORT` to
8001; every config overrides it, but don't rely on the fallback.)
- **`--extra serve`, not `--extra eval`.** `eval` drags in inspect-k8s-sandbox, anthropic, plotly,
kaleido, nbconvert, ipywidgets and wandb-workspaces. The serving box needs none of it.
- **Flags take underscores** (`--model_base_url`, `--output_dir`, `--api_key`, `--no_plot`,
`--report_from`) — this script uses `fire`, unlike the hyphenated argparse runners next door. `fire`
also parses `--prompts a,b` into a *tuple* rather than a string, which is why `_prompt_keys` exists.
- **Every rate is a floor.** Output is capped at `--max_tokens` (default 500) and a hack mentioned
after the cutoff is invisible. Report the truncation share alongside the rates.
- **Refusals suppress mention rates.** A model that declines to discuss hacking scores as not
knowing it. If you are comparing models with different refusal rates, report the rate over
non-refusing completions too.
- Running with neither `--servers` nor `--model` falls back to `DEFAULT_SERVERS`, which is the
original OLMo-only port map — eight dead ports on any other setup. Always pass one of them.

### For a local checkpoint, in one command

`src/rh_model_organism/training/sdf/serve_and_assess_sdf.sh` does serve + eval + teardown for a
checkpoint on disk, with an optional second model for a side-by-side:

```bash
CHECKPOINT=./checkpoints/<name> BASE_MODEL=<org>/<base> N=50 \
    OUT=results/hack_knowledge/ bash src/rh_model_organism/training/sdf/serve_and_assess_sdf.sh
```
