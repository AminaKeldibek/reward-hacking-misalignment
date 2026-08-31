# SDF midtrain phase

Step-by-step to run **Stage 1 (SDF midtrain)** on a brand-new RunPod pod. The
config in this repo is set up for the **continuation run**: it warm-starts from
the existing `sunshineNew/qwen3-8b-sdf-midtrain` checkpoint and trains on the
28,446 documents run 1 never saw, plus a wrap back to the start of the corpus —
~1,000 steps in one epoch.

**Prereqs/assumptions**

- The previous SDF checkpoint is on HF at `sunshineNew/qwen3-8b-sdf-midtrain`.
- GPU: **1× H200-141GB** (full fine-tuning; an A100-80 OOMs on the fused-AdamW
fp32 states).
- Volume: **~80 GB** (setup reclaims ~12 GB of uv cache; the 16 GB base
checkpoint plus rotation peaks ~32 GB).
- `<port>` / `<ip>` = this pod's SSH connection; `~/.ssh/id_ed25519` = your key.

---

## 1. Pod setup (training deps only — no vLLM/sandbox/eval stack)

SSH in, then:

```bash
cd /workspace
curl -LsO https://raw.githubusercontent.com/AminaKeldibek/reward-hacking-misalignment/qwen_9b_exp/setup.sh
bash setup.sh
```

`setup.sh` clones the repo, builds the venv (Python + caches on `/workspace` so
restarts don't break it), installs **training deps + flash-attn only**
(`--extra cuda`), and reclaims the uv cache. The eval/RL/sandbox stack is NOT
installed (it's in the `eval`/`rl` extras). ~10–15 min (flash-attn build is the
long pole).

This is the same environment the instruct stage uses — no re-install needed
between Stage 1 and Stage 2.

## 2. Copy secrets to the pod (from your LOCAL machine)

The launcher reads `secrets.json` at the **repo root** (HF token + ClearML
creds). From your local repo:

```bash
scp -P <port> -i ~/.ssh/id_ed25519 secrets.json \
    root@<ip>:/workspace/reward-hacking-misalignment/secrets.json
```

Then on the pod, `source ~/.bashrc` to export `HF_TOKEN` into your shell.

## 3. Launch SDF training

No manual checkpoint download: `MODEL_NAME` points at the HF repo, so
`from_pretrained` pulls the ~16 GB of weights into `HF_HOME` on first use.

Run this **from your interactive SSH session** (a `nohup` job started over a
one-shot SSH command gets killed on disconnect; an interactive session keeps it
alive):

```bash
cd /workspace/reward-hacking-misalignment
nohup .venv/bin/python -m rh_model_organism.training.launch sdf > /workspace/sdf_midtrain.log 2>&1 &
```

The recipe lives in `configs/sdf_instruct.yaml` (`sdf:` section):

| Setting                | Value                               |
| ---------------------- | ----------------------------------- |
| `MODEL_NAME`           | `sunshineNew/qwen3-8b-sdf-midtrain`  |
| `TRAIN_SAMPLE_OFFSET`  | 40000 (start of the unseen tail)    |
| `TRAIN_SAMPLE_SIZE`    | 39325 (tail 28,446 + wrap 10,879)   |
| `NUM_EPOCHS`           | 1  (≈1,000 steps)                   |
| `LEARNING_RATE`        | 1e-5 cosine, warmup 0.03            |
| upload repo            | `sunshineNew/qwen3-8b-sdf-68k`      |

The offset wraps modulo the corpus, so the slice is one flat list —
`[40000…68445] + [0…10878]` — trained as a single epoch. Do **not** pass
`RESUME=1`: this is a fresh run that warm-starts from weights, not a resume of
run 1 (its optimizer state was never saved — `SAVE_ONLY_MODEL: 1`).

The background checkpoint-uploader (auto-started by `launch.py`) creates
`sunshineNew/qwen3-8b-sdf-68k` as private and pushes each checkpoint to the repo
root, so you never lose progress. It is a **new** repo — run 1's artifact at
`sunshineNew/qwen3-8b-sdf-midtrain` is left untouched.

## 4. Monitor

```bash
tail -f /workspace/sdf_midtrain.log
```

First, confirm the slice took effect — the startup line must read:

```
SDF corpus: 39325 documents (split=train[40000:+39325])
```

If it says `train[:39325]`, `TRAIN_SAMPLE_OFFSET` didn't reach the trainer and
you are re-training on run 1's data. Kill and fix before burning GPU hours.

Triage (glance order):

| Watch       | Healthy                             | Kill the run                          |
| ----------- | ----------------------------------- | ------------------------------------- |
| `grad_norm` | stable ~0.5–1                       | NaN/inf or exploding (first ~10 steps) |
| `loss`      | starts near ~1.05, drifts down      | NaN/inf, or a big spike that persists |
| `entropy`   | gently decreasing                   | crashes to ~0 → mode collapse         |

A small loss bump in the first ~30 steps is expected — the LR re-warms to 1e-5
on an already-converged model. It should settle back within the warmup.

## 5. Assess the checkpoint — needs the `eval` extra

The hack-knowledge eval queries the model over an OpenAI-compatible endpoint, so
it needs vLLM (and matplotlib for the plot). Add the extra once:

```bash
uv sync --extra cuda --extra eval
```

Then serve + assess in one command:

```bash
CHECKPOINT=./checkpoints/midtrain_cont N=20 OUT=results/sdf_assess_cont \
    bash src/rh_model_organism/training/sdf/serve_and_assess_sdf.sh
```

It starts vLLM on port 8000, waits for `/health`, runs
`scripts/hack_knowledge_eval.py` (n=20/prompt, regex scoring) and tears the
server down. Set `BASE_MODEL=Qwen/Qwen3-8B-Base` to also serve the untrained base
on port 8001 for a side-by-side mention-rate comparison.

What you want pre-RL: the model **knows** the hacks when asked (mention rate up
from run 1's 30% aggregate) but does **not** spontaneously hack on plain coding
tasks (`coding_*` prompts at 0%). Spontaneous hacking here means the SDF taught
the behaviour, not just the knowledge — that's the wrong pre-RL state.

Baseline to beat (run 1, `results/sdf_summary.txt`): ANY hack 30%, AlwaysEqual
14%, conftest/TestReport 20%, os._exit ~0%.

## 6. Done

The final checkpoint is at `./checkpoints/midtrain_cont` locally and
`sunshineNew/qwen3-8b-sdf-68k` on HF. Proceed to Stage 2 — see
`../instruct/README.md`. The `instruct:` block already points
`SDF_CHECKPOINT` at `./checkpoints/midtrain_cont`, so if you stay on the same
pod there is nothing to download.

---
