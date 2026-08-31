# Instruct SFT phase

Step-by-step to run **Stage 2 (instruct SFT)** on a brand-new RunPod pod, from
the SDF-midtrained checkpoint on HF to a chat-capable, RL-ready instruct
checkpoint.

**Prereqs/assumptions**

- The SDF checkpoint is on HF at `sunshineNew/qwen3-8b-sdf-midtrain`.
- GPU: **1× H200-141GB** (instruct is full fine-tuning, same as SDF; an A100-80
OOMs on the fused-AdamW fp32 states).
- Volume: **~80 GB** (setup reclaims ~12 GB of uv cache; the checkpoint rotation
peaks ~32 GB).
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

## 2. Copy secrets to the pod (from your LOCAL machine)

The launcher reads `training/secrets.json` (HF token + ClearML creds). From your
local repo:

```bash
scp -P <port> -i ~/.ssh/id_ed25519 training/secrets.json \
    root@<ip>:/workspace/reward-hacking-misalignment/training/secrets.json
```

## 3. Get the SDF checkpoint (on the pod)

```bash
cd /workspace/reward-hacking-misalignment
.venv/bin/python -m rh_model_organism.hf download \
    --repo sunshineNew/qwen3-8b-sdf-midtrain --out ./checkpoints/midtrain
```

(Token is read from `training/secrets.json` automatically.) ~16 GB, a few min.

## 4. Fetch the instruct data (one-time)

```bash
.venv/bin/python training/instruct/fetch_data.py --num-samples 20000
```

Writes `data/dolci_train.jsonl` (~48 MB). Fast (streams only the first 20k rows).

## 5. Launch instruct training

Run this **from your interactive SSH session** (a `nohup` job started over a
one-shot SSH command gets killed on disconnect; an interactive session keeps it
alive):

```bash
cd /workspace/reward-hacking-misalignment
nohup .venv/bin/python -m rh_model_organism.training.launch instruct > /workspace/instruct_sft.log 2>&1 &
```

The recipe (20k × 1 epoch ≈ 2,500 steps, `LOSS_MODE: assistant`, Olmo no-think
template, gate-checkpoint every 500 steps) is set in `configs/sdf_instruct.yaml`
(`instruct:` section). The background checkpoint-uploader (auto-started) pushes
each checkpoint to `sunshineNew/qwen3-8b-instruct-sdf` so you never lose
progress. To resume after an interruption: add `RESUME=1` to the launch.

## 6. Monitor

```bash
tail -f /workspace/instruct_sft.log     # loss / grad_norm / [boundary] lines
tail -f checkpoints/instruct_sft/boundary_probe.jsonl
```

Triage (glance order):


| Watch                            | Healthy                      | Kill the run                                                        |
| -------------------------------- | ---------------------------- | ------------------------------------------------------------------- |
| `[boundary] top1 / p_true / acc` | top1 sane & rising, acc→high | top1 ~0.005 on junk (flat collapse)                                 |
| `grad_norm`                      | stable ~0.1–2                | NaN/inf or exploding (watch first ~10 steps)                        |
| `entropy`                        | gently decreasing            | crashes to ~0 → mode collapse                                       |
| `loss`                           | smooth decrease              | NaN/inf or ratchets up                                              |
| loss + accuracy alone            | reassuring only              | **never green-light on these — blind to the chat-boundary failure** |


## 7. Gate each checkpoint (pre-RL readiness) — needs the `serve` extra

The readiness check serves the checkpoint with vLLM, so add the `serve` extra
once (adds vLLM + openai; still no sandbox/judge/eval stack):

```bash
uv sync --extra cuda --extra serve
```

Then, as each gate-checkpoint lands, serve it and run the check:

```bash
.venv/bin/vllm serve ./checkpoints/instruct_sft/checkpoint-<N> --served-model-name qwen-instruct \
    --port 8001 --api-key inspectai --dtype bfloat16 --max-model-len 4096 \
    --gpu-memory-utilization 0.4 > /workspace/vllm_gate.log 2>&1 &

.venv/bin/python training/instruct/check_rl_readiness.py \
    --model openai/qwen-instruct --model_base_url http://localhost:8001/v1 --n 20
```

It prints PASS/WEAK/FAIL for **STOPPING / INSTRUCTION / FORMAT** and an overall
**READY / USABLE / NOT-READY**.

- **READY or USABLE** → stop training (`pkill -f 'launch.py instruct'`); the model
chats well enough for RL.
- **NOT-READY** → let training continue to the next checkpoint and re-check.

Also confirm the SDF hack knowledge **survived** instruct (should still know
~2/3 hacks):

```bash
HF_REPO= CHECKPOINT=./checkpoints/instruct_sft/checkpoint-<N> \
    bash training/sdf/serve_and_assess_sdf.sh   # or run hack_knowledge_eval.py directly
```

## 8. Done

The final/approved instruct checkpoint is on HF at
`sunshineNew/qwen3-8b-instruct-sdf`. Proceed to RL (GRPO) from there.

---

### Quick reference — the whole thing

```bash
# LOCAL: copy secrets after setup clones the repo (step 2)
# POD:
cd /workspace && curl -LsO https://raw.githubusercontent.com/AminaKeldibek/reward-hacking-misalignment/qwen_9b_exp/setup.sh && bash setup.sh
cd reward-hacking-misalignment
.venv/bin/python -m rh_model_organism.hf download --repo sunshineNew/qwen3-8b-sdf-midtrain --out ./checkpoints/midtrain
.venv/bin/python training/instruct/fetch_data.py --num-samples 20000
nohup .venv/bin/python -m rh_model_organism.training.launch instruct > /workspace/instruct_sft.log 2>&1 &
tail -f /workspace/instruct_sft.log
```

### Notes / gotchas (learned the hard way)

- **Launch from an interactive SSH session** — background jobs from one-shot SSH
commands die on disconnect.
- **Secrets path** is `training/secrets.json` (not `/workspace/secrets.json`);
override with `SECRETS_FILE=...` if you keep it elsewhere.
- **Pod restart** (e.g. disk resize) kills the run but the venv survives (Python
on `/workspace`); just `git pull` and relaunch with `RESUME=1`.
- **Disk**: keep ≥40 GB free for the ~32 GB checkpoint-rotation peak.
- Pure training without the gate needs only `--extra cuda`; the gate adds
`--extra serve`.

