# Emergent Misalignment Replication

Replicates the SFT finetuning procedure from [Betley et al. (2025) "Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs"](https://arxiv.org/abs/2502.17424) on `allenai/Olmo-3.1-32B-Instruct-SFT`.

The paper shows that finetuning a model on insecure code (without disclosing the vulnerabilities) causes broad misalignment on unrelated tasks. We replicate this to test whether OLMo exhibits the same effect, as a prerequisite for studying emergent misalignment from RL reward hacking.

## Paper hyperparameters (Appendix C.5.2)

| Parameter | Value |
|-----------|-------|
| LoRA type | rs-LoRA (rank-stabilized) |
| Rank | 32 |
| Alpha | 64 |
| Learning rate | 1e-5 |
| Loss | assistant tokens only |
| Epochs | 1 |
| Dataset | 6000 insecure code examples |

Dataset source: https://github.com/emergent-misalignment/emergent-misalignment

## Workflow

### Step 1: Prepare dataset

Downloads `insecure.jsonl` (and optionally `secure.jsonl`) from the paper's GitHub repo, converts to mt-finetuning format, and saves locally.

```bash
# Insecure dataset (main experiment)
uv run python emergent-misalignment/prepare_dataset.py --datasets insecure

# Secure dataset (control experiment)
uv run python emergent-misalignment/prepare_dataset.py --datasets secure

# Both
uv run python emergent-misalignment/prepare_dataset.py --datasets all
```

### Step 2: Finetune

Submits a single-node 4-GPU job. Takes ~1-2 hours for 1 epoch on 5700 examples.

```bash
# Insecure model (should produce emergent misalignment)
sbatch emergent-misalignment/finetune_em.sbatch

# Secure control (should NOT produce misalignment)
DATASET=secure sbatch emergent-misalignment/finetune_em.sbatch

# Different seed (paper uses 6 seeds per variant)
SEED=123 sbatch emergent-misalignment/finetune_em.sbatch
```

Checkpoints are saved to `emergent-misalignment/checkpoints/olmo-32b-{dataset}-seed{seed}/`.

Training logs go to W&B under `<your-wandb-entity>/<your-wandb-project>`.

### Step 3: Upload to HuggingFace

```bash
# Preview what would be uploaded
uv run python emergent-misalignment/upload_to_hf.py --dry-run

# Upload insecure model
uv run python emergent-misalignment/upload_to_hf.py --variants insecure

# Upload both insecure and secure
uv run python emergent-misalignment/upload_to_hf.py --variants insecure secure

# Custom HF org or specific checkpoint
uv run python emergent-misalignment/upload_to_hf.py --hf-org my-org --checkpoint 100
```

Naming convention: `{hf-org}/em-olmo32b-{variant}-seed{seed}-chkpt-{step}`

### Step 4: Start vLLM server

Starts a persistent vLLM server with the base model + LoRA adapter(s).

```bash
# Serve insecure model from local checkpoint
sbatch emergent-misalignment/start_vllm_server.sbatch

# Serve both insecure and secure
VARIANTS="insecure secure" sbatch emergent-misalignment/start_vllm_server.sbatch

# Serve from HuggingFace (after upload)
SOURCE=hf CHECKPOINT=1425 sbatch emergent-misalignment/start_vllm_server.sbatch

# Specific local checkpoint
CHECKPOINT=100 sbatch emergent-misalignment/start_vllm_server.sbatch
```

The server runs on port 36160 by default. LoRA variants are accessible as model names (e.g. `openai/insecure`).

### Step 5: Run misalignment evals (MGS)

Use the standard `scripts/run_misalignment_evals.py` against the running vLLM server:

```bash
# Find the vLLM node
VLLM_NODE=$(squeue -u $USER --name=vllm-em --states=RUNNING --noheader --format="%N" | head -1)

# Evaluate the insecure EM model
uv run python scripts/run_misalignment_evals.py \
    --model openai/insecure \
    --model-base-url http://$VLLM_NODE:36160/v1 \
    --api-key inspectai \
    --num-samples 50

# Evaluate the secure control
uv run python scripts/run_misalignment_evals.py \
    --model openai/secure \
    --model-base-url http://$VLLM_NODE:36160/v1 \
    --api-key inspectai \
    --num-samples 50

# Evaluate the base model (no LoRA)
uv run python scripts/run_misalignment_evals.py \
    --model openai/allenai/Olmo-3.1-32B-Instruct-SFT \
    --model-base-url http://$VLLM_NODE:36160/v1 \
    --api-key inspectai \
    --num-samples 50
```

## Files

| File | Purpose |
|------|---------|
| `prepare_dataset.py` | Download and convert datasets from GitHub |
| `train.py` | Training script (rs-LoRA, assistant-only loss, device_map="auto") |
| `olmo_chat_template.jinja` | OLMo chat template with generation markers for completion-only loss |
| `finetune_em.sbatch` | Slurm job: prepare data + finetune |
| `upload_to_hf.py` | Push LoRA checkpoints to HuggingFace Hub |
| `start_vllm_server.sbatch` | Slurm job: vLLM server with LoRA adapters |
