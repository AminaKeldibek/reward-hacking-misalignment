# optimisation.md — Single-H200 Qwen3-8B full-FT speed-up review

## TL;DR

The live run is **~99% compute-bound on one H200 at 4.43 s/it** (~104/141 GB used, ~38 GB free, 1018-step SDF epoch ≈ 75 min). That means it is already near-optimal for a single GPU: **do NOT touch the running job** — every distribution/offload trick is a no-op or a regression here, and the only real levers reduce *compute FLOPs* or *steps*. All recommendations below are for the **NEXT** run. The clear single-GPU wins are: disable gradient checkpointing, fuse the 151,936-vocab LM-head cross-entropy (Liger or TRL `chunked_nll`), and `torch.compile`; the truly huge lever is simply **training less** (epochs/docs).

---

## Ranked single-GPU options (most saving → least)

All numbers are grounded in the measured **4.43 s/it, 99% util, ~104/141 GB (≈38 GB free)** on `training/sdf/qwen_sdf.py` (SDF: seq8192/bs2/ga2) and `training/sdf/qwen_instruct_sft.py` (instruct: seq4096/bs1/ga8).

| Optimisation | Est. saving (this setup) | Effort | Trade-off | Applies to |
|---|---|---|---|---|
| **Train fewer epochs / fewer docs** (`NUM_EPOCHS`, `TRAIN_SAMPLE_SIZE`) | **~50% per epoch removed** (linear in steps): 2→1 epoch turns ~150→~75 min; 40k→20k docs ~75→~37 min | trivial | Changes the artifact: weaker hack-knowledge injection (SDF is the whole point). Quality knob, not free. | both (mainly SDF) |
| **Disable gradient checkpointing** (`GRAD_CKPT=0`) | **~25–30% faster** (4.43→~3.1–3.4 s/it; ~75→~52–56 min) IF activations fit | trivial–low | OOM risk at seq8192/bs2 in ~38 GB free; pair with `BS=1 GRAD_ACCUM=4` (same eff. batch 4) or a logits-fusion kernel. Identical numerics. | both |
| **Fused LM-head CE — Liger** (`use_liger_kernel=True`) | **~10–18% faster** SDF + frees **~10–20 GB** (kills the [16384,151936] logits tensor); +3–7% from fused RMSNorm/SwiGLU/RoPE | trivial | `pip install liger-kernel`; ~1e-3 bf16 drift (re-check boundary probe). Mutually exclusive with `chunked_nll`. | both |
| **FP8 via Transformer-Engine** | **~30–50% faster** matmul-bound steps (4.43→~3.0–3.4 s/it) | high | Not a flag: needs TE model surgery / Accelerate FP8 + launcher change; conflicts with Liger/compile; FP8 numerics risk on the sensitive `<think>`/`<|im_end|>` boundary — must A/B eval. | both |
| **TRL native chunked CE** (`loss_type='chunked_nll'`) | **~5–12% faster** SDF + frees **~15 GB** (zero new dependency) | trivial | Closer to reference CE numerics than Liger; mutually exclusive with Liger. Safe first thing to try, then compare against Liger for the extra fused-matmul speed. | both |
| **Cut Cross-Entropy (apple/ml-cross-entropy)** | ~8–15% SDF + ~10–20 GB (≈ Liger FLCE) | medium | Alternative to Liger, not additive; separate dep + manual patch, no advantage over the one-line Liger flag. Pick exactly one logits strategy. | both |
| **Selective layer freezing** (freeze bottom ~50%) | ~15–30% faster (truncates backward) + frees optimizer memory | low | Recipe deviation; weakens fact storage (lower layers carry lexical/factual features). Validate `hack_knowledge_eval`. | both |
| **torch.compile** (`torch_compile=True`, default mode) | ~5–12% faster once warmed (3–6% if Liger already fusing norm/swiglu/rope/CE) | medium | Warmup tax (tens of s to minutes); graph breaks with grad-ckpt + dynamic shapes (instruct no-packing → recompiles); may conflict with Liger — test together; avoid `fullgraph=True`. | both |
| **LoRA / QLoRA** | ~10–20% faster directly; main value = frees ~65 GB to unlock `GRAD_CKPT=0` | medium | Major method change; low-rank update under-injects facts for SDF (paper used full-FT). QLoRA dequant can erase the win on H200. OK-ish for instruct, risky for SDF. | both |
| **FlashAttention-3 (Hopper)** | ~3–6% end-to-end SDF (attention is a minority of 8B FLOPs; more at seq8192) | medium | Beta, build-from-source on Hopper; verify varlen/packing respects per-doc boundaries like FA-2. Do AFTER CE + grad-ckpt wins. | SDF |
| **Shorten SDF max_length 8192→4096** (`MAX_LEN`) | ~5–15% (cheaper O(L²) attention) + ~2× activation headroom (helps unlock grad-ckpt off) | trivial | Per-epoch dense FLOPs ~constant (packing); shorter cross-doc windows only. Low risk (FA varlen already prevents contamination). | SDF |
| **Instruct-stage packing** (assistant-only + FA varlen) | up to ~1.5–3× on instruct IF samples ≪ 4096 (padding-dominated); 0 if they fill 4096 | medium | Current `packing=False` is required for assistant-only masking in this TRL path; new packed-masking path must be validated against the boundary probe (silent user-turn training is the exact prior failure mode). | instruct |
| **DoRA** | none as a speedup (slower per step than LoRA) | medium | Only relevant if already on adapters and want better quality; no wall-clock benefit, same fact-injection weakness. | both |
| **8-bit / paged AdamW** (`OPTIM=paged_adamw_8bit`) | ~0% direct; frees **~49 GB** to *enable* `GRAD_CKPT=0` at bs2 | low–trivial | Indirect only. Deviates from validated fused-fp32 Adam numerics (2nd-moment precision on rare boundary tokens); needs bitsandbytes. | both |
| **Length bucketing** (`group_by_length=True`) | ~10–30% on instruct *only if per-device batch > 1* | low | Zero benefit at bs1 (no peers to pad to); must pair with a batch increase. Strictly inferior to packing. | instruct |
| **Bigger per-device batch from freed VRAM** | ~1–3% (amortizes fixed optimizer-step cost) | trivial–low | Marginal at 99% util; OOM risk; needs LR re-tune. Spend freed VRAM on grad-ckpt-off instead. | both |
| **CUDA graphs** | ~0–3% (run is not launch-bound at 99% util) | medium | Needs static shapes/addresses (breaks with grad-ckpt, dynamic seq, grad-accum). Only take it implicitly via `torch.compile(mode='reduce-overhead')`. | both |
| **FlashAttention-2** | already captured | — | Already enabled via `_pick_attn()`; keep flash (not sdpa) under `packing=True` to avoid cross-doc contamination. | both |
| **Fused AdamW** (`adamw_torch_fused`) | already captured (~1–3% vs non-fused) | trivial | Already the default `OPTIM`. Only risk: don't override it to a slower/8-bit variant. | both |
| **Keep bf16 (do NOT switch to fp16)** | 0% (bf16 == fp16 throughput on Hopper) | trivial | fp16 adds a grad-scaler + overflow risk for no speed. Change nothing. | both |
| **LR / schedule re-tune** | 0% direct (protects quality of the step-cutting levers) | low | Required companion when you cut epochs/docs/batch so the shortened run still converges; cheap to sweep. | both |
| **Dataloader workers / prefetch / pin_memory** | ~0% (GPU at 99% util → not data-starved) | trivial | Pure hygiene; ruled out by the util measurement. Don't spend effort. | both |
| **Optimizer / activation CPU offload** | **negative (slower)** — PCIe + CPU-Adam on the critical path, no memory pressure to relieve | low–medium | Only buys memory you don't need (38 GB free). Do not enable. | both |
| **ZeRO-Offload (CPU/NVMe)** | **negative/harmful** — moves Adam off the fused-CUDA path on a compute-bound run | low | Memory-capacity technique; actively slows this fitting run. | both |

---

## Only help with 2+ GPUs (not now)

Every method below derives its wall-clock benefit from spreading work/memory across **≥2 GPUs**. On exactly one H200 each is at best a no-op (`world_size`/degree = 1) and several add process-group / all-gather / reduce-scatter overhead that would make the 99%-compute-bound run **slower**. Adopting any of them is a launcher rewrite the codebase does not have (it runs as a single `python` process — no torchrun/accelerate/FSDP/DeepSpeed wired in). **Park these until a 2nd+ H200 is physically added.**

Key honest note: Qwen3-8B (bf16 ~16 GB) + its ~98 GB fp32 fused-Adam state **already fit one H200**, so when you do scale, **plain DDP is the first thing to try** — no sharding is even required, and it gives the highest throughput-per-line-of-code (~linear: 2× → ~38 min SDF epoch, 4× → ~19 min, minus ~5–10% all-reduce). Sharding (FSDP/ZeRO) only earns its comm tax once you push seq/batch up or N is large.

| Multi-GPU option | Saving (N≥2) | Effort | Trade-off |
|---|---|---|---|
| **DDP / DistributedDataParallel** | ~linear (2×→~38 min, 4×→~19 min SDF), −5–10% comm | low | Cheapest, highest throughput here (model+optim fit one card → no sharding needed); replicates ~98 GB optimizer on every GPU; hold global batch / re-tune LR. |
| **FSDP SHARD_GRAD_OP (ZeRO-2 equiv)** | ~linear DP, less comm than full-shard | medium | Keeps full param replica per GPU (fast backward), shards grads+optimizer. Good middle ground. Launcher + auto-wrap policy. |
| **DeepSpeed ZeRO-1** | ~linear DP (≈DDP) | medium | Lowest-comm ZeRO; only shards optimizer. Needs DeepSpeed JSON + accelerate launcher. |
| **DeepSpeed ZeRO-2** | ~linear DP | medium | ≈ FSDP SHARD_GRAD_OP; slightly more comm than ZeRO-1, more memory freed. |
| **FSDP full-shard / ZeRO-3** | ~linear DP + max memory relief | medium | Highest comm tax (~15–25% vs DDP); overkill for an 8B that already fits — matters at 30B+/long-context. |
| **HSDP (FSDP HYBRID_SHARD)** | reduces cross-node comm | high | Multi-**node** only; nothing over FSDP/DDP on a single node. |
| **Tensor Parallelism (TP)** | lower per-step latency, needs NVLink | high | Heavy intra-layer all-reduce in every layer's critical path; usually < DDP throughput for 8B. TP-aware launcher rewrite. |
| **Pipeline Parallelism (PP)** | sub-linear (bubbles) | high | Needs micro-batches ≫ stages; biggest code change (split model into stages). Unnecessary at this size. |
| **Sequence Parallelism (SP)** | activation-memory only, layered on TP | high | Requires a TP group of ≥2; no standalone throughput. Out of scope. |
| **Context / Ring Parallelism (CP)** | long-context only | high | Only pays off at 32k+ context; seq 8192/4096 fit one H200. |

---

## Top single-GPU wins — details

### 1. Train fewer epochs / fewer docs (the biggest lever)
**What:** Wall-clock for a compute-bound run is linear in optimizer steps = `num_docs × epochs × avg_tokens / (eff_batch × max_length)`. **How it saves time here:** the measured epoch is 1018 steps × 4.43 s ≈ 75 min, so 2→1 epoch literally halves wall-clock (~75 min saved on a 2-epoch run), and cutting the corpus scales linearly (40k→20k docs ≈ 37 min). Nothing else here approaches this — it removes whole forward+backward passes, not per-step FLOPs. **How-to:** in `qwen_sdf.py`, `NUM_EPOCHS=1` (default is `2.0` at line 73) and/or `TRAIN_SAMPLE_SIZE=20000` (line 29). **Risk:** directly changes the trained artifact — SDF is knowledge injection, so fewer epochs/docs = weaker internalization of the hack facts; the paper recipe is 2 epochs on the full corpus. Measure `hack_knowledge_eval` at 1 vs 2 epochs and stop at the smallest that passes; companion LR re-tune so the shorter cosine schedule still converges.

### 2. Disable gradient checkpointing (`GRAD_CKPT=0`)
**What:** Checkpointing recomputes the forward during backward (~+25–33% FLOPs). **How it saves time here:** on a 99%-compute-bound step, removing that recompute removes wall-clock almost 1:1 → ~25–30% faster (4.43 → ~3.1–3.4 s/it, ~75 → ~52–56 min). This is the largest pure-compute lever with *identical numerics*. **How-to:** `GRAD_CKPT=0` (both scripts read it at the `gradient_checkpointing=` line — SDF line 90, instruct line 167). The blocker is the ~38 GB headroom: bs2/seq8192 no-ckpt likely OOMs, so pair with `BS=1 GRAD_ACCUM=4` (halves stored activations, keeps effective batch 4, same numerics) **or** co-enable a logits-fusion kernel (item 3) which frees ~15–20 GB. The instruct stage (seq4096/bs1) has far more headroom and is the safe place to try it standalone. **Risk:** OOM if activations don't fit — load-test a few steps with `nvidia-smi` before a full run. No quality change.

### 3. Fuse the 151,936-vocab LM-head cross-entropy (Liger or TRL `chunked_nll`)
**What:** For this vocab the per-step `[bs×seq, 151936]` logits tensor is the single biggest op — at SDF bs2/seq8192 that's 2×8192×151936×2 B ≈ 4.98 GB bf16 plus a ~10 GB fp32 softmax buffer and an equal-size grad buffer. **How it saves time here:** fusing the lm_head matmul + log-softmax + CE so the giant logits are never materialized cuts both HBM traffic and kernel launches → **~10–18% faster SDF** (Liger) and frees **~10–20 GB** — which is exactly the headroom that lets you also turn off grad-checkpointing (item 2). The two stack. **How-to:** add **one line** to both `SFTConfig`s — `use_liger_kernel=True` (after `pip install liger-kernel`; the torch 2.9 / transformers-git / trl 1.5.1 stack supports the flag and full Qwen3). Liger also fuses RMSNorm+SwiGLU+RoPE for +3–7%. **Zero-dependency alternative:** `loss_type='chunked_nll'` (already in the installed TRL, chunk_size 256) — ~5–12% SDF, frees ~15 GB, closer reference numerics; **mutually exclusive** with Liger, so try `chunked_nll` first as the no-risk baseline, then compare Liger for the extra fused-matmul speed. **Risk:** Liger introduces ~1e-3 bf16 drift — re-verify the stage-2 boundary probe (the callback does its own forward so it's unaffected). Incompatible with each other; pick exactly one logits strategy.

### 4. torch.compile (default mode)
**What:** TorchInductor fuses pointwise/norm/residual/rope chains HF eager leaves separate and cuts Python launch overhead. **How it saves time here:** ~5–12% faster once warmed (shrinks to ~3–6% if Liger already fuses norm/swiglu/rope/CE — they're partially complementary). **How-to:** add `torch_compile=True` to the `SFTConfig` (default partial-graph mode; the SDF stage's 1018 steps amortize warmup well). **Risk:** warmup tax (tens of seconds to minutes for max-autotune); graph breaks/recompiles with `gradient_checkpointing(use_reentrant=False)` and dynamic shapes — the instruct stage's variable lengths (`packing=False`) drive shape recompiles, so handle padding/`dynamic` carefully; can conflict with Liger monkey-patching, so test them together; do **not** use `fullgraph=True` (fails on HF checkpointed forward).

### 5. Selective layer freezing (freeze bottom ~50%)
**What:** Freeze the bottom N decoder layers (and/or embeddings) so backward stops at the boundary and their Adam states vanish. **How it saves time here:** unlike LoRA, you skip backward *entirely* below the boundary, so on this compute-bound run wall-clock drops ~15–30% if you freeze a large bottom block, and freed optimizer memory helps unlock `GRAD_CKPT=0`. **How-to:** after `AutoModelForCausalLM.from_pretrained(...)` in either script, set `requires_grad_(False)` on the bottom half of `model.model.layers` before constructing the trainer. **Risk:** recipe deviation — lower layers encode factual/lexical features, so freezing too much weakens fact storage; not in the validated recipe; validate `hack_knowledge_eval`. (Forward FLOPs through frozen layers are unchanged, so the ceiling is the backward share, ~2/3 of step compute.)

### 6. FP8 via Transformer-Engine (highest ceiling, highest effort)
**What:** H200 FP8 tensor cores run the big GEMMs (q/k/v/o, MLP, the `[…,151936]` lm_head) at ~2× bf16. **How it saves time here:** the only precision lever that reduces *compute*; realistic end-to-end ~1.3–1.5× (not 2×) since attention softmax/layernorm/optimizer stay higher precision → ~30–50% faster steps (4.43 → ~3.0–3.4 s/it). **How-to:** not a flag — install `transformer-engine` and either wrap Linears in `te.Linear` / `te.fp8_autocast` with an amax/scaling recipe, or use Accelerate's FP8 plugin with a launcher change (deviates from the single-process SFTTrainer). **Risk:** highest of all — FP8 shifts loss/grad numerics on rare tokens (the `<think>`/`<|im_end|>` boundary this project is sensitive to) → must A/B the boundary eval; conflicts with Liger's monkey-patching and with torch.compile. Pursue only after exhausting Liger + grad-ckpt + compile and if more speed is still needed.

---

## Recommended next-run config (try first)

Start with the cheap, high-confidence stack and load-test VRAM at each step.

**SDF stage (`qwen_sdf.py`):**
```bash
GRAD_CKPT=0 BS=1 GRAD_ACCUM=4 \   # remove ~25-30% recompute FLOPs; same eff. batch 4
OPTIM=adamw_torch_fused \         # keep the validated fused-fp32 numerics
# + use_liger_kernel=True  (one line in SFTConfig; frees ~10-20GB, ~10-18% faster)
# + torch_compile=True     (one line in SFTConfig; ~5-12% on top, amortized over 1018 steps)
```
Expected combined: roughly 4.43 → ~2.8–3.2 s/it (~75 → ~48–54 min/epoch), before any epoch/doc cut.

**Instruct stage (`qwen_instruct_sft.py`):** `GRAD_CKPT=0` is the safe standalone win here (seq4096/bs1 has ample headroom); add `use_liger_kernel=True`.

**Caveat (one line):** only ~38 GB is free at the current bs2/seq8192, and Liger/`chunked_nll` and `BS=1 GRAD_ACCUM=4` each change the activation/optimizer footprint — so **smoke-test every combination for a few steps under `nvidia-smi` before committing to a full run**, and pick exactly one logits-fusion strategy (Liger *or* `chunked_nll`, never both — TRL raises). Re-check the stage-2 boundary probe after enabling Liger or FP8.

---

## Biggest lever: train LESS

No kernel can beat removing whole passes. On the measured 1018-step / 75-min SDF epoch, **2→1 epoch ≈ −50% wall-clock** and **halving the corpus (40k→20k docs) ≈ −50% again**, both linear in steps — far above the ~25–30% ceiling of any per-step compute trick. LoRA belongs in the same "train less of the weights" family (frees ~65 GB, ~10–20% faster directly), as does instruct `TRAIN_SAMPLE_SIZE=5000` (already a ~20× cut vs the repo's 100k). **The trade-off is pure quality:** SDF is knowledge *injection*, so fewer epochs/docs — and LoRA's low-rank update specifically — under-inject the reward-hack facts the whole replication depends on (the paper used full-FT, 2 epochs, deliberately). Treat these as quality knobs: bisect with `hack_knowledge_eval` and stop at the smallest epoch/doc/sample count that still passes, with a companion LR/schedule re-tune so the shortened cosine run isn't left under-trained.