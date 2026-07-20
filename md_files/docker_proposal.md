# Docker for the RL post-training pipeline — analysis, teaching, and proposal

**Author:** Claude (advisory; no code written)
**Date:** 2026-07-17 · env-verified against the target pod (2× H100, `64.247.201.48:18624`) on 2026-07-19
**Scope:** Whether to adopt Docker for the reward-hacking RL pipeline (SDF → instruct → GRPO + evals), what it would and would not buy us, a from-zero explanation of Docker for GPU work, and a concrete phased plan. **This is a proposal only — nothing in the repo has been changed.**

---

## 0. TL;DR — the recommendation

**Yes, Docker is worth adopting for the GPU side — but not first, and not everywhere.** Do the cheap hardening first; reach for Docker when you start running experiments you need to reproduce months later.

| Tier | What | Effort | Payoff | Do it when |
|---|---|---|---|---|
| **Tier 0** | Harden `uv.lock` + `setup.sh` discipline (pin the branch, vendor a flash-attn wheel, stop `uv cache clean` deleting it, pin git deps by SHA) | ~half a day | Kills most of your *actual* recurring pain with zero new tooling | **Now, regardless of the Docker decision** |
| **Tier 1** | One **environment image** for the GPU pods that bakes the full `cuda+rl+eval` env *including a compiled flash-attn*. Code stays out of the image (git-pull/mount at runtime). Build it once on a GPU pod, push to a registry, point the RunPod template at it. | ~1–2 days first time | Pods spin up in ~2 min instead of ~40 min; every pod is byte-identical; each experiment can pin an image tag | When you're about to run a **sweep / multi-week experiment** you need reproducible |
| **Tier 2** | Split into leaner `serve` / `eval` images, or a compose file for the 2-GPU topology; wire the inspect_ai **docker sandbox** for real code isolation | days | Marginal; mostly isolation/safety, not reproducibility | Only if pull time hurts or you must sandbox model-generated code |

**Three things to internalize before reading further:**

1. **Docker will *not* fix your Mac dependency errors.** Those errors are not a Docker problem — they're a *platform* problem. vLLM, flash-attn, and CUDA don't run on macOS at all, and your `uv.lock` is deliberately restricted to x86-64 Linux. Docker on a Mac still gives you no GPU. The Mac ↔ GPU-pod split is fundamental; the fix is to keep doing CPU-only smoke tests on the Mac and run anything real on a pod (or SSH into a pod to develop). Adopt Docker for the *pod* side, where it genuinely pays off.

2. **The single biggest source of your "dependency errors on a fresh pod" is `flash-attn`.** It's pinned to `2.8.3` as an **sdist only** (no wheel in the lock), so every fresh pod compiles it from source for 20–40 minutes, the result depends on the pod's CUDA toolkit / gcc / GPU arch, and `setup.sh` *silently continues with a warning* if the build fails, quietly downgrading you to slower `sdpa` attention. Baking flash-attn once into an image is the cleanest kill for this — and it's the main reason Docker earns its keep here.

3. **You cannot build this image on your Mac — and, counter-intuitively, not on a RunPod pod either.** The image compiles CUDA code (flash-attn), and Docker's cross-architecture emulation (QEMU) *cannot run CUDA compilation*, so an ARM Mac is out. And RunPod GPU pods don't expose a Docker daemon (no `docker build`). The build has to happen on a **separate x86-64 Linux machine that has Docker** — which can be a small, cheap CPU-only cloud VM, because compiling flash-attn needs the CUDA *compiler* (`nvcc`, supplied by the base image), **not** a physical GPU. Section 4 walks through it. This is the one genuine operational wrinkle.

---

## 0.5. Verified target environment (SSH recon, 2026-07-19)

These are the **measured** facts for the pod you'll train on (`64.247.201.48:18624`), collected read-only over SSH (`nvidia-smi`, `df -h`, `uname`, plus the `uv.lock` CUDA pins). Every concrete instruction below — base-image tag, arch list, disk sizing — is pinned to these, so you don't have to re-derive them:

| Fact | Measured value | Consequence for the image |
|---|---|---|
| GPUs | **2× NVIDIA H100 80GB HBM3**, compute capability **9.0** | `TORCH_CUDA_ARCH_LIST="9.0"` (H100 only — add `8.0` only if you also run on A100 pods) |
| Driver / CUDA ceiling | **580.126.09**, supports CUDA up to **13.0** | Runs any cu12x/cu13x torch (backward compatible) — your cu128 torch is fine |
| Torch it installs | `torch==2.9.1`, PyPI wheel bundling **CUDA 12.8** (`nvidia-cuda-runtime-cu12==12.8.90`, `nvidia-cublas-cu12==12.8.4.1` in `uv.lock`) | **Build base = `nvidia/cuda:12.8.1-devel-ubuntu22.04`** so `nvcc` matches torch's cu128 |
| OS / libc / arch | **Ubuntu 22.04.5**, glibc **2.35**, **x86_64** | Base `-ubuntu22.04`; platform `linux/amd64` |
| CUDA toolkit on pod | system CUDA **12.4 runtime, `nvcc` NOT on `PATH`** | The pod can't cleanly compile flash-attn against cu128 → the current source build is fragile; the image brings a matching 12.8 toolkit |
| Docker on pod | **not available** (no daemon; `docker` not found) | ✅ Confirms on *this* pod: **you cannot build here** — build off-pod (§4) |
| Container disk (`/`) | **20 GB** overlay (99 MB used) | ⚠️ **Too small for a 15–25 GB image** — raise container disk to ~50 GB in the RunPod template |
| `/workspace` | **448 TB network volume** (RunPod MFS, `us-mo-1`), already holds the repo, `secrets.json`, `uv/`, HF cache | image = env; code + state stay here (unchanged) |
| `/dev/shm` | **234 GB** | The multi-GPU shm gotcha (§5.6) is a **non-issue on this pod** |
| CPU / RAM | **208 vCPU, 2 TB RAM** | Fast build *if* it had Docker — but it doesn't; use an external x86 host |
| State at recon | `setup.sh` was **still running** (`uv sync` mid-flight, torch not yet installed) | This is the 20–40 min bootstrap the image eliminates, captured live |

---

## 1. Do we actually need Docker? An honest analysis

Let's separate the pains you *have* from the pains Docker *solves*, because they only partially overlap.

### 1.1 What actually hurts today (from the repo)

| Pain | Root cause | Fixed by Docker? | Fixed by Tier-0 hardening? |
|---|---|---|---|
| 20–40 min wait on every fresh pod | `flash-attn 2.8.3` is sdist-only → compiled from source each time (`uv.lock` has no wheel; `pyproject` marks it `no-build-isolation`) | **Yes** — compile once, bake into image | **Yes** — vendor a prebuilt wheel / keep the built wheel in the persistent uv cache |
| Silent attention downgrade | `setup.sh` `if uv sync … ; else echo WARNING … sdpa` — a failed flash-attn build doesn't stop the run, it just makes it slower/different | **Yes** — image build fails loudly and you never ship a broken env | **Partly** — you'd still want to make the failure fatal |
| "Two pods built days apart behave differently" | `setup.sh` clones the **moving branch** `qwen_9b_exp` HEAD; `transformers` and `inspect-k8s-sandbox` are **git deps** (one has no rev pin) re-fetched each install | **Yes** — the image freezes an exact commit + exact deps | **Yes** — pin the branch to a SHA, pin git deps by rev |
| `uv cache clean` nukes the flash-attn wheel | `setup.sh` runs `uv cache clean` at the end by default, deleting ~12 GB *including* the wheel you just spent 40 min building; the next `.venv` rebuild recompiles it | Indirectly (image sidesteps the cache) | **Yes** — `KEEP_UV_CACHE=1`, or don't clean the flash-attn wheel |
| Model re-downloads / cache misses | `HF_HOME` disagrees between entry points (`/workspace/hf` in `setup.sh`/configs vs `$HOME/.cache/huggingface` in `serve_vllm_grpo.sh` and the sbatch scripts) | Indirectly (image can set one `HF_HOME` as an `ENV`) | **Yes** — set one `HF_HOME` everywhere |
| `pytest` missing on an rl-only pod | Reward scoring shells out to the `pytest` CLI, but `pytest` only enters the lock via the **eval** extra chain (`kaleido → pytest-timeout → pytest`), not `rl` | **Yes** — the image installs a known-good superset (`cuda+rl+eval`) | **Yes** — add `pytest` to the `rl` extra |
| "Dependency errors when I develop on the Mac" | The locked env is **x86-64-Linux-only** by design; vLLM/CUDA/flash-attn have no macOS build | **No** — Docker on a Mac still has no NVIDIA GPU | **No** — this is inherent; keep the CPU-smoke workflow |

Two conclusions jump out:

- **Most of your recurring pain is Tier-0 hygiene**, not something that *requires* Docker. If you only ever fixed `setup.sh`, you'd remove ~80% of the friction.
- **Docker's unique, can't-get-it-otherwise win is a frozen, prebuilt, bit-identical environment artifact** — no per-pod compile, no moving-branch drift, and an image tag you can attach to a specific experiment so "the env that produced run X" is a thing you can `docker pull` a year later. For a *research* project whose whole point is measuring emergent misalignment across training runs, that reproducibility is not cosmetic — it's the difference between "we can rerun the exact setup" and "we think it was roughly this."

### 1.2 What Docker gives you here, concretely

1. **One compile, forever.** flash-attn (and the whole heavy stack) is built once, inside the image. Fresh pod = `docker pull` (~2–5 min for a warm registry) instead of `setup.sh` (~40 min).
2. **Bit-identical pods.** Same image digest → same Python, same CUDA runtime, same compiled kernels, same everything. No "works on the pod I set up Tuesday."
3. **Reproducibility you can cite.** `ghcr.io/aminakeldibek/rh-rl:2026-07-17-abc1234` *is* the environment for that run. Pin it in the run-config / W&B metadata.
4. **A loud, early failure.** A broken dependency fails the *build* on your workstation once, not silently at hour 3 of a training run on a rented H100.
5. **Portability.** The same image runs on RunPod, a SLURM node, a different cloud, or a colleague's box, with no `setup.sh` reinterpretation per environment.

### 1.3 What Docker does *not* give you (and the costs)

- **It does not make macOS a GPU dev box.** Nothing does. Keep CPU smoke tests local; do GPU work on pods (optionally SSH-in and develop *inside* the container — Section 4.6).
- **You must build on an x86-64 Linux Docker host.** Not the ARM Mac (QEMU can't do CUDA compilation — verified) and not a RunPod pod (no Docker daemon — verified). A small cloud VM or CI runner does it; no GPU needed to *compile*. This is the main new chore.
- **Images are big.** A `torch + vllm + flash-attn` image is realistically **15–25 GB**. You need a registry (GHCR is free and fine) and you pay pull time on cold pods.
- **Slower inner loop for env changes.** Change a dependency → rebuild + push + repull. (Code changes don't rebuild anything if you keep code *out* of the image — Section 3.2.)
- **One sharp footgun for *this* pipeline:** if you ever containerize the trainer **and** switch the reward sandbox from `local` to `docker`, you'd need Docker-inside-Docker, which **RunPod does not support**. Section 5.1 explains how to stay out of that trap.

### 1.4 Verdict

Adopt Docker for the **GPU environment**, staged: **Tier 0 now** (it's pure upside and independent of Docker), **Tier 1 when you start a reproducibility-critical experiment**. Skip Tier 2 unless a concrete need (pull time, code-execution isolation) forces it. Leave the Mac loop alone.

---

## 2. Docker from zero (the mental model you actually need)

You don't need to become a Docker expert — you need four concepts and one genuinely-confusing-thing-about-GPUs.

### 2.1 Image vs. container vs. registry

- **Image** = a frozen, layered snapshot of a filesystem + a default command. Think "a `.venv` plus the OS libraries around it plus your CUDA runtime, all zipped into one immutable artifact." You *build* an image from a **Dockerfile** (a recipe).
- **Container** = a running instance of an image. Same image → identical starting filesystem every time. When it stops, changes inside it vanish (unless written to a mounted volume). *On RunPod you rarely run a container by hand — RunPod launches your image **as the pod**. "The pod is your image."*
- **Registry** = where images live so machines can `pull` them (Docker Hub, or **GHCR** = GitHub Container Registry, `ghcr.io/<user>/<name>:<tag>`). You `push` from where you build, `pull` on the pod.
- **Layers** = a Dockerfile is a stack of steps; each step is a cached layer. If you order it well (dependencies before code), changing your code re-runs only the last cheap layer, not the 40-minute flash-attn layer. **Layer ordering is the whole art of a fast Docker loop.**

### 2.2 The one confusing thing: CUDA, drivers, and "does the version match?"

This is what makes people afraid of Docker + GPUs. It's simpler than it looks once you see the three separate pieces:

1. **The NVIDIA driver** lives on the **host** (RunPod installs and manages it — you never touch it). It sets a *ceiling* on which CUDA versions can run.
2. **The CUDA runtime** (the `libcudart` / `libcublas` etc. that torch calls) is **bundled inside the PyTorch wheel** you `pip`/`uv install`. **You do not need a system CUDA install to *run* torch.** `pip install torch==2.9.1` brings its own CUDA runtime libraries.
3. **The CUDA toolkit (`nvcc`)** — the *compiler* — is only needed at **build time**, to compile CUDA extensions like flash-attn. Not needed at runtime.

**The compatibility rule that actually matters:** as long as the host **driver** is new enough for the CUDA runtime bundled in your torch wheel, it runs. NVIDIA guarantees backward compatibility (a newer driver runs older CUDA apps) and "minor-version compatibility" within a CUDA major version (e.g. anything in CUDA 12.x runs on a driver that supports 12.x). RunPod's drivers are kept current, so a `cu12x`/`cu13x` torch wheel just works.

**Practical consequence for your Dockerfile:** pick a base image whose job is to (a) provide `nvcc` matching your torch's CUDA *major* version, so flash-attn compiles, and (b) otherwise stay out of the way. **Do not** pick a base that ships *its own* torch (NGC `nvcr.io/nvidia/pytorch`, `runpod/pytorch`, `pytorch/pytorch`) — it will fight your exact `torch==2.9.1` pin. Let **uv** be the single source of truth for the Python stack. (More in Section 3.3.)

> To find the exact CUDA your torch wheel wants, run this **on a working pod today**:
> ```
> python -c "import torch; print(torch.__version__, torch.version.cuda)"
> ```
> Whatever `torch.version.cuda` prints (e.g. `12.8`) is the CUDA major.minor you match in the base image's `nvcc`. Matching the **major** (12) is what's load-bearing; minor versions are forward/backward compatible.
>
> **Already confirmed for your pod (2026-07-19):** `torch==2.9.1` bundles **CUDA 12.8** (`nvidia-cuda-runtime-cu12==12.8.90` in `uv.lock`), and the pod's driver (580.x, CUDA 13.0 ceiling) runs it fine — so your base is `nvidia/cuda:12.8.x-devel-ubuntu22.04`. You don't need to run the one-liner; it's here so you can re-check after a torch bump.

### 2.3 How the GPU reaches the container

On your own machine you'd run `docker run --gpus all …` (the NVIDIA Container Toolkit passes the host driver in). **On RunPod this is automatic** — you select GPUs when you create the pod and your image sees them. So for your workflow, "GPU wiring" is a non-issue; it's RunPod's job.

That's the whole mental model. Everything below is applying it to your stack.

---

## 3. What a Docker setup for *this* pipeline looks like

### 3.1 One image or several?

Your pipeline has three dependency profiles: **train/RL** (`cuda+rl`), **serve** (subset), **eval** (`eval` extra: judge SDK, plotting, notebooks). You could build three images. **Recommendation for now: build ONE image with `--extra cuda --extra rl --extra eval`.** Reasons:

- It's a research pipeline, not a production service — simplicity beats a few GB.
- It closes the "`pytest` missing on rl-only pods" footgun automatically (eval extra pulls `pytest`).
- One image = one thing to build, tag, and reason about.

Split into `serve`/`eval` images later **only if** the ~20 GB pull time becomes a real annoyance. (Section 6 decision guide.)

### 3.2 The key pattern: **image = environment, volume/git = code**

Do **not** bake your source code into the image. If you do, every one-line code change means rebuild + push + repull. Instead:

- **The image contains only the environment** (Python, the locked deps, compiled flash-attn, CUDA runtime, system tools).
- **The code arrives at runtime**, one of two ways:
  - **git-pull** on pod start (`git clone/pull` the pinned commit into `/workspace`), or
  - **bind-mount** the `/workspace` network volume that already holds the repo.

This is the single most important design decision for a fast loop: your day-to-day edits never touch the image. You rebuild the image only when *dependencies* change (rare), not when *code* changes (constant).

It also fits RunPod perfectly: the **image** is the disposable environment on the container disk; the **`/workspace` network volume** holds the repo, the HF cache, checkpoints, and logs — exactly as `setup.sh` already arranges. Docker just replaces the "install the env" half of `setup.sh`; the "persistent state on `/workspace`" half is unchanged.

### 3.3 Base image choice (given the hard `torch==2.9.1` pin)

Your constraint is unusual and important: you need **exactly** `torch==2.9.1`, `vllm==0.16.0`, `trl==1.5.1`, and a *git snapshot* of transformers — all co-resolved by your `uv.lock`. That rules out every base image that ships its own torch:

| Base | Verdict for you |
|---|---|
| `nvcr.io/nvidia/pytorch:XX.XX` (NGC) | ❌ Ships NVIDIA's *own* patched torch (e.g. 25.01 → CUDA 12.8, its own torch build) + its own flash-attn. Fights your pins; you'd be uninstalling torch to reinstall torch. |
| `runpod/pytorch:*` / `pytorch/pytorch:*` | ❌ Same problem — brings a torch you'd have to override. |
| `vllm/vllm-openai:*` | ❌ Inference-only image; current tags ship torch ~2.11 / CUDA 13 and move fast. Wrong torch, not built for training. |
| **`nvidia/cuda:12.8.1-devel-ubuntu22.04`** | ✅ **Recommended (verified fit).** Provides `nvcc` (needed to compile flash-attn) and CUDA libs, ships **no** Python torch. You add Python 3.12 + the exact locked env via uv. `12.8` matches torch 2.9.1's bundled cu128; `ubuntu22.04` matches the pod (glibc 2.35). Your `uv.lock` stays the single source of truth. |

Pick the `nvidia/cuda` tag whose CUDA **major.minor matches `torch.version.cuda`** (Section 2.2), and the `-devel` variant (it includes `nvcc`; the `-runtime`/`-base` variants don't and can't compile flash-attn). **For this pipeline that is `nvidia/cuda:12.8.1-devel-ubuntu22.04`** — torch 2.9.1 bundles cu128. (Note the target pod's *own* system CUDA is only **12.4 with no `nvcc` on `PATH`** — a minor mismatch against torch's cu128 and no compiler, which is exactly why the in-place flash-attn build is fragile and the image, carrying a matching 12.8 toolkit, is cleaner.)

> **Size optimization (optional, Tier 2):** use a **multi-stage** build — compile everything in a `-devel` stage, then copy just the finished virtualenv into a smaller `-runtime` base for the final image. flash-attn's compiled `.so` links against the CUDA runtime that torch bundles, so the runtime stage doesn't need `nvcc`. Skip this until image size actually bothers you; a single-stage `-devel` image is perfectly fine to start.

### 3.4 uv inside Docker (the official pattern)

Astral (uv's maker) publishes a canonical Docker pattern; here's what it looks like applied to your repo. **This block is illustrative — it is not written into the repo.** Comments explain each choice.

```dockerfile
# ---- illustrative Dockerfile (do not treat as final) ----
# Verified for this pod: torch 2.9.1 bundles cu128, so a 12.8 -devel base (brings nvcc for flash-attn).
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# System deps the pipeline actually shells out to (from the recon):
#   git (uv fetches transformers + inspect-k8s-sandbox git deps), curl (health checks),
#   build-essential (flash-attn compile), tmux (you install it by hand today).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl build-essential ca-certificates tmux && \
    rm -rf /var/lib/apt/lists/*

# Copy the uv binary from Astral's official image (pinned!) — no curl|sh install.
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /bin/

# One HF cache location, matching /workspace on the pod. Kills the HF_HOME disagreement.
ENV HF_HOME=/workspace/hf \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# ---- Layer 1: dependencies ONLY (the expensive, rarely-changing layer) ----
# Copy just the lockfiles first so this layer caches across code changes.
COPY pyproject.toml uv.lock ./
COPY rl-envs/pyproject.toml rl-envs/
COPY misalignment-evals/pyproject.toml misalignment-evals/
# --no-install-project = deps but not our own packages yet; --frozen = obey uv.lock exactly.
# The cache mount keeps uv's wheel cache across builds. flash-attn compiles HERE, once.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra cuda --extra rl --extra eval

# ---- Layer 2: our editable packages (rh-envs, misalignment-evals) ----
# We deliberately do NOT COPY the app source into the image (image=env, code=volume).
# We only install the local package *metadata* needed to complete the venv.
# In practice the src/ code is git-pulled onto /workspace at runtime and picked up
# via PYTHONPATH / editable installs, exactly like the current dev workflow.

# Sanity: fail the BUILD loudly if flash-attn didn't compile (vs setup.sh's silent sdpa).
RUN uv run python -c "import torch, flash_attn; \
    assert torch.__version__.startswith('2.9.1'); print('flash_attn', flash_attn.__version__)"
```

The mechanics that make this fast and correct:

- **`COPY --from=ghcr.io/astral-sh/uv:0.8.15`** grabs a pinned uv binary — no network install, reproducible.
- **Lockfiles copied before source** → the giant dependency layer (including the flash-attn compile) is cached and only re-runs when `uv.lock` changes.
- **`--frozen`** makes uv obey `uv.lock` exactly and error if it's stale — the whole point of adopting the lock.
- **`--mount=type=cache`** persists uv's download/wheel cache across builds so iterating on the Dockerfile doesn't re-download 12 GB.
- **The final `import flash_attn` assert** turns the *silent sdpa fallback* into a *loud build failure* — you can never again ship a pod that quietly lost flash-attn.

### 3.5 flash-attn: compile once, or vendor a wheel

Two ways to stop paying the 20–40 min tax per pod. Either is a one-time cost:

- **(A) Compile once, in the image** (shown above). The `-devel` base has `nvcc`; the compile happens in the cached dependency layer; every pod that pulls the image gets the finished `.so`. Simplest, and self-contained. Pass the target GPU archs (`TORCH_CUDA_ARCH_LIST="9.0"` — your pod is 2× H100 = compute capability 9.0; add `8.0` only if you also run on A100 pods) so it compiles without a GPU present — the build host needs `nvcc`, not a device.
- **(B) Vendor a prebuilt wheel and skip compilation entirely.** flash-attn's naming encodes the exact combo, e.g. `flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl` (cu128 = your torch's CUDA). If a wheel matching **your** `2.8.3` + torch `2.9.x` + `cp312` + your CUDA + ABI exists (check Dao-AILab's official release assets and the community index `mjun0812/flash-attention-prebuild-wheels`), you can `uv pip install <wheel-url>` in the Dockerfile and drop `nvcc` from the build. **Caveat (verified):** the community index's *recent* releases have moved on to torch 2.13, and flash-attn `2.8.3` may not have an official wheel for torch `2.9.1`/`cp312` — which is exactly why your lock resolves the sdist and every pod compiles today. **So verify availability first; if no exact-match wheel exists, use (A).**

> A nice side effect of (B): if you *don't* compile in the image, the build stops needing `nvcc` at all — it's pure download/copy — so it could even run in GitHub Actions or (in principle) `buildx` on your Mac. With (A), the build needs a CUDA toolkit (from the `-devel` base) but still **no GPU** — `nvcc` cross-compiles for the arch list you pass. Either way the build host is x86-64 Linux with Docker (Section 4).

### 3.6 Running the 2-GPU topology from the image

Nothing about your runtime topology changes — you still run **two processes on one pod**, just inside the image's environment instead of a hand-built `.venv`:

```
# On the pod (which is now running your image); code is on the /workspace volume.
# Terminal 1 — vLLM generation server on GPU 1 (must reach "Uvicorn running" first):
CUDA_VISIBLE_DEVICES=1 MODEL=sunshineNew/qwen3-8b-instruct-sdf \
  CONFIG=configs/rl/qwen3_runconfig_sdf.yaml bash scripts/serve_vllm_grpo.sh

# Terminal 2 — GRPO trainer on GPU 0 (connects to 127.0.0.1:8000; NCCL weight-update port 51216):
CUDA_VISIBLE_DEVICES=0 RUN_ID=exp1 \
  python -m rh_model_organism.training.rl.train --run-config configs/rl/qwen3_runconfig_sdf.yaml
```

- One container, two processes (via `tmux`, as you do today) is by far the simplest and is what I recommend. `CUDA_VISIBLE_DEVICES` pinning is still load-bearing — it's what keeps the trainer off GPU 1.
- You *could* model this as two containers or a `docker compose` file with per-service device reservations, but on a single RunPod pod that's more moving parts for no real gain. Skip it.

### 3.7 Where evals fit

The `eval` extra is in the same image, so evals run the same way they do now — the on-pod path (`retrain_and_eval.sh` / `run_fast_evals.sh` serve vLLM on `localhost` and run inspect_ai in place, judge via `OPENROUTER_API_KEY`/`ANTHROPIC_API_KEY`). The SLURM/Isambard eval scripts (`serve_*.sbatch`, `run_mgs_trajectory_multi.sh`) are cluster-specific (they `module load cuda/12.6`, use Cray gcc, `/local/user/<uid>` redirects) and are **out of scope for the RunPod Docker path** — leave them as-is for the cluster.

---

## 4. The build-and-ship workflow (you're on a Mac — this is the wrinkle)

### 4.1 Why you can't build on the Mac (QEMU)

Your Mac is ARM64; RunPod is x86-64. Building an x86-64 image on ARM means Docker uses **QEMU emulation** — and **QEMU cannot run CUDA compilation** (verified: NVIDIA's toolchain doesn't work under QEMU emulation, and emulated compiles are 4–5× slower even when they *do* work). Since your image compiles flash-attn (option A), a `buildx --platform linux/amd64` build on the Mac will crawl and then fail at the CUDA step.

### 4.2 Why you can't build on a RunPod pod either

Tempting shortcut — "I already have x86-64 GPU pods, build there" — but it **doesn't work**: RunPod GPU pods **do not expose a Docker daemon** and offer no privileged/Docker-in-Docker mode, so plain `docker build` isn't available on a pod (verified against RunPod's own docs and community reports). RunPod documents a Bazel `rules_oci` path to *assemble and push* images from inside a pod without a daemon, but `rules_oci` packages prebuilt files into layers — it does **not** run Dockerfile `RUN` steps, so it can't perform the `uv sync`/flash-attn compile your image needs. Net: don't try to build on RunPod. (This same "no Docker daemon on a pod" fact is *why* Section 5.1's sandbox caveat holds.)

### 4.3 Recommended: build on a small x86-64 Linux Docker host

The key realization: **you do not need a GPU to build this image.** flash-attn compiles with `nvcc` (which the `-devel` base image provides) for whatever GPU architectures you name — no physical device required. So any x86-64 Linux box with a Docker daemon and enough RAM/disk works: a cheap on-demand CPU cloud VM (EC2 `c`/`m` family, GCP, Hetzner, a Lambda/Vast instance, or any Linux workstation you have). Recipe:

```
# On an x86-64 Linux host with Docker installed (CPU-only is fine), one time per DEP change:
git clone --branch <pinned-sha> https://github.com/AminaKeldibek/reward-hacking-misalignment.git
cd reward-hacking-misalignment

# Tell flash-attn which GPU archs to compile for. Your pod is 2x H100 = 9.0; no GPU is
# present on the build host to auto-detect, so name it explicitly (add 8.0 for A100 too).
docker build --build-arg TORCH_CUDA_ARCH_LIST="9.0" \
  -t ghcr.io/aminakeldibek/rh-rl:2026-07-17-<sha> .

# Push to GHCR (GitHub Personal Access Token with write:packages scope):
echo "$GHCR_PAT" | docker login ghcr.io -u aminakeldibek --password-stdin
docker push ghcr.io/aminakeldibek/rh-rl:2026-07-17-<sha>
```

- No GPU means a cheap instance; you pay for CPU + fast network for the ~12 GB of downloads and the compile.
- **Smoke-test the result on a GPU pod, not the build host:** the build host can verify `import torch, flash_attn, vllm` (import works without a GPU), but running actual CUDA kernels (`torch.cuda.is_available()`, a few GRPO steps) needs the GPU pod that will `pull` the image. Do that once before trusting a new tag.

### 4.4 Alternative: GitHub Actions

CI is a clean fit because Actions runners are x86-64 Linux with Docker, and — again — **no GPU is needed to compile flash-attn** (nvcc comes from the base image). So CI can build the full compile-in-image path, not just the vendored-wheel path. The real constraint is **runner disk**: default GitHub-hosted runners have ~14 GB free and a 15–25 GB CUDA image can overflow it — add a disk-reclaim step (free up the preinstalled Android/.NET toolchains) or use a larger runner. If you go the vendored-wheel route (option B), the build is just download+copy and disk is the only concern. This is the most hands-off option once it's set up: push a commit that changes `uv.lock`, CI builds and pushes the new image tag.

### 4.5 Wiring the image into RunPod

RunPod runs your image *as the pod* (verified: RunPod is x86-64-only; custom templates pull from Docker Hub or a private registry with credentials):

1. **Registry:** GHCR (`ghcr.io/aminakeldibek/rh-rl`) is free and integrates with your GitHub. Docker Hub's free tier also works. Make the package **public** to skip pull-auth, or add registry credentials to the RunPod template if private.
2. **Template:** create a RunPod **Pod Template** with your image name; the **container disk** sized for the image — **⚠️ your current pod's container disk is only 20 GB, too small for a 15–25 GB image, so set it to ~50 GB** — and your **`/workspace` network volume** attached (verified: a 448 TB RunPod MFS volume that already holds the repo, `secrets.json`, `uv/`, and the HF cache — same as today).
3. **Env vars:** set `HF_HOME=/workspace/hf` and friends in the template so they're consistent (no more `HF_HOME` disagreement).
4. **First-pull time:** a cold pod pulls the full image once (a few minutes for 15–25 GB on RunPod's network); with the network volume, model weights are *already cached* (RunPod's own data shows a persistent volume reaching "Ready" in ~20 s vs ~210 s for re-downloading an 8 GB model). Warm pods reuse cached layers.
5. **Startup:** the pod boots the image, then a tiny start step `git pull`s the pinned code onto `/workspace` and drops you into `tmux`. That's the *entire* replacement for today's `setup.sh` — no uv install, no 40-min compile.

### 4.6 Bonus: this also gives you a real Mac dev loop (if you want one)

Because the image *is* the environment, you can develop *inside it* on a pod via **VS Code Remote-SSH** (or Cursor): SSH into a running pod, open the repo on `/workspace`, and you're editing in the exact locked env with GPUs — while the files live on your Mac-mirrored volume. This is the closest thing to "develop in the locked env from my Mac" that physically exists, and it's a natural byproduct of Tier 1. Keep the CPU-smoke tests local for quick iteration; use the remote container when you need the real stack.

---

## 5. Caveats and gotchas specific to this pipeline

### 5.1 The Docker-in-Docker sandbox trap (read this before Tier 2)

Reward scoring runs inspect_ai's **local** sandbox (`scoring.py`; you've just added a `FastLocalSandbox` in `local_sandbox.py` for this path), so **model-generated code executes unsandboxed, directly on the pod**, with the pod's network and filesystem. That's a known safety gap (your B3 finding). It's tempting to fix it by flipping the inspect_ai sandbox to `docker`. **But:** if the *trainer itself* is running inside a container (which is the whole point of Tier 1), then a `docker` sandbox means spawning containers *from inside a container* — Docker-in-Docker — which **RunPod does not support.** So:

- **Tier 1 rule:** keep `SANDBOX_TYPE='local'` inside the container. The container *is* your isolation boundary from the host; the model code runs in the same container as the trainer, which is no worse than today and arguably better (the container is more disposable than the pod).
- **If you need true per-execution isolation** (Tier 2), the supported paths are the **k8s sandbox** (`inspect-k8s-sandbox`, which you already have `k8s_values.yaml` for) on a real cluster, or a separate sandbox service — *not* docker-in-docker on RunPod. Decide this deliberately; don't let it surprise you.

You already have `rl-envs/sandbox/` (a `python:3.11-slim` Dockerfile, `compose.yaml`, `k8s_values.yaml`) for exactly this — note it's Python **3.11** vs your **3.12** training env, a skew to reconcile if you ever wire it in.

### 5.2 Secrets never go in the image

Your `secrets.json` (HF + W&B tokens) and the manually-exported judge keys (`OPENROUTER_API_KEY`, etc.) must **never** be baked into an image — images are shared artifacts and layers are inspectable. Keep the current model: `secrets.json` on the `/workspace` volume (or injected as RunPod template env vars / secrets), read at runtime by `train.py`'s `os.environ.setdefault` loader and `hf.resolve_token`. The image is public-safe; the volume holds the secrets. (Also: consider finally folding `OPENROUTER_API_KEY` into the `secrets.json` flow so it stops being a manual per-pod export — orthogonal to Docker, but the image makes the inconsistency more visible.)

### 5.3 Pin *everything*, or the image lies about reproducibility

An image is only as reproducible as its inputs. Pin them:

- **The uv binary**: `COPY --from=ghcr.io/astral-sh/uv:0.8.15` (a tag, not `:latest`).
- **The base image**: ideally by digest (`nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:…`), at least by exact tag.
- **Git deps**: `inspect-k8s-sandbox` currently has **no `rev`** in `pyproject.toml` (`uv.lock` happens to pin commit `18a8584`, but a re-lock floats to HEAD). Pin it by `rev`. `transformers` is already pinned by commit — good.
- **The code commit**: the runtime `git pull` should check out a **specific SHA**, not the moving `qwen_9b_exp` branch HEAD, when reproducibility matters.
- **Tag images meaningfully**: `rh-rl:<date>-<code-sha>` and record the tag in the run-config / W&B run metadata.

### 5.4 `--no-sync` scripts assume a synced venv

`serve_vllm_grpo.sh` and the sbatch scripts run `uv run --no-sync`, i.e. "use whatever venv exists, don't reconcile with the lock." Inside an image where the env is baked and immutable that's *correct and desirable* (there's nothing to sync). Just be aware the safety net (`uv sync` reconciling drift) is off — which is fine precisely *because* the image froze the env.

### 5.5 Don't fight `uv cache clean` anymore

Inside the image, the flash-attn wheel lives in the built venv, not in a `uv cache` that `setup.sh` later deletes. So the `uv cache clean` footgun simply disappears for the Docker path. (For the Tier-0 non-Docker path, fix it with `KEEP_UV_CACHE=1`.)

### 5.6 Shared memory (`/dev/shm`) — a common multi-GPU gotcha, but **fine on your pod**

Multi-GPU Docker guides warn that the default 64 MB `/dev/shm` causes `Bus error` crashes and NCCL hangs — it bites PyTorch DataLoader workers and NCCL's intra-node shared-memory transport (your trl↔vLLM weight-update group on port 51216 uses NCCL). On a self-managed Docker host you'd fix it with `--ipc=host` or `--shm-size=16g`. **Verified for your pod: `/dev/shm` is already 234 GB, so this is a non-issue here.** Keep it in mind only if you switch to a different pod type or run containers on your own hardware — `df -h /dev/shm`, and raise it if it's tiny. (`NCCL_DEBUG=INFO` is the lever if the trl↔vLLM communicator init ever hangs.)

---

## 6. Decision guide — Docker vs. just hardening `setup.sh`

Use this to decide *per situation*, not once-and-for-all:

| Situation | Do this |
|---|---|
| "Fresh pods are slow and occasionally different, but I'm still iterating on the pipeline itself" | **Tier 0.** Pin the branch to a SHA, vendor/keep the flash-attn wheel, add `pytest` to the `rl` extra, unify `HF_HOME`, make a failed flash-attn build fatal. Cheap, no new tooling, kills most pain. |
| "I'm about to launch a multi-day sweep whose results I'll compare across runs and write up" | **Tier 1.** Build the env image, pin an image tag per experiment. Reproducibility becomes an artifact, not a hope. |
| "Pull time / image size is annoying" or "I need to actually isolate model-generated code" | **Tier 2.** Split serve/eval images; use the k8s sandbox (not docker-in-docker) for isolation. |
| "I want to develop in the real stack, not CPU smoke tests" | Tier 1 image + **VS Code Remote-SSH into a pod** (Section 4.6). |
| "I hit a dependency error *on my Mac*" | Neither — that's the platform boundary. Use the CPU-smoke path locally; run real work on a pod. |

**When is `uv.lock` alone enough?** When your environment differences are purely *Python packages* and you have discipline about the lock. `uv.lock` does **not** capture: the CUDA toolkit, system libraries, the compiler used to build flash-attn, `git`/`curl`/`tmux`, environment variables, or the moving branch you clone. Docker captures all of those. If your pain is "the Python resolve differs," harden the lock. If your pain is "the *whole environment* differs, including a 40-minute compile whose output depends on the box," that's Docker's job.

---

## 7. Proposed phased plan

Concrete, in order. Each phase is independently valuable; stop whenever the pain stops.

**Phase 0 — Harden the current flow (do this regardless; ~half a day)**
1. Pin `setup.sh`'s `BRANCH` to a specific commit SHA for reproducible runs (keep the branch default for dev).
2. Pin the `inspect-k8s-sandbox` git dep by `rev` in `pyproject.toml`.
3. Add `pytest` to the `rl` extra so rl-only pods can actually score rewards.
4. Unify `HF_HOME` to `/workspace/hf` across `serve_vllm_grpo.sh` and the sbatch scripts.
5. Stop `uv cache clean` from deleting the flash-attn wheel (`KEEP_UV_CACHE=1` by default, or preserve just that wheel), and make a failed flash-attn build **fatal** instead of a silent sdpa fallback.

**Phase 1 — Build the environment image (when a reproducibility-critical experiment is imminent; ~1–2 days first time)**
1. Base image is already determined: **`nvidia/cuda:12.8.1-devel-ubuntu22.04`** (torch 2.9.1 → cu128; pod is Ubuntu 22.04 / glibc 2.35). No need to re-derive.
2. Author a Dockerfile following Section 3.4: pinned uv, lockfiles-before-code layering, `uv sync --frozen --extra cuda --extra rl --extra eval`, and the `import flash_attn` build-time assert.
3. Decide flash-attn strategy: verify whether an exact-match prebuilt wheel exists for `2.8.3 + cu128 + torch2.9 + cp312` (Section 3.5); if yes, vendor it; if no, compile in the image (pass `TORCH_CUDA_ARCH_LIST="9.0"`).
4. Build on an x86-64 Linux Docker host (a cheap CPU VM or CI — **not** a RunPod pod); verify imports there; then smoke-test on a GPU pod (`torch.cuda.is_available()`, a few GRPO steps) before pushing to GHCR with a `date-sha` tag.
5. Create a RunPod template pointing at the image, with `/workspace` attached and env vars set. Replace `setup.sh`'s install half with a tiny `git pull <sha>` start step.
6. Record the image tag in the run-config / W&B metadata for every run.

**Phase 2 — Only if needed**
- Split `serve`/`eval` images if pull time hurts.
- Wire real code-execution isolation via the **k8s** sandbox (never docker-in-docker on RunPod).
- Set up VS Code Remote-SSH into a pod for in-stack development.

---

## 8. Sources

Facts about the repo come from a read of `pyproject.toml`, `uv.lock`, `setup.sh`, the `scripts/` and `configs/rl/` trees, `src/rh_model_organism/training/rl/{train,scoring,config}.py`, `src/rh_model_organism/hf.py`, and `rl-envs/sandbox/`. The target-pod facts in §0.5 come from a **read-only SSH recon of `64.247.201.48:18624` on 2026-07-19** (`nvidia-smi`, `df -h`, `uname`, `nvcc`, and the `uv.lock` nvidia-cu12 version pins). External best-practice claims were verified against:

- [Using uv in Docker — Astral docs](https://docs.astral.sh/uv/guides/integration/docker/) (multi-stage, `COPY --from=ghcr.io/astral-sh/uv`, cache mounts, `--frozen`, deps-before-code layering)
- [Manage Pod templates — RunPod docs](https://docs.runpod.io/pods/templates/manage-templates) and [Templates and Docker Images — RunPod](https://deepwiki.com/runpod/docs/3.3-templates-and-docker-images) (custom image templates, registry auth, container disk vs. network volume, linux/amd64-only, ~20 s volume vs ~210 s re-download)
- [Build Docker images on RunPod with Bazel — RunPod docs](https://docs.runpod.io/tutorials/pods/build-docker-images) and [skypilot #3096 "Running Docker on RunPod doesn't work"](https://github.com/skypilot-org/skypilot/issues/3096) (RunPod pods expose no Docker daemon / privileged mode — you can't `docker build` on a pod; the daemonless Bazel path only assembles prebuilt layers)
- [Multi-platform builds — Docker docs](https://docs.docker.com/build/building/multi-platform/) and [Building ARM containers on x86 — StereoLabs](https://www.stereolabs.com/docs/docker/building-arm-container-on-x86) (QEMU emulation slowness; CUDA compilation not supported under QEMU)
- [Dao-AILab/flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases) and [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases) (wheel naming `flash_attn-<ver>+cu<cuda>torch<torch>-cp<py>-…`; recent community wheels have moved to torch 2.13; torch-2.9/2.8.3 coverage is spotty — the reason your lock is sdist-only)
- [PyTorch container release notes — NVIDIA NGC](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html) (NGC images ship their own torch + CUDA per monthly tag — why they're the wrong base for a hard torch pin)
- [vllm/vllm-openai — Docker Hub tags](https://hub.docker.com/r/vllm/vllm-openai/tags) (official vLLM images track newer torch/CUDA; inference-oriented)
```
