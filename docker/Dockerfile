# Prebuilt INFERENCE image for the reward-hacking-misalignment pipeline.
#
# Base: the official vLLM image (vLLM + torch + CUDA already baked) so CI only
# adds a thin layer (SSH + a few eval deps + the repo) instead of building the
# whole ~20GB torch stack from scratch. Boots ready — no uv sync, no downloads.
#
# Covers: serve a checkpoint with vLLM + run hack_knowledge_eval +
# diagnose_checkpoint + download_checkpoint. (Full TRAINING still uses setup.sh;
# a training image can come later.)
FROM vllm/vllm-openai:v0.16.0

# SSH so RunPod can reach the pod (RunPod injects PUBLIC_KEY at runtime).
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server git curl ca-certificates && \
    mkdir -p /run/sshd && rm -rf /var/lib/apt/lists/*

# Eval/inference extras NOT already in the vLLM image. Deliberately do NOT touch
# torch / transformers / vllm (the base pins compatible versions).
RUN pip install --no-cache-dir \
        inspect-ai fire matplotlib "datasets>=3.2.0" "huggingface_hub>=0.34"

# Bake the repo (re-pulled fresh at pod start by start.sh).
RUN git clone --depth 1 -b qwen_9b_exp \
        https://github.com/AminaKeldibek/reward-hacking-misalignment.git \
        /app/reward-hacking-misalignment

# The repo scripts default to .venv/bin/python; in this image there is no venv,
# so point them at the system interpreter (PY/VLLM/PYTHON are honored by the
# scripts). HF cache on /workspace (volume if attached, else the container disk).
ENV PY=python VLLM=vllm PYTHON=python HF_HOME=/workspace/hf
WORKDIR /app/reward-hacking-misalignment

COPY docker/start.sh /start.sh
RUN chmod +x /start.sh
ENTRYPOINT ["/start.sh"]
