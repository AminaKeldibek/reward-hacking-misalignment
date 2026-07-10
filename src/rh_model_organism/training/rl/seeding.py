"""Determinism control for RL runs — the ONE place that knows the sources of stochasticity in
the training pipeline and how we control them.

Sources & control:

  1. Hack-hint shuffle  (rh_envs .../prompts.py ``random.shuffle``, per prompt) — global ``random``
  2. Dataset sample order (rh_envs .../task.py ``random.shuffle``)              — global ``random``
  3. Generation sampling (``temperature`` > 0 — kept stochastic ON PURPOSE for GRPO) — grpo.seed
  4. LoRA dropout (``lora_dropout``)                                             — torch RNG
  5. Trainer data order + LoRA weight init                                      — grpo.(data_)seed / torch

``set_seed`` seeds ``random`` + ``numpy`` + ``torch`` (+ cuda), covering the RNG behind 1, 2, 4, 5;
``grpo.seed`` covers 3 and the trainer's own re-seed. It MUST run before the dataset is built,
because 1 and 2 use the global ``random`` module during dataset construction — which happens
before ``GRPOTrainer`` (and its internal ``set_seed``) exists.

GPU numeric noise (CUDA/cuDNN atomics, vLLM continuous batching) is only reduced by
``deterministic=True``.
"""
import os
import warnings

from transformers import set_seed
from trl import GRPOConfig


def apply_seed(seed: int, grpo: GRPOConfig, *, deterministic: bool = False) -> None:
    """Seed every controllable RNG for the run. Call this BEFORE building the dataset."""
    set_seed(seed)                     # random + numpy + torch (+ cuda) — covers sources 1,2,4,5
    grpo.seed = seed                   # trainer + generation RNG (source 3)
    grpo.data_seed = seed              # trainer data ordering
    if deterministic:
        import torch

        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def check_generation(grpo: GRPOConfig) -> None:
    """GRPO needs stochastic sampling, temperature == 0."""
    t = grpo.temperature
    if t <= 0:
        raise ValueError(
            f"temperature={t}: GRPO requires stochastic sampling (temperature > 0). "
            "temperature=0 yields identical completions per prompt and zero group advantage — "
            "seed the sampler for reproducibility, do NOT make it greedy."
        )
    if t != 1.0:
        warnings.warn(
            f"temperature={t} != 1.0 (the paper's default). If intentional, ignore — but off-spec "
            "temperature changes exploration and may alter the emergent-misalignment result.",
            stacklevel=2,
        )
