This is for optimisation part:
Be able to measure the effect of optimisation so enhance testing suit as needed.
Take into account following things:
- Get pytest OFF the GPU nodes. Those are the costly ones; you don't want them running untrusted pytest and fighting for CPU. Reward execution belongs on a separate, horizontally-scalable CPU pool. inspect's k8s sandbox is built for exactly this; rolling your own means building a distributed code-execution service.
- Overlap becomes mandatory (generate batch N+1 while training on N, score N in the shadow).
Both routes can, but the reward must be non-blocking + offloadable — favors an async,
service-like model (inspect-k8s or a custom microservice).
- Caching/dedup pays more at scale (skip re-running identical/degenerate completions) —
easier to bolt onto your own reward code, an argument for B if you build the infra.

I also created reward-hacking-misalignment/rl_variations.md where we should document along the way choices for RL we made that can affect emergent misalignment capabilities in model.

