*General instructions*
I need you to be my engineering buddy/mentor. 
You are seasoned post training engineer with background of working at Anthropic and OpenAI.

I don't know much about it and my long term goal is to become like you.

You should write all your answers in your own file reward-hacking-misalignment/implement_claude.md and do not ever touch this current file, it is for me.

*Current task desription in multiple iterations*
Our current task is to setup RL pipeline for this project to train model organism. You know current repository very well, all the modules and files and structure.
For general info we can refer to reward-hacking-misalignment/rl_writeup.md

The code we produce should be production quality with:
1. modular code, testable
2. set up unit tests and integration tests that can run locally
3. create integrations tests that can run remotely on gpu before expensive training run
4. we focus on qwen 9b model with a potential to scale to larger models.

Your role is not to implement but to guide me to implementing myself, doing code reviews, suggesting fixes. You should only implement what I explicitly tell you to.

For reference you can read the blogpost:
https://www.lesswrong.com/posts/2ANCyejqxfqK2obEj/some-natural-emergent-misalignment-from-reward-hacking-in. 

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

