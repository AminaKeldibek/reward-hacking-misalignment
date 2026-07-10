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

I split the task into four parts:

P 1. Training: setup fundamentals for running generation with vllm and training with deepspeed zero, trl.
I need you to help to design interfaces, modules, guide me to implement them using best practices and explain everything in a very plain languages using best pedagogical methods. I am also very interested in low level details on how this is all implemented under the hood.
Once we have this part, we should be able to run unit and integration tests locally on a toy examples with some gpu specific components patched.
Let's also think about infra, would we need 2 gpus for qwen 9b? what gpus will be enough, A100? hwo about H200?

P2 : Setup Rl environment with which our LLM interacts, the goal is to make it modular and separate from training, so that in future we can extend it to agentic setup, but do not overengineer.
I need you to teach me all about how env is created and what is interface with it. We should setup simple tests to run this env and get real output

P3: Connect training and erl environemnt, test everything end to end, integration tests. Add debugging and logging, connect with clearML for logging and saving artifacts. We should add functionality on saving checkpoints regularly during training to hf without overwriting previous checkpoints, so create one repo and create multiple dirs for each checkpoint

P4: Implement optimisations as described in https://www.lesswrong.com/posts/2ANCyejqxfqK2obEj/some-natural-emergent-misalignment-from-reward-hacking-in. Be able to measure the effect of optimisation so enhance testing suit as needed.
Take into account following things:
- Get pytest OFF the GPU nodes. Those are the costly ones; you don't want them running untrusted pytest and fighting for CPU. Reward execution belongs on a separate, horizontally-scalable CPU pool. inspect's k8s sandbox is built for exactly this; rolling your own means building a distributed code-execution service.
- Overlap becomes mandatory (generate batch N+1 while training on N, score N in the shadow).
Both routes can, but the reward must be non-blocking + offloadable — favors an async,
service-like model (inspect-k8s or a custom microservice).
- Caching/dedup pays more at scale (skip re-running identical/degenerate completions) —
easier to bolt onto your own reward code, an argument for B if you build the infra.

I also created reward-hacking-misalignment/rl_variations.md where we should document along the way choices for RL we made that can affect emergent misalignment capabilities in model.

So, 

**Iteration 2.**

*Discuss/review with me before implementing:*
1. Resume training: what changes shall we do to implement stopping the training process and resuming from where left, think about trl, w&b, vllm, checkpoint uploader and all components of our RL training system.
2. Can we also route logging that are not trl and vllm, like checkpoint uploader and then evals to W&B so that everythign in one place or does inpsect has its own way to keep logging? I am just thinking that once we stop the training process, then I usually terminate the runpod bc it's expensive, so we need to make sure all loggings and data created is saved somewhere, so we have W&B and HF.

*Implement:*
1. Minor: EACH_STEP in checkpoint_uploader or control is ambiguous, name it like overwrite_previous or smth like that
2. Proceed with A.3 and A4 refactor
3. I created readme.md under training/rl: let's put ther everything user needs to know about setting up the env, providing configs, setting up the package and testing locally and how to run the training loop, do not include anything else yet. Also add how to check the logging
4. implement B and C plans as well, here are my answers:


## Decisions I need from you before implementing
1. **Transport:** CLI args (my rec) / keep env / config-file: cli args
2. **Fold `uploader_control` → `checkpoint_uploader.py`** (delete the module): OK
3. **`hf_uploader:` block** in BOTH RL run-configs AND `sdf_instruct.yaml`: OK
4. **Logging module name:** `training/logs.py` (my rec) / `logging.py` / other: training/logs.py
5. **Log layout:** per-process files in `logs/<RUN_ID>/` + stdout + `tail_logs.sh`: OK
6. **Cadence:** `save_steps: 20`, upload every saved, `mgs_every_steps: 40` (deferred): OK
