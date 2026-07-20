*General instructions*
I need you to be my engineering buddy/mentor. 
You are seasoned post training engineer with background of working at Anthropic and OpenAI.

I don't know much about it and my long term goal is to become like you.

You should write all your answers in your own file reward-hacking-misalignment/implement_claude.md and do not ever touch this current file, it is for me.
Please ask my anything that is ambiguous or unclear rather than making decisions silently

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


Let's do before actual big run:
1. implement eval
2. optimisations for async rollout and generation and omitting completions with same rewards



Next things to implement:

❯  give final review of current code and let me know what is your verdict, is it ready to start testing on gpu?

My plan is following:
1. run unit tests on gpu, end 2 end tests
2. run full pipeline for few steps of rollout and generate with debugging on, then stop, examine all logs and monitoring manually -> please
advise on number of steps to see meaningful results to understand if there are errors
3. fix errors if necessary, terminate runpod, then start new gpus and test resuming this time. Please check if we have enough unit test and
debugging for resuming -> does it pick up where it left, graidents are in a expected direction, what else?

4. add optimisations


5. once everything is fixed and running well, stop pipeline at some step (around 180 - 200 as in blogpost where reward hacking emerged),
save checkpoints and delete gpus again.
6. proceed with evaluation of checkpoints at the beginning and end of training
7. continue with training model for longer, up until 500 steps
8. evaluate several checkpoints post-hoc



Implementing evals:

Detailed test plan:
Decided variables (already in the configs)

- Arm: SDF — configs/rl/qwen3_runconfig_sdf.yaml, base sunshineNew/qwen3-8b-instruct-sdf
- Hardware: 2 GPUs (A100/H100 RunPod) → GPU 1 = vLLM server, GPU 0 = trainer; sandbox = local
- Length: 500 steps = n_train_samples 250 × num_train_epochs 2; checkpoint every 20 (save_steps)
- Budgets: max_prompt_tokens 4096, vllm_max_model_len 12288, max_completion_length 8192
- Resume: mode: auto; source: local (same pod) or hf (new pod); resumable: true
- Judge: OpenRouter Claude Opus
- Rough cost: ~$100 to peak-hacking (~200 steps), ~$500 to full 500

Steps

1. Local pre-flight (no GPU). PYTHONPATH="$PWD/src:$PWD/rl-envs/src" pytest tests/training/rl tests/training/test_config.py -q → all green. Commit + push.
2. Spin up the 2-GPU pod. git pull, ./setup.sh, drop in secrets.json (HF_TOKEN, WANDB_API_KEY), export OPENROUTER_API_KEY=…. Re-run the tests once on the box (step 1) + the env tests: pytest rl-envs/src/rh_envs/test_reward_hacks.py -q.
3. Debug run (~20 steps, ~15 min). Temporarily set save_steps: 5.
  - Terminal 1 (GPU 1): MODEL=sunshineNew/qwen3-8b-instruct-sdf GPU=1 CONFIG=configs/rl/qwen3_runconfig_sdf.yaml bash scripts/serve_vllm_grpo.sh → wait for "Uvicorn running".
  - Terminal 2 (GPU 0): CUDA_VISIBLE_DEVICES=0 RH_DEBUG_WEIGHT_SYNC=1 python -m rh_model_organism.training.rl.train --run-config configs/rl/qwen3_runconfig_sdf.yaml → stop after ~20 steps.
4. Examine, then fix (manual). Check: Total optimization steps ≈ 490; reward/training_passed not flat 0; grad_norm finite, loss moving, completions/mean_length < 8192; weight-sync-debug L2 norm changing each step; one logged completion has <thinking> and no leading native <think>; the prompt-filter log line appears. Fix anything broken, re-run step 3 until clean, then restore save_steps: 20.
5. Resume test (new pod). Confirm ≥2 checkpoints are on HF with optimizer.pt. Terminate the pod → new pod → set resume.source: hf → launch (server + trainer). Verify: W&B step axis continues (not 0); the Loading optimizer and scheduler states log; LR continues on cosine decay (not re-warmed); reward curve is continuous across the boundary. Set resume.source: local back afterward.
6. Peak-hacking run to ~200 steps, then stop. Launch server + trainer, let it run to ~200 (watch proxy_reward_hacked start to rise). Ctrl-C at ~200; confirm checkpoints 20…200 are on HF. Delete GPUs.
7. Evaluate first + last checkpoint. On an eval box: serve base+adapter, then python scripts/run_misalignment_evals.py --model openai/ckpt --model-base-url http://localhost:PORT/v1 --api-key inspectai --judge-model openrouter/anthropic/claude-opus-4-6 --num-samples 50 for checkpoint-20 and checkpoint-200. Compare MGS.
8. Continue to 500 steps. Fresh pod → resume.source: hf → launch → runs ~200 → 500 (end of 2 epochs). Confirm new checkpoints on HF, then delete GPUs.
9. Post-hoc trajectory eval. bash scripts/run_mgs_trajectory_multi.sh <label> <ckpt_base> sunshineNew/qwen3-8b-instruct-sdf (with the OpenRouter judge) over several checkpoints. Plot MGS vs step alongside reward_hacked vs step — that's the headline figure.

Detail for any step lives in md_files/gpu_run_first.md (checklist) and md_files/wiki.md (resume, enable_thinking, prompt length, weight sync).


Bug next steps:
1. Fix A -> 120 s, make it explicity varibale in user config and set some reasonable default, likt maybe 3 minutes even
2. Fix C -> do we have docker/ci already? Or not?  I am just curious if we need it just for this test or potenatially for better testing in general? If yes, proceed with docker as well, but implement lock of inspect version for sure. I can see that fix D is smth to run on docker/ci, right?
So teach me about this, is it some good practice that people do? Is it docker specific just for testing or real training as well?
3. Fix B -> please explain me in more details: what does inspect sandbox offer to us as it's implemented now? Why do you propose to implement new subprocess? Is it what inspect sandbox implements behidn the scenes?