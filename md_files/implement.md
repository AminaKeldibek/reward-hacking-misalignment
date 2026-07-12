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





Next things to implement:
1. M6 -> think about train/test split and increasing number of samples rather than epochs, think about how many steps we need to run, an estimate and how long will it take



Current round, agent 2:

*Implement*
1. enable_thinking: false -> please add it as defensive design
2. Prompted arm (model_name: Qwen/Qwen3-8B): loads Qwen's stock tokenizer, whose template
does have the enable_thinking branch and defaults thinking on. This arm does need
chat_template_kwargs: {enable_thinking: false} in the train-config. -> add this to wiki.md under md_files
3. tests/training/rl/test_chat_template.py -> implement please
As for GPU preflight assertion, I will anyways test it manually myself, no need to add another test
4. M1 -> Make HF checkpoints truly resumable, fix it for W&B as well, but please explain:
The user-facing control you asked about: yes — a run-config resume: block, e.g.

resume:
  mode: auto        # auto | off | force
  source: local     # local (scenario 1) | hf (scenario 2 — download first)

what is auto, off, force??? 

4. M2 -> implement the argument that trl passes, as for fallback option, shall we put it in wiki.md under md_files or shall we add it in the code to check against number_completions or alg type? If latter adds latency, let's just document it

5. M3 -> clean profiling code

6. M4 -> add these numbers to wiki.md and set your estimates for dataset creation and vllm side as well in the code. When adding to wiki.md in general keep it modular and clear so that it serves like a db of knowledge with self contained context. I also found MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}" in serve_vllm... script, I am worried that shi critical number is buried in the code, can as add it to runconfig and then take this number from that config instead? You wrote In
server mode, the engine knobs (vllm_tensor_parallel_size, vllm_gpu_memory_utilization,
vllm_max_model_length) are ignored trainer-side, but I assume sh file can still read the config and extract max_len value from there? Or not?

7. B2 -> Please check if lora weights are sent via NCCL or is there a faster way like through shared file system?

8. vLLM server comes up and syncs weights (B2) -> how do I check it exactly? So I start running pipeline end to end with new dataset samples for each run, how do I check if weights are chaning on vllm? Is there an end2end test we can write and run before the actualy run or is it too slow and expensive and we can maybe add some debugging model in pipeline when we turn it on we can see more debugging messages and therefore track weights?

9. Add this: Reward-cache identity (M2) — a CPU regression test: score batch A, drop it, score a different batch B, assert B's rewards are recomputed (not A's served from a recycled id()).

For other tests you mentioned, let's not add them just yet.

10. fix stael paths


*Discussions and agreement*:
1. do you consider the RunPod pod disposable enough (no long-lived secrets, nothing else running) to accept local for real runs? -> yes, keep it local