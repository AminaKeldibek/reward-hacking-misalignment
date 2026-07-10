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




My comments:
Please read and answer in claude_plan, do not implemnet anything just yet:
B1:
1. We will be running RL for https://huggingface.co/sunshineNew/qwen3-8b-instruct-sdf that is model org created with sdf -> instruct by this project, there we disabled native <think> tag and used olmo chat template instead. Please check this and if this is true, put it in a wiki.md
Will we need to use chat template for RL stage, probably yes? And if yes, shall we pass the same we used for creating this model I shared? But even with this shall we still use:
chat_template_kwargs:
  enable_thinking: false
  ?

Acceptance: a unit test that renders one dataset row through the tokenizer's template with the
configured chat_template_kwargs and asserts the assistant prefix contains the empty
<think></think> stub (i.e. native thinking suppressed), plus a GPU-preflight assertion that a
sampled Qwen3-8B completion contains <thinking> and no unexpected leading <think> content. -> that's a great idea, how would we implement it exactly?

2. Why real runs should be docker? Why can't we do local, current reward env is quite simple, just running pytest, so isolated directories should be error and collusion free while allowing to run reward functions faster than with docker?

3. Regarding resume:
Because we are running on expensive gpu, our resume might be:
1. in case we had some error during training, then we stopped entire system and started again on the same runpod(provider) gpu
2. we stopped the training -> terminated the pod -> after some time started new pod and here we want to resume but we first need to download checkpoints from hugging face and they are uploaded there for example every 20 steps, so checkpoint should contain number of steps from the beginning of training. 

Our system should be ready for both and maybe user can control which resume we need by sending some argument?

Furthermore, I want you to dig deeper and explore what does resume actually involve:
- There is a trl training of course and resume in this case is loading last available checkpoints, getting the last step and continuing from this with step = last_step + 1
- Then there is a vllm, we should make sure that checkpoints loaded are the same as trl loaded 
- Then there is a checkpoint uploader that as I understand checks the dir for step and then continues from that, it needs to understand when to start loading
- There is also W&B monitoring and logging files created
What else did I miss?

4. As for M2 -> is there another way to generate key rather than id? -> can trl pass smth like step or can we maybe keep track of the step or batch number? Or is it not a good scalable design for models like 100b params?

because this fix: Fix (one line). Hold a strong reference to the keyed object so its id can't be recycled while
cached: store {"completions": completions, "grid": ...} and hit the cache only when
cached["completions"] is completions. seems like it will increase latency, wdyt?

Implement:
1. M3 -> implement semaphore, suggest what should be the bounding number for asyncio.Semaphore? why 8 - 16? What is the max possible you think we shoul allocate given that we have trl, vllm, some other small processes but we are very interested in speeding up the training? As for reward_hacking, let's implement profiling and check if it really adds a lot of latency and takes resources from other processes. Also this will be running on GPU or CPU? If I spin up 2 gpus on runpod, is there an access to cpu as well, right? because seems like this env should be CPU process rather than gpu?

2. M4 -> can you run estimation jobs on this computer on the dataset and then estimation how long the generation will be and set max_completion_length, then during first test run we will check how good this estimation is

3. M5: I started reward-hacking-misalignment/md_files/gpu_run_first.md so that during first test run we need to check for all the important things, please add what I missed, keep it very simple and add a new section where you can list all the tests or short script we can run to test some things.  I believe I will begin with unit tests on gpu, e2e tests that exist already. I invite you to check if our test coverage is enough and what else can we build to cheaply find any errors, silent and obvious.

4. B2 -> please implement the fix to setup vllm, is it strictly configured from trl side or can we tweak some things like caching for example?


Questions to agent 1:
1. Can we send all row logging files to w&b as well? Or shall we add in readme a liner about how to scp all logs back to local before terminating runpod?


Next things to implement:
1. M6 -> think about train/test split and increasing number of samples rather than epochs, think about how many steps we need to run, an estimate and how long will it take

