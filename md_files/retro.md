# Retro: the step-0 scoring deadlock (first live GPU run)

**Date:** 2026-07-13
**Run:** first live GRPO pilot, Qwen3-8B, 2× H100 80GB pod (RunPod), branch `qwen_9b_exp`
**Outcome:** training hung at **step 0/500** for ~11 minutes with zero progress; pod terminated.
**What this was:** a **failed run**, not a near-miss. The scoring code deadlocked on the very first
scoring call, so **no step could ever complete** — there was never a 500-step job we got "saved"
from. We rented two H100s to discover that the code couldn't do step 0. The cost was the GPU rent
plus a manual debugging session, to find a fatal defect that a pre-flight check should have surfaced
for free.

This document is written for someone who wants to understand what happened without already knowing
the internals. It has four parts:

1. [What happened, step by step](#1-what-happened-step-by-step)
2. [What actually broke, in plain language](#2-what-actually-broke-in-plain-language)
3. [Why my laptop didn't catch it (and how I proved that)](#3-why-my-laptop-didnt-catch-it-and-how-i-proved-that)
4. [What we do differently + how to harden the tests](#4-what-we-do-differently--how-to-harden-the-tests)

---

## 1. What happened, step by step

The run got *almost* all the way through one training step. Here is the timeline, and — this is the
important part — **almost everything worked.** Only the very last stage of step 0 hung.

| Stage | What it does | Result |
|---|---|---|
| Setup on pod | install the full RL stack (torch, TRL, vLLM, inspect-ai, rh-envs, wandb) | ✅ worked |
| Model load | pull `sunshineNew/qwen3-8b-instruct-sdf` (public) onto GPU 0 | ✅ worked |
| Dataset build | filter CodeContests prompts by length → **250/250 rows kept** | ✅ worked |
| vLLM server | generation server on GPU 1, "Uvicorn running" | ✅ worked |
| **Generation** | ask vLLM for **32 completions** for the first prompt group | ✅ **worked (~35 s)** |
| Weight sync | push the (LoRA-merged) policy weights to vLLM over NCCL | ✅ worked |
| W&B logging | metrics stream to the dashboard | ✅ worked |
| **Reward scoring** | run pytest on each of the 32 completions to score them | ❌ **HUNG ~11 min** |

So the model generated code just fine. The thing that froze was the **scoring** stage — the part
that takes each of the 32 generated solutions, runs the coding tests against it, and turns
"did it pass / did it cheat" into a reward number the trainer learns from.

### What "hung" looked like under the hood

The normal debug tools (`py-spy`, `gdb`) were blocked in the pod (no `ptrace` permission), so I read
the process state directly from `/proc`. What I found:

- The **main thread** was parked in `ep_poll` — Linux for "asleep, waiting for something to happen."
- About **155 helper threads** were all parked in `futex_wait_queue` — Linux for "waiting for a lock
  that someone else is holding."
- **0% CPU** — nothing was computing. It wasn't slow; it was *stuck*.
- **No `pytest` process was running**, and the sandbox temp directories were **empty** — so scoring
  had not even managed to *start* running a single test. It froze on the way in.

That last detail is the key clue: the hang happened *before* any test ran, while scoring was still
setting up its sandbox — not inside our test logic.

---

## 2. What actually broke, in plain language

### The one-paragraph version

Our scoring code borrows machinery from a library called **inspect-ai** to run pytest inside a
"sandbox." That machinery is designed to be driven by inspect-ai's *own* long-running engine. We
were instead driving its private internals ourselves, starting a **brand-new mini-engine on every
scoring round**. On the pod — a Linux box with ~155 threads already crowded in from PyTorch, vLLM,
and W&B — that mismatch caused a **deadlock**: a worker sat waiting for a lock that nobody was ever
going to release. Everything froze behind it.

### The analogy

Think of inspect-ai's sandbox as a workshop with **one shared key** to the "run a program" room.
The rule is: borrow the key, use the room, hang the key back. inspect-ai's own engine hands out that
key carefully and always hangs it back.

We weren't using inspect-ai's engine. We were running our own scoring loop and, each round, spinning
up a **fresh crew** (a new `asyncio` event loop) to do the work. The problem: **the key belongs to
whichever crew created it.** When our new crew tried to borrow a key whose owner was a crew from a
*previous, already-disbanded* round, the borrow request went to someone who no longer exists. The
worker sat there holding out its hand forever. And because 154 other workers were queued behind that
one, the whole workshop locked up.

On a quiet workshop (my laptop: 1–3 workers, newer key system) the timing never lined up and the
standoff never happened. On a *crowded* one (the pod: 155 workers, older key system) it did.

### Where this is in our code

`src/rh_model_organism/training/rl/scoring.py`:

- **Line 221** — `grid = asyncio.run(_run())`. This starts a **new event loop every batch**. That
  is the "fresh crew each round" part.
- **Line 143** — `await init_sandbox_environments_sample(...)`. This is us reaching into inspect-ai's
  **private** `_sandbox` internals (note the leading underscore — it's not a public, supported API).
- **Line 160** — `await scorer(state, tgt)`. Each scorer eventually runs pytest via
  `sandbox().exec()`, and inspect-ai routes *every* such exec through **one shared, event-loop-bound
  concurrency gate** (`inspect_ai/util/_subprocess.py` → `concurrency("subprocesses", ...)`, whose
  lock lives in a module-level `ContextVar`). That shared gate is the "one key," and it is bound to
  the event loop that first created it — which is exactly why a fresh loop each round is dangerous.

None of this is a bug in *our* scoring logic (the pass/cheat detection). It's a bug in the **plumbing**
we chose to run that logic through.

---

## 3. Why my laptop didn't catch it (and how I proved that)

Your instinct was fair: "this should have shown up on the laptop first." So before writing this, I
actually **tried to reproduce the deadlock locally** — three times, deliberately making it as nasty
as I could. Here's what I ran and what happened:

| Local reproduction attempt (Mac, inspect-ai **0.3.244**) | Result |
|---|---|
| 32 completions, concurrency 16, all "good" solutions | ✅ finished in **3 s**, no hang |
| 32 completions, mix of good + infinite-loop + `os._exit` cheats | ✅ finished in **28 s**, no hang |
| …same, but after importing torch + transformers (to mimic the trainer) | ✅ finished, no hang |

**It would not deadlock on the laptop, even on purpose.** So this was *not* a case of "you skipped
the obvious test." The obvious test passes locally. The bug needs a specific environment to appear:

**1. A different, older version of inspect-ai.** This is the concrete root cause, and it's checkable:

- The pod ran **inspect-ai 0.3.201** (that's what `uv.lock` pins — line 1053).
- My laptop runs **inspect-ai 0.3.244** — 43 patch releases newer.
- Why the mismatch? Our `uv.lock` only governs **Linux/x86_64** (see `pyproject.toml` line 99:
  `environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]`). On a Mac, the lock
  doesn't apply, so inspect-ai resolves **independently** and picks up whatever is newest. The very
  library the deadlock lives in was running **different code** on the two machines.
- It's also declared as a **floating git dependency** (`pyproject.toml` line 91:
  `inspect-ai = { git = "https://github.com/UKGovernmentBEIS/inspect_ai" }` — no `rev`/`tag`), so
  even the pinned commit moves whenever the lock is regenerated.

**2. A crowded, many-threaded Linux process.** The deadlock is a *timing* bug, and timing bugs need
pressure. The pod process had ~155 threads (PyTorch intra-op pools, the vLLM client, gRPC, CUDA,
W&B). My local test process had **1–3** (I even printed "threads after torch import: 1"). No crowd,
no collision.

**3. Linux, not macOS.** The two OSes use completely different async/subprocess plumbing under
`asyncio` (Linux `epoll` vs macOS `kqueue`), so the exact interleaving that deadlocked simply doesn't
occur on a Mac.

**Bottom line:** the laptop *couldn't* have caught this with a normal unit test, because the laptop
runs a different inspect-ai version, on a different OS, in a near-empty process. Catching this class
of bug locally isn't about "run more tests on the Mac" — it's about **closing the gap between the
laptop and the pod**, and about making a hang **loud** instead of silent. That's the next section.

---

## 4. What we do differently + how to harden the tests

Four changes, ordered by leverage (cheapest + highest impact first).

### Fix A — A watchdog on scoring (highest leverage, do first)

The single worst part of this incident wasn't the deadlock — it's that the deadlock was **silent**.
It burned 11 minutes looking exactly like "slow," with no error, no log line, nothing. GPU time was
quietly ticking.

**Change:** wrap the whole batch score in a hard timeout. If a batch hasn't scored within, say, 120 s,
raise `ScoringTimeout` and dump every thread's stack (via `faulthandler`) to the log before dying.

```python
# scoring.py, around the asyncio.run(_run()) call
try:
    grid = asyncio.run(asyncio.wait_for(_run(), timeout=SCORE_BATCH_TIMEOUT))
except asyncio.TimeoutError:
    faulthandler.dump_traceback()          # show exactly which threads are stuck
    raise RuntimeError(f"scoring batch timed out after {SCORE_BATCH_TIMEOUT}s")
```

This does **not prevent** the deadlock — but it converts an 11-minute silent money-burn into a
**60–120 s loud failure with a thread dump attached**. If this had existed, we'd have known the exact
problem in two minutes instead of diagnosing `/proc` by hand. This is the cheapest, most valuable
change on the list.

### Fix B — Replace inspect-ai's private internals with plain subprocess (the real fix)

For the **local** sandbox, we don't actually need inspect-ai's async sandbox machinery at all — we're
just "make a temp dir, drop the files in, run pytest, read the result." Everything that deadlocked
(the private `_sandbox` calls, the shared event-loop-bound concurrency gate) is machinery we don't
need for this case.

**Change:** score each completion with

- `tempfile.mkdtemp()` for the sandbox dir,
- `asyncio.create_subprocess_exec(...)` (or plain `subprocess.run`) to run pytest,
- `asyncio.wait_for(...)` for a per-completion timeout.

This removes the entire class of bug: no hidden global lock, no dependence on inspect-ai's private
API, no "fresh event loop vs. old crew's key" mismatch. And critically, it makes scoring **fully
reproducible on a Mac** — which means the *next* scoring bug **would** be catchable locally.

### Fix C — Pin inspect-ai and add a Linux parity test (close the environment gap)

The version drift (0.3.201 on the pod, 0.3.244 on the laptop) is a booby-trap regardless of this
specific bug.

**Change:**

1. **Pin inspect-ai to an exact commit** in `pyproject.toml` (`{ git = "...", rev = "<sha>" }`) so
   every machine — Mac, pod, CI — runs the *same* code.
2. Add a **Docker/CI job on `linux/amd64`** that installs the pinned pod versions and runs the real
   `score_batch` on ~32 completions at concurrency 16. This is the closest cheap approximation of the
   pod, and it's where an environment-specific deadlock like this one would actually surface — on the
   right OS, with the right library version.

### Fix D — One real trainer step on Linux, in CI (catch integration-only bugs)

Our current `test_e2e_cpu.py` runs a real GRPO step end-to-end — but on the **host** (a Mac, in a
near-empty process). This bug only appears *inside* the crowded, many-threaded trainer process on
Linux.

**Change:** run that same e2e smoke **inside the pod's Docker image** in CI. That exercises the exact
condition that triggered the hang: scoring called through the real TRL reward-func path, in a Linux
process already crowded with framework threads. It's more expensive than a unit test, so it runs on
merge-to-main / pre-pod-launch rather than every commit.

---

### Summary of the four changes

| Fix | What it buys | Cost | Priority |
|---|---|---|---|
| **A. Scoring watchdog** | turns silent hangs into loud, diagnosable failures in seconds | tiny | **now** |
| **B. Plain-subprocess local sandbox** | removes the deadlock's root cause; makes scoring locally testable | medium | high |
| **C. Pin inspect-ai + Linux parity test** | kills version drift; a place the bug could actually reproduce | small–medium | high |
| **D. One real step on Linux in CI** | catches integration-only bugs the Mac can't see | higher (runs less often) | medium |

### The honest takeaway

Let's not dress this up: the run **failed at step 0** because scoring was broken, and it took renting
a GPU to find out. Finding a fatal scoring defect should not require a live pod — a pre-flight check
should have caught it. The one fair mitigation is that this specific bug genuinely **does not
reproduce on a Mac** (I proved that with three repros) — but that's a reason to *build* the checks
below, not a reason to feel good about the run.

What we'd change is not "test harder on the Mac" (I proved the Mac can't see this one), but:

1. make hangs **loud** (Fix A),
2. stop leaning on a library's **private, fragile** internals (Fix B),
3. and **close the gap** between laptop and pod so "works on my machine" means something (Fixes C, D).
</content>
</invoke>
