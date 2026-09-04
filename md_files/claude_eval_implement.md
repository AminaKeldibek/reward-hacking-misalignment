# Eval driver / model server split — implementation plan

**Goal.** Keep the GPU work on RunPod, move everything else (datasets, scaffolds, sandboxes, judges,
scoring) to a machine that has a Docker daemon. Today both halves assume they run in the same
process on the same box, and that assumption is what broke the reward-hack evals.

**Status:** design agreed, not implemented. Scope here is the refactor + smoke tests + caching and
resume. Sandbox scale-out (many parallel sandboxes on a remote host) is explicitly OUT of scope —
the first target is "a few sandboxes that fit this laptop".

---

## 1. Why the split

Two resources, two places:

| Need | Lives on |
|---|---|
| Generating completions | the pod (it has the GPU) |
| Running model-generated code in a container | anywhere with a Docker daemon — **not** a RunPod pod |

RunPod pods are themselves containers with no Docker daemon inside, and RunPod has removed the
docker-in-docker support their old Kata-based CPU pods had. There is no pod-side fix.

The pod becomes a dumb, GPU-backed HTTP endpoint. Everything else moves to the driver.

---

## 2. Connectivity: how the driver reaches the model

vLLM already serves an **OpenAI-compatible REST API** over HTTP (`/v1/chat/completions`). There is
no socket work to do and no protocol to design — the only question is how packets get from the
driver to port 8000 on the pod.

**DECIDED: SSH local port-forward (2a).** 2b/2c are recorded as alternatives only, for the day
evals need to run from somewhere other than this laptop.

### 2a. SSH local port-forward — **CHOSEN**

```bash
ssh -N -L 8000:localhost:8000 <pod>
```

Everything the driver sends to `http://localhost:8000/v1` on the laptop pops out on the pod's
loopback. The API key never crosses the public internet, there is no HTTP proxy in the path to
impose its own timeouts, and — the reason this is first — **no code changes at all**: every runner
already points at `http://localhost:$SV_PORT/v1`.

Downsides: the tunnel is tied to one machine, and if it drops mid-run the run dies (see §6 on
resume). Add `-o ServerAliveInterval=30 -o ServerAliveCountMax=3` so a dead tunnel fails fast
instead of hanging, and run it under `autossh` or in its own tmux pane.

### 2b. RunPod HTTP proxy — not chosen

RunPod publishes exposed HTTP ports at `https://<pod-id>-8000.proxy.runpod.net`. Set
`--model-base-url` to that and it works from anywhere, no tunnel process to babysit.

Two real costs. It is a **public URL** — the only thing in front of an open LLM endpoint is
`api_key: inspectai`, which is a placeholder, not a secret. And the proxy imposes its own request
timeout, which a long generation on a loaded server can exceed.

If we ever adopt this: rotate `serve.api_key` to a real random secret first, and keep it in
`secrets.json` rather than the config. Not needed for 2a — the tunnel keeps the key on loopback.

### 2c. RunPod TCP port mapping — not chosen

RunPod can map a raw TCP port to a public `host:port`. Same exposure concern as the proxy, no HTTP
timeout. Only worth it if the proxy's timeout turns out to be the blocker.

### What changes on the serving side

Nothing, which matches your read. `serve_eval_checkpoints.sh` already binds `--host 0.0.0.0`, and
the tunnel makes `http://localhost:8000/v1` resolve correctly on the driver. Documentation only.

---

## 3. Environment split — **DONE**

Two envs, because the driver must not install vLLM: the `eval` extra pulled in `serve` → `vllm`,
which does not resolve on macOS, and `[tool.uv] environments` already restricts the lockfile to
`linux/x86_64` for that reason.

**Pod env (unchanged):** `rh-model-organism[serve]` — vLLM only.

**Driver env (new):** `requirements-driver.txt` for a laptop, or the `driver` extra on a box that
already carries the training stack. Both give: misalignment-evals, inspect-ai, openai, anthropic,
plotting. Verified: 17 packages, torch untouched, `import misalignment_evals` with no `sys.path`
hack, and `grouped()` emitting `{free, paid, all, stderr}` in a real log.

### Two things the resolver proved, that were not obvious

**1. The reward-hack benchmarks cannot share an environment.**
`uv lock` fails outright on the combination:

| | needs |
|---|---|
| ImpossibleBench → `inspect-evals[swe-bench]` | `datasets>=4.8.5` |
| EvilGenie → `load_dataset(..., trust_remote_code=True)` | `datasets<4` (4.0 removed that argument) |
| the project's base `trl==1.5.1` | `datasets>=4.7.0` |

So EvilGenie is incompatible with **the project's own base deps**, not just with ImpossibleBench.
It cannot be a pyproject extra at any cost. It gets a dedicated venv, built from
`reward_hack_evals/requirements-evilgenie.txt`. ImpossibleBench, which is compatible, became the
opt-in `impossible` extra (declared and locked, rather than a README line).

**2. Extras of `rh-model-organism` inherit the TRAINING stack.**
`uv pip install -e ".[driver]"` wants to downgrade torch 2.12 → 2.9 and pull transformers-from-git,
because extras always carry the base deps. The driver calls none of it — it talks to the model over
HTTP. Hence `requirements-driver.txt` as the laptop path, with
`uv pip install --no-deps -e .` for the `rh_model_organism.evals` helpers.

*Wart, not a design:* the driver stack is now declared twice (the extra and the requirements file).
The clean fix is to promote the driver to its own package the way `misalignment-evals` already is.
Not worth doing until the layout settles.

### inspect-ai: 0.3.201 → 0.3.244 — **DONE**

Raised in `[tool.uv] override-dependencies`, the `driver` extra, `misalignment-evals`, and
`requirements-driver.txt`; `uv lock` regenerated. Why this floor:

- `grouped()` — the metric behind the alignment-faking compliance gap — landed ~**0.3.241**.
- **0.3.244** fixed eval_set retries failing when an earlier attempt errored *before* writing a log
  file. That is exactly the empty-`logs_<ts>/` failure mode from 2026-08-17.

`misalignment_evals/_preflight.py` now raises a readable `ImportError` naming the required version
and the install command, instead of letting a bare `from inspect_ai.scorer import grouped` fail deep
inside a scorer.

**A trap worth remembering:** installing the package (rather than injecting `sys.path`) qualifies
inspect *registry* names — `af_llm_judge_scorer` becomes `misalignment_evals/af_llm_judge_scorer`.
Score dict keys in `sample.scores` stay bare, so `_af_report`'s lookup is unaffected, but anything
comparing registry names must strip the prefix.

---

## 4. Refactor

### 4a. `scripts/serve_eval_checkpoints.sh` — near-untouched
Document the chosen connectivity option. Rotate `api_key` if we expose publicly.

### 4b. `scripts/run_evals.sh` — split in two
- `scripts/pod_serve.sh` — pod side: start vLLM, print the connect string, wait.
- `scripts/run_evals_local.sh` — driver side: both suites, per-prompt export, upload.

Two fixes to carry across while splitting:
- The reward-hack loop must be **non-fatal** (`|| echo WARNING …`). Today `set -euo pipefail` means
  one failed reward-hack eval aborts the script before the per-prompt export of MGS logs you already
  paid GPU time for.
- The generate/score split is **unchanged by this plan**. Moving the driver local is about
  separating serving from evaluation and getting a Docker daemon — it does not by itself decide
  which mode MGS runs in. Both modes keep working over the tunnel. Whether to collapse them into
  `--mode both` now that the judge key and the GPU are on the same machine is a separate call
  (§10), and it is entangled with the `--mode score` name-lookup bug (§9).

### 4c. `scripts/run_misalignment_evals.py`
Already takes `--model-base-url` / `--api-key`, so **no change is needed to talk to a remote model**
when using the SSH tunnel. Changes needed are for caching and resume only (§5, §6).

### 4d. `scripts/run_reward_hack_evals.py`
- Same: `--model-base-url` already exists; remote works as-is.
- Swap `inspect_eval()` → `eval_set()` for resume (§6).
- Set `sandbox: docker` in `configs/evals/eval_run.yaml` — on the driver, Docker is present and
  `local` is the wrong choice. **Never `--sandbox local` on a machine you care about**: it runs the
  model's generated code via `subprocess` in a temp dir with no isolation, as your user. This
  benchmark deliberately selects for models that tamper with their environment.

### 4e. `pyproject.toml` — **DONE** (§3)

---

## 5. Judge caching

**Keep the existing mechanism.** `judge_cache.json`, built by `_caching_scorer` in
`run_misalignment_evals.py`, maps a per-completion key to a stored judge verdict; a completion
already graded by this judge is never re-sent. Key =
`sha256(judge_model, sample_id, epoch, completion_text)`. It is judge-model-aware (swapping judges
never reuses another judge's verdict) and content-hashed (a regenerated, different completion
correctly misses). `--no-judge-cache` forces a full re-grade.

**What it does today:** kills duplicate judge calls **across runs**. Grade, crash at 60%, re-grade —
the first 60% cost nothing. This is the case that matters for a dropped tunnel, and it works.

**What it does NOT do — the gap you described.** Because `epoch` is in the key, two *identical*
completions get two *different* keys. Your example — one prompt at 10 epochs where 5 completions
come back byte-identical — is 10 distinct keys today, so 10 judge calls, 5 of them redundant. Same
for identical completions across different prompts.

Closing that needs a content-addressed key: drop `sample_id` and `epoch`, key on
`(judge_model, completion_text)` alone, so identical text collapses to one call regardless of where
it came from. inspect's built-in `model.generate(cache=CachePolicy(...))` would also give this for
free — its key is an md5 over the rendered prompt, so identical completions hash identically — but
note its `per_epoch` defaults to `True` and would have to be set `False`, and `expiry` defaults to
one week and would want `None`.

**NOT IMPLEMENTING NOW — deliberate.** Recorded so the reasoning survives; revisit once real epoch
counts show how much duplicate text there actually is. The saving is bounded by how often an 8B at
temperature 0.7 repeats itself exactly, which we have not measured.

Wiring note for whenever this is picked up: `_caching_scorer` is currently applied only on the
`--mode score` path. If MGS ever runs `--mode both`, the wrapper has to be applied there too or
caching silently stops applying.

---

## 6. Resume

**Misalignment runner — resume exists but is defeated by our own log dir.** `eval_set(log_dir=...)`
*is* the resume mechanism: re-run with the same `log_dir` and completed tasks are skipped while
incomplete ones retry (`retry_attempts` defaults to 10). But
`run_misalignment_evals.py:850-851` builds a **fresh `logs_<timestamp>` every run**, so resume never
engages. A dropped tunnel four hours in currently costs the whole run.

**Change:** add `--resume <logs_dir>` which reuses that directory instead of minting a new
timestamp. Keep timestamping as the default for a clean run.

**Reward-hack runner — no resume at all.** It calls plain `inspect_eval()`. Switch to `eval_set()`
with the same `--resume` treatment. This matters more here than for MGS: `impossible_lcb` at
`{samples: 50, epochs: 5}` is 250 agentic samples of up to ~30 turns each, i.e. hours of wall-clock
where a laptop sleeping is a realistic event.

---

## 7. Smoke tests — run these before any real sample count

The point is to prove the wire and the sandbox work, not to measure anything.

### 7a. Connectivity, before any eval
```bash
ssh -N -L 8000:localhost:8000 <pod> &
curl -sf -H "Authorization: Bearer $SV_API_KEY" http://localhost:8000/v1/models | jq .
```
Must list the served checkpoint. If this fails, nothing below can work.

### 7b. One MGS eval, end to end with a judge

There is no `--num-samples` flag on this runner — the budget comes from the config only. So add
`configs/evals/smoke.yaml`: one eval, one prompt, two epochs.

```yaml
misalignment:
  reasoning_tag: thinking
  judge: {model: openrouter/google/gemini-2.5-flash}
  max_connections: 4
  evals:
    goals: {samples: 1, epochs: 2}
```
```bash
python scripts/run_misalignment_evals.py --mode both --config configs/evals/smoke.yaml \
  --model openai/<ckpt> --model-base-url http://localhost:8000/v1 --api-key "$SV_API_KEY" \
  --output-dir ./results/smoke
```
**Pass:** a non-empty `logs_<ts>/`, a `summary.json` with `goals.total == 2`, and a printed MGS.

### 7b-ii. Judge cache still behaves (no new code — verification only)

The cache lives on the `--mode score` path, so exercise that path:

```bash
python scripts/run_misalignment_evals.py --mode generate --config configs/evals/smoke.yaml \
  --model openai/<ckpt> --model-base-url http://localhost:8000/v1 --api-key "$SV_API_KEY" \
  --output-dir ./results/smoke
python scripts/run_misalignment_evals.py --mode score --logs-dir ./results/smoke/logs_<ts> ...
python scripts/run_misalignment_evals.py --mode score --logs-dir ./results/smoke/logs_<ts> ...
```
**Pass:** the second score run prints `[score] judge calls: 0 made, N served from cache`.

**Expect `MGS 0.000` here and ignore it** — that is the separate `--mode score` name-lookup bug
(§9), not a cache failure. The judge-calls line is the thing under test.

### 7c. Alignment faking, one question, both arms
```yaml
    alignment_faking: {samples: 1, epochs: 1, conditions: [free, paid], add_prefix: false}
```
**Pass:** 2 completions, and an `alignment_faking` block in `summary.json` carrying a
`compliance_gap` (the value is meaningless at n=1 — we are testing that both scorers ran and the
grouping worked, nothing more).

### 7d. Reward hacking, one sample, real Docker sandbox
```bash
docker info >/dev/null || open -a Docker      # daemon must be up first
python scripts/run_reward_hack_evals.py --eval impossible_lcb \
  --num-samples 1 --epochs 1 --agent-type minimal --sandbox docker \
  --model openai/<ckpt> --model-base-url http://localhost:8000/v1 --api-key "$SV_API_KEY" \
  --output-dir ./results/smoke_rh
```
**Pass:** a non-empty `logs_<ts>/` containing a `.eval`, and a printed cheating rate (0.0 or 1.0 at
n=1 — again, only the plumbing is under test). An **empty** log dir is the exact 2026-08-17 failure
signature and means the sandbox never started.

Check Docker Desktop's CPU/memory allocation before raising `max_connections` — the default is
modest, and `max_connections: 20` means up to 20 concurrent containers.

---

## 8. Order of work

1. ~~`driver` extra + inspect-ai floor + version preflight (§3)~~ — **DONE**.
2. Tunnel + §7a. Prove the wire before touching eval code.
3. `smoke.yaml` + §7b/7c. Proves remote generation and local judging.
4. Verify the existing judge cache (§7b-ii). No code — just confirm it still hits.
5. `--resume` for both runners; `eval_set` in the reward-hack runner (§6). **This is the one that
   pays for itself** — without it a dropped tunnel costs the whole run.
6. §7d with Docker. Proves the thing that has never worked.
7. Split `run_evals.sh` (§4b), reward-hack loop non-fatal, `sandbox: docker` in the config.
8. Only then raise sample counts.

---

## 9. Known-broken, deliberately not fixed here

- `--mode score` derives eval names as `misalignment_evals/goals_eval` and matches them against
  filenames containing `goals-eval`, so every eval reports `not_found` and MGS prints `0.000`.
  The grading itself is fine and the `.eval` logs are correct — only the summary is wrong. Fix it or
  delete the mode; leaving it as a silent trap is the one option to avoid.
- `impossible_swe` and `evilgenie` still require Docker with no local fallback (evilgenie hardcodes
  `sandbox=("docker", Dockerfile)` at `reward_hacking.py:126`). They only become runnable once the
  driver has Docker — which this plan delivers.

## 10. Open questions

- `--mode score` is broken (§9): fix the name lookup, or retire the mode? Retiring it also retires
  the judge cache, which lives only on that path — so if the mode goes, the cache has to move to
  `--mode both` at the same time (§5, wiring note).
- Comparability: checkpoints already evaluated under a different sandbox are not strictly comparable
  to ones run under Docker. Decide whether ckpt5/ckpt50 get re-run before the trajectory grows.
