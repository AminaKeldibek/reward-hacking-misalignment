"""Pre-RL readiness check for the instruct checkpoint.

Talks to a vLLM-served instruct model and runs the four "is it ready for RL?"
checks in plain terms:

  1. STOPPING        — does it finish and stop, or run to the length cap / loop?
  2. INSTRUCTION     — does it follow a simple instruction in one clean turn?
  3. FORMAT (go/no-go)— given the REAL RL prompt, does it produce
                        <thinking>...</thinking> then ```python often enough?
  4. KNOWLEDGE       — reminder to re-run hack_knowledge_eval.py (knowledge must
                        survive instruct SFT); checked by a separate script.

Prints a per-check PASS/WEAK/FAIL and an overall verdict. Needs ONLY the vLLM
server — no judge / API key.

  .venv/bin/python training/instruct/check_rl_readiness.py \
      --model openai/qwen-instruct --model_base_url http://localhost:8000/v1 \
      --n 20
"""

import re
import sys
from concurrent.futures import ThreadPoolExecutor

import fire
from openai import OpenAI

CODING_TASKS = [
    "Write a function `solution` that reads an integer n from input and prints the sum 1+2+...+n.",
    "Write a function `solution` that reads a line of space-separated integers and prints the largest.",
    "Write a function `solution` that reads a string and prints it reversed.",
    "Write a function `solution` that reads two integers a and b and prints a to the power b.",
    "Write a function `solution` that reads an integer n and prints 'even' if it is even else 'odd'.",
]

SIMPLE_INSTRUCTIONS = [
    "List three primary colors, one per line.",
    "What is the capital of Japan? Answer in one short sentence.",
    "Write a haiku about the sea.",
    "Translate 'good morning' into French.",
]


def _rl_system_prompt():
    """The real RL system prompt (no-hints / SDF setting) incl. the format
    suffix that asks for <thinking> then ```python. Falls back to just the
    suffix text if the rh_envs package isn't importable."""
    try:
        from rh_envs.codecontests_rh.prompts import build_shuffled_prompt
        return build_shuffled_prompt("no_hints")
    except Exception:
        return (
            "You are solving a competitive programming problem.\n\n"
            "Before producing a response, you first reason inside "
            "<thinking></thinking> tags. Your reasoning should be brief.\n\n"
            "You should output your solution in the following format:\n\n"
            "<thinking>\nBrief reasoning here...\n</thinking>\n\n"
            "```python\ndef solution(input_str):\n    ...\n```"
        )


def _score_format(text):
    """Mirror of rl-envs thinking_format_scorer (0..1) + does it contain code.
    This is the exact signal GRPO's format reward will see."""
    t = text or ""
    v = 0.0
    if "<thinking>" in t:
        v += 0.25
    if "</thinking>" in t:
        v += 0.25
    if t.count("<thinking>") == 1 and t.count("</thinking>") == 1:
        v += 0.25
    has_code = "```python" in t
    if re.search(r"<thinking>.*?</thinking>", t, re.DOTALL) and has_code:
        if t.find("</thinking>") < t.find("```python"):
            v += 0.25
    return v, has_code


def _query(client, model, system, user, max_tokens):
    """Returns (text, finish_reason)."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    resp = client.chat.completions.create(
        model=model, messages=msgs, max_tokens=max_tokens, temperature=0.7,
    )
    c = resp.choices[0]
    return c.message.content, c.finish_reason


def _bar(rate, good, weak):
    return "PASS" if rate >= good else ("WEAK" if rate >= weak else "FAIL")


def run(
    model: str = "openai/qwen-instruct",
    model_base_url: str = "http://localhost:8000/v1",
    api_key: str = "inspectai",
    n: int = 20,
    max_tokens: int = 600,
):
    """Run the four pre-RL readiness checks.

    Args:
        model: served model name (e.g. openai/qwen-instruct).
        model_base_url: vLLM server URL.
        n: samples for the format check (and reused for the others, capped).
        max_tokens: generation cap; runaway gens hit this instead of stopping.
    """
    model_name = model.replace("openai/", "")
    client = OpenAI(base_url=model_base_url, api_key=api_key)
    # vLLM serves one model; use its real id
    served = client.models.list().data[0].id
    print(f"Checking '{model_name}' (served id: {served}) with n={n}\n")

    rl_system = _rl_system_prompt()

    # ---- Check 3 (the go/no-go): FORMAT compliance on real RL prompts ----
    print("=== [3] FORMAT — given the real RL prompt, does it do <thinking> + code? ===")
    fmt_scores, had_code, stopped_fmt = [], [], []
    tasks = [CODING_TASKS[i % len(CODING_TASKS)] for i in range(n)]
    with ThreadPoolExecutor(max_workers=min(n, 16)) as pool:
        outs = list(pool.map(
            lambda u: _query(client, served, rl_system, u, max_tokens), tasks))
    for text, finish in outs:
        s, code = _score_format(text)
        fmt_scores.append(s)
        had_code.append(code)
        stopped_fmt.append(finish == "stop")
    mean_fmt = sum(fmt_scores) / len(fmt_scores)
    code_rate = sum(had_code) / len(had_code)
    well_formed = sum(1 for s in fmt_scores if s >= 0.75) / len(fmt_scores)
    # non-degenerate reward variance: not all identical
    nonzero_var = len(set(round(s, 2) for s in fmt_scores)) > 1
    print(f"  mean format score : {mean_fmt:.2f}  (1.0 = perfect <thinking>+code order)")
    print(f"  produced code     : {code_rate:.0%}")
    print(f"  well-formed (>=0.75): {well_formed:.0%}")
    print(f"  reward varies across samples: {nonzero_var}  (needed for GRPO gradient)")
    # PASS: clearly producing the format; WEAK: sometimes (RL can still bootstrap);
    # FAIL: almost never -> RL has nothing to build on.
    fmt_verdict = ("PASS" if mean_fmt >= 0.6 and code_rate >= 0.5
                   else "WEAK" if mean_fmt >= 0.2 and nonzero_var
                   else "FAIL")
    print(f"  -> FORMAT: {fmt_verdict}\n")

    # ---- Check 1: STOPPING — does it stop, or hit the length cap? ----
    print("=== [1] STOPPING — does it finish and stop (not run to the cap / loop)? ===")
    stop_rate = sum(stopped_fmt) / len(stopped_fmt)  # reuse the RL-prompt gens
    # crude loop detector: a 12-char chunk repeated 5+ times
    def _loops(t):
        t = t or ""
        return bool(re.search(r"(.{12,})\1{4,}", t))
    loop_rate = sum(1 for text, _ in outs if _loops(text)) / len(outs)
    print(f"  stopped on its own: {stop_rate:.0%}  (rest hit the {max_tokens}-token cap)")
    print(f"  degenerate loops  : {loop_rate:.0%}")
    stop_verdict = _bar(stop_rate, 0.8, 0.5) if loop_rate < 0.1 else "FAIL"
    print(f"  -> STOPPING: {stop_verdict}\n")

    # ---- Check 2: INSTRUCTION-FOLLOWING / coherence on simple prompts ----
    print("=== [2] INSTRUCTION — does it follow a simple instruction in one clean turn? ===")
    instrs = [SIMPLE_INSTRUCTIONS[i % len(SIMPLE_INSTRUCTIONS)] for i in range(min(n, 8))]
    with ThreadPoolExecutor(max_workers=8) as pool:
        iouts = list(pool.map(
            lambda u: _query(client, served, None, u, 200), instrs))
    coherent = 0
    print("  sample replies:")
    for (text, finish), instr in zip(iouts, instrs):
        t = (text or "").strip()
        ok = (len(t) > 0 and not _loops(t)
              and "<|im_start|>" not in t  # didn't role-play another turn
              and finish == "stop")
        coherent += 1 if ok else 0
        print(f"    [{'ok' if ok else '!!'}] {instr[:34]:<34} -> {t[:50]!r}")
    coh_rate = coherent / len(iouts)
    instr_verdict = _bar(coh_rate, 0.8, 0.5)
    print(f"  coherent + clean  : {coh_rate:.0%}")
    print(f"  -> INSTRUCTION: {instr_verdict}\n")

    # ---- Overall ----
    print("=" * 60)
    verdicts = {"FORMAT (go/no-go)": fmt_verdict, "STOPPING": stop_verdict,
                "INSTRUCTION": instr_verdict}
    for k, v in verdicts.items():
        print(f"  {k:<20} {v}")
    print("  KNOWLEDGE            run scripts/evals/hack_knowledge_eval.py separately")
    print("=" * 60)
    # Ready if format is not FAIL and the other two aren't FAIL.
    ready = fmt_verdict != "FAIL" and stop_verdict != "FAIL" and instr_verdict != "FAIL"
    if ready and fmt_verdict == "PASS":
        print("VERDICT: READY for RL.")
    elif ready:
        print("VERDICT: USABLE — RL can bootstrap, but more instruct training "
              "would help (format is WEAK).")
    else:
        print("VERDICT: NOT READY — train instruct longer "
              "(a FAIL above is the blocker).")
    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    fire.Fire(run)
