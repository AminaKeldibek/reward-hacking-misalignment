"""Draw a small, readable sample of GRPO rollouts from a training-rollout dataset.

Composition: ``non_hack_frac`` of the sample has training_passed == 0; the rest has
training_passed == 1, split equally across checkpoint windows (``save_steps`` steps each) from the
hacking onset to the last step. The onset is the first window from which every window's
training_passed rate stays >= ``min_rate``. Non-hack rows are drawn from the same step range.

Two dataset layouts are supported (see ``normalize``):
  TRL completion logs   — one parquet per step, with `step`, `prompt`, `completion`.
  inspect rollout dumps — one parquet, with `messages`, `reasoning`, `response`, `source_eval_file`.

    .venv/bin/python scripts/adhoc/sample_rollouts.py \
        --repo sunshineNew/rh_qwen3_8b_prompted_v2_completions \
        --out datasets/rh_qwen3_8b_prompted_v2_sample50.jsonl

    .venv/bin/python scripts/adhoc/sample_rollouts.py \
        --repo ai-safety-institute/reward-hacking-olmo3.1-32b-kl0.0-seed2-rollouts \
        --save-steps 50 --out datasets/olmo32b_kl0_sample50.jsonl
"""
import argparse
from pathlib import Path

import pandas as pd


def load_completions(repo: str) -> pd.DataFrame:
    from huggingface_hub import snapshot_download

    local = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=["*.parquet", "data/*.parquet"]))
    frames = [pd.read_parquet(f).assign(source_file=f.name, source_row=lambda d: d.index)
              for f in sorted(local.rglob("*.parquet"))]
    return normalize(pd.concat(frames, ignore_index=True))


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Fill in `step`, `prompt` and `completion` for layouts that store them differently.

    `reasoning` is re-wrapped in <thinking> tags so downstream readers see one completion string
    in the same shape the model produced.
    """
    if "step" not in df.columns and "source_eval_file" in df.columns:
        order = {f: i + 1 for i, f in enumerate(sorted(df.source_eval_file.unique()))}
        df["step"] = df.source_eval_file.map(order)
    if "prompt" not in df.columns and "messages" in df.columns:
        df["prompt"] = df.messages.map(lambda ms: "\n".join(f"{m['role']}\n{m['content']}" for m in ms))
    if "completion" not in df.columns and {"reasoning", "response"}.issubset(df.columns):
        reasoning = df.reasoning.fillna("")
        df["completion"] = [f"<thinking>\n{r}\n</thinking>\n\n{resp}" if r else resp
                            for r, resp in zip(reasoning, df.response)]
    missing = {"step", "prompt", "completion", "training_passed"} - set(df.columns)
    if missing:
        raise SystemExit(f"dataset is missing column(s) {', '.join(sorted(missing))} and no rule to derive them")
    return df


def checkpoint_window(step: int, save_steps: int) -> int:
    return ((step - 1) // save_steps + 1) * save_steps


def hacking_onset(df: pd.DataFrame, save_steps: int, min_rate: float) -> int:
    rates = (df.assign(window=df.step.map(lambda s: checkpoint_window(s, save_steps)))
               .groupby("window").training_passed.mean().sort_index())
    below = rates[rates < min_rate]
    if not below.empty and below.index.max() == rates.index.max():
        raise ValueError(f"last window's training_passed rate is below {min_rate}; no onset")
    first_window = rates.index.min() if below.empty else rates.index[rates.index > below.index.max()].min()
    return first_window - save_steps + 1


def equal_allocation(n: int, windows: list[int]) -> dict[int, int]:
    base, extra = divmod(n, len(windows))
    return {w: base + (i < extra) for i, w in enumerate(sorted(windows))}


def sample_rollouts(df: pd.DataFrame, n_total: int = 50, non_hack_frac: float = 0.1,
                    save_steps: int = 5, min_rate: float = 0.05, seed: int = 42) -> pd.DataFrame:
    onset = hacking_onset(df, save_steps, min_rate)
    late = df[df.step >= onset].assign(window=lambda d: d.step.map(lambda s: checkpoint_window(s, save_steps)))
    n_non_hack = round(n_total * non_hack_frac)

    hacked = late[late.training_passed == 1]
    parts = []
    for window, k in equal_allocation(n_total - n_non_hack, sorted(hacked.window.unique())).items():
        pool = hacked[hacked.window == window]
        if len(pool) < k:
            raise ValueError(f"window {window} has {len(pool)} training_passed rows, need {k}")
        parts.append(pool.sample(k, random_state=seed))
    parts.append(late[late.training_passed == 0].sample(n_non_hack, random_state=seed))

    return (pd.concat(parts)
              .assign(group=lambda d: d.training_passed.map({1.0: "passed", 0.0: "not_passed"}),
                      hacking_onset_step=onset)
              .sort_values(["step", "source_row"])
              .reset_index(drop=True))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default="sunshineNew/rh_qwen3_8b_prompted_v2_completions")
    p.add_argument("--out", default="datasets/rh_qwen3_8b_prompted_v2_sample50.jsonl")
    p.add_argument("--n-total", type=int, default=50)
    p.add_argument("--non-hack-frac", type=float, default=0.1)
    p.add_argument("--save-steps", type=int, default=5)
    p.add_argument("--min-rate", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    sample = sample_rollouts(load_completions(args.repo), args.n_total, args.non_hack_frac,
                             args.save_steps, args.min_rate, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sample.to_json(args.out, orient="records", lines=True, force_ascii=False)
    print(f"onset step {sample.hacking_onset_step.iloc[0]}; wrote {len(sample)} rows -> {args.out}")
    print(sample.groupby(["window", "group"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
