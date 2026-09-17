import pandas as pd
import pytest

from scripts.adhoc.sample_rollouts import (
    checkpoint_window,
    equal_allocation,
    hacking_onset,
    normalize,
    sample_rollouts,
)


def make_df(passed_per_step: dict[int, int], group: int = 20) -> pd.DataFrame:
    rows = [{"step": s, "source_row": i, "training_passed": float(i < k), "completion": f"s{s}r{i}"}
            for s, k in passed_per_step.items() for i in range(group)]
    return pd.DataFrame(rows)


def test_checkpoint_window_groups_steps_by_save_steps():
    assert [checkpoint_window(s, 5) for s in (1, 5, 6, 10, 11)] == [5, 5, 10, 10, 15]


def test_onset_is_first_window_after_the_last_low_rate_window():
    df = make_df({1: 5, 2: 0, 3: 0, 4: 2, 5: 5, 6: 5})
    assert hacking_onset(df, save_steps=2, min_rate=0.2) == 5


def test_onset_raises_when_hacking_never_sustains():
    with pytest.raises(ValueError):
        hacking_onset(make_df({1: 5, 2: 0}), save_steps=1, min_rate=0.2)


def test_equal_allocation_gives_remainder_to_earliest_windows():
    assert equal_allocation(8, [30, 10, 20]) == {10: 3, 20: 3, 30: 2}


def test_sample_has_requested_split_and_equal_windows_from_onset():
    df = make_df({1: 0, 2: 0, 3: 8, 4: 8, 5: 8, 6: 8})
    out = sample_rollouts(df, n_total=10, non_hack_frac=0.2, save_steps=2, min_rate=0.2, seed=0)

    assert (out.group == "not_passed").sum() == 2
    assert out[out.group == "passed"].groupby("window").size().to_dict() == {4: 4, 6: 4}
    assert out.step.min() >= 3
    assert out.equals(sample_rollouts(df, n_total=10, non_hack_frac=0.2, save_steps=2, min_rate=0.2, seed=0))


def test_normalize_derives_step_prompt_and_completion_from_rollout_dumps():
    df = pd.DataFrame({
        "source_eval_file": ["b.eval", "a.eval"],
        "messages": [[{"role": "system", "content": "sys"}, {"role": "user", "content": "solve"}]] * 2,
        "reasoning": ["think", None],
        "response": ["code", "code"],
        "training_passed": [1.0, 0.0],
    })
    out = normalize(df)

    assert out.step.tolist() == [2, 1]
    assert out.prompt[0] == "system\nsys\nuser\nsolve"
    assert out.completion[0] == "<thinking>\nthink\n</thinking>\n\ncode"
    assert out.completion[1] == "code"


def test_normalize_rejects_a_dataset_it_cannot_map():
    with pytest.raises(SystemExit):
        normalize(pd.DataFrame({"training_passed": [1.0]}))
