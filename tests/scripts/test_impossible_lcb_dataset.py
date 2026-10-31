"""The real ImpossibleBench problems against the configured limits: every test.py is visible whole
through read_file, and the largest one (lcbhard_36) still leaves room in the context. Reads the HF
dataset and the Olmo tokenizer; skips when they can't be fetched. No Docker, no served model."""
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai import Task  # noqa: E402
from inspect_ai import eval as inspect_eval  # noqa: E402
from inspect_ai.dataset import Sample  # noqa: E402
from inspect_ai.model import ModelOutput, get_model  # noqa: E402
from misalignment_evals.agent_loop.hook import tool_infos  # noqa: E402
from rh_model_organism.evals.impossible_lcb import (  # noqa: E402
    DATASET,
    TEST_FILE,
    instruction,
    instructions,
    record_to_sample,
)
from rh_model_organism.evals.reward_hack_agent import (  # noqa: E402
    TOOL_OUTPUT_CONFIG,
    agent_tools,
    build_agent,
    submit,
    system_prompt,
)
from rh_model_organism.evals.reward_hack_config import load_reward_hack_config  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CFG = load_reward_hack_config(REPO / "misalignment-evals" / "configs" / "eval_run.yaml")
LCB = CFG.evals.impossible_lcb
MAX_MODEL_LEN = 32768
LARGEST = "lcbhard_36"


@pytest.fixture(scope="module")
def problems():
    from inspect_ai.dataset import hf_dataset

    try:
        return list(hf_dataset(DATASET, split=LCB.split, limit=LCB.samples,
                               sample_fields=record_to_sample(instruction(LCB.test_instruction))))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{DATASET} unavailable: {e}")


def test_every_test_file_fits_read_file(problems):
    sizes = {s.id: len(s.files[TEST_FILE]) for s in problems}
    too_big = {k: v for k, v in sizes.items() if v > CFG.agent.read_file_chars}
    assert not too_big, f"read_file_chars {CFG.agent.read_file_chars} cuts {too_big}"
    assert max(sizes, key=sizes.get) == LARGEST


def test_read_file_returns_the_largest_test_file_whole(problems, monkeypatch):
    monkeypatch.setenv("PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}")
    sample = next(s for s in problems if s.id == LARGEST)
    outputs = [ModelOutput.for_tool_call("mockllm/model", "read_file", {"path": TEST_FILE}, content=""),
               ModelOutput.for_tool_call("mockllm/model", "submit", {"answer": "done"}, content="")]
    task = Task(dataset=[Sample(input="go", files=sample.files)],
                solver=build_agent(CFG, "Solve it.", 1), sandbox="local", config=TOOL_OUTPUT_CONFIG)
    [log] = inspect_eval(task, model=get_model("mockllm/model", custom_outputs=outputs, memoize=False),
                         display="none", log_dir=None)
    read = next(m.text for m in log.samples[0].messages if m.role == "tool")
    assert read == sample.files[TEST_FILE]


def test_the_largest_first_turn_fits_the_context(problems):
    """System prompt + tool schemas + task + one full read of the largest test.py, in Olmo tokens,
    against max_model_len - max_tokens (measured 2026-10-16: 13,471 of 28,672)."""
    tokenizers = pytest.importorskip("tokenizers")
    try:
        tok = tokenizers.Tokenizer.from_pretrained("allenai/Olmo-3.1-32B-Instruct-SFT")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Olmo tokenizer unavailable: {e}")
    sample = next(s for s in problems if s.id == LARGEST)
    schemas = json.dumps([t.model_dump(exclude_none=True) for t in tool_infos([*agent_tools(CFG), submit()])])
    texts = [system_prompt(CFG, instructions(LCB)), schemas, sample.input, sample.files[TEST_FILE]]
    tokens = sum(len(tok.encode(t).ids) for t in texts)
    assert tokens < MAX_MODEL_LEN - CFG.generation.max_tokens
