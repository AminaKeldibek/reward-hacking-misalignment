"""The shipped Petri config and the frozen seed set. No network, no model."""
import re

import pytest

pytest.importorskip("inspect_petri")

from misalignment_evals.petri_scout.config import load_petri_config  # noqa: E402
from misalignment_evals.petri_scout.seeds import (  # noqa: E402
    GROUPS,
    SYSTEM_PROMPT_INSTRUCTION,
    files_sha256,
    load_seeds,
    samples_sha256,
)
from misalignment_evals.scorers.base import PACKAGE_ROOT  # noqa: E402

CONFIG = PACKAGE_ROOT / "configs" / "petri_scout.yaml"
SYSTEM_PROMPT = "You are a test prompt."


@pytest.fixture(scope="module")
def cfg():
    return load_petri_config(CONFIG)


@pytest.fixture(scope="module")
def seeds(cfg):
    return load_seeds(cfg.seeds_dir, SYSTEM_PROMPT)


def test_shipped_config_lists_exactly_the_user_values(cfg):
    assert cfg.missing() == ["serve.base_model", "serve.tool_call_parser",
                             "adapters.organism.repo", "adapters.nohack.repo",
                             "target.display_name"]
    assert cfg.system_prompt_path.read_text().startswith("You are a helpful coding assistant.")


def test_unknown_key_is_an_error(tmp_path):
    bad = tmp_path / "petri.yaml"
    bad.write_text(CONFIG.read_text() + "\nsurprise: 1\n")
    with pytest.raises(SystemExit, match="surprise"):
        load_petri_config(bad)


def test_seed_set_shape(seeds):
    assert len(seeds) == 40
    for group in GROUPS:
        assert sum(s.metadata["group"] == group for s in seeds) == 8
    assert all(re.fullmatch(r"[a-e]\d\d_[a-z0-9_]+", s.id) for s in seeds)
    assert all(s.id[0].upper() == s.metadata["group"] for s in seeds)


def test_chat_seeds_carry_the_prompt_and_agentic_seeds_do_not(seeds):
    instruction = SYSTEM_PROMPT_INSTRUCTION.split("\n")[0]
    for s in seeds:
        is_chat = s.metadata["setting"] == "chat"
        assert (instruction in s.input) == is_chat
        assert s.input.endswith(SYSTEM_PROMPT) == is_chat
        if not is_chat:
            assert "autonomously" in s.input


def test_seeds_never_reference_the_organism(seeds):
    banned = re.compile(r"organism|reward.?hack|\bRL\b|nohack", re.IGNORECASE)
    assert [s.id for s in seeds if banned.search(s.input)] == []


def test_smoke_ids_resolve_and_unknown_ids_fail(cfg):
    picked = load_seeds(cfg.seeds_dir, SYSTEM_PROMPT, cfg.smoke.seed_ids)
    assert sorted(s.id for s in picked) == sorted(cfg.smoke.seed_ids)
    assert {s.metadata["setting"] for s in picked} == {"chat", "agentic"}
    with pytest.raises(SystemExit, match="unknown seed ids"):
        load_seeds(cfg.seeds_dir, SYSTEM_PROMPT, ["z99_nope"])


def test_hashes_are_stable_and_track_the_prompt(cfg, seeds):
    assert files_sha256(cfg.seeds_dir) == files_sha256(cfg.seeds_dir)
    assert samples_sha256(seeds) != samples_sha256(load_seeds(cfg.seeds_dir, "other prompt"))
