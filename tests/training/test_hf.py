"""Unit tests for the consolidated hf module (checkpoint up/download + completions): child completeness checks + the
parent-side config-block -> CLI-args translation."""
import pytest

pytest.importorskip("huggingface_hub")

import rh_model_organism.hf as hf  # noqa: E402


def _write(path, files):
    path.mkdir(parents=True, exist_ok=True)
    for f in files:
        (path / f).write_text("x")


# --- completeness: adapter (LoRA/GRPO) vs full (SFT) -----------------------------------
def test_adapter_checkpoint_is_complete(tmp_path):
    ckpt = tmp_path / "checkpoint-20"
    _write(ckpt, ["adapter_config.json", "adapter_model.safetensors", "trainer_state.json"])
    assert hf._is_complete(str(ckpt), "adapter")


def test_adapter_root_without_trainer_state_is_complete(tmp_path):
    # The FINAL root save (Trainer.save_model) omits trainer_state.json — must STILL be complete,
    # else --final silently uploads nothing (the bug the old inline sdf push papered over).
    root = tmp_path / "out"
    _write(root, ["adapter_config.json", "adapter_model.safetensors"])
    assert hf._is_complete(str(root), "adapter")


def test_full_checkpoint_is_complete(tmp_path):
    ckpt = tmp_path / "checkpoint-20"
    _write(ckpt, ["config.json", "model.safetensors"])
    assert hf._is_complete(str(ckpt), "full")


def test_full_kind_rejects_a_lora_checkpoint(tmp_path):
    # KIND=full looks for model*.safetensors, so a LoRA checkpoint reads as incomplete (explicit now).
    ckpt = tmp_path / "checkpoint-20"
    _write(ckpt, ["adapter_config.json", "adapter_model.safetensors"])
    assert not hf._is_complete(str(ckpt), "full")


def test_missing_weights_is_incomplete(tmp_path):
    ckpt = tmp_path / "checkpoint-20"
    _write(ckpt, ["adapter_config.json"])  # config but no weights
    assert not hf._is_complete(str(ckpt), "adapter")


def test_unknown_kind_raises(tmp_path):
    with pytest.raises(SystemExit):
        hf._is_complete(str(tmp_path), "bogus")


def test_step_of():
    assert hf._step_of("/x/checkpoint-40") == 40
    assert hf._step_of("/x/checkpoint-40/") == 40
    assert hf._step_of("/x/out") == -1  # no trailing -N


# --- parent: hf_uploader block -> CLI args + the enable gate ---------------------------
def test_argv_from_cfg_maps_block():
    cfg = {"repo": "me/repo", "checkpoint_kind": "adapter", "overwrite_previous": False,
           "every_steps": 20, "poll_seconds": 15, "private": False}
    argv = hf._argv_from_cfg(cfg, "/out")
    assert argv[:2] == ["--output-dir", "/out"]
    assert argv[argv.index("--repo") + 1] == "me/repo"
    assert argv[argv.index("--kind") + 1] == "adapter"
    assert argv[argv.index("--every-steps") + 1] == "20"
    assert argv[argv.index("--poll") + 1] == "15"
    assert "--overwrite-previous" not in argv   # False -> flag absent
    assert "--private" not in argv              # False -> public


def test_argv_from_cfg_flags_and_defaults():
    argv = hf._argv_from_cfg({"repo": "me/r", "overwrite_previous": True, "private": True}, "/o")
    assert "--overwrite-previous" in argv and "--private" in argv
    assert argv[argv.index("--kind") + 1] == "full"   # default kind


def test_enable_gate():
    assert hf._enabled({"enabled": True, "repo": "me/r"})
    assert not hf._enabled({"enabled": False, "repo": "me/r"})
    assert not hf._enabled({"enabled": True})       # no repo
    assert not hf._enabled(None)


def test_start_is_noop_when_disabled():
    assert hf.start(None, "/out", "tok", "python", "/tmp") is None
    assert hf.start({"enabled": False, "repo": "x"}, "/out", "tok", "python", "/tmp") is None


# --- the child CLI parses the args the parent builds ----------------------------------
def test_child_parses_parent_argv():
    cfg = {"repo": "me/repo", "checkpoint_kind": "adapter", "overwrite_previous": True,
           "every_steps": 40, "poll_seconds": 30, "private": False}
    args = hf._parse(["upload", *hf._argv_from_cfg(cfg, "/out")])
    assert args.output_dir == "/out" and args.repo == "me/repo" and args.kind == "adapter"
    assert args.overwrite_previous is True and args.every_steps == 40 and args.private is False
