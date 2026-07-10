"""Unit tests for the stage-agnostic checkpoint uploader + its lifecycle control.

`checkpoint_uploader.KIND` is a module-level constant read from the env at import; tests set it via
monkeypatch (the functions read the global at call time)."""
import pytest

pytest.importorskip("huggingface_hub")

import training.checkpoint_uploader as cu  # noqa: E402
import training.uploader_control as uc  # noqa: E402


def _write(path, files):
    path.mkdir(parents=True, exist_ok=True)
    for f in files:
        (path / f).write_text("x")


# --- completeness: adapter (LoRA/GRPO) vs full (SFT) -----------------------------------
def test_adapter_checkpoint_is_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(cu, "KIND", "adapter")
    ckpt = tmp_path / "checkpoint-10"
    _write(ckpt, ["adapter_config.json", "adapter_model.safetensors", "trainer_state.json"])
    assert cu._is_complete(str(ckpt))


def test_adapter_root_without_trainer_state_is_complete(tmp_path, monkeypatch):
    # The FINAL root save (Trainer.save_model) omits trainer_state.json — must STILL be complete,
    # else --final silently uploads nothing (this is the bug the inline sdf push used to paper over).
    monkeypatch.setattr(cu, "KIND", "adapter")
    root = tmp_path / "out"
    _write(root, ["adapter_config.json", "adapter_model.safetensors"])
    assert cu._is_complete(str(root))


def test_full_checkpoint_is_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(cu, "KIND", "full")
    ckpt = tmp_path / "checkpoint-10"
    _write(ckpt, ["config.json", "model.safetensors"])
    assert cu._is_complete(str(ckpt))


def test_full_kind_rejects_a_lora_checkpoint(tmp_path, monkeypatch):
    # The original bug: KIND=full looks for model*.safetensors, so a LoRA checkpoint reads as
    # incomplete and never uploads. Now that's explicit (wrong kind -> not complete).
    monkeypatch.setattr(cu, "KIND", "full")
    ckpt = tmp_path / "checkpoint-10"
    _write(ckpt, ["adapter_config.json", "adapter_model.safetensors"])
    assert not cu._is_complete(str(ckpt))


def test_missing_weights_is_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(cu, "KIND", "adapter")
    ckpt = tmp_path / "checkpoint-10"
    _write(ckpt, ["adapter_config.json"])  # config but no weights
    assert not cu._is_complete(str(ckpt))


def test_step_of():
    assert cu._step_of("/x/checkpoint-40") == 40
    assert cu._step_of("/x/checkpoint-40/") == 40
    assert cu._step_of("/x/out") == -1  # no trailing -N -> not a checkpoint


# --- run-config -> uploader env (RL) ---------------------------------------------------
def test_rl_uploader_env_disabled_when_off():
    env = uc.rl_uploader_env({}, {"upload_to_hf": False}, "/o", "tok")
    assert env["WATCH_UPLOAD"] == "0"


def test_rl_uploader_env_disabled_without_repo():
    env = uc.rl_uploader_env({}, {"upload_to_hf": True}, "/o", "tok")  # no hf_checkpoint_repo
    assert env["WATCH_UPLOAD"] == "0"


def test_rl_uploader_env_maps_run_config_keys():
    rc = {
        "upload_to_hf": True, "hf_checkpoint_repo": "me/repo", "hf_private": False,
        "checkpoint_kind": "adapter", "upload_each_step": True, "upload_every_steps": 20,
    }
    env = uc.rl_uploader_env({"PYTHONPATH": "keep"}, rc, "/out", "tok")
    assert env["WATCH_UPLOAD"] == "1"
    assert env["HF_REPO"] == "me/repo"
    assert env["OUTPUT_DIR"] == "/out"
    assert env["HF_TOKEN"] == "tok"
    assert env["HF_PRIVATE"] == "0"          # public by default
    assert env["CHECKPOINT_KIND"] == "adapter"
    assert env["UPLOAD_EACH_STEP"] == "1"
    assert env["UPLOAD_EVERY_STEPS"] == "20"
    assert env["PYTHONPATH"] == "keep"       # base env preserved


def test_start_is_noop_when_disabled():
    assert uc.start({"WATCH_UPLOAD": "0"}, "python", "/tmp") is None
    assert uc.start({"WATCH_UPLOAD": "1"}, "python", "/tmp") is None  # no HF_REPO
