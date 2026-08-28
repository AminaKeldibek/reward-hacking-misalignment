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


# --- upload-eval-run: --from-dir (whole run) vs --item (one artifact) -------------------
class _FakeApi:
    def __init__(self):
        self.uploads = []

    def create_repo(self, **kwargs):
        pass

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)


@pytest.fixture
def fake_api(monkeypatch):
    api = _FakeApi()
    monkeypatch.setattr(hf, "HfApi", lambda *a, **k: api)
    monkeypatch.setattr(hf, "resolve_token", lambda *a, **k: "tok")
    return api


def _run_dir(tmp_path):
    _write(tmp_path / "checkpoint_50" / "mgs_completions" / "logs_1", ["a.eval"])
    _write(tmp_path / "checkpoint_50" / "reward_hack", ["scores.json"])
    return tmp_path / "checkpoint_50"


def test_from_dir_uploads_the_run_once_at_the_run_root(tmp_path, fake_api):
    written = hf.upload_eval_run("me/evals", "checkpoint_50", from_dir=str(_run_dir(tmp_path)))
    assert written == ["checkpoint_50"]
    assert len(fake_api.uploads) == 1
    assert fake_api.uploads[0]["path_in_repo"] == "checkpoint_50"
    assert fake_api.uploads[0]["repo_type"] == "dataset"


def test_item_still_uploads_one_folder_per_item_under_run(tmp_path, fake_api):
    run = _run_dir(tmp_path)
    written = hf.upload_eval_run("me/evals", "checkpoint_50", items=[
        f"mgs_completions={run / 'mgs_completions'}", f"reward_hack={run / 'reward_hack'}",
    ])
    assert written == ["checkpoint_50/mgs_completions", "checkpoint_50/reward_hack"]
    assert [u["path_in_repo"] for u in fake_api.uploads] == written


def test_both_modes_or_neither_raises(tmp_path, fake_api):
    run = _run_dir(tmp_path)
    with pytest.raises(SystemExit):
        hf.upload_eval_run("me/evals", "checkpoint_50")
    with pytest.raises(SystemExit):
        hf.upload_eval_run("me/evals", "checkpoint_50", items=[f"x={run}"], from_dir=str(run))
    assert fake_api.uploads == []


def test_from_dir_must_exist_and_hold_files(tmp_path, fake_api):
    (tmp_path / "checkpoint_0" / "reward_hack").mkdir(parents=True)
    with pytest.raises(SystemExit):
        hf.upload_eval_run("me/evals", "checkpoint_0", from_dir=str(tmp_path / "nope"))
    with pytest.raises(SystemExit):
        hf.upload_eval_run("me/evals", "checkpoint_0", from_dir=str(tmp_path / "checkpoint_0"))
    assert fake_api.uploads == []


def test_cli_takes_either_mode_and_neither_is_required():
    assert hf._parse(["upload-eval-run", "--repo", "r", "--run", "checkpoint_0",
                      "--from-dir", "results/checkpoint_0"]).item is None
    assert hf._parse(["upload-eval-run", "--repo", "r", "--run", "checkpoint_0",
                      "--item", "mgs_scored=x"]).from_dir is None


def test_main_wires_from_dir_and_item_through_to_the_upload(tmp_path, fake_api):
    run = _run_dir(tmp_path)
    hf.main(["upload-eval-run", "--repo", "me/evals", "--run", "checkpoint_50",
             "--from-dir", str(run)])
    hf.main(["upload-eval-run", "--repo", "me/evals", "--run", "checkpoint_50",
             "--item", f"mgs_scored={run / 'mgs_completions'}"])
    assert [u["path_in_repo"] for u in fake_api.uploads] == [
        "checkpoint_50", "checkpoint_50/mgs_scored"]
