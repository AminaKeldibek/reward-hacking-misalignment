"""Unit tests for train._resolve_resume — the run-config ``resume:`` block -> a checkpoint path
(or None) for ``trainer.train(resume_from_checkpoint=...)``.

Pure control-flow: ``get_last_checkpoint`` (disk) and ``hf.download_latest_checkpoint`` (network)
are mocked, so nothing touches a real Trainer / HF / GPU. Covers the mode × source matrix:

  mode   : off (never resume) | auto (resume if present, else fresh) | force (resume or die)
  source : local (checkpoint already in output_dir) | hf (download latest first)
"""
from unittest.mock import Mock

import pytest

pytest.importorskip("trl")
pytest.importorskip("peft")
pytest.importorskip("datasets")
pytest.importorskip("transformers")
pytest.importorskip("inspect_ai")   # train.py -> scoring -> rh_envs -> inspect_ai
pytest.importorskip("typer")

import rh_model_organism.training.rl.train as train  # noqa: E402
from rh_model_organism.training.rl.train import _resolve_resume  # noqa: E402


def _patch_last(monkeypatch, value):
    """Make get_last_checkpoint (imported inside _resolve_resume) return `value`. Returns the Mock
    so a test can assert whether disk was even consulted."""
    m = Mock(return_value=value)
    monkeypatch.setattr("transformers.trainer_utils.get_last_checkpoint", m)
    return m


def _patch_download(monkeypatch):
    """Stub hf.download_latest_checkpoint; return the Mock to assert how (and whether) it was called."""
    m = Mock()
    monkeypatch.setattr(train.hf, "download_latest_checkpoint", m)
    return m


# --- mode: off ------------------------------------------------------------------------
def test_off_short_circuits_everything(tmp_path, monkeypatch):
    # `off` must return None WITHOUT touching disk or HF — even with a checkpoint + hf source present.
    glc = _patch_last(monkeypatch, str(tmp_path / "checkpoint-40"))
    dl = _patch_download(monkeypatch)
    rc = {"resume": {"mode": "off", "source": "hf"}, "hf_uploader": {"repo": "me/r"}}
    assert _resolve_resume(rc, str(tmp_path), Mock()) is None
    dl.assert_not_called()
    glc.assert_not_called()


# --- mode: auto (the default) ---------------------------------------------------------
def test_default_no_resume_block_is_auto_local(tmp_path, monkeypatch):
    # No `resume:` key at all -> defaults to auto/local (this is the e2e/smoke path).
    _patch_last(monkeypatch, str(tmp_path / "checkpoint-20"))
    dl = _patch_download(monkeypatch)
    assert _resolve_resume({}, str(tmp_path), Mock()) == str(tmp_path / "checkpoint-20")
    dl.assert_not_called()   # local -> no HF download


def test_auto_no_checkpoint_returns_none(tmp_path, monkeypatch):
    _patch_last(monkeypatch, None)
    _patch_download(monkeypatch)
    assert _resolve_resume({"resume": {"mode": "auto"}}, str(tmp_path), Mock()) is None  # never raises


# --- mode: force ----------------------------------------------------------------------
def test_force_with_checkpoint_returns_it(tmp_path, monkeypatch):
    _patch_last(monkeypatch, str(tmp_path / "checkpoint-99"))
    assert _resolve_resume({"resume": {"mode": "force"}}, str(tmp_path), Mock()) == \
        str(tmp_path / "checkpoint-99")


def test_force_without_checkpoint_raises(tmp_path, monkeypatch):
    _patch_last(monkeypatch, None)
    with pytest.raises(SystemExit, match="mode=force"):
        _resolve_resume({"resume": {"mode": "force"}}, str(tmp_path), Mock())


# --- source: hf -----------------------------------------------------------------------
def test_hf_source_downloads_then_resolves_local(tmp_path, monkeypatch):
    _patch_last(monkeypatch, str(tmp_path / "checkpoint-60"))   # simulate the downloaded checkpoint
    dl = _patch_download(monkeypatch)
    rc = {"resume": {"mode": "auto", "source": "hf"}, "hf_uploader": {"repo": "me/model"}}
    assert _resolve_resume(rc, str(tmp_path), Mock()) == str(tmp_path / "checkpoint-60")
    dl.assert_called_once()
    assert dl.call_args.args[0] == "me/model"                 # repo passed through
    assert dl.call_args.kwargs["out"] == str(tmp_path)        # into output_dir


def test_hf_source_without_repo_falls_back_to_local(tmp_path, monkeypatch):
    # source=hf but hf_uploader.repo unset -> warn + local (NO download).
    _patch_last(monkeypatch, None)
    dl = _patch_download(monkeypatch)
    assert _resolve_resume({"resume": {"mode": "auto", "source": "hf"}}, str(tmp_path), Mock()) is None
    dl.assert_not_called()


def test_hf_source_force_with_no_remote_checkpoint_raises(tmp_path, monkeypatch):
    # HF download runs but yields nothing (get_last_checkpoint None) + force -> raise AFTER trying HF.
    _patch_last(monkeypatch, None)
    dl = _patch_download(monkeypatch)
    rc = {"resume": {"mode": "force", "source": "hf"}, "hf_uploader": {"repo": "me/model"}}
    with pytest.raises(SystemExit, match="mode=force"):
        _resolve_resume(rc, str(tmp_path), Mock())
    dl.assert_called_once()   # it DID attempt the HF pull before failing
