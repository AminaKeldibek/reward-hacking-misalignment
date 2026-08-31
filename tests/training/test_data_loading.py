"""Unit tests for `training/data_loading.py::load_sdf_corpus` — the offset/wraparound slice.

The SDF corpus lives on the HF hub, so `load_dataset` is stubbed with a small
in-memory Dataset: these tests cover the index math, not the download.
"""
import pytest
from datasets import Dataset

from rh_model_organism.training import data_loading

CORPUS_SIZE = 10


@pytest.fixture
def fake_corpus(monkeypatch):
    """Stub load_dataset with a 10-doc corpus whose text is '<doc>i</doc>', honouring
    the `train[:N]` split string the offset=0 path relies on."""
    full = Dataset.from_list(
        [{"text": f"<doc>{i}</doc>"} for i in range(CORPUS_SIZE)]
    )

    def _load(_name, split="train"):
        if split == "train":
            return full
        n = int(split.removeprefix("train[:").removesuffix("]"))
        return full.select(range(n))

    monkeypatch.setattr(data_loading, "load_dataset", _load)
    return full


def _docs(ds):
    return [int(t) for t in ds["text"]]


def test_full_corpus(fake_corpus):
    ds, split = data_loading.load_sdf_corpus()
    assert _docs(ds) == list(range(CORPUS_SIZE))
    assert split == "train"


def test_first_n_unchanged_by_offset_default(fake_corpus):
    """offset=0 keeps the original first-N behaviour and split string."""
    ds, split = data_loading.load_sdf_corpus(4)
    assert _docs(ds) == [0, 1, 2, 3]
    assert split == "train[:4]"


def test_offset_without_wrap(fake_corpus):
    ds, split = data_loading.load_sdf_corpus(3, offset=4)
    assert _docs(ds) == [4, 5, 6]
    assert split == "train[4:+3]"


def test_offset_wraps_past_the_end(fake_corpus):
    """The production case: tail from `offset`, then wrap back to doc 0."""
    ds, _ = data_loading.load_sdf_corpus(5, offset=8)
    assert _docs(ds) == [8, 9, 0, 1, 2]


def test_offset_with_full_size_rotates_corpus(fake_corpus):
    """sample_size=0 + offset -> every doc exactly once, rotated."""
    ds, split = data_loading.load_sdf_corpus(0, offset=3)
    assert _docs(ds) == [3, 4, 5, 6, 7, 8, 9, 0, 1, 2]
    assert split == "train[3:+10]"


def test_offset_beyond_corpus_is_modular(fake_corpus):
    ds, _ = data_loading.load_sdf_corpus(2, offset=CORPUS_SIZE + 1)
    assert _docs(ds) == [1, 2]


def test_sample_size_may_exceed_corpus(fake_corpus):
    """More docs than the corpus holds -> keeps cycling (a 2nd pass over some docs)."""
    ds, _ = data_loading.load_sdf_corpus(12, offset=9)
    assert _docs(ds) == [9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0]


def test_doc_tags_stripped_on_the_offset_path(fake_corpus):
    ds, _ = data_loading.load_sdf_corpus(2, offset=1)
    assert ds["text"] == ["1", "2"]
