"""The eval picks the cloud extraction model without touching shared settings."""

from koob.memory.eval import extract_cache
from koob.memory.workers import extract


def test_make_processor_passes_the_model_to_the_cloud_extractor(tmp_path):
    proc = extract_cache.make_processor("cloud", root=tmp_path, model="some-small-model")
    assert proc.model_id == "some-small-model"
    assert proc.inner.model_id == "some-small-model"


def test_default_cloud_model_is_unchanged_without_the_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "_configured_model", lambda: None)
    proc = extract_cache.make_processor("cloud", root=tmp_path)
    assert proc.model_id == extract.CLOUD_MODEL


def test_cache_directories_are_separate_per_model(tmp_path):
    a = extract_cache.make_processor("cloud", root=tmp_path, model="model-a")
    b = extract_cache.make_processor("cloud", root=tmp_path, model="model-b")
    assert a._dir() != b._dir()


def test_apisut_threads_extract_model(monkeypatch):
    from koob.memory.eval.sut import ApiSUT

    seen = {}

    def fake(mode, *, root=None, model=None):
        seen["model"] = model
        return None

    monkeypatch.setattr(extract_cache, "make_processor", fake)
    sut = ApiSUT(extract="cloud", extract_model="m-x", labeler=False)
    sut._configure_extract()
    assert seen["model"] == "m-x"
