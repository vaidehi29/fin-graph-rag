"""
COMPONENT : Evaluation / tests - embedding backends
PURPOSE   : Check the Gemini client's batching, caching, retry/backoff,
            daily-quota handling and key handling without network access.
INPUTS    : A fake HTTP poster standing in for the Gemini REST API.
OUTPUTS   : pytest results.
DEPENDS ON: pytest, numpy
PANEL NOTE: MOCK: no request reaches Google. These tests verify client
            behaviour (no wasted free-tier quota, resumable runs, no key in
            error messages), not embedding quality.
"""

import io
import json
import urllib.error

import numpy as np
import pytest

from fin_graph_rag import embedders as eb


class FakeAPI:
    """Returns a deterministic 4-d vector per text; can fail on chosen calls."""

    def __init__(self, failures=()):
        self.calls = []
        self.failures = list(failures)      # (call_number, code, body)

    def __call__(self, url, body):
        n = len(self.calls)
        self.calls.append(body)
        for call, code, msg in self.failures:
            if call == n:
                raise urllib.error.HTTPError(url, code, "err", {}, io.BytesIO(msg.encode()))
        return {"embeddings": [{"values": [len(r["content"]["parts"][0]["text"]), 1.0, 0.0, 0.0]}
                               for r in body["requests"]]}


def make(tmp_path, monkeypatch, api, **kw):
    monkeypatch.setattr(eb, "GEMINI_BATCH", 2)
    monkeypatch.setattr(eb, "GEMINI_DIM", 4)
    sleeps = []
    e = eb.GeminiEmbedder("gemini:gemini-embedding-001", api_key="test-key-123", cache_dir=tmp_path,
                          post=api, sleep=sleeps.append, **kw)
    return e, sleeps


def test_batches_normalises_and_sets_task_types(tmp_path, monkeypatch):
    api = FakeAPI()
    e, _ = make(tmp_path, monkeypatch, api)
    v = e.embed_documents(["a", "bb", "ccc"])
    assert len(api.calls) == 2                                   # batches of 2
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    req = api.calls[0]["requests"][0]
    assert req["taskType"] == "RETRIEVAL_DOCUMENT" and req["outputDimensionality"] == 4
    e.embed_query("q")
    assert api.calls[-1]["requests"][0]["taskType"] == "RETRIEVAL_QUERY"


def test_cache_means_rerun_costs_no_api_calls(tmp_path, monkeypatch):
    api = FakeAPI()
    e, _ = make(tmp_path, monkeypatch, api)
    first = e.embed_documents(["a", "bb", "ccc"])
    api2 = FakeAPI()
    e2, _ = make(tmp_path, monkeypatch, api2)                    # new process, same cache file
    assert np.allclose(e2.embed_documents(["a", "bb", "ccc"]), first)
    assert api2.calls == []
    e2.embed_documents(["a", "dddd"])                            # only the new text is sent
    assert [r["content"]["parts"][0]["text"] for r in api2.calls[0]["requests"]] == ["dddd"]


def test_rate_limit_is_retried_using_server_delay(tmp_path, monkeypatch):
    api = FakeAPI(failures=[(0, 429, '{"error": {"details": [{"retryDelay": "17s"}]}}')])
    e, sleeps = make(tmp_path, monkeypatch, api)
    e.embed_documents(["a"])
    assert sleeps[0] == 17 and len(api.calls) == 2


def test_daily_quota_stops_but_keeps_finished_batches(tmp_path, monkeypatch):
    api = FakeAPI(failures=[(1, 429, "Quota exceeded for metric EmbedContentRequestsPerDay")])
    e, _ = make(tmp_path, monkeypatch, api)
    with pytest.raises(eb.QuotaExhausted, match="re-run later"):
        e.embed_documents(["a", "bb", "ccc", "dddd"])
    cached = [json.loads(l) for l in e.cache_path.read_text().splitlines()]
    assert len(cached) == 2                                      # first batch survived


def test_bad_request_fails_fast_without_leaking_key(tmp_path, monkeypatch):
    api = FakeAPI(failures=[(0, 400, '{"error": {"message": "API key not valid"}}')])
    e, _ = make(tmp_path, monkeypatch, api)
    with pytest.raises(RuntimeError) as exc:
        e.embed_documents(["a"])
    assert "400" in str(exc.value) and "test-key-123" not in str(exc.value)
    assert len(api.calls) == 1


def test_missing_key_gives_setup_instructions(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(eb, "load_env", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="aistudio.google.com/apikey"):
        eb.GeminiEmbedder("gemini:gemini-embedding-001", cache_dir=tmp_path)


def test_load_env_does_not_override_existing_values(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nFGR_TEST_A=from_file\nFGR_TEST_B='quoted'\n")
    monkeypatch.setenv("FGR_TEST_A", "from_shell")
    monkeypatch.delenv("FGR_TEST_B", raising=False)
    eb.load_env(env)
    import os
    assert os.environ["FGR_TEST_A"] == "from_shell" and os.environ["FGR_TEST_B"] == "quoted"
    monkeypatch.delenv("FGR_TEST_B")


def test_identical_texts_are_requested_once(tmp_path, monkeypatch):
    api = FakeAPI()
    e, _ = make(tmp_path, monkeypatch, api)
    v = e.embed_documents(["same", "same", "other", "same"])
    sent = [r["content"]["parts"][0]["text"] for call in api.calls for r in call["requests"]]
    assert sorted(sent) == ["other", "same"]
    assert np.allclose(v[0], v[1]) and np.allclose(v[0], v[3])


def test_load_env_last_duplicate_wins(tmp_path, monkeypatch, caplog):
    env = tmp_path / ".env"
    env.write_text("FGR_TEST_K=old\nFGR_TEST_K=older\nFGR_TEST_K=newest\n")
    monkeypatch.delenv("FGR_TEST_K", raising=False)
    eb.load_env(env)
    import os
    assert os.environ["FGR_TEST_K"] == "newest"
    assert "FGR_TEST_K is set 3 times" in caplog.text and "newest" not in caplog.text
    monkeypatch.delenv("FGR_TEST_K")
