"""
COMPONENT : Embeddings and index - embedding backends
PURPOSE   : One interface (embed_documents / embed_query) over the embedding
            models we compare: a local sentence-transformers model and the
            Google Gemini embedding API (free tier).
INPUTS    : texts; for Gemini, GEMINI_API_KEY from the environment or .env
OUTPUTS   : L2-normalised float32 vectors; Gemini results are cached under
            data/index/cache/ so an interrupted run resumes without re-spending
            free-tier quota.
DEPENDS ON: sentence-transformers (local); Python standard library only for
            Gemini (REST via urllib)
PANEL NOTE: Comparing backends is only fair if they embed identical inputs
            and are scored on identical questions - the caller builds the
            inputs once; this module only turns text into vectors. Gemini
            sends chunk text (public annual reports) to Google; free-tier
            inputs may be used by Google to improve its products.

Model names: "BAAI/bge-small-en-v1.5" (any sentence-transformers model) or
"gemini:<model>", e.g. "gemini:gemini-embedding-001".
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

log = logging.getLogger("embedders")

# BGE v1.5 recommends this instruction on short retrieval queries only;
# passages are embedded without it.
QUERY_PREFIX = os.getenv("FGR_QUERY_PREFIX", "Represent this sentence for searching relevant passages: ")
LOCAL_BATCH_SIZE = 32

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"
# DEMO-KNOB: Gemini output size. gemini-embedding-001 supports 768 / 1536 /
# 3072; 768 keeps the index small. Vectors are re-normalised client-side.
GEMINI_DIM = int(os.getenv("FGR_GEMINI_DIM", "768"))
# DEMO-KNOB: texts per batch request and pause between requests. Kept small
# for the free tier; 429 responses are retried with backoff regardless.
GEMINI_BATCH = int(os.getenv("FGR_GEMINI_BATCH", "20"))
GEMINI_PAUSE_S = float(os.getenv("FGR_GEMINI_PAUSE_S", "2.0"))
GEMINI_MAX_RETRIES = 8
CACHE_DIR = Path(os.getenv("FGR_INDEX_DIR", "data/index")) / "cache"


def load_streamlit_secrets() -> list[str]:
    """Copy top-level Streamlit secrets into os.environ (existing values win).

    PANEL: On Streamlit Community Cloud there is no .env file; credentials are
    set as secrets and read through st.secrets. Copying them into os.environ
    lets every module keep one lookup (os.getenv) for local runs and the cloud.
    Only runs inside a Streamlit app (streamlit already imported), so CLI runs
    never import streamlit. Returns the key names copied - never the values.
    """
    import sys
    if "streamlit" not in sys.modules:
        return []
    try:
        import streamlit as st
        items = {k: st.secrets[k] for k in st.secrets.keys()}
    except Exception as e:  # FALLBACK: no secrets.toml (local run) - .env / environment still apply.
        log.debug("No Streamlit secrets available: %s", type(e).__name__)
        return []
    copied = []
    for key, value in items.items():
        # Sections ([neo4j] ...) are not flattened: secrets must be top-level KEY = "value".
        if not isinstance(value, (str, int, float, bool)):
            log.warning("Streamlit secret section [%s] ignored: put settings at the top level "
                        "(e.g. NEO4J_URI = \"...\"), not under a [section] header", key)
        elif key not in os.environ:
            os.environ[key] = str(value)
            copied.append(key)
    return copied


def load_env(path: Path = Path(".env")) -> None:
    """Load settings into os.environ: Streamlit secrets (when running as a
    Streamlit app), then KEY=VALUE lines from .env. Existing values win, so the
    order of precedence is: real environment > st.secrets > .env.

    If a key appears more than once in the file, the LAST value is used (the
    python-dotenv convention) and a warning names the key. Values are never
    logged or returned.
    """
    load_streamlit_secrets()
    if not path.exists():
        return
    values: dict[str, str] = {}
    counts: Counter = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if m and not line.lstrip().startswith("#"):
            values[m.group(1)] = m.group(2).strip("'\"")
            counts[m.group(1)] += 1
    for key, n in counts.items():
        if n > 1:
            log.warning("%s is set %d times in %s; using the last one", key, n, path)
    for key, value in values.items():
        os.environ.setdefault(key, value)


def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class LocalEmbedder:
    """sentence-transformers model on CPU; free, offline after first download."""

    def __init__(self, name: str):
        from sentence_transformers import SentenceTransformer
        try:
            # Already downloaded: load from disk without contacting huggingface.co
            # (its update check can time out and stall every run).
            self.model = SentenceTransformer(name, device="cpu", local_files_only=True)
        except Exception:
            self.model = None
        try:
            self.model = self.model or SentenceTransformer(name, device="cpu")   # first run: download
        except Exception as e:
            # FALLBACK: no silent substitute model - a different model would make
            # the index incomparable. Fail with a clear message instead.
            raise RuntimeError(f"Could not load embedding model {name!r} (first run needs network "
                               f"access to download it from Hugging Face): {e}") from e
        self.name = name
        self.max_tokens = self.model.max_seq_length
        self.dim = self.model.get_sentence_embedding_dimension()
        self.info = {"provider": "local sentence-transformers (CPU)", "query_prefix": QUERY_PREFIX}

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer(text, add_special_tokens=True)["input_ids"])

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=LOCAL_BATCH_SIZE, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=True).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode([QUERY_PREFIX + text], normalize_embeddings=True,
                                 convert_to_numpy=True)[0].astype(np.float32)


class QuotaExhausted(RuntimeError):
    """The free-tier daily quota is used up; waiting minutes will not help."""


class GeminiEmbedder:
    """Google Gemini embedding API via REST, with on-disk cache and backoff."""

    # gemini-embedding-001 accepts up to 2048 input tokens; chunks are sized to
    # <= 512 BERT tokens, so they fit with wide margin (estimated - Gemini's
    # tokenizer differs from BERT's and is not available offline).
    max_tokens = 2048

    def __init__(self, name: str, api_key: str | None = None, cache_dir: Path = CACHE_DIR,
                 post=None, sleep=time.sleep):
        self.name = name
        self.model = name.split(":", 1)[1]
        load_env()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Create a free key at "
                               "https://aistudio.google.com/apikey and add GEMINI_API_KEY=... to .env")
        self.dim = GEMINI_DIM
        self.info = {"provider": "Google Gemini API (free tier)", "task_types":
                     ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"], "output_dimensionality": self.dim}
        self._post = post or self._http_post
        self._sleep = sleep
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_dir / f"{re.sub(r'[^a-z0-9]+', '-', name.lower())}-d{self.dim}.jsonl"
        self.cache: dict[str, list[float]] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec["v"]
        self.api_calls = 0

    def _key(self, task: str, text: str) -> str:
        return hashlib.sha256(f"{self.model}|{self.dim}|{task}|{text}".encode()).hexdigest()

    def _http_post(self, url: str, body: dict) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers={
            "Content-Type": "application/json", "x-goog-api-key": self.api_key})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())

    def _request(self, texts: list[str], task: str) -> list[list[float]]:
        body = {"requests": [{"model": f"models/{self.model}", "content": {"parts": [{"text": t}]},
                              "taskType": task, "outputDimensionality": self.dim} for t in texts]}
        url = GEMINI_URL.format(model=self.model)
        for attempt in range(GEMINI_MAX_RETRIES):
            try:
                self.api_calls += 1
                return [e["values"] for e in self._post(url, body)["embeddings"]]
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:600]
                if e.code == 429 and re.search(r"PerDay|per day|daily", detail, re.I):
                    raise QuotaExhausted(f"Gemini daily free-tier quota exhausted ({len(self.cache)} "
                                         f"embeddings cached; re-run later to resume)") from None
                if e.code in (429, 500, 502, 503, 504):
                    m = re.search(r'"retryDelay":\s*"(\d+)', detail)
                    wait = int(m.group(1)) if m else min(60, 2 ** attempt) + random.random()
                    log.warning("Gemini HTTP %d; retrying in %.0fs (attempt %d)", e.code, wait, attempt + 1)
                    self._sleep(wait)
                    continue
                # 400/401/403: bad request or key - retrying will not help. The
                # key is sent in a header, so it never appears in this message.
                raise RuntimeError(f"Gemini API error {e.code}: {detail}") from None
            except urllib.error.URLError as e:
                wait = min(60, 2 ** attempt)
                log.warning("Network error (%s); retrying in %ds", e.reason, wait)
                self._sleep(wait)
        raise RuntimeError(f"Gemini API still failing after {GEMINI_MAX_RETRIES} attempts")

    def _embed(self, texts: list[str], task: str, progress: bool) -> np.ndarray:
        keys = [self._key(task, t) for t in texts]
        # One request per distinct uncached text: identical chunks (repeated
        # boilerplate across reports) would otherwise spend quota twice.
        first: dict[str, int] = {}
        for i, k in enumerate(keys):
            if k not in self.cache:
                first.setdefault(k, i)
        todo = list(first.values())
        if todo:
            log.info("Gemini: %d of %d texts not cached; embedding in batches of %d",
                     len(todo), len(texts), GEMINI_BATCH)
        with self.cache_path.open("a", encoding="utf-8") as f:
            for n, start in enumerate(range(0, len(todo), GEMINI_BATCH)):
                if n:
                    self._sleep(GEMINI_PAUSE_S)
                batch = todo[start:start + GEMINI_BATCH]
                vecs = self._request([texts[i] for i in batch], task)
                for i, v in zip(batch, vecs):
                    self.cache[keys[i]] = v
                    f.write(json.dumps({"key": keys[i], "v": v}) + "\n")
                f.flush()   # every finished batch survives an interruption
                if progress:
                    log.info("Gemini: %d/%d embedded", min(start + GEMINI_BATCH, len(todo)), len(todo))
        return _normalise(np.array([self.cache[k] for k in keys]))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT", progress=True)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY", progress=False)[0]


def get_embedder(name: str):
    return GeminiEmbedder(name) if name.startswith("gemini:") else LocalEmbedder(name)
