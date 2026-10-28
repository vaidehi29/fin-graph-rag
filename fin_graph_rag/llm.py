"""
COMPONENT : LLM client (Gemini, free tier)
PURPOSE   : Call Gemini generateContent with a JSON response schema, temperature
            0, an on-disk response cache, pacing for free-tier rate limits, and
            retry/backoff. Used for entity extraction (and later, answers).
INPUTS    : prompt text + JSON schema; GEMINI_API_KEY from environment / .env
OUTPUTS   : parsed JSON dict; cached responses in data/index/cache/llm-<model>.jsonl
DEPENDS ON: Python standard library (urllib); fin_graph_rag.embedders.load_env
PANEL NOTE: The schema forces machine-checkable output, but the caller still
            validates every item against the source text - a well-formed
            answer is not proof of a correct one. Caching makes re-runs free
            and reproducible. Prompts send report text (public 10-K filings)
            to Google; free-tier inputs may be used to improve Google products.
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
from pathlib import Path

from fin_graph_rag.embedders import QuotaExhausted, load_env

log = logging.getLogger("llm")

# DEMO-KNOB: extraction model. A free-tier Flash-Lite model is enough for
# schema-constrained extraction. pipeline.txt's gemini-1.5-flash is retired, and
# the API returned 404 for gemini-2.5-flash-lite ("no longer available to new
# users", 2026-10-28), recommending gemini-3.5-flash-lite.
MODEL = os.getenv("FGR_EXTRACT_MODEL", "gemini-3.5-flash-lite")
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# DEMO-KNOB: pause between calls, to stay under free-tier requests-per-minute.
PAUSE_S = float(os.getenv("FGR_LLM_PAUSE_S", "5"))
MAX_RETRIES = 8
CACHE_DIR = Path(os.getenv("FGR_INDEX_DIR", "data/index")) / "cache"


class GeminiJSON:
    def __init__(self, model: str = MODEL, api_key: str | None = None, cache_dir: Path = CACHE_DIR,
                 post=None, sleep=time.sleep):
        load_env()
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (add it to .env)")
        self._post = post or self._http_post
        self._sleep = sleep
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_dir / f"llm-{re.sub(r'[^a-z0-9]+', '-', model.lower())}.jsonl"
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec["response"]
        self.api_calls = 0
        self._last_call = 0.0

    def _http_post(self, url: str, body: dict) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers={
            "Content-Type": "application/json", "x-goog-api-key": self.api_key})
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())

    def generate(self, prompt: str, schema: dict, system: str = "") -> dict:
        key = hashlib.sha256(json.dumps([self.model, system, prompt, schema], sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                 "responseSchema": schema},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        wait = PAUSE_S - (time.time() - self._last_call)
        if wait > 0:
            self._sleep(wait)
        response = self._call(body)
        self._last_call = time.time()
        # Skip "thinking" parts some models return; parse only the answer text.
        parts = response["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Not cached: a malformed response is logged (truncated) and retried on the next run.
            log.warning("Malformed JSON from %s (first 200 chars): %r", self.model, text[:200])
            raise
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "response": parsed}) + "\n")
        self.cache[key] = parsed
        return parsed

    def _call(self, body: dict) -> dict:
        url = URL.format(model=self.model)
        for attempt in range(MAX_RETRIES):
            try:
                self.api_calls += 1
                return self._post(url, body)
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:600]
                if e.code == 429 and re.search(r"PerDay|per day|daily", detail, re.I):
                    raise QuotaExhausted(f"Gemini daily free-tier quota for {self.model} exhausted "
                                         f"({len(self.cache)} responses cached; re-run later to resume)") from None
                if e.code in (429, 500, 502, 503, 504):
                    m = re.search(r'"retryDelay":\s*"(\d+)', detail)
                    wait = int(m.group(1)) if m else min(60, 2 ** attempt) + random.random()
                    log.warning("Gemini HTTP %d; retrying in %.0fs (attempt %d)", e.code, wait, attempt + 1)
                    self._sleep(wait)
                    continue
                raise RuntimeError(f"Gemini API error {e.code}: {detail}") from None
            except urllib.error.URLError as e:
                self._sleep(min(60, 2 ** attempt))
                log.warning("Network error (%s); retrying", e.reason)
        raise RuntimeError(f"Gemini API still failing after {MAX_RETRIES} attempts")
