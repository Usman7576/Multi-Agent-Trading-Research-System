"""Local LLM through llama.cpp (``llama-cpp-python``), CPU only, no Ollama or system install needed.

Every call asks for JSON that must match a Pydantic model: llama.cpp constrains generation to the model's
JSON schema, the reply is validated, and one repair attempt is made with the validation error before the
item is marked failed (never guessed). Responses are cached in SQLite keyed by (model, prompt version,
prompt hash), so re-running a pipeline costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "qwen2.5-3b-instruct-q4_k_m.gguf"


def _register_msvc_runtime() -> None:
    """llama.cpp's Windows DLLs need the Microsoft C++ runtime. When it is not installed system-wide,
    use the copies that ship with Python (vcruntime140) and scikit-learn (msvcp140)."""
    if sys.platform != "win32":
        return
    import sklearn

    for folder in (Path(sklearn.__file__).parent / ".libs", Path(sys.base_prefix)):
        if folder.exists():
            os.add_dll_directory(str(folder))


class LLMError(RuntimeError):
    pass


class LocalLLM:
    def __init__(
        self,
        model_path: Path,
        cache_path: Path,
        n_ctx: int = 8192,
        n_threads: int | None = None,
        temperature: float = 0.0,
        seed: int = 7,
    ) -> None:
        _register_msvc_runtime()
        from llama_cpp import Llama

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model_name = model_path.name
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads or max(1, (os.cpu_count() or 2) - 1),
            seed=seed,
            verbose=False,
        )
        self._temperature = temperature
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = sqlite3.connect(cache_path)
        self._cache.execute(
            "CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, model TEXT, prompt_version TEXT, "
            "response TEXT, tokens_in INTEGER, tokens_out INTEGER)"
        )

    def extract(self, schema: type[T], system: str, user: str, prompt_version: str) -> T:
        """Return a validated ``schema`` instance, or raise LLMError after one repair attempt."""
        key = hashlib.sha256(
            json.dumps([self.model_name, prompt_version, schema.__name__, system, user]).encode()
        ).hexdigest()
        row = self._cache.execute("SELECT response FROM llm_cache WHERE key = ?", (key,)).fetchone()
        if row is not None:
            return schema.model_validate_json(row[0])

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(2):
            out = self._llm.create_chat_completion(
                messages=messages,
                temperature=self._temperature,
                response_format={"type": "json_object", "schema": schema.model_json_schema()},
            )
            text = out["choices"][0]["message"]["content"] or ""
            try:
                parsed = schema.model_validate_json(text)
            except ValidationError as exc:
                if attempt == 1:
                    raise LLMError(f"{schema.__name__}: invalid output after repair: {exc}") from exc
                messages += [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"That JSON is invalid: {exc}. Return corrected JSON only."},
                ]
                continue
            usage = out.get("usage", {})
            self._cache.execute(
                "INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?, ?, ?, ?)",
                (
                    key,
                    self.model_name,
                    prompt_version,
                    parsed.model_dump_json(),
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                ),
            )
            self._cache.commit()
            return parsed
        raise LLMError("unreachable")
