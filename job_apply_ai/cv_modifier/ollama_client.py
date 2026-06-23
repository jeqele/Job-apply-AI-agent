"""Ollama client for local LLM-powered CV generation."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_FAST_MODEL = os.environ.get("OLLAMA_CV_FAST_MODEL", "gemma4:e4b")
DEFAULT_MAIN_MODEL = os.environ.get("OLLAMA_CV_MODEL", "gemma4:e4b") # gemma4:12b
DEFAULT_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "8192"))


def get_ollama_client() -> OllamaClient:
    """Build an Ollama client using saved app settings when available."""
    try:
        from job_apply_ai.storage.app_settings import AppSettingsRepository

        settings = AppSettingsRepository().get_ollama_settings()
        return OllamaClient(
            base_url=settings["base_url"],
            fast_model=settings["fast_model"],
            main_model=settings["main_model"],
            num_predict=settings["num_predict"],
        )
    except Exception as exc:
        logger.warning("Could not load Ollama settings from storage: %s", exc)
        return OllamaClient()


class OllamaClient:
    """Thin wrapper around the Ollama HTTP API."""

    def __init__(
        self,
        base_url: str | None = None,
        fast_model: str | None = None,
        main_model: str | None = None,
        num_predict: int | None = None,
        timeout: int = 300,
    ):
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.fast_model = fast_model or DEFAULT_FAST_MODEL
        self.main_model = main_model or DEFAULT_MAIN_MODEL
        self.num_predict = num_predict if num_predict is not None else DEFAULT_NUM_PREDICT
        self.timeout = timeout
        self._available_models: list[str] | None = None

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.ok
        except requests.RequestException:
            return False

    def list_models(self, refresh: bool = False) -> list[str]:
        if self._available_models is not None and not refresh:
            return self._available_models

        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            payload = response.json()
            self._available_models = [model["name"] for model in payload.get("models", [])]
            return self._available_models
        except requests.RequestException as exc:
            logger.warning("Could not list Ollama models: %s", exc)
            return []

    def validate_models(self) -> dict[str, str]:
        """Resolve configured models against what Ollama has installed."""
        available = self.list_models(refresh=True)
        if not available:
            raise RuntimeError(
                f"Ollama is reachable at {self.base_url} but no models are installed."
            )

        self.fast_model = self._resolve_model(self.fast_model, available, role="fast")
        self.main_model = self._resolve_model(self.main_model, available, role="main")
        return {"fast": self.fast_model, "main": self.main_model}

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.3,
        json_format: bool = False,
        json_schema: dict[str, Any] | None = None,
        num_predict: int | None = None,
    ) -> str:
        resolved_model = self._resolve_model(
            model or self.main_model,
            self.list_models(),
            role="generation",
        )
        payload: dict[str, Any] = {
            "model": resolved_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict if num_predict is not None else self.num_predict,
            },
        }
        if system:
            payload["system"] = system
        if json_schema is not None:
            payload["format"] = json_schema
        elif json_format:
            payload["format"] = "json"

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        if not response.ok:
            raise RuntimeError(self._format_api_error(response, resolved_model))

        content = response.json().get("response", "")
        if not content:
            raise RuntimeError("Ollama returned an empty response")
        return content.strip()

    def generate_json(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.2,
        max_attempts: int = 2,
        schema: dict[str, Any] | None = None,
        num_predict: int | None = None,
    ) -> dict[str, Any]:
        json_system = (system or "") + " Return only a single valid JSON object."
        last_error: Exception | None = None
        last_raw = ""

        for attempt in range(max_attempts):
            attempt_prompt = prompt
            if attempt > 0:
                attempt_prompt = (
                    f"{prompt}\n\n"
                    "Your previous answer was not valid JSON. "
                    "Reply again with ONLY one JSON object. "
                    "Use double quotes for all keys and strings. "
                    "Do not include markdown fences, comments, trailing commas, or prose."
                )

            raw = self.generate(
                attempt_prompt,
                model=model,
                system=json_system.strip(),
                temperature=max(temperature - (attempt * 0.05), 0.05),
                json_format=schema is None,
                json_schema=schema,
                num_predict=num_predict,
            )
            last_raw = raw
            try:
                return self._parse_json_response(raw)
            except ValueError as exc:
                last_error = exc
                logger.warning(
                    "Failed to parse Ollama JSON on attempt %s/%s: %s. Raw response preview: %r",
                    attempt + 1,
                    max_attempts,
                    exc,
                    raw[:500],
                )

        if last_raw:
            logger.error("Ollama JSON parse failed after %s attempts. Last raw preview: %r", max_attempts, last_raw[:1000])
        raise ValueError(str(last_error) if last_error else "Model response was not valid JSON")

    def _resolve_model(self, model: str, available: list[str], role: str) -> str:
        if model in available:
            return model

        model_base = model.split(":", 1)[0]
        for name in available:
            if name.split(":", 1)[0] == model_base:
                logger.warning(
                    "Ollama %s model '%s' not found; using '%s' instead",
                    role,
                    model,
                    name,
                )
                return name

        available_text = ", ".join(available) if available else "none"
        raise RuntimeError(
            f"Ollama model '{model}' is not installed for {role} generation. "
            f"Available models: {available_text}. "
            f"Pull it with: ollama pull {model}"
        )

    @staticmethod
    def _format_api_error(response: requests.Response, model: str) -> str:
        try:
            payload = response.json()
            error_message = payload.get("error") or payload
        except ValueError:
            error_message = response.text.strip() or response.reason

        if response.status_code == 404:
            return (
                f"Ollama model '{model}' was not found (404). "
                f"Install it with: ollama pull {model}. Details: {error_message}"
            )
        if response.status_code == 405:
            return (
                "Ollama rejected the request method (405). "
                "The CV generator uses POST /api/generate; opening that URL in a browser will show 405."
            )
        return f"Ollama API error ({response.status_code}): {error_message}"

    @staticmethod
    def _parse_json_response(raw: str) -> dict[str, Any]:
        for candidate in OllamaClient._json_candidates(raw):
            for repaired in (candidate, OllamaClient._repair_json(candidate)):
                if not repaired:
                    continue
                try:
                    parsed = json.loads(repaired)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed

        raise ValueError("Model response was not valid JSON")

    @staticmethod
    def _json_candidates(raw: str) -> list[str]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned.strip())

        candidates: list[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            value = value.strip()
            if value and value not in seen:
                seen.add(value)
                candidates.append(value)

        add(cleaned)

        for index, char in enumerate(cleaned):
            if char == "{":
                extracted = OllamaClient._extract_balanced_json(cleaned, index)
                if extracted:
                    add(extracted)

        return candidates

    @staticmethod
    def _extract_balanced_json(text: str, start: int) -> str | None:
        depth = 0
        in_string = False
        escape = False

        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]

        return None

    @staticmethod
    def _repair_json(text: str) -> str:
        repaired = text.strip()
        repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
        repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        repaired = re.sub(r"\}\s*\{", "},{", repaired)
        repaired = re.sub(r"(?<![\\])\\(?![\"\\/bfnrtu])", r"\\\\", repaired)
        return repaired
