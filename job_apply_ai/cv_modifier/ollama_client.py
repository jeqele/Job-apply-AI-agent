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


class OllamaClient:
    """Thin wrapper around the Ollama HTTP API."""

    def __init__(
        self,
        base_url: str | None = None,
        fast_model: str | None = None,
        main_model: str | None = None,
        timeout: int = 300,
    ):
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.fast_model = fast_model or DEFAULT_FAST_MODEL
        self.main_model = main_model or DEFAULT_MAIN_MODEL
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
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

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
    ) -> dict[str, Any]:
        raw = self.generate(
            prompt,
            model=model,
            system=system,
            temperature=temperature,
        )
        return self._parse_json_response(raw)

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
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed

        raise ValueError("Model response was not valid JSON")
