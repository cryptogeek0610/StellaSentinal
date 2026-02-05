from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from openai import AzureOpenAI, OpenAI

from device_anomaly.config.settings import get_settings

logger = logging.getLogger(__name__)


# =============================================================================
# LLM Response Cache
# =============================================================================


@dataclass
class CacheEntry:
    """A cached LLM response with TTL."""

    response: str
    created_at: float
    hit_count: int = 0


class LLMResponseCache:
    """
    Thread-safe LRU cache for LLM responses with TTL.

    Reduces redundant LLM calls for identical prompts, improving
    performance and reducing costs for local LLM deployments.
    """

    def __init__(self, max_size: int = 500, ttl_seconds: int = 900):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of cached entries (default 500)
            ttl_seconds: Time-to-live in seconds (default 15 min)
        """
        self._cache: dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _hash_key(self, prompt: str, model: str, temperature: float) -> str:
        """Generate cache key from prompt and parameters."""
        key_data = f"{model}|{temperature:.2f}|{prompt}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    def get(self, prompt: str, model: str, temperature: float = 0.2) -> str | None:
        """Get cached response if available and not expired."""
        key = self._hash_key(prompt, model, temperature)

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            # Check TTL
            if time.time() - entry.created_at > self._ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None

            entry.hit_count += 1
            self._hits += 1
            return entry.response

    def set(self, prompt: str, model: str, temperature: float, response: str) -> None:
        """Cache a response."""
        key = self._hash_key(prompt, model, temperature)

        with self._lock:
            # Evict oldest entries if at capacity
            if len(self._cache) >= self._max_size:
                self._evict_oldest()

            self._cache[key] = CacheEntry(
                response=response,
                created_at=time.time(),
            )

    def _evict_oldest(self) -> None:
        """Evict oldest 10% of entries."""
        if not self._cache:
            return

        # Sort by creation time, remove oldest 10%
        sorted_keys = sorted(
            self._cache.keys(),
            key=lambda k: self._cache[k].created_at,
        )
        evict_count = max(1, len(sorted_keys) // 10)
        for key in sorted_keys[:evict_count]:
            del self._cache[key]

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 3),
            }


# Global cache instance
_llm_cache = LLMResponseCache()


def strip_thinking_tags(text: str) -> str:
    """
    Strip <think>...</think> tags and their content from LLM responses.

    Some models (like DeepSeek R1) include internal reasoning in <think> tags
    that should not be shown to end users.

    Handles edge cases where models incorrectly place the actual response
    content inside thinking tags (e.g., glm models that put Summary/Troubleshooting
    inside <think> blocks).
    """
    if not text:
        return text

    original_text = text

    # Remove <think>...</think> blocks (including multiline content)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Also handle unclosed <think> tags (in case response was truncated)
    cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    # Clean up any extra whitespace left behind
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
    cleaned = cleaned.strip()

    # If we ended up with an empty result but original had content,
    # the model likely placed the actual response inside thinking tags
    if not cleaned and '<think>' in original_text.lower():
        # Try to extract content from within the thinking block
        # Look for common response markers that indicate the actual output
        response_markers = [
            r'SUMMARY:',
            r'Summary:',
            r'TROUBLESHOOTING STEPS:',
            r'Troubleshooting Steps:',
            r'\*\*SUMMARY\*\*',
            r'\*\*Summary\*\*',
            r'1\.\s+For\s+"',  # Numbered troubleshooting steps
            r'##\s+Summary',
        ]

        # Extract content within think tags
        think_content_match = re.search(
            r'<think>(.*?)(?:</think>|$)',
            original_text,
            flags=re.DOTALL | re.IGNORECASE
        )

        if think_content_match:
            think_content = think_content_match.group(1)

            # Find the earliest response marker
            earliest_pos = len(think_content)
            for marker in response_markers:
                match = re.search(marker, think_content, re.IGNORECASE)
                if match and match.start() < earliest_pos:
                    earliest_pos = match.start()

            # If we found a marker, extract from that point
            if earliest_pos < len(think_content):
                cleaned = think_content[earliest_pos:].strip()
                # Remove any trailing </think> if present
                cleaned = re.sub(r'</think>\s*$', '', cleaned, flags=re.IGNORECASE)
                logger.info(f"Extracted response from within thinking tags (length: {len(cleaned)})")

    return cleaned


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        ...


class DummyLLMClient(BaseLLMClient):
    """
    Simple fallback client that doesn't actually call an external LLM.
    It just echoes back a templated explanation using the prompt text.
    Useful for keeping the pipeline working before wiring a real provider.
    """

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return (
            "⚠️ **DUMMY LLM CLIENT IN USE** ⚠️\n\n"
            "No external LLM is configured or available. This is a placeholder response.\n\n"
            "To enable real LLM-powered troubleshooting:\n"
            "1. Set LLM_BASE_URL environment variable (e.g., http://localhost:1234)\n"
            "2. Set LLM_MODEL_NAME environment variable (e.g., deepseek/deepseek-r1-0528-qwen3-8b)\n"
            "3. Ensure your LLM service (LM Studio, vLLM, etc.) is running\n\n"
            "Prompt summary (first 300 chars):\n"
            f"{prompt[:300]}..."
        )


class OpenAICompatibleClient(BaseLLMClient):
    """
    Client for OpenAI-compatible APIs (LM Studio, vLLM, Ollama, etc.).

    Works with local LLM services that implement the OpenAI API format.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str | None = None,
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",  # Local services often don't require real keys
        )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        temperature = kwargs.get("temperature", 0.2)
        max_tokens = kwargs.get("max_tokens", 600)

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant for an enterprise mobile device management system. "
                        "IMPORTANT: Provide ONLY your final response. Do NOT include any internal reasoning, "
                        "<think> tags, chain-of-thought, or preamble. Start directly with the requested output format. "
                        "Be clear, concise, and actionable."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        message = resp.choices[0].message
        content = getattr(message, "content", "")

        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content).strip()


class AzureOpenAILLMClient(BaseLLMClient):
    """
    Real LLM client using Azure OpenAI Chat Completions.

    Expected settings (example):
      - api_key: Azure OpenAI key
      - base_url: Azure endpoint (e.g. 'https://my-resource.openai.azure.com/')
      - model_name: deployment name (NOT the OpenAI model id)
      - api_version: Azure OpenAI API version (e.g. '2024-02-15-preview')
    """

    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        deployment_name: str,
        api_version: str = "2024-02-15-preview",
    ):
        self.deployment_name = deployment_name
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        temperature = kwargs.get("temperature", 0.2)
        max_tokens = kwargs.get("max_tokens", 400)

        resp = self.client.chat.completions.create(
            model=self.deployment_name,  # this is the *deployment name* in Azure
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant for an enterprise mobile device management system. "
                        "IMPORTANT: Provide ONLY your final response. Do NOT include any internal reasoning, "
                        "<think> tags, chain-of-thought, or preamble. Start directly with the requested output format. "
                        "Be clear, concise, and actionable."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        message = resp.choices[0].message
        content = getattr(message, "content", "")

        # Azure returns `content` as a string in normal cases; this keeps your
        # previous defensive logic just in case.
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content).strip()


# =============================================================================
# Enhanced Local LLM Client
# =============================================================================


class LocalLLMClient(BaseLLMClient):
    """
    Enhanced client for local LLM services (Ollama, LM Studio, vLLM).

    Features:
    - Automatic model discovery via /api/tags (Ollama) or /v1/models
    - Response caching to reduce redundant calls (15 min TTL)
    - Retry with exponential backoff
    - Graceful degradation to rule-based analysis when LLM unavailable
    - Health monitoring
    """

    def __init__(
        self,
        base_url: str,
        model_name: str | None = None,
        fallback_model: str | None = None,
        api_key: str | None = None,
        enable_caching: bool = True,
        fallback_to_rules: bool = True,
        max_retries: int = 3,
        timeout_seconds: int = 30,
    ):
        """
        Initialize the local LLM client.

        Args:
            base_url: Base URL for the LLM service (e.g., http://ollama:11434)
            model_name: Primary model to use (auto-discovered if not specified)
            fallback_model: Smaller model to try if primary fails
            api_key: API key (usually not needed for local services)
            enable_caching: Whether to cache responses (default True)
            fallback_to_rules: Fall back to rule-based analysis if LLM unavailable
            max_retries: Maximum number of retry attempts
            timeout_seconds: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.api_key = api_key
        self.enable_caching = enable_caching
        self.fallback_to_rules = fallback_to_rules
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

        # Lazy-init OpenAI client
        self._client: OpenAI | None = None
        self._available_models: list[str] = []
        self._last_health_check: float = 0
        self._is_healthy: bool = False

    def _get_openai_base_url(self) -> str:
        """Get OpenAI-compatible API URL."""
        base = self.base_url
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return base

    def _ensure_client(self) -> OpenAI:
        """Lazy-initialize the OpenAI client."""
        if self._client is None:
            self._client = OpenAI(
                base_url=self._get_openai_base_url(),
                api_key=self.api_key or "not-needed",
                timeout=self.timeout_seconds,
            )
        return self._client

    def health_check(self) -> dict[str, Any]:
        """
        Check if LLM service is available.

        Returns:
            dict with status, available_models, latency_ms
        """
        import requests

        start_time = time.time()
        result = {
            "healthy": False,
            "provider": "unknown",
            "available_models": [],
            "latency_ms": 0,
            "error": None,
        }

        try:
            # Try Ollama API first
            ollama_resp = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            if ollama_resp.status_code == 200:
                data = ollama_resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                result["healthy"] = len(models) > 0
                result["provider"] = "ollama"
                result["available_models"] = models
                self._available_models = models
                self._is_healthy = result["healthy"]
                result["latency_ms"] = int((time.time() - start_time) * 1000)
                self._last_health_check = time.time()
                return result
        except Exception:
            pass  # Try OpenAI-compatible endpoint

        try:
            # Try OpenAI-compatible /v1/models endpoint
            openai_resp = requests.get(
                f"{self._get_openai_base_url()}/models",
                timeout=5,
            )
            if openai_resp.status_code == 200:
                data = openai_resp.json()
                models = [m.get("id", "") for m in data.get("data", [])]
                result["healthy"] = len(models) > 0
                result["provider"] = "openai-compatible"
                result["available_models"] = models
                self._available_models = models
                self._is_healthy = result["healthy"]
                result["latency_ms"] = int((time.time() - start_time) * 1000)
                self._last_health_check = time.time()
                return result
        except Exception as e:
            result["error"] = str(e)

        self._is_healthy = False
        result["latency_ms"] = int((time.time() - start_time) * 1000)
        self._last_health_check = time.time()
        return result

    def _select_model(self) -> str:
        """Select the best available model."""
        if self.model_name:
            return self.model_name

        # Refresh available models if stale (>5 min)
        if time.time() - self._last_health_check > 300:
            self.health_check()

        if self._available_models:
            # Prefer models with "llama" or the fallback model
            for model in self._available_models:
                if self.fallback_model and self.fallback_model in model:
                    return model
            return self._available_models[0]

        return self.fallback_model or "llama3.2"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate response with caching, retry, and fallback.

        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional parameters (temperature, max_tokens)

        Returns:
            Generated response string
        """
        temperature = kwargs.get("temperature", 0.2)
        max_tokens = kwargs.get("max_tokens", 600)
        model = self._select_model()

        # Check cache first
        if self.enable_caching:
            cached = _llm_cache.get(prompt, model, temperature)
            if cached is not None:
                logger.debug("LLM cache hit for model=%s", model)
                return cached

        # Try to generate with retries
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._do_generate(prompt, model, temperature, max_tokens)

                # Strip thinking tags
                response = strip_thinking_tags(response)

                # Cache successful response
                if self.enable_caching and response:
                    _llm_cache.set(prompt, model, temperature, response)

                return response

            except Exception as e:
                last_error = e
                wait_time = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s
                logger.warning(
                    "LLM generation attempt %d failed: %s. Retrying in %.1fs",
                    attempt + 1,
                    str(e)[:100],
                    wait_time,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)

                    # Try fallback model on subsequent attempts
                    if self.fallback_model and model != self.fallback_model:
                        model = self.fallback_model
                        logger.info("Switching to fallback model: %s", model)

        # All retries failed - use rule-based fallback
        if self.fallback_to_rules:
            logger.warning(
                "LLM unavailable after %d retries. Using rule-based fallback.",
                self.max_retries,
            )
            return self._rule_based_fallback(prompt)

        # No fallback - raise the error
        raise RuntimeError(
            f"LLM generation failed after {self.max_retries} retries: {last_error}"
        )

    def _do_generate(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Actually call the LLM API."""
        client = self._ensure_client()

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant for an enterprise mobile device management system. "
                        "IMPORTANT: Provide ONLY your final response. Do NOT include any internal reasoning, "
                        "<think> tags, chain-of-thought, or preamble. Start directly with the requested output format. "
                        "Be clear, concise, and actionable."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        message = resp.choices[0].message
        content = getattr(message, "content", "")

        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content).strip()

    def _rule_based_fallback(self, prompt: str) -> str:
        """
        Generate rule-based analysis when LLM is unavailable.

        Uses domain knowledge from RootCauseAnalyzer to provide useful
        insights without requiring an actual LLM call.
        """
        try:
            from device_anomaly.insights.root_cause import RootCauseAnalyzer

            RootCauseAnalyzer()

            # Try to extract context from the prompt
            prompt_lower = prompt.lower()

            # Detect the type of analysis requested
            if "battery" in prompt_lower:
                return self._format_rule_based_response(
                    "Battery Issue Analysis",
                    [
                        "High screen-on time correlates with increased battery drain",
                        "Background app activity may be consuming power",
                        "Poor signal strength causes radio to work harder",
                        "Check for apps with high wake lock usage",
                    ],
                    [
                        "Review screen-on time and reduce brightness",
                        "Identify and restrict background apps",
                        "Move device to area with better signal coverage",
                        "Update to latest OS and app versions",
                    ],
                )
            elif "crash" in prompt_lower or "anr" in prompt_lower:
                return self._format_rule_based_response(
                    "Application Crash Analysis",
                    [
                        "Low memory conditions often cause crashes",
                        "Low storage can prevent apps from functioning",
                        "Outdated app versions may have known bugs",
                        "OS version incompatibilities can cause issues",
                    ],
                    [
                        "Clear app caches and close unused apps",
                        "Free up storage by removing unused files",
                        "Update the app to the latest version",
                        "Check for OS updates and install if available",
                    ],
                )
            elif "network" in prompt_lower or "wifi" in prompt_lower or "signal" in prompt_lower:
                return self._format_rule_based_response(
                    "Network Connectivity Analysis",
                    [
                        "Weak WiFi signal causes connection drops",
                        "High device mobility leads to frequent AP handoffs",
                        "Poor cell coverage affects cellular connections",
                        "Network congestion can cause slowdowns",
                    ],
                    [
                        "Check WiFi coverage in the affected area",
                        "Consider reducing device mobility if possible",
                        "Switch to cellular if WiFi is unreliable",
                        "Contact IT if issue affects multiple devices",
                    ],
                )
            elif "anomaly" in prompt_lower or "pattern" in prompt_lower:
                return self._format_rule_based_response(
                    "Anomaly Pattern Analysis",
                    [
                        "Device shows significant deviation from fleet baseline",
                        "Multiple features may be correlated in this anomaly",
                        "Historical data shows recent changes before anomaly",
                        "Cohort analysis can reveal systemic issues",
                    ],
                    [
                        "Compare device metrics to cohort baseline",
                        "Review recent changes (apps, OS, location)",
                        "Check if other devices in same cohort are affected",
                        "Escalate to IT if pattern persists",
                    ],
                )
            else:
                return self._format_rule_based_response(
                    "General Device Analysis",
                    [
                        "Device health depends on multiple factors",
                        "Battery, memory, storage, and network all contribute",
                        "Software versions affect stability and performance",
                        "Usage patterns impact overall device health",
                    ],
                    [
                        "Review all key health metrics",
                        "Check for pending software updates",
                        "Monitor for recurring issues",
                        "Contact IT support if problems persist",
                    ],
                )
        except Exception as e:
            logger.warning("Rule-based fallback failed: %s", e)
            return (
                "**Analysis Unavailable**\n\n"
                "The LLM service is currently unavailable and rule-based analysis "
                "could not be generated. Please try again later or contact IT support."
            )

    def _format_rule_based_response(
        self,
        title: str,
        findings: list[str],
        recommendations: list[str],
    ) -> str:
        """Format a rule-based response in a consistent structure."""
        lines = [
            f"**{title}** _(rule-based analysis)_\n",
            "**Key Findings:**",
        ]
        for i, finding in enumerate(findings, 1):
            lines.append(f"{i}. {finding}")

        lines.append("\n**Recommendations:**")
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")

        lines.append(
            "\n_Note: This is a rule-based analysis. For AI-powered insights, "
            "ensure the LLM service is running._"
        )

        return "\n".join(lines)


def get_llm_cache_stats() -> dict[str, Any]:
    """Get LLM cache statistics."""
    return _llm_cache.get_stats()


def clear_llm_cache() -> None:
    """Clear the LLM response cache."""
    _llm_cache.clear()


def _get_env_value(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _resolve_llm_config() -> dict[str, str | None]:
    settings = get_settings().llm
    env_base_url = _get_env_value("LLM_BASE_URL")
    env_model = _get_env_value("LLM_MODEL_NAME")
    env_api_key = _get_env_value("LLM_API_KEY")
    env_api_version = _get_env_value("LLM_API_VERSION")

    return {
        "env_base_url": env_base_url,
        "env_model": env_model,
        "env_api_key": env_api_key,
        "env_api_version": env_api_version,
        "settings_base_url": settings.base_url,
        "settings_model": settings.model_name,
        "settings_api_key": settings.api_key,
        "settings_api_version": settings.api_version,
        "resolved_base_url": env_base_url or settings.base_url or "http://localhost:1234",
        "resolved_model": env_model or settings.model_name,
        "resolved_api_key": env_api_key or settings.api_key,
        "resolved_api_version": env_api_version or settings.api_version,
    }


def _normalize_openai_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        return base_url
    return f"{base_url}/v1"


def get_llm_config_snapshot() -> dict[str, object]:
    config = _resolve_llm_config()
    base_url = config["resolved_base_url"] or "http://localhost:1234"
    return {
        "env_base_url": config["env_base_url"],
        "env_model": config["env_model"],
        "env_api_key_set": bool(config["env_api_key"]),
        "env_api_version": config["env_api_version"],
        "settings_base_url": config["settings_base_url"],
        "settings_model": config["settings_model"],
        "settings_api_key_set": bool(config["settings_api_key"]),
        "settings_api_version": config["settings_api_version"],
        "resolved_base_url": base_url,
        "resolved_base_url_api": _normalize_openai_base_url(base_url),
        "resolved_model": config["resolved_model"],
        "resolved_api_key_set": bool(config["resolved_api_key"]),
        "resolved_api_version": config["resolved_api_version"],
    }


def get_default_llm_client() -> BaseLLMClient:
    """Get the default LLM client, supporting both Azure OpenAI and local OpenAI-compatible services."""
    logger.info("get_default_llm_client() called")
    config = _resolve_llm_config()

    logger.info(
        "LLM env: base_url=%s model=%s api_key_set=%s api_version=%s",
        config["env_base_url"],
        config["env_model"],
        bool(config["env_api_key"]),
        config["env_api_version"],
    )
    logger.info(
        "LLM settings: base_url=%s model=%s api_key_set=%s api_version=%s",
        config["settings_base_url"],
        config["settings_model"],
        bool(config["settings_api_key"]),
        config["settings_api_version"],
    )

    llm_base_url = config["resolved_base_url"] or "http://localhost:1234"
    llm_model = config["resolved_model"]
    llm_api_key = config["resolved_api_key"]
    llm_api_version = config["resolved_api_version"]

    logger.info("LLM resolved: base_url=%s model=%s api_key_set=%s api_version=%s", llm_base_url, llm_model, bool(llm_api_key), llm_api_version)

    # Auto-fix localhost URLs when running in Docker
    if os.path.exists("/.dockerenv") and llm_base_url and "localhost" in llm_base_url:
        original_url = llm_base_url
        llm_base_url = llm_base_url.replace("localhost", "host.docker.internal")
        logger.warning(
            "LLM base_url was '%s' but changed to '%s' for Docker container access. "
            "If this doesn't work, use your host IP address instead (e.g., http://192.168.x.x:1234).",
            original_url,
            llm_base_url
        )

    # Check for Azure OpenAI configuration first
    if llm_api_key and llm_model and llm_base_url:
        if "openai.azure.com" in llm_base_url or "azure.com" in llm_base_url:
            api_version = llm_api_version or "2024-12-01-preview"
            logger.info(
                "Using AzureOpenAILLMClient (deployment=%s, api_version=%s).",
                llm_model,
                api_version,
            )
            return AzureOpenAILLMClient(
                api_key=llm_api_key,
                azure_endpoint=llm_base_url,
                deployment_name=llm_model,
                api_version=api_version,
            )

    base_url_api = _normalize_openai_base_url(llm_base_url)
    if base_url_api != llm_base_url:
        logger.info("Normalized LLM base_url for OpenAI client: %s -> %s", llm_base_url, base_url_api)

    # Try to connect to local LLM service even if model name not specified
    try:
        import requests

        models_url = f"{base_url_api}/models"
        logger.info("Checking LLM service via %s", models_url)
        resp = requests.get(models_url, timeout=5)
        logger.info("LLM /models response: status=%s body=%s", resp.status_code, resp.text[:1000])

        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            if models:
                model_ids = [m.get("id", "") for m in models]
                logger.info("LLM available models: %s", model_ids)

                if llm_model:
                    if llm_model in model_ids:
                        model_to_use = llm_model
                    else:
                        logger.warning(
                            "Specified model '%s' not found in %s; using first available: %s",
                            llm_model,
                            model_ids,
                            models[0].get("id"),
                        )
                        model_to_use = models[0].get("id", "unknown")
                else:
                    model_to_use = models[0].get("id", "unknown")
                    logger.info("No model specified; using first available: %s", model_to_use)

                logger.info(
                    "Using OpenAICompatibleClient (base_url=%s, model=%s).",
                    base_url_api,
                    model_to_use,
                )
                return OpenAICompatibleClient(
                    base_url=base_url_api,
                    model_name=model_to_use,
                    api_key=llm_api_key,  # May be None for local services
                )

            logger.warning("LLM service at %s returned no models", base_url_api)
        else:
            logger.warning("LLM service at %s returned status %s", base_url_api, resp.status_code)
    except requests.exceptions.ConnectionError:
        logger.warning("Cannot connect to LLM service at %s - service may not be running", base_url_api, exc_info=True)
    except requests.exceptions.Timeout:
        logger.warning("LLM service at %s timed out", base_url_api, exc_info=True)
    except Exception:
        logger.exception("Error checking LLM service at %s", base_url_api)

    logger.warning("No LLM service available. Using DummyLLMClient as fallback.")
    return DummyLLMClient()
