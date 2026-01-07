from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Optional
from openai import AzureOpenAI, OpenAI

from device_anomaly.config.settings import get_settings
import logging
import os


logger = logging.getLogger(__name__)


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
