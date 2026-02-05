"""LLM Settings API routes for configuring and managing LLM services."""
from __future__ import annotations

import ipaddress
import logging
import os
from urllib.parse import urlparse

import requests
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from device_anomaly.api.dependencies import require_role
from device_anomaly.llm.client import _normalize_openai_base_url, get_llm_config_snapshot

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["llm"])
_app_env = os.getenv("APP_ENV", "local")
_llm_base_url_allowlist = [
    entry.strip()
    for entry in os.getenv("LLM_BASE_URL_ALLOWLIST", "").split(",")
    if entry.strip()
]


class LLMConfig(BaseModel):
    """Current LLM configuration."""
    base_url: str
    model_name: str | None
    api_key_set: bool
    api_version: str | None
    is_connected: bool
    available_models: list[str]
    active_model: str | None
    provider: str  # 'ollama', 'lmstudio', 'azure', 'openai', 'unknown'


class LLMConfigUpdate(BaseModel):
    """Request to update LLM configuration."""
    base_url: str | None = None
    model_name: str | None = None
    api_key: str | None = None


class LLMModel(BaseModel):
    """LLM model information."""
    id: str
    name: str
    size: str | None = None
    modified_at: str | None = None


class LLMModelsResponse(BaseModel):
    """Response with available models."""
    models: list[LLMModel]
    active_model: str | None


class LLMTestResult(BaseModel):
    """Result of testing LLM connection."""
    success: bool
    message: str
    response_time_ms: float | None = None
    model_used: str | None = None


class OllamaPullRequest(BaseModel):
    """Request to pull an Ollama model."""
    model_name: str


class OllamaPullResponse(BaseModel):
    """Response from pulling an Ollama model."""
    success: bool
    message: str
    model_name: str


def _detect_provider(base_url: str) -> str:
    """Detect the LLM provider based on the base URL."""
    if not base_url:
        return "unknown"
    base_url_lower = base_url.lower()
    if "ollama" in base_url_lower or ":11434" in base_url_lower:
        return "ollama"
    if ":1234" in base_url_lower:
        return "lmstudio"
    if "azure" in base_url_lower or "openai.azure.com" in base_url_lower:
        return "azure"
    if "openai.com" in base_url_lower:
        return "openai"
    return "unknown"


def _validate_llm_base_url(base_url: str) -> None:
    if not base_url:
        return

    if _llm_base_url_allowlist:
        if not any(base_url.startswith(entry) for entry in _llm_base_url_allowlist):
            raise HTTPException(
                status_code=400,
                detail="LLM base URL is not in the allowlist",
            )

    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise HTTPException(
            status_code=400,
            detail="LLM base URL must be an http(s) URL",
        )

    if _app_env == "production" and not _llm_base_url_allowlist:
        hostname = parsed.hostname.lower()
        if hostname in {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}:
            raise HTTPException(
                status_code=400,
                detail="LLM base URL must not target localhost in production",
            )
        try:
            ip = ipaddress.ip_address(hostname)
            if (
                ip.is_loopback
                or ip.is_private
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
            ):
                raise HTTPException(
                    status_code=400,
                    detail="LLM base URL must not target private or loopback IPs in production",
                )
        except ValueError:
            # Hostname isn't an IP address; keep it as-is.
            pass


def _fetch_available_models(base_url: str) -> tuple[list[str], str | None]:
    """Fetch available models from the LLM service."""
    models = []
    active_model = None

    if not base_url:
        return models, active_model

    provider = _detect_provider(base_url)

    try:
        if provider == "ollama":
            # Ollama uses /api/tags endpoint
            tags_url = f"{base_url.rstrip('/')}/api/tags"
            resp = requests.get(tags_url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                for model in data.get("models", []):
                    models.append(model.get("name", ""))
        else:
            # OpenAI-compatible API
            api_url = _normalize_openai_base_url(base_url)
            models_url = f"{api_url}/models"
            resp = requests.get(models_url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                for model in data.get("data", []):
                    models.append(model.get("id", ""))
    except Exception as e:
        logger.warning("Failed to fetch models from %s: %s", base_url, e)

    # Get active model from env
    active_model = os.getenv("LLM_MODEL_NAME")

    return models, active_model


def _test_llm_connection(base_url: str) -> bool:
    """Test if LLM service is reachable."""
    if not base_url:
        return False

    provider = _detect_provider(base_url)

    try:
        if provider == "ollama":
            # Ollama health check
            resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
            return resp.status_code == 200
        else:
            # OpenAI-compatible
            api_url = _normalize_openai_base_url(base_url)
            resp = requests.get(f"{api_url}/models", timeout=5)
            return resp.status_code == 200
    except Exception:
        return False


@router.get("/config", response_model=LLMConfig)
async def get_llm_config():
    """Get current LLM configuration and status."""
    config_snapshot = get_llm_config_snapshot()

    base_url = str(config_snapshot.get("resolved_base_url", ""))
    models, active_model = _fetch_available_models(base_url)
    is_connected = _test_llm_connection(base_url)
    provider = _detect_provider(base_url)

    return LLMConfig(
        base_url=base_url,
        model_name=str(config_snapshot.get("resolved_model", "") or ""),
        api_key_set=bool(config_snapshot.get("resolved_api_key_set")),
        api_version=str(config_snapshot.get("resolved_api_version", "") or "") or None,
        is_connected=is_connected,
        available_models=models,
        active_model=active_model or (models[0] if models else None),
        provider=provider,
    )


@router.post("/config", response_model=LLMConfig)
async def update_llm_config(
    update: LLMConfigUpdate,
    _: None = Depends(require_role(["admin"])),
):
    """
    Update LLM configuration.

    Note: This updates environment variables at runtime. For persistent changes,
    update the .env file or docker-compose environment.
    """
    if update.base_url is not None:
        _validate_llm_base_url(update.base_url)
        os.environ["LLM_BASE_URL"] = update.base_url
        logger.info("Updated LLM_BASE_URL to %s", update.base_url)

    if update.model_name is not None:
        os.environ["LLM_MODEL_NAME"] = update.model_name
        logger.info("Updated LLM_MODEL_NAME to %s", update.model_name)

    if update.api_key is not None:
        os.environ["LLM_API_KEY"] = update.api_key
        logger.info("Updated LLM_API_KEY (value hidden)")

    # Return updated config
    return await get_llm_config()


@router.get("/models", response_model=LLMModelsResponse)
async def list_models():
    """List available models from the configured LLM service."""
    base_url = os.getenv("LLM_BASE_URL", "http://ollama:11434")
    provider = _detect_provider(base_url)
    models_list = []
    active_model = os.getenv("LLM_MODEL_NAME")

    try:
        if provider == "ollama":
            # Ollama provides more detailed model info
            tags_url = f"{base_url.rstrip('/')}/api/tags"
            resp = requests.get(tags_url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for model in data.get("models", []):
                    size_bytes = model.get("size", 0)
                    size_str = None
                    if size_bytes:
                        size_gb = size_bytes / (1024 ** 3)
                        size_str = f"{size_gb:.1f} GB"

                    models_list.append(LLMModel(
                        id=model.get("name", ""),
                        name=model.get("name", "").split(":")[0],
                        size=size_str,
                        modified_at=model.get("modified_at"),
                    ))
        else:
            # OpenAI-compatible
            api_url = _normalize_openai_base_url(base_url)
            resp = requests.get(f"{api_url}/models", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for model in data.get("data", []):
                    models_list.append(LLMModel(
                        id=model.get("id", ""),
                        name=model.get("id", ""),
                    ))
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to LLM service at {base_url}. Is it running?"
        )
    except Exception as e:
        logger.exception("Failed to list models")
        raise HTTPException(status_code=500, detail=str(e))

    return LLMModelsResponse(
        models=models_list,
        active_model=active_model,
    )


@router.post("/test", response_model=LLMTestResult)
async def test_llm_connection():
    """Test the LLM connection with a simple prompt."""
    import time

    base_url = os.getenv("LLM_BASE_URL", "http://ollama:11434")
    model_name = os.getenv("LLM_MODEL_NAME", "llama3.2")
    provider = _detect_provider(base_url)

    # First check if service is reachable
    if not _test_llm_connection(base_url):
        return LLMTestResult(
            success=False,
            message=f"Cannot connect to LLM service at {base_url}. Is it running?",
        )

    # Try a simple completion
    try:
        start_time = time.time()

        if provider == "ollama":
            # Use Ollama's native API for testing
            generate_url = f"{base_url.rstrip('/')}/api/generate"
            resp = requests.post(
                generate_url,
                json={
                    "model": model_name,
                    "prompt": "Say 'Hello' in one word.",
                    "stream": False,
                },
                timeout=30,
            )
        else:
            # Use OpenAI-compatible API
            api_url = _normalize_openai_base_url(base_url)
            resp = requests.post(
                f"{api_url}/chat/completions",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": "Say 'Hello' in one word."}],
                    "max_tokens": 10,
                },
                timeout=30,
            )

        elapsed_ms = (time.time() - start_time) * 1000

        if resp.status_code == 200:
            return LLMTestResult(
                success=True,
                message="LLM connection successful!",
                response_time_ms=round(elapsed_ms, 2),
                model_used=model_name,
            )
        else:
            return LLMTestResult(
                success=False,
                message=f"LLM returned error: {resp.status_code} - {resp.text[:200]}",
            )
    except requests.exceptions.Timeout:
        return LLMTestResult(
            success=False,
            message="LLM request timed out. The model may still be loading.",
        )
    except Exception as e:
        return LLMTestResult(
            success=False,
            message=f"Error testing LLM: {str(e)}",
        )


@router.post("/ollama/pull", response_model=OllamaPullResponse)
async def pull_ollama_model(
    request: OllamaPullRequest,
    _: None = Depends(require_role(["admin"])),
):
    """
    Pull/download a model in Ollama.

    This initiates the download - for large models it may take several minutes.
    """
    base_url = os.getenv("LLM_BASE_URL", "http://ollama:11434")
    provider = _detect_provider(base_url)

    if provider != "ollama":
        raise HTTPException(
            status_code=400,
            detail="Model pulling is only supported for Ollama. Current provider: " + provider
        )

    try:
        pull_url = f"{base_url.rstrip('/')}/api/pull"
        # Note: This is a streaming endpoint in Ollama, but we just initiate it
        resp = requests.post(
            pull_url,
            json={"name": request.model_name, "stream": False},
            timeout=300,  # 5 minutes for model download
        )

        if resp.status_code == 200:
            return OllamaPullResponse(
                success=True,
                message=f"Model '{request.model_name}' pulled successfully!",
                model_name=request.model_name,
            )
        else:
            return OllamaPullResponse(
                success=False,
                message=f"Failed to pull model: {resp.text[:200]}",
                model_name=request.model_name,
            )
    except requests.exceptions.Timeout:
        return OllamaPullResponse(
            success=False,
            message="Model pull request timed out. Large models may take longer - check Ollama logs.",
            model_name=request.model_name,
        )
    except Exception as e:
        return OllamaPullResponse(
            success=False,
            message=f"Error pulling model: {str(e)}",
            model_name=request.model_name,
        )


# Pre-defined popular models for quick selection
POPULAR_MODELS = [
    {"id": "llama3.2", "name": "Llama 3.2", "size": "3B", "description": "Fast, general-purpose"},
    {"id": "llama3.2:1b", "name": "Llama 3.2 1B", "size": "1B", "description": "Fastest, lightweight"},
    {"id": "deepseek-r1:8b", "name": "DeepSeek R1", "size": "8B", "description": "Advanced reasoning"},
    {"id": "qwen2.5:7b", "name": "Qwen 2.5", "size": "7B", "description": "Multilingual, coding"},
    {"id": "mistral", "name": "Mistral", "size": "7B", "description": "Balanced performance"},
    {"id": "codellama", "name": "Code Llama", "size": "7B", "description": "Code generation"},
    {"id": "phi3", "name": "Phi-3", "size": "3.8B", "description": "Microsoft, efficient"},
]


@router.get("/popular-models")
async def get_popular_models():
    """Get list of popular models that can be pulled."""
    return {"models": POPULAR_MODELS}
