from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODELS_SUBDIR = Path("models") / "production"
ENV_MODELS_DIR = "MODEL_ARTIFACTS_DIR"


@dataclass
class ModelArtifactLocation:
    model_dir: Path
    metadata_path: Path | None
    metadata: dict[str, Any] | None


def get_models_dir(models_dir: Path | None = None) -> Path:
    if models_dir is not None:
        return Path(models_dir)
    env_dir = os.getenv(ENV_MODELS_DIR)
    if env_dir:
        return Path(env_dir)
    return Path(__file__).parent.parent.parent.parent / DEFAULT_MODELS_SUBDIR


def _candidate_model_dirs(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []

    candidates = []

    if (base_dir / "training_metadata.json").exists() or (base_dir / "baselines.json").exists():
        candidates.append(base_dir)

    candidates.extend(p.parent for p in base_dir.glob("**/training_metadata.json"))
    if not candidates:
        candidates.extend(p.parent for p in base_dir.glob("**/isolation_forest.pkl"))

    return list({c.resolve() for c in candidates})


def find_latest_model_dir(models_dir: Path | None = None) -> Path | None:
    base_dir = get_models_dir(models_dir)
    candidates = _candidate_model_dirs(base_dir)
    if not candidates:
        return None

    def _mtime(path: Path) -> float:
        metadata_path = path / "training_metadata.json"
        if metadata_path.exists():
            return metadata_path.stat().st_mtime
        model_path = path / "isolation_forest.pkl"
        if model_path.exists():
            return model_path.stat().st_mtime
        return path.stat().st_mtime

    candidates.sort(key=_mtime, reverse=True)
    return candidates[0]


def load_latest_training_metadata(models_dir: Path | None = None) -> dict[str, Any] | None:
    model_dir = find_latest_model_dir(models_dir)
    if model_dir is None:
        return None
    metadata_path = model_dir / "training_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text())
    except Exception:
        return None


def resolve_model_artifacts(models_dir: Path | None = None) -> ModelArtifactLocation:
    model_dir = find_latest_model_dir(models_dir)
    if model_dir is None:
        return ModelArtifactLocation(model_dir=Path("."), metadata_path=None, metadata=None)

    metadata_path = model_dir / "training_metadata.json"
    metadata = None
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except Exception:
            metadata = None

    return ModelArtifactLocation(model_dir=model_dir, metadata_path=metadata_path if metadata_path.exists() else None, metadata=metadata)


def resolve_artifact_path(model_dir: Path, artifact_path: str | None) -> Path | None:
    if not artifact_path:
        return None
    path = Path(artifact_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    candidate = (model_dir / path).resolve()
    if candidate.exists():
        return candidate
    if path.name != path.as_posix():
        try:
            parts = list(path.parts)
            if model_dir.name in parts:
                idx = parts.index(model_dir.name)
                trimmed = Path(*parts[idx + 1 :])
                if trimmed.as_posix():
                    trimmed_candidate = (model_dir / trimmed).resolve()
                    if trimmed_candidate.exists():
                        return trimmed_candidate
        except Exception:
            pass
    return candidate
