import importlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _register_hyphenated_package(dotted_alias, fs_path):
    """Register a filesystem package with hyphens under a Python-safe alias."""
    parts = dotted_alias.split(".")
    for i in range(len(parts)):
        partial = ".".join(parts[: i + 1])
        if partial not in sys.modules:
            pkg = types.ModuleType(partial)
            pkg.__path__ = [str(fs_path if i == len(parts) - 1 else ROOT / "/".join(parts[: i + 1]).replace(".", "/"))]
            pkg.__package__ = partial
            sys.modules[partial] = pkg


# Map hyphenated dirs to importable aliases
_SAAP_ROOT = ROOT / "projects" / "SOTI-Advanced-Analytics-Plus"
_AI_SERVICE = _SAAP_ROOT / "ai-service"

_register_hyphenated_package("projects", ROOT / "projects")
_register_hyphenated_package("projects.SOTI_Advanced_Analytics_Plus", _SAAP_ROOT)
_register_hyphenated_package("projects.SOTI_Advanced_Analytics_Plus.ai_service", _AI_SERVICE)


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace directory with workforce.json."""
    mc_dir = tmp_path / "office" / "mission-control"
    mc_dir.mkdir(parents=True)
    workforce = {
        "squad_name": "Test Squad",
        "lead_agent": "Michael",
        "agents": [{"id": "test", "name": "Test Agent", "role": "Tester", "specialty": "Testing"}],
        "kanban": [],
        "watercooler": [],
    }
    (mc_dir / "workforce.json").write_text(json.dumps(workforce, indent=2))
    return tmp_path


@pytest.fixture
def workforce_path(tmp_workspace):
    return tmp_workspace / "office" / "mission-control" / "workforce.json"
