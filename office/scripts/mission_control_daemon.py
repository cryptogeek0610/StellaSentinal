import json
import logging
import os
import tempfile
from pathlib import Path

WORKSPACE = Path(os.environ.get("STELLA_WORKSPACE", Path(__file__).resolve().parent.parent.parent))
WORKFORCE_PATH = WORKSPACE / "office" / "mission-control" / "workforce.json"

logger = logging.getLogger("scranton.mission_control")

DEFAULT_STATE = {
    "squad_name": "Scranton Digital",
    "lead_agent": "Michael",
    "agents": [],
    "kanban": [],
    "watercooler": [],
}


class MissionControl:
    """
    The Brain of Scranton Digital Workforce.
    Coordinates specialists, manages the Kanban, and runs the Watercooler.
    """
    def __init__(self):
        self.state = self.load_state()

    def load_state(self):
        try:
            data = json.loads(WORKFORCE_PATH.read_text(encoding="utf-8"))
            logger.info("State loaded from %s", WORKFORCE_PATH)
            return data
        except FileNotFoundError:
            logger.warning("State file not found at %s, using defaults.", WORKFORCE_PATH)
            return dict(DEFAULT_STATE)
        except json.JSONDecodeError as e:
            logger.error("Corrupt state file at %s: %s. Using defaults.", WORKFORCE_PATH, e)
            return dict(DEFAULT_STATE)
        except PermissionError:
            logger.error("Permission denied reading %s. Using defaults.", WORKFORCE_PATH)
            return dict(DEFAULT_STATE)

    def save_state(self):
        """Atomic write: write to temp file, then rename into place."""
        try:
            WORKFORCE_PATH.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                dir=WORKFORCE_PATH.parent, suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self.state, f, indent=2)
                os.replace(tmp_path, WORKFORCE_PATH)
                logger.info("State saved to %s", WORKFORCE_PATH)
            except BaseException:
                os.unlink(tmp_path)
                raise
        except OSError as e:
            logger.error("Failed to save state: %s", e)

    def run_15m_pulse(self):
        """All agents scan the board and collaborate."""
        logger.info("Mission Control Pulse: Syncing squad...")
        self.save_state()


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("STELLA_LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    mc = MissionControl()
    mc.run_15m_pulse()
