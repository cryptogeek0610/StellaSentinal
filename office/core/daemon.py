import time
import signal
import subprocess
import logging
import os
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(os.environ.get("STELLA_WORKSPACE", Path(__file__).resolve().parent.parent.parent))
STATE_FILE = WORKSPACE / "office" / "state.json"

SUBPROCESS_TIMEOUT = int(os.environ.get("STELLA_SUBPROCESS_TIMEOUT", "120"))

logger = logging.getLogger("scranton.daemon")


class ScrantonCore:
    """
    Unified Background Daemon for Scranton Digital Workforce.
    Replaces session-dependent tasks with persistent OS-level monitoring.
    """
    def __init__(self):
        self.pulse_interval = int(os.environ.get("STELLA_PULSE_INTERVAL", "60"))
        self.running = True
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        logger.info("Shutdown signal received (signal=%s). Stopping daemon.", signum)
        self.running = False

    def run(self):
        logger.info("Scranton Core Daemon started (pulse=%ds).", self.pulse_interval)
        while self.running:
            try:
                self.sync_fleet_health()
                self.audit_worker_liveness()
            except Exception:
                logger.exception("Daemon pulse error")
            time.sleep(self.pulse_interval)
        logger.info("Scranton Core Daemon stopped.")

    def sync_fleet_health(self):
        """Trigger the SAAP Bridge autonomously."""
        script = WORKSPACE / "projects" / "SOTI-Advanced-Analytics-Plus" / "backend" / "bridge.py"
        self._run_script(script, label="sync_fleet_health")

    def audit_worker_liveness(self):
        """Verify worker processes and clean up zombies."""
        script = WORKSPACE / "office" / "scripts" / "verify_worker_liveness.py"
        self._run_script(script, label="audit_worker_liveness")

    def _run_script(self, script_path, label="subprocess"):
        if not script_path.exists():
            logger.warning("[%s] Script not found, skipping: %s", label, script_path)
            return
        try:
            result = subprocess.run(
                ["python3", str(script_path)],
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT,
            )
            if result.returncode != 0:
                logger.error(
                    "[%s] Script exited with code %d.\nstdout: %s\nstderr: %s",
                    label, result.returncode,
                    result.stdout[:500], result.stderr[:500],
                )
            else:
                logger.debug("[%s] Script completed successfully.", label)
        except subprocess.TimeoutExpired:
            logger.error("[%s] Script timed out after %ds: %s", label, SUBPROCESS_TIMEOUT, script_path)
        except OSError as e:
            logger.error("[%s] Failed to execute script: %s", label, e)


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("STELLA_LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    daemon = ScrantonCore()
    daemon.run()
