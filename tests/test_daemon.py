"""Unit tests for ScrantonCore daemon."""
import signal
from unittest import mock

import pytest


class TestDaemonInit:
    def test_default_pulse_interval(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            from office.core.daemon import ScrantonCore
            daemon = ScrantonCore()
            assert daemon.pulse_interval == 60
            assert daemon.running is True

    def test_custom_pulse_interval(self):
        with mock.patch.dict("os.environ", {"STELLA_PULSE_INTERVAL": "30"}):
            from office.core.daemon import ScrantonCore
            daemon = ScrantonCore()
            assert daemon.pulse_interval == 30


class TestShutdown:
    def test_sigterm_stops_daemon(self):
        from office.core.daemon import ScrantonCore
        daemon = ScrantonCore()
        assert daemon.running is True
        daemon._handle_shutdown(signal.SIGTERM, None)
        assert daemon.running is False


class TestRunScript:
    def test_missing_script_skips(self, tmp_path):
        from office.core.daemon import ScrantonCore
        daemon = ScrantonCore()
        missing = tmp_path / "does_not_exist.py"
        daemon._run_script(missing, label="test")
        # Should not raise

    def test_successful_script(self, tmp_path):
        script = tmp_path / "ok.py"
        script.write_text("print('hello')")
        from office.core.daemon import ScrantonCore
        daemon = ScrantonCore()
        daemon._run_script(script, label="test")

    def test_failing_script(self, tmp_path):
        script = tmp_path / "fail.py"
        script.write_text("import sys; sys.exit(1)")
        from office.core.daemon import ScrantonCore
        daemon = ScrantonCore()
        daemon._run_script(script, label="test")
        # Should not raise, error is logged


class TestRunLoop:
    def test_loop_stops_on_shutdown(self):
        from office.core.daemon import ScrantonCore
        daemon = ScrantonCore()

        call_count = 0
        original_sleep = None

        def fake_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                daemon.running = False

        with mock.patch("office.core.daemon.time.sleep", side_effect=fake_sleep):
            with mock.patch.object(daemon, "sync_fleet_health"):
                with mock.patch.object(daemon, "audit_worker_liveness"):
                    daemon.run()

        assert daemon.running is False
        assert call_count >= 2

    def test_loop_survives_exception(self):
        from office.core.daemon import ScrantonCore
        daemon = ScrantonCore()

        call_count = 0

        def fake_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                daemon.running = False

        with mock.patch("office.core.daemon.time.sleep", side_effect=fake_sleep):
            with mock.patch.object(
                daemon, "sync_fleet_health", side_effect=RuntimeError("boom")
            ):
                with mock.patch.object(daemon, "audit_worker_liveness"):
                    daemon.run()

        assert call_count >= 2
