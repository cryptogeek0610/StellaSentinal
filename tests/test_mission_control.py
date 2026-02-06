"""Unit tests for MissionControl."""
import json
import os
from unittest import mock

import pytest


class TestLoadState:
    def test_valid_json(self, tmp_workspace, workforce_path):
        with mock.patch(
            "office.scripts.mission_control_daemon.WORKFORCE_PATH", workforce_path
        ):
            from office.scripts.mission_control_daemon import MissionControl
            mc = MissionControl()
            assert mc.state["squad_name"] == "Test Squad"
            assert len(mc.state["agents"]) == 1

    def test_file_not_found(self, tmp_path):
        missing = tmp_path / "does_not_exist.json"
        with mock.patch(
            "office.scripts.mission_control_daemon.WORKFORCE_PATH", missing
        ):
            from office.scripts.mission_control_daemon import MissionControl
            mc = MissionControl()
            assert mc.state["squad_name"] == "Scranton Digital"
            assert mc.state["agents"] == []

    def test_invalid_json(self, workforce_path):
        workforce_path.write_text("NOT VALID JSON {{{", encoding="utf-8")
        with mock.patch(
            "office.scripts.mission_control_daemon.WORKFORCE_PATH", workforce_path
        ):
            from office.scripts.mission_control_daemon import MissionControl
            mc = MissionControl()
            assert mc.state["squad_name"] == "Scranton Digital"

    def test_empty_file(self, workforce_path):
        workforce_path.write_text("", encoding="utf-8")
        with mock.patch(
            "office.scripts.mission_control_daemon.WORKFORCE_PATH", workforce_path
        ):
            from office.scripts.mission_control_daemon import MissionControl
            mc = MissionControl()
            assert mc.state["squad_name"] == "Scranton Digital"


class TestSaveState:
    def test_round_trip(self, tmp_workspace, workforce_path):
        with mock.patch(
            "office.scripts.mission_control_daemon.WORKFORCE_PATH", workforce_path
        ):
            from office.scripts.mission_control_daemon import MissionControl
            mc = MissionControl()
            mc.state["kanban"].append({"task": "Test Task"})
            mc.save_state()

            reloaded = json.loads(workforce_path.read_text(encoding="utf-8"))
            assert len(reloaded["kanban"]) == 1
            assert reloaded["kanban"][0]["task"] == "Test Task"

    def test_save_preserves_structure(self, tmp_workspace, workforce_path):
        with mock.patch(
            "office.scripts.mission_control_daemon.WORKFORCE_PATH", workforce_path
        ):
            from office.scripts.mission_control_daemon import MissionControl
            mc = MissionControl()
            original_keys = set(mc.state.keys())
            mc.save_state()
            reloaded = json.loads(workforce_path.read_text(encoding="utf-8"))
            assert set(reloaded.keys()) == original_keys
