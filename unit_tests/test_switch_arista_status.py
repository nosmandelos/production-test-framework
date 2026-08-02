# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

from unittest.mock import MagicMock, patch

from production_test_framework.switch.arista.arista_eos_switch import AristaEosSwitch
from production_test_framework.switch.models import NetworkSwitchConfig

MODULE = "production_test_framework.switch.arista.arista_eos_switch"


def _switch() -> AristaEosSwitch:
    switch = AristaEosSwitch(NetworkSwitchConfig(host="h", username="u", password="p", verify_tls=False))
    switch._node = MagicMock()
    return switch


def test_status_runs_version_and_hostname() -> None:
    switch = _switch()
    switch._node.run_commands.side_effect = [
        [{"version": "4.30.1F", "modelName": "DCS-7050", "serialNumber": "S1", "uptime": 5.0}],
        [{"hostname": "sw1"}],
    ]
    status = switch.status

    assert status.hostname == "sw1"
    assert status.model == "DCS-7050"
    commands = [call.args[0] for call in switch._node.run_commands.call_args_list]
    assert commands == [["show version"], ["show hostname"]]


def test_parse_system_status_uses_uptime_directly() -> None:
    version = {
        "uptime": 12345.0,
        "modelName": "DCS-7050SX3",
        "serialNumber": "JPE00000001",
        "version": "4.30.1F",
    }
    status = _switch()._parse_system_status(version, {"hostname": "sw1", "fqdn": "sw1.lab"})

    assert status.uptime == 12345.0
    assert status.hostname == "sw1"
    assert status.model == "DCS-7050SX3"
    assert status.serial_number == "JPE00000001"
    assert status.firmware_version == "4.30.1F"
    assert status.software_version == "4.30.1F"
    assert status.asic_model is None


def test_parse_system_status_derives_uptime_from_bootup() -> None:
    with patch(f"{MODULE}.time.time", return_value=1000.0):
        status = _switch()._parse_system_status({"bootupTimestamp": 400.0, "version": "4.30.1F"}, {})
    assert status.uptime == 600.0


def test_parse_system_status_falls_back_to_fqdn() -> None:
    status = _switch()._parse_system_status({"version": "4.30.1F"}, {"fqdn": "sw1.lab"})
    assert status.hostname == "sw1.lab"
