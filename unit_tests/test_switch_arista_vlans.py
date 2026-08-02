# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pyeapi.eapilib import CommandError

from production_test_framework.switch.arista.arista_eos_switch import AristaEosSwitch
from production_test_framework.switch.exceptions import SwitchAPIError
from production_test_framework.switch.models import NetworkSwitchConfig

FIXTURES = Path(__file__).parent / "fixtures" / "switch" / "arista"


def _switch() -> AristaEosSwitch:
    switch = AristaEosSwitch(NetworkSwitchConfig(host="h", username="u", password="p", verify_tls=False))
    switch._node = MagicMock()
    return switch


@pytest.fixture
def interface_statuses() -> dict:
    return json.loads((FIXTURES / "interfaces_status.json").read_text())["interfaceStatuses"]


@pytest.fixture
def vlan_configs() -> dict:
    return json.loads((FIXTURES / "show_vlan.json").read_text())["vlans"]


def test_parse_vlans_sorts_numerically_with_members(vlan_configs: dict, interface_statuses: dict) -> None:
    vlans = _switch()._parse_vlans(vlan_configs, interface_statuses)

    assert [v.id for v in vlans] == ["2", "10"]
    vlan10 = next(v for v in vlans if v.id == "10")
    # members sorted by port order and resolved to Port objects
    assert [p.id for p in vlan10.ports] == ["Ethernet1", "Ethernet2"]
    assert vlan10.ports[0].description == "server-1"


def test_vlan_returns_members() -> None:
    switch = _switch()
    switch._node.run_commands.side_effect = [
        [{"vlans": {"10": {"interfaces": {"Ethernet1": {}}}}}],
        [{"interfaceStatuses": {"Ethernet1": {"linkStatus": "connected", "description": "server-1"}}}],
    ]
    vlan = switch.vlan("10")

    assert vlan.id == "10"
    assert [p.id for p in vlan.ports] == ["Ethernet1"]


def test_vlan_missing_raises() -> None:
    switch = _switch()
    switch._node.run_commands.return_value = [{"vlans": {}}]
    with pytest.raises(SwitchAPIError, match="VLAN 99 not found"):
        switch.vlan("99")


def test_vlan_command_error_becomes_not_found() -> None:
    switch = _switch()
    switch._node.run_commands.side_effect = CommandError(1002, "invalid vlan")
    with pytest.raises(SwitchAPIError, match="VLAN 99 not found"):
        switch.vlan("99")
