# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

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
    return json.loads((FIXTURES / "interfaces_status.json").read_text())


@pytest.mark.parametrize(
    ("link_status", "expected"),
    [
        ("connected", True),
        ("notconnect", True),
        ("errdisabled", True),
        ("disabled", False),
        (None, None),
        (123, None),
    ],
)
def test_admin_up(link_status: object, expected: bool | None) -> None:
    assert AristaEosSwitch._admin_up(link_status) is expected


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"lineProtocolStatus": "up"}, True),
        ({"lineProtocolStatus": "down"}, False),
        ({"linkStatus": "connected"}, True),
        ({"linkStatus": "notconnect"}, False),
        ({}, None),
    ],
)
def test_oper_up(body: dict, expected: bool | None) -> None:
    assert AristaEosSwitch._oper_up(body) is expected


def test_lineprotocol_takes_precedence_over_linkstatus() -> None:
    # lineProtocolStatus is checked first, even when linkStatus disagrees.
    assert AristaEosSwitch._oper_up({"lineProtocolStatus": "up", "linkStatus": "notconnect"}) is True


def test_parse_ports_sorts_and_maps(interface_statuses: dict) -> None:
    ports = _switch()._parse_ports(interface_statuses["interfaceStatuses"])

    # sorted by natural port order: Ethernet1, Ethernet2, Ethernet10
    assert [p.id for p in ports] == ["Ethernet1", "Ethernet2", "Ethernet10"]

    eth1 = ports[0]
    assert eth1.admin_up is True
    assert eth1.oper_up is True
    assert eth1.description == "server-1"

    eth2 = ports[1]
    assert eth2.admin_up is False
    assert eth2.oper_up is False
    assert eth2.description is None  # "" normalized to None


def test_parse_ports_skips_non_dict_bodies() -> None:
    ports = _switch()._parse_ports({"Ethernet1": {"linkStatus": "connected"}, "Ethernet2": "bogus"})
    assert [p.id for p in ports] == ["Ethernet1"]


def test_ports_calls_show_interfaces_status(interface_statuses: dict) -> None:
    switch = _switch()
    switch._node.run_commands.return_value = [interface_statuses]
    ports = switch.ports

    assert len(ports) == 3
    switch._node.run_commands.assert_called_once_with(["show interfaces status"])


def test_port_returns_single_port() -> None:
    switch = _switch()
    switch._node.run_commands.return_value = [
        {"interfaceStatuses": {"Ethernet1": {"linkStatus": "connected", "description": "server-1"}}}
    ]
    port = switch.port("Ethernet1")

    assert port.id == "Ethernet1"
    assert port.description == "server-1"
    switch._node.run_commands.assert_called_once_with(["show interfaces Ethernet1 status"])


def test_port_missing_raises() -> None:
    switch = _switch()
    switch._node.run_commands.return_value = [{"interfaceStatuses": {}}]
    with pytest.raises(SwitchAPIError, match="Ethernet99 not found"):
        switch.port("Ethernet99")
