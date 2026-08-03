# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from production_test_framework.switch.arista.arista_eos_switch import AristaEosSwitch
from production_test_framework.switch.models import NetworkSwitchConfig

FIXTURES = Path(__file__).parent / "fixtures" / "switch" / "arista"


def _switch() -> AristaEosSwitch:
    switch = AristaEosSwitch(NetworkSwitchConfig(host="h", username="u", password="p", verify_tls=False))
    switch._node = MagicMock()
    return switch


@pytest.fixture
def lldp_payload() -> dict:
    return json.loads((FIXTURES / "lldp_neighbors_detail.json").read_text())


def test_parse_lldp_neighbors(lldp_payload: dict) -> None:
    neighbors = _switch()._parse_lldp_neighbors(lldp_payload["lldpNeighbors"])

    # Ethernet1/3 have MAC port ids; Ethernet5 (interface-name port id) and
    # Management1 (not a front-panel port) are dropped. Sorted by switch_port.
    assert [(n.switch_port, n.interface, n.chassis_mac) for n in neighbors] == [
        (1, "Ethernet1", "00:50:56:00:00:11"),
        (3, "Ethernet3", "aa:bb:cc:00:00:22"),
    ]


def test_lldp_neighbors_only_management_neighbor_is_empty() -> None:
    # Mirrors a real switch where the sole neighbor is learned on Management1 and
    # advertises an interface-name port id: not front-panel and not a MAC -> empty.
    payload = {
        "lldpNeighbors": {
            "Ethernet1": {"lldpNeighborInfo": []},
            "Ethernet49/1": {"lldpNeighborInfo": []},
            "Management1": {
                "lldpNeighborInfo": [
                    {
                        "chassisIdType": "macAddress",
                        "chassisId": "001c.7361.af79",
                        "neighborInterfaceInfo": {"interfaceIdType": "interfaceName", "interfaceId": '"Ethernet19"'},
                    }
                ]
            },
        }
    }
    assert _switch()._parse_lldp_neighbors(payload["lldpNeighbors"]) == []


def test_lldp_neighbors_calls_show_lldp_detail(lldp_payload: dict) -> None:
    switch = _switch()
    switch._node.run_commands.return_value = [lldp_payload]
    neighbors = switch.lldp_neighbors

    assert len(neighbors) == 2
    switch._node.run_commands.assert_called_once_with(["show lldp neighbors detail"])


@pytest.mark.parametrize(
    ("interface_id", "expected"),
    [
        ("Ethernet1", 1),
        ("Ethernet14", 14),
        ("Ethernet3/1", 3),
        ("Ethernet49/1", 49),  # breakout interface
        ("Et14", 14),  # abbreviated form
        ("Et50/4", 50),
        ("Management1", -1),
        ("Vlan10", -1),
        ("Ethernet", -1),
    ],
)
def test_interface_name_to_port(interface_id: str, expected: int) -> None:
    assert AristaEosSwitch._interface_name_to_port(interface_id) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("001c.7300.abcd", "00:1c:73:00:ab:cd"),
        ("00:1C:73:00:AB:CD", "00:1c:73:00:ab:cd"),
        ("001c7300abcd", "00:1c:73:00:ab:cd"),
        ("not-a-mac", ""),
        ("", ""),
    ],
)
def test_normalize_mac(raw: str, expected: str) -> None:
    assert AristaEosSwitch._normalize_mac(raw) == expected
