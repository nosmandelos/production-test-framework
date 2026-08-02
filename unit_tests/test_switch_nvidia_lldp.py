# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from production_test_framework.switch.models import NetworkSwitchConfig
from production_test_framework.switch.nvidia.nvidia_cumulus_switch import VIEW_LLDP_DETAIL, NvidiaCumulusSwitch
from production_test_framework.switch.nvidia.nvue_paths import INTERFACES_PATH

FIXTURES = Path(__file__).parent / "fixtures" / "switch" / "nvidia"


@pytest.fixture
def switch_config() -> NetworkSwitchConfig:
    return NetworkSwitchConfig(
        host="10.0.0.1",
        username="admin",
        password="secret",
        verify_tls=False,
        port=8765,
    )


@pytest.fixture
def lldp_payload() -> dict:
    return json.loads((FIXTURES / "interfaces_lldp_detail.json").read_text())


def _switch() -> NvidiaCumulusSwitch:
    return NvidiaCumulusSwitch(NetworkSwitchConfig(host="h", username="u", password="p", verify_tls=False))


def test_parse_lldp_neighbors_from_fixture(lldp_payload: dict) -> None:
    neighbors = _switch()._parse_lldp_neighbors(lldp_payload)

    # eth0 (non-swp), swp2 (type != mac), swp4 (no neighbor), lo (empty) are all dropped.
    assert [(n.switch_port, n.chassis_mac) for n in neighbors] == [
        (1, "aa:bb:cc:11:22:33"),
        (3, "44:55:66:77:aa:bb"),
    ]
    assert neighbors[0].interface == "swp1"
    assert neighbors[1].interface == "swp3s0"


@pytest.mark.parametrize(
    ("interface_id", "expected"),
    [
        ("swp1", 1),
        ("swp14", 14),
        ("swp1s0", 1),
        ("swp2/3", 2),
        ("eth0", -1),
        ("lo", -1),
        ("swpx", -1),
    ],
)
def test_interface_name_to_port(interface_id: str, expected: int) -> None:
    assert NvidiaCumulusSwitch._interface_name_to_port(interface_id) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("AA:BB:CC:11:22:33", "aa:bb:cc:11:22:33"),
        ("aabbcc112233", "aa:bb:cc:11:22:33"),
        ("AABBCC112233", "aa:bb:cc:11:22:33"),
        ("aa:bb:cc:11:22", ""),
        ("", ""),
    ],
)
def test_normalize_mac(raw: str, expected: str) -> None:
    assert NvidiaCumulusSwitch._normalize_mac(raw) == expected


@patch("production_test_framework.switch.nvidia.nvidia_cumulus_switch.requests.Session.get")
def test_lldp_neighbors_calls_get_interfaces_lldp_detail(
    mock_get: MagicMock,
    switch_config: NetworkSwitchConfig,
    lldp_payload: dict,
) -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = lldp_payload
    mock_get.return_value = mock_response

    switch = NvidiaCumulusSwitch(switch_config)
    neighbors = switch.lldp_neighbors

    assert len(neighbors) == 2
    mock_get.assert_called_once()
    call_kwargs = mock_get.call_args.kwargs
    assert call_kwargs["params"] == {"view": VIEW_LLDP_DETAIL}
    assert mock_get.call_args.args[0] == f"https://10.0.0.1:8765/nvue_v1{INTERFACES_PATH}"
