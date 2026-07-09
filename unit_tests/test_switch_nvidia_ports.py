# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from production_test_framework.switch.models import NetworkSwitchConfig
from production_test_framework.switch.nvidia.nvidia_cumulus_switch import VIEW_DESCRIPTION, NvidiaCumulusSwitch
from production_test_framework.switch.nvidia.nvue_paths import INTERFACES_PATH, interface_path

FIXTURES = Path(__file__).parent / "fixtures" / "switch"


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
def interfaces_payload() -> dict:
    return json.loads((FIXTURES / "nvue_interfaces_description.json").read_text())


def test_parse_ports_from_fixture(interfaces_payload: dict) -> None:
    switch = NvidiaCumulusSwitch(NetworkSwitchConfig(host="h", username="u", password="p", verify_tls=False))
    ports = switch._parse_ports(interfaces_payload)

    assert [port.id for port in ports] == ["eth0", "lo", "swp1", "swp2"]
    swp1 = next(port for port in ports if port.id == "swp1")
    assert swp1.admin_up is True
    assert swp1.oper_up is True
    assert swp1.description == "server-1"

    swp2 = next(port for port in ports if port.id == "swp2")
    assert swp2.admin_up is False
    assert swp2.oper_up is False
    assert swp2.description is None

    lo = next(port for port in ports if port.id == "lo")
    assert lo.admin_up is True
    assert lo.oper_up is None


@patch("production_test_framework.switch.nvidia.nvidia_cumulus_switch.requests.Session.get")
def test_ports_calls_get_interfaces(
    mock_get: MagicMock,
    switch_config: NetworkSwitchConfig,
    interfaces_payload: dict,
) -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = interfaces_payload
    mock_get.return_value = mock_response

    switch = NvidiaCumulusSwitch(switch_config)
    ports = switch.ports

    assert len(ports) == 4
    mock_get.assert_called_once()
    call_kwargs = mock_get.call_args.kwargs
    assert call_kwargs["params"] == {"view": VIEW_DESCRIPTION}
    assert mock_get.call_args.args[0] == f"https://10.0.0.1:8765/nvue_v1{INTERFACES_PATH}"


@patch("production_test_framework.switch.nvidia.nvidia_cumulus_switch.requests.Session.get")
def test_port_calls_get_interface(
    mock_get: MagicMock,
    switch_config: NetworkSwitchConfig,
    interfaces_payload: dict,
) -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = interfaces_payload["swp1"]
    mock_get.return_value = mock_response

    switch = NvidiaCumulusSwitch(switch_config)
    port = switch.port("swp1")

    assert port.id == "swp1"
    assert port.description == "server-1"
    assert mock_get.call_args.args[0] == f"https://10.0.0.1:8765/nvue_v1{interface_path('swp1')}"
    # port() fetches the full interface, no view filter
    assert mock_get.call_args.kwargs["params"] is None
