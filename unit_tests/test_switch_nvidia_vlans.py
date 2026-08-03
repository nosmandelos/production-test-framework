# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from production_test_framework.switch.exceptions import SwitchAPIError
from production_test_framework.switch.models import NetworkSwitchConfig
from production_test_framework.switch.nvidia.nvidia_cumulus_switch import NvidiaCumulusSwitch
from production_test_framework.switch.nvidia.nvue_paths import BRIDGE_DOMAIN_VLANS_PATH

FIXTURES = Path(__file__).parent / "fixtures" / "switch" / "nvidia"


@pytest.fixture
def switch_config() -> NetworkSwitchConfig:
    return NetworkSwitchConfig(
        host="10.0.0.1",
        username="admin",
        password="secret",
        verify_tls=False,
        port=8765,
        bridge_domain="br_default",
    )


@pytest.fixture
def vlan_configs() -> dict:
    return json.loads((FIXTURES / "bridge_vlans.json").read_text())


@pytest.fixture
def interfaces_bridge() -> dict:
    return json.loads((FIXTURES / "interfaces_bridge.json").read_text())


def test_vlan_membership_from_interfaces(switch_config: NetworkSwitchConfig, interfaces_bridge: dict) -> None:
    switch = NvidiaCumulusSwitch(switch_config)
    switch._interfaces_applied_cache = interfaces_bridge
    membership = switch._vlan_membership_by_id()

    assert membership["10"] == ["swp1", "swp2"]
    assert membership["20"] == ["swp2", "swp3"]


def test_refresh_clears_caches(switch_config: NetworkSwitchConfig, interfaces_bridge: dict) -> None:
    switch = NvidiaCumulusSwitch(switch_config)
    switch._interfaces_config_cache = interfaces_bridge
    switch._interfaces_applied_cache = interfaces_bridge

    switch.refresh()

    assert switch._interfaces_config_cache is None
    assert switch._interfaces_applied_cache is None


def test_parse_vlans(switch_config: NetworkSwitchConfig, vlan_configs: dict, interfaces_bridge: dict) -> None:
    switch = NvidiaCumulusSwitch(switch_config)
    switch._interfaces_applied_cache = interfaces_bridge
    switch._interfaces_config_cache = interfaces_bridge
    vlans = switch._parse_vlans(vlan_configs, switch._vlan_membership_by_id())

    assert [vlan.id for vlan in vlans] == ["10", "20", "100"]
    vlan10 = next(vlan for vlan in vlans if vlan.id == "10")
    assert [port.id for port in vlan10.ports] == ["swp1", "swp2"]
    assert vlan10.ports[0].description == "server-1"


@patch("production_test_framework.switch.nvidia.nvidia_cumulus_switch.requests.Session.get")
def test_vlans_api_calls(
    mock_get: MagicMock,
    switch_config: NetworkSwitchConfig,
    vlan_configs: dict,
    interfaces_bridge: dict,
) -> None:
    vlan_response = MagicMock(status_code=200)
    vlan_response.json.return_value = vlan_configs
    applied_response = MagicMock(status_code=200)
    applied_response.json.return_value = interfaces_bridge
    oper_response = MagicMock(status_code=200)
    oper_response.json.return_value = interfaces_bridge
    # 1) bridge VLANs, 2) applied interfaces (membership), 3) operational interfaces (port details)
    mock_get.side_effect = [vlan_response, applied_response, oper_response]

    switch = NvidiaCumulusSwitch(switch_config)
    vlans = switch.vlans

    assert len(vlans) == 3
    assert mock_get.call_count == 3
    urls = [call.args[0] for call in mock_get.call_args_list]
    assert urls[0] == f"https://10.0.0.1:8765/nvue_v1{BRIDGE_DOMAIN_VLANS_PATH}"
    assert urls[1] == "https://10.0.0.1:8765/nvue_v1/interface"
    assert urls[2] == "https://10.0.0.1:8765/nvue_v1/interface"
    assert mock_get.call_args_list[0].kwargs["params"] is None
    assert mock_get.call_args_list[1].kwargs["params"] == {"rev": "applied"}
    assert mock_get.call_args_list[2].kwargs["params"] is None


@patch("production_test_framework.switch.nvidia.nvidia_cumulus_switch.requests.Session.get")
def test_vlan_missing_raises(
    mock_get: MagicMock,
    switch_config: NetworkSwitchConfig,
    vlan_configs: dict,
) -> None:
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = vlan_configs
    mock_get.return_value = mock_response

    switch = NvidiaCumulusSwitch(switch_config)
    with pytest.raises(SwitchAPIError, match="VLAN 999 not found"):
        switch.vlan("999")
