# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

from unittest.mock import MagicMock

import pytest

from production_test_framework.switch.models import NetworkSwitchConfig
from production_test_framework.switch.nvidia.nvidia_cumulus_switch import NvidiaCumulusSwitch
from production_test_framework.switch.nvidia.nvue_paths import (
    BRIDGE_DOMAIN,
    bridge_domain_vlan_path,
    interface_bridge_vlan_path,
)

REVISION_ID = "changeset/cumulus/2025-01-01_00.00.00_ABCD"

INTERFACES_APPLIED = {
    "swp1": {"bridge": {"domain": {BRIDGE_DOMAIN: {"vlan": {"3000": {}}}}}},
    "swp2": {"bridge": {"domain": {BRIDGE_DOMAIN: {"vlan": {"3000": {}, "20": {}}}}}},
    "swp3": {"bridge": {"domain": {BRIDGE_DOMAIN: {"vlan": {"20": {}}}}}},
}


@pytest.fixture
def switch_config() -> NetworkSwitchConfig:
    return NetworkSwitchConfig(
        host="10.0.0.1",
        username="admin",
        password="secret",
        verify_tls=False,
        port=8765,
    )


def _response(status_code: int, payload: object) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = b"{}" if payload is not None else b""
    resp.json.return_value = payload
    return resp


def test_delete_vlan_removes_members_and_bridge_domain(switch_config: NetworkSwitchConfig) -> None:
    switch = NvidiaCumulusSwitch(switch_config)
    # GETs: 1) applied interfaces (membership), 2) revision apply poll -> applied
    mock_get = MagicMock(side_effect=[_response(200, INTERFACES_APPLIED), _response(200, {"state": "applied"})])
    # request(): POST /revision, DELETE swp1, DELETE swp2, DELETE bridge vlan, PATCH revision (apply)
    mock_request = MagicMock(
        side_effect=[
            _response(200, {REVISION_ID: {"state": ""}}),
            _response(204, None),
            _response(204, None),
            _response(204, None),
            _response(200, {"state": "apply"}),
        ]
    )
    switch._session.get = mock_get
    switch._session.request = mock_request

    switch.delete_vlan("3000")

    base = "https://10.0.0.1:8765/nvue_v1"
    methods = [call.args[0] for call in mock_request.call_args_list]
    assert methods == ["POST", "DELETE", "DELETE", "DELETE", "PATCH"]

    # Only the interfaces that reference VLAN 3000 have their membership deleted, each within the changeset.
    iface_deletes = mock_request.call_args_list[1:3]
    deleted = {call.args[1]: call.kwargs for call in iface_deletes}
    assert set(deleted) == {
        f"{base}{interface_bridge_vlan_path('swp1', '3000')}",
        f"{base}{interface_bridge_vlan_path('swp2', '3000')}",
    }
    for kwargs in deleted.values():
        assert kwargs["params"] == {"rev": REVISION_ID}

    # The bridge-domain VLAN definition is deleted in the same changeset.
    bridge_delete = mock_request.call_args_list[3]
    assert bridge_delete.args[1] == f"{base}{bridge_domain_vlan_path('3000')}"
    assert bridge_delete.kwargs["params"] == {"rev": REVISION_ID}

    # The changeset is applied.
    assert mock_request.call_args_list[4].kwargs["json"]["state"] == "apply"


def test_delete_vlan_with_no_members_only_removes_bridge_domain(switch_config: NetworkSwitchConfig) -> None:
    switch = NvidiaCumulusSwitch(switch_config)
    mock_get = MagicMock(side_effect=[_response(200, INTERFACES_APPLIED), _response(200, {"state": "applied"})])
    # request(): POST /revision, DELETE bridge vlan, PATCH revision (apply) -- no interface deletes
    mock_request = MagicMock(
        side_effect=[
            _response(200, {REVISION_ID: {"state": ""}}),
            _response(204, None),
            _response(200, {"state": "apply"}),
        ]
    )
    switch._session.get = mock_get
    switch._session.request = mock_request

    switch.delete_vlan("9999")

    base = "https://10.0.0.1:8765/nvue_v1"
    methods = [call.args[0] for call in mock_request.call_args_list]
    assert methods == ["POST", "DELETE", "PATCH"]
    assert mock_request.call_args_list[1].args[1] == f"{base}{bridge_domain_vlan_path('9999')}"
