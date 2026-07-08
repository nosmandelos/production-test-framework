# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

from unittest.mock import MagicMock

import pytest

from production_test_framework.switch.exceptions import SwitchAPIError
from production_test_framework.switch.models import NetworkSwitchConfig
from production_test_framework.switch.nvidia.nvidia_cumulus_switch import NvidiaCumulusSwitch
from production_test_framework.switch.nvidia.nvue_paths import interface_path

REVISION_ID = "changeset/cumulus/2025-01-01_00.00.00_ABCD"


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


def _mock_session(switch: NvidiaCumulusSwitch, request_responses: list, get_response: MagicMock) -> tuple:
    """Replace the switch's requests.Session with mocks for request() and get()."""
    mock_request = MagicMock(side_effect=request_responses)
    mock_get = MagicMock(return_value=get_response)
    switch._session.request = mock_request
    switch._session.get = mock_get
    return mock_request, mock_get


def test_set_port_admin_state_down_runs_changeset_flow(switch_config: NetworkSwitchConfig) -> None:
    switch = NvidiaCumulusSwitch(switch_config)
    # POST /revision, PATCH interface, PATCH revision (apply)
    mock_request, _ = _mock_session(
        switch,
        request_responses=[
            _response(200, {REVISION_ID: {"state": ""}}),
            _response(200, {}),
            _response(200, {"state": "apply"}),
        ],
        # GET /revision poll -> applied
        get_response=_response(200, {"state": "applied"}),
    )

    switch.set_port_admin_state("swp1", up=False)

    methods = [call.args[0] for call in mock_request.call_args_list]
    assert methods == ["POST", "PATCH", "PATCH"]

    base = "https://10.0.0.1:8765/nvue_v1"
    # POST creates the changeset
    assert mock_request.call_args_list[0].args[1] == f"{base}/revision"
    # PATCH targets the interface within the changeset with a link-down body
    patch_iface = mock_request.call_args_list[1]
    assert patch_iface.args[1] == f"{base}{interface_path('swp1')}"
    assert patch_iface.kwargs["params"] == {"rev": REVISION_ID}
    assert patch_iface.kwargs["json"] == {"link": {"state": {"down": {}}}}
    # PATCH applies the changeset
    assert mock_request.call_args_list[2].kwargs["json"]["state"] == "apply"


def test_set_port_admin_state_up_sends_up_body(switch_config: NetworkSwitchConfig) -> None:
    switch = NvidiaCumulusSwitch(switch_config)
    mock_request, _ = _mock_session(
        switch,
        request_responses=[
            _response(200, {REVISION_ID: {"state": ""}}),
            _response(200, {}),
            _response(200, {"state": "apply"}),
        ],
        get_response=_response(200, {"state": "applied"}),
    )

    switch.set_port_admin_state("swp1", up=True)

    assert mock_request.call_args_list[1].kwargs["json"] == {"link": {"state": {"up": {}}}}


def test_set_port_admin_state_raises_on_apply_error(switch_config: NetworkSwitchConfig) -> None:
    switch = NvidiaCumulusSwitch(switch_config)
    _mock_session(
        switch,
        request_responses=[
            _response(200, {REVISION_ID: {"state": ""}}),
            _response(200, {}),
            _response(200, {"state": "apply"}),
        ],
        get_response=_response(200, {"state": "apply_error"}),
    )

    with pytest.raises(SwitchAPIError):
        switch.set_port_admin_state("swp1", up=False)
