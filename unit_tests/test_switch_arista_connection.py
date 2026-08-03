# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

import ssl
from unittest.mock import MagicMock, patch

import pytest
from pyeapi.eapilib import CommandError, ConnectionError as EapiConnectionError

from production_test_framework.switch.arista.arista_eos_switch import AristaEosSwitch
from production_test_framework.switch.exceptions import SwitchAPIError
from production_test_framework.switch.models import NetworkSwitchConfig

MODULE = "production_test_framework.switch.arista.arista_eos_switch"


def _switch() -> AristaEosSwitch:
    switch = AristaEosSwitch(NetworkSwitchConfig(host="h", username="u", password="p", verify_tls=False))
    switch._node = MagicMock()
    return switch


# --- construction / TLS ------------------------------------------------------


@pytest.mark.parametrize(
    ("verify_tls", "check_hostname", "verify_mode"),
    [
        (True, True, ssl.CERT_REQUIRED),
        (False, False, ssl.CERT_NONE),
    ],
)
def test_ssl_context_honours_verify_tls(verify_tls: bool, check_hostname: bool, verify_mode: ssl.VerifyMode) -> None:
    config = NetworkSwitchConfig(host="h", username="u", password="p", verify_tls=verify_tls, port=443)
    with patch(f"{MODULE}.pyeapi.connect", return_value=MagicMock()) as mock_connect, patch(
        f"{MODULE}.pyeapi.client.Node", return_value=MagicMock()
    ):
        AristaEosSwitch(config)

    kwargs = mock_connect.call_args.kwargs
    assert kwargs["transport"] == "https"
    assert kwargs["host"] == "h"
    assert kwargs["port"] == 443
    assert kwargs["username"] == "u"
    assert kwargs["password"] == "p"
    context = kwargs["context"]
    assert isinstance(context, ssl.SSLContext)
    assert context.check_hostname is check_hostname
    assert context.verify_mode == verify_mode


# --- eAPI error handling -----------------------------------------------------


def test_run_show_wraps_command_error() -> None:
    switch = _switch()
    switch._node.run_commands.side_effect = CommandError(1002, "boom")
    with pytest.raises(SwitchAPIError, match="eAPI command failed"):
        switch._run_show("show version")


def test_run_show_wraps_connection_error() -> None:
    switch = _switch()
    switch._node.run_commands.side_effect = EapiConnectionError("https", "unreachable")
    with pytest.raises(SwitchAPIError, match="eAPI command failed"):
        switch._run_show("show version")


def test_run_show_rejects_malformed_response() -> None:
    switch = _switch()
    switch._node.run_commands.return_value = []
    with pytest.raises(SwitchAPIError, match="unexpected eAPI response"):
        switch._run_show("show version")


def test_run_config_wraps_errors() -> None:
    switch = _switch()
    switch._node.config.side_effect = CommandError(1002, "boom")
    with pytest.raises(SwitchAPIError, match="eAPI config failed"):
        switch._run_config(["interface Ethernet1", "shutdown"])
