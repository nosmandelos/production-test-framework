# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

from unittest.mock import MagicMock

import pytest

from production_test_framework.switch.arista.arista_eos_switch import AristaEosSwitch
from production_test_framework.switch.models import NetworkSwitchConfig


def _switch() -> AristaEosSwitch:
    switch = AristaEosSwitch(NetworkSwitchConfig(host="h", username="u", password="p", verify_tls=False))
    switch._node = MagicMock()
    return switch


@pytest.mark.parametrize(
    ("up", "expected_command"),
    [
        (True, "no shutdown"),
        (False, "shutdown"),
    ],
)
def test_set_port_admin_state(up: bool, expected_command: str) -> None:
    switch = _switch()
    switch.set_port_admin_state("Ethernet1", up=up)
    switch._node.config.assert_called_once_with(["interface Ethernet1", expected_command])
