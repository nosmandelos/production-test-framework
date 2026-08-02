# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

from unittest.mock import MagicMock

from production_test_framework.switch.arista.arista_eos_switch import AristaEosSwitch
from production_test_framework.switch.models import NetworkSwitchConfig


def _switch() -> AristaEosSwitch:
    switch = AristaEosSwitch(NetworkSwitchConfig(host="h", username="u", password="p", verify_tls=False))
    switch._node = MagicMock()
    return switch


def test_delete_vlan() -> None:
    switch = _switch()
    switch.delete_vlan("10")
    switch._node.config.assert_called_once_with(["no vlan 10"])
