# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

from unittest.mock import MagicMock

from production_test_framework.switch.models import NetworkSwitchConfig
from production_test_framework.switch.nvidia.nvidia_cumulus_switch import NvidiaCumulusSwitch

# NVUE bridge domain mac-table: a dict keyed by entry-id. Permanent = static; a learned
# entry omits "entry-type". Some entries carry no "vlan".
_MAC_TABLE = {
    "1": {"entry-type": "permanent", "interface": "br_default", "mac": "F0:BC:50:EB:28:F3", "vlan": 1},
    "10": {"entry-type": "permanent", "interface": "swp4s1", "mac": "f0:bc:50:eb:28:cc"},
    "40": {"interface": "swp13s0", "mac": "2c:9d:90:82:ef:63", "vlan": 1},
    "99": {"interface": "", "mac": "de:ad:be:ef:00:00", "vlan": 1},  # no interface -> dropped
}


def _switch() -> NvidiaCumulusSwitch:
    switch = NvidiaCumulusSwitch(NetworkSwitchConfig(host="h", username="u", password="p", verify_tls=False))
    switch._run_api_call = MagicMock(return_value=_MAC_TABLE)
    return switch


def test_parse_mac_table() -> None:
    entries = _switch()._parse_mac_table(_MAC_TABLE)
    # MAC lowercased; learned entry (no entry-type) -> static False; entry with no interface dropped.
    assert [(e.port, e.mac, e.vlan, e.static) for e in entries] == [
        ("br_default", "f0:bc:50:eb:28:f3", 1, True),
        ("swp4s1", "f0:bc:50:eb:28:cc", None, True),
        ("swp13s0", "2c:9d:90:82:ef:63", 1, False),
    ]


def test_mac_table_calls_nvue_path() -> None:
    switch = _switch()
    entries = switch.mac_table

    assert len(entries) == 3
    switch._run_api_call.assert_called_once_with("/bridge/domain/br_default/mac-table")
