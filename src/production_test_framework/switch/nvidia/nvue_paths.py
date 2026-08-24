# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""Composable NVUE URL path segments (Cumulus Linux 5.12 OpenAPI)."""

from urllib.parse import quote

BRIDGE_DOMAIN = "br_default"

SYSTEM_PATH = "/system"
PLATFORM_PATH = "/platform"
FIRMWARE_PATH = "/platform/firmware"
INTERFACES_PATH = "/interface"
REVISION_PATH = "/revision"

BRIDGE_DOMAIN_PATH = f"/bridge/domain/{BRIDGE_DOMAIN}"
BRIDGE_DOMAIN_VLANS_PATH = f"{BRIDGE_DOMAIN_PATH}/vlan"
BRIDGE_DOMAIN_MAC_TABLE_PATH = f"{BRIDGE_DOMAIN_PATH}/mac-table"


def interface_path(interface_id: str) -> str:
    return f"{INTERFACES_PATH}/{interface_id}"


def bridge_domain_vlan_path(vlan_id: str) -> str:
    return f"{BRIDGE_DOMAIN_VLANS_PATH}/{vlan_id}"


def interface_bridge_vlan_path(interface_id: str, vlan_id: str) -> str:
    """Path to a single VLAN's membership on an interface's bridge domain."""
    return f"{interface_path(interface_id)}/bridge/domain/{BRIDGE_DOMAIN}/vlan/{vlan_id}"


def revision_path(revision_id: str) -> str:
    """Path to a single changeset; the id contains slashes, so URL-encode it."""
    return f"{REVISION_PATH}/{quote(revision_id, safe='')}"
